"""Build and independently reload TrueType and WOFF2 artifacts."""

import io
import math
import re
import string
from pathlib import Path

import cv2
import numpy as np
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._k_e_r_n import KernTable_format_0
from PIL import Image, ImageDraw, ImageFont

from handwriting.imaging import ImageError, ink_mask, tight_crop
from handwriting.models import GlyphBox, PreviewRequest

ALIASES = {
    "\u00a0": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2212": "-",
}


def placement(character: str) -> tuple[int, int]:
    if character == "j":
        return 850, -200
    if character in "gpqy":
        return 680, -200
    if character in "bdfhkl":
        return 700, 0
    if character == "t":
        return 620, 0
    if character == "i":
        return 650, 0
    if character.islower():
        return 480, 0
    if character.isupper() or character.isdigit() or character == "₹":
        return 700, 0
    return {
        ".": (65, 0),
        ",": (110, -75),
        ":": (310, 100),
        ";": (400, -75),
        "'": (180, 500),
        '"': (180, 500),
        "-": (40, 250),
        "_": (35, -100),
        "(": (730, -80),
        ")": (730, -80),
        "!": (700, 0),
        "?": (700, 0),
        "+": (450, 120),
        "=": (200, 220),
    }.get(character, (650, 0))


def topology(mask: np.ndarray) -> tuple[int, int]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for node in hierarchy[0]:
            parent, depth = int(node[3]), 0
            while parent != -1:
                depth += 1
                parent = int(hierarchy[0][parent][3])
            holes += depth % 2
    return len(contours) - holes, holes


def balance_weight(mask: np.ndarray, character: str) -> tuple[np.ndarray, dict]:
    """Estimate stroke width on a morphological skeleton, then limit offsets.

    The working raster has two font units per pixel. Detached small marks are
    excluded from the weight pass. Topology must survive any proposed offset.
    """
    if not character.isalnum():
        return mask, {"weight_adjustment": 0}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    main = (labels == biggest).astype(np.uint8) * 255
    remaining = cv2.bitwise_and(mask, cv2.bitwise_not(main))
    skeleton, work = np.zeros_like(main), main.copy()
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, cross)
        opened = cv2.dilate(eroded, cross)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = eroded
    distance = cv2.distanceTransform(main, cv2.DIST_L2, 5)
    neighbors = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    samples = distance[(skeleton > 0) & (neighbors == 3)] * 4
    if len(samples) < 5:
        samples = distance[skeleton > 0] * 4
    lower, upper = np.percentile(samples, [15, 85])
    measured = float(np.median(samples[(samples >= lower) & (samples <= upper)]))
    target = (60 if character.islower() else 62) + 0.10 * (measured - 60)
    radius = int(np.clip(round((target - measured) / 4), -8, 8))
    original_topology = topology(mask)
    applied = 0
    for offset in range(abs(radius), 0, -1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * offset + 1, 2 * offset + 1))
        adjusted = cv2.dilate(main, kernel) if radius > 0 else cv2.erode(main, kernel)
        candidate = cv2.bitwise_or(adjusted, remaining)
        if topology(candidate) == original_topology:
            mask, applied = candidate, offset * (1 if radius > 0 else -1) * 2
            break
    return mask, {
        "measured_stroke": round(measured, 2),
        "target_stroke": round(target, 2),
        "weight_adjustment": applied,
    }


def trace_glyph(mask: np.ndarray, box: GlyphBox):
    height, baseline = placement(box.character)
    height = box.target_height if box.target_height is not None else height
    baseline = box.baseline if box.baseline is not None else baseline
    # Preserve aspect ratio. Bound pathological crops before allocating a raster.
    ys, xs = np.nonzero(mask)
    mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target_width = max(2, round(mask.shape[1] * height / mask.shape[0] / 2))
    if target_width > 1100:
        raise ImageError(f"The crop for {box.character!r} is too wide. Check the crop or height.")
    raster = cv2.resize(mask, (target_width, max(2, height // 2)), interpolation=cv2.INTER_CUBIC)
    raster = np.pad(raster, 12)
    smooth = cv2.GaussianBlur(raster, (3, 3), 0.65)
    raster = (smooth > 110).astype(np.uint8) * 255
    if not np.any(raster):
        raise ImageError(f"The crop for {box.character!r} is too thin to trace.")
    raster, measurements = balance_weight(raster, box.character)
    contours, _ = cv2.findContours(raster, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pen = TTGlyphPen(None)
    ys, xs = np.nonzero(raster)
    x_min, y_max = int(xs.min()), int(ys.max())
    for contour in contours:
        if cv2.contourArea(contour) < 0.5:
            continue
        points = cv2.approxPolyDP(contour, 0.45, True).reshape(-1, 2)[::-1]
        points = [
            (50 + (float(x) - x_min) * 2, baseline + (y_max - float(y)) * 2) for x, y in points
        ]
        if len(points) < 3:
            continue
        midpoint = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)  # noqa: E731
        pen.moveTo(midpoint(points[-1], points[0]))
        for index, point in enumerate(points):
            pen.qCurveTo(point, midpoint(point, points[(index + 1) % len(points)]))
        pen.closePath()
    glyph = pen.glyph()
    if glyph.numberOfContours == 0:
        raise ImageError(f"No outline survived tracing for {box.character!r}.")
    glyph.recalcBounds(None)
    advance = glyph.xMax + 25
    return glyph, (advance, glyph.xMin), measurements


def pair_kerning(font_path: Path, characters: list[str]) -> dict[tuple[str, str], int]:
    """Measure the rasterized vector outlines, with clearance on every common row."""
    font = ImageFont.truetype(str(font_path), 1000, layout_engine=ImageFont.Layout.BASIC)
    with TTFont(font_path) as tt:
        cmap = tt.getBestCmap()
        advances = {c: tt["hmtx"][cmap[ord(c)]][0] for c in characters}
        baseline = tt["hhea"].ascent + 24
        canvas_height = baseline + abs(tt["hhea"].descent) + 24
        canvas_width = max(advances.values(), default=64) + 64
    profiles = {}
    for character in characters:
        raster = Image.new("L", (canvas_width, canvas_height))
        ImageDraw.Draw(raster).text((0, baseline), character, font=font, fill=255, anchor="ls")
        pixels = np.asarray(raster) > 32
        exists = pixels.any(axis=1)
        left = np.argmax(pixels, axis=1).astype(float)
        right = (pixels.shape[1] - np.argmax(pixels[:, ::-1], axis=1)).astype(float)
        # Adjacent rows constrain clearance too. A half-resolution scan can miss
        # a one-unit overlap between the tips of two slanted strokes.
        safety = cv2.dilate((np.asarray(raster) > 8).astype(np.uint8), np.ones((3, 1), np.uint8))
        safety_exists = safety.any(axis=1)
        safety_left = np.argmax(safety, axis=1).astype(float)
        safety_right = (safety.shape[1] - np.argmax(safety[:, ::-1], axis=1)).astype(float)
        profiles[character] = (exists, left, right, safety_exists, safety_left, safety_right)
    positions = np.arange(canvas_height)
    reading_band = (positions >= baseline - 420) & (positions <= baseline - 35)
    result = {}
    for a in characters:
        for b in characters:
            a_rows, _, a_right, a_safety_rows, _, a_safety_right = profiles[a]
            b_rows, b_left, _, b_safety_rows, b_safety_left, _ = profiles[b]
            common = a_rows & b_rows
            band = common & reading_band
            if np.count_nonzero(band) < 3:
                continue
            gaps = advances[a] + b_left - a_right
            desired = 130 - float(np.percentile(gaps[band], 25))
            safety_common = a_safety_rows & b_safety_rows
            safety_gaps = advances[a] + b_safety_left - a_safety_right
            clearance = 56 - float(np.min(safety_gaps[safety_common]))
            # Round upward for clearance. Rounding to nearest could introduce overlap.
            value = int(math.ceil(max(desired, clearance, -280) / 2) * 2)
            value = min(value, 40)
            if abs(value) >= 6:
                result[(a, b)] = value
    return result


def build_font(
    source: Path, output: Path, boxes: list[dict], mode: str, family: str, unique_id: str
) -> dict:
    with Image.open(source) as image:
        mask, _ = ink_mask(image.convert("RGB"), mode)
    glyphs, metrics = {}, {}
    cmap = {32: "space"}
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((50, 700))
    pen.lineTo((450, 700))
    pen.lineTo((450, 0))
    pen.closePath()
    pen.moveTo((100, 50))
    pen.lineTo((400, 50))
    pen.lineTo((400, 650))
    pen.lineTo((100, 650))
    pen.closePath()
    glyphs[".notdef"], metrics[".notdef"] = pen.glyph(), (500, 50)
    glyphs["space"], metrics["space"] = TTGlyphPen(None).glyph(), (500, 0)
    measurements = {}
    duplicates = []
    for value in boxes:
        box = GlyphBox.model_validate(value)
        code = ord(box.character)
        if code in cmap:
            duplicates.append(box.character)
            continue
        glyph_name = f"uni{code:04X}"
        glyphs[glyph_name], metrics[glyph_name], measurements[box.character] = trace_glyph(
            tight_crop(mask, box), box
        )
        cmap[code] = glyph_name
    for alias, original in ALIASES.items():
        if ord(original) in cmap and ord(alias) not in cmap:
            cmap[ord(alias)] = cmap[ord(original)]
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(list(glyphs))
    builder.setupCharacterMap(cmap)
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(metrics)
    top = max(850, *(getattr(g, "yMax", 0) for g in glyphs.values()))
    bottom = min(-250, *(getattr(g, "yMin", 0) for g in glyphs.values()))
    builder.setupHorizontalHeader(ascent=top, descent=bottom, lineGap=100)
    builder.setupNameTable(
        {
            "familyName": family,
            "styleName": "Regular",
            "uniqueFontIdentifier": unique_id,
            "fullName": family,
            "psName": f"Handwriting-{unique_id}",
            "version": "Version 1.000",
        }
    )
    builder.setupOS2(
        version=4,
        sTypoAscender=top,
        sTypoDescender=bottom,
        sTypoLineGap=100,
        usWinAscent=top,
        usWinDescent=abs(bottom),
        fsSelection=0xC0,
        sxHeight=480,
        sCapHeight=700,
        usWeightClass=400,
    )
    builder.setupPost()
    ttf_path = output / "font.ttf"
    builder.save(ttf_path)
    characters = [c for c in string.ascii_letters + string.digits if ord(c) in cmap]
    pairs = pair_kerning(ttf_path, characters)
    if pairs:
        lines = [f"pos {cmap[ord(a)]} {cmap[ord(b)]} {value};" for (a, b), value in pairs.items()]
        features = "languagesystem DFLT dflt; languagesystem latn dflt; feature kern {\n"
        addOpenTypeFeaturesFromString(builder.font, features + "\n".join(lines) + "\n} kern;")
        kern = newTable("kern")
        kern.version = 0
        subtable = KernTable_format_0()
        subtable.version, subtable.coverage = 0, 1
        subtable.kernTable = {(cmap[ord(a)], cmap[ord(b)]): v for (a, b), v in pairs.items()}
        kern.kernTables = [subtable]
        builder.font["kern"] = kern
    builder.save(ttf_path)
    builder.font.flavor = "woff2"
    builder.save(output / "font.woff2")
    validate_exports(ttf_path, output / "font.woff2")
    render_preview(ttf_path, PreviewRequest()).save(output / "sample.png")
    (output / "font.css").write_text(
        f'@font-face {{ font-family: "{family}"; src: url("font.woff2") format("woff2"); '
        "font-style: normal; font-weight: 400; font-display: swap; }\n"
        f'.handwriting {{ font-family: "{family}", cursive; font-kerning: normal; '
        "letter-spacing: normal; word-spacing: normal; font-synthesis: none; "
        "text-align: left; font-size: 1.75rem; line-height: 1.7; }\n",
        encoding="utf-8",
    )
    return {
        "family": family,
        "supported_characters": "".join(chr(c) for c in sorted(cmap)),
        "kerning_pairs": len(pairs),
        "measurements": measurements,
        "duplicate_samples": duplicates,
        "units_per_em": 1000,
        "ascent": top,
        "descent": bottom,
        "line_gap": 100,
    }


def validate_exports(ttf_path: Path, woff_path: Path):
    """Reload exports before publication, or verify a downloaded font kit."""
    with TTFont(ttf_path) as ttf, TTFont(woff_path) as woff:
        if ttf.getBestCmap() != woff.getBestCmap():
            raise ImageError("The web font failed character mapping validation.")
        for name in ttf.getGlyphOrder():
            a = ttf["glyf"][name].getCoordinates(ttf["glyf"])
            b = woff["glyf"][name].getCoordinates(woff["glyf"])
            if a != b:
                raise ImageError("The web font failed outline validation.")


def render_preview(font_path: Path, request: PreviewRequest) -> Image.Image:
    """Render each glyph once; apply stored pair spacing once, without RAQM."""
    font = ImageFont.truetype(
        str(font_path), request.font_size, layout_engine=ImageFont.Layout.BASIC
    )
    with TTFont(font_path) as tt:
        cmap = tt.getBestCmap()
        metrics = dict(tt["hmtx"].metrics)
        kern = dict(tt["kern"].kernTables[0].kernTable) if "kern" in tt else {}
        ascent, descent = tt["hhea"].ascent, abs(tt["hhea"].descent)
    scale = request.font_size / 1000
    glyph_height = math.ceil((ascent + descent) * scale)
    line_height = max(math.ceil(request.font_size * 1.7), glyph_height + 8)
    lines: list[list[tuple[float, str]]] = [[]]
    x, previous = 24.0, None
    for token in re.findall(r"[^\s]+|[^\S\n]+|\n", request.text):
        if token == "\n":
            lines.append([])
            x, previous = 24.0, None
            continue
        token_width, last_name = 0.0, previous
        for char in token:
            name = cmap.get(ord(char), ".notdef")
            token_width += (metrics[name][0] + kern.get((last_name, name), 0)) * scale
            last_name = name
        if x + token_width > request.width - 24 and lines[-1]:
            lines.append([])
            x, previous = 24.0, None
            if token.isspace():
                continue
        for char in token:
            name = cmap.get(ord(char), ".notdef")
            adjustment = kern.get((previous, name), 0) * scale
            advance = metrics[name][0] * scale
            if x + adjustment + advance > request.width - 24 and lines[-1]:
                lines.append([])
                x, previous, adjustment = 24.0, None, 0
            x += adjustment
            lines[-1].append((x, char))
            x += advance
            previous = name
    height = 48 + glyph_height + (len(lines) - 1) * line_height
    if height > 4096:
        raise ImageError("Preview is too tall. Use less text, a wider canvas, or a smaller font.")
    image = Image.new("RGB", (request.width, height), "#fcfaf5")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        baseline = 24 + index * line_height + math.ceil(ascent * scale)
        for x, char in line:
            draw.text((x, baseline), char, font=font, fill="#243b55", anchor="ls")
    return image


def preview_bytes(font_path: Path, request: PreviewRequest) -> bytes:
    output = io.BytesIO()
    render_preview(font_path, request).save(output, format="PNG")
    return output.getvalue()

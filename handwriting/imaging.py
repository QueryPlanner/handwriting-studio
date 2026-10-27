"""Conservative ink extraction and layout-based proposals, without OCR."""

import io
import warnings
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from handwriting.config import Settings
from handwriting.models import GlyphBox
from handwriting.vision import VisionError


class ImageError(ValueError):
    """An image cannot be safely decoded or traced."""


def decode_image(data: bytes, settings: Settings) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG", "MPO", "WEBP"}:
                    raise ImageError("Use a JPEG, PNG, or WebP image.")
                # Some camera JPEGs carry an auxiliary gain/depth frame. Trace
                # the primary photograph instead of rejecting its MPO label.
                source.seek(0)
                is_jpeg = source.format in {"JPEG", "MPO"}
                limit = settings.max_jpeg_pixels if is_jpeg else settings.max_image_pixels
                if source.width * source.height > limit:
                    raise ImageError(
                        f"The decoded image exceeds the {limit / 1_000_000:g} megapixel limit."
                    )
                if source.width < 32 or source.height < 32:
                    raise ImageError("The image must be at least 32 pixels in each dimension.")
                if is_jpeg:
                    # The decoder scales by 1/2, 1/4 or 1/8 while reading, never below
                    # the requested size, so a 24 megapixel photo does not need 72 MB.
                    source.draft("RGB", (settings.max_image_side, settings.max_image_side))
                image = ImageOps.exif_transpose(source).convert("RGBA")
                white = Image.new("RGBA", image.size, "white")
                white.alpha_composite(image)
                image = white.convert("RGB")
                image.thumbnail((settings.max_image_side, settings.max_image_side))
                return image
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageError("The file is not a supported, readable image.") from exc


LINE_STEP = 4
LINE_BAND = 4


def _line_reach(
    mask: np.ndarray, x: float, y: float, slope: float, direction: int, reach: int
) -> tuple[float, float]:
    """Follow a ruled line from one detected end while ink still lies under it."""
    height, width = mask.shape
    start, gap = x, 0
    while 0 <= x + direction * LINE_STEP < width:
        x += direction * LINE_STEP
        row = round(y + slope * (x - start))
        if not 0 <= row < height:
            break
        hit = mask[max(0, row - LINE_BAND) : row + LINE_BAND + 1, int(x)].any()
        gap = 0 if hit else gap + LINE_STEP
        if gap > reach:
            x -= direction * gap
            break
    return x, y + slope * (x - start)


def remove_ruled_lines(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Erase long, nearly horizontal rules from a dark-ink mask.

    Rules are found geometrically, because black pen and printed lines share a
    colour. Ink inside the erased zone comes back where it is blue-shifted (ruled
    lines are warm or neutral) or where a stroke continues across the line.
    """
    height, width = mask.shape
    scale = min(1.0, 1200 / max(height, width))
    small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    segments = cv2.HoughLinesP(
        (small > 127).astype(np.uint8) * 255,
        1,
        np.pi / 720,
        threshold=int(0.18 * small.shape[1]),
        minLineLength=int(0.30 * small.shape[1]),
        maxLineGap=int(0.04 * small.shape[1]),
    )
    if segments is None:
        return mask
    zone = np.zeros_like(mask)
    thickness, reach = max(5, round(width / 260)), int(0.04 * width)
    for x0, y0, x1, y1 in segments.reshape(-1, 4) / scale:
        (x0, y0), (x1, y1) = sorted([(x0, y0), (x1, y1)])
        if x1 == x0 or abs(np.degrees(np.arctan2(y1 - y0, x1 - x0))) > 8:
            continue
        slope = (y1 - y0) / (x1 - x0)
        left = _line_reach(mask, x0, y0, slope, -1, reach)
        right = _line_reach(mask, x1, y1, slope, 1, reach)
        cv2.line(
            zone,
            (round(left[0]), round(left[1])),
            (round(right[0]), round(right[1])),
            255,
            thickness,
        )
    stripped = cv2.bitwise_and(mask, cv2.bitwise_not(zone))
    # A stroke that crossed a rule has a gap now. Close it where ink meets on both sides.
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (1, thickness * 2 + 1))
    bridged = cv2.morphologyEx(stripped, cv2.MORPH_CLOSE, bridge)
    blue_shifted = ((rgb[:, :, 2] - rgb[:, :, 0]) > 5).astype(np.uint8) * 255
    restored = stripped | ((bridged | blue_shifted) & zone & mask)
    # A curved rule can poke out of the zone as a hairline. Open only near rules so real
    # strokes elsewhere keep their shape.
    near = cv2.dilate(zone, np.ones((thickness * 2 + 1, 1), np.uint8))
    opened = cv2.morphologyEx(restored, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return np.where(near > 0, opened, restored).astype(np.uint8)


def ink_mask(
    image: Image.Image, mode: str = "auto", *, for_inference: bool = False
) -> tuple[np.ndarray, str]:
    rgb = np.asarray(image).astype(np.float32)
    red, green, blue = (rgb[:, :, i] for i in range(3))
    blue_pixels = (blue - red > 18) & (blue - green > 7) & (blue > 35)
    selected = "blue" if mode == "auto" and np.count_nonzero(blue_pixels) > 80 else mode
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 21)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=255)
    if selected == "blue":
        # Shadow can make blue ink nearly black. Only include strong local contrast.
        dark_fallback = (normalized < 155) & (gray < 115) if mode == "auto" else False
        mask = (blue_pixels | dark_fallback).astype(np.uint8) * 255
    else:
        selected = "dark"
        mask = ((normalized < 185) & (gray < 180)).astype(np.uint8) * 255
    mask = remove_ruled_lines(mask, rgb)
    # Ink must be surrounded by neutral or warm paper. This excludes blue fabric
    # and dark page bindings without assuming fixed coordinates for the sheet.
    if for_inference:
        paper = ((red >= blue * 0.97) & (red >= green * 0.95)).astype(np.float32)
        support_size = max(41, int(image.width * 0.07) | 1)
        paper_support = cv2.blur(paper, (support_size, support_size)) > 0.65
        mask[~paper_support] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    height, width = mask.shape
    minimum = max(2, width * height // 1_200_000)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < minimum:
            continue
        if for_inference:
            if area > width * height * 0.25:
                continue
            if x < width * 0.025 or x + w > width * 0.975:
                continue
            if y == 0 or y + h >= height:
                continue
        clean[labels == index] = 255
    return clean, selected


def tight_crop(mask: np.ndarray, box: GlyphBox) -> np.ndarray:
    height, width = mask.shape
    x0, y0 = int(box.x * width), int(box.y * height)
    x1 = min(width, max(x0 + 1, round((box.x + box.width) * width)))
    y1 = min(height, max(y0 + 1, round((box.y + box.height) * height)))
    crop = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(crop)
    if len(xs) < 3:
        raise ImageError(f"No ink was found in the crop for {box.character!r}.")
    return np.pad(crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], 3)


def layout_groups(
    mask: np.ndarray, rows: list[str], speck: float = 0.0
) -> tuple[list[list[list[int]]], float] | None:
    """Group ink into glyph-sized boxes per text row, in reading order.

    Returns None when the mask has no ink. `rows` only guides how many rows to expect.
    Marks with an area below (speck x typical glyph height) squared are dust and ignored.
    Each box is [x0, y0, x1, y1] in pixels, with the typical glyph height.
    """
    height, width = mask.shape
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    components = [
        (stats[i].tolist(), centers[i].tolist())
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= 3
    ]
    if not components:
        return None
    # Each component votes by area. The projection rewards aligned text rows.
    ys, xs = np.nonzero(mask[::3, ::3])
    best_score, slope = -1.0, 0.0
    for candidate in np.linspace(-0.22, 0.22, 89):
        transformed = np.round(ys - candidate * (xs - width / 6) + height / 6).astype(int)
        histogram = np.bincount(np.maximum(transformed, 0), minlength=height)
        smooth = cv2.GaussianBlur(histogram.astype(np.float32).reshape(-1, 1), (1, 15), 0)
        score = float(np.sum(smooth**2))
        if score > best_score:
            best_score, slope = score, float(candidate)
    # Group tall/main strokes first. Add detached dots after row estimation.
    typical_height = float(np.percentile([c[0][3] for c in components], 75))
    components = [c for c in components if c[0][4] >= (speck * typical_height) ** 2]
    main = [c for c in components if c[0][3] >= max(7, typical_height * 0.45)]
    bands: list[list] = []
    for comp in sorted(main, key=lambda c: c[1][1] - slope * c[1][0]):
        cy = comp[1][1] - slope * comp[1][0]
        if not bands or cy - np.median([c[1][1] - slope * c[1][0] for c in bands[-1]]) > max(
            typical_height * 1.05, height * 0.019
        ):
            bands.append([comp])
        else:
            bands[-1].append(comp)
    # Noise outside the written alphabet often forms sparse bands.
    minimum_strokes = max(1, min(3, min(map(len, rows))))
    band_positions = [np.median([c[1][1] - slope * c[1][0] for c in band]) for band in bands]
    strong_positions = [
        position
        for position, band in zip(band_positions, bands, strict=True)
        if len(band) >= minimum_strokes
    ]
    bands = [
        band
        for position, band in zip(band_positions, bands, strict=True)
        if len(band) >= minimum_strokes
        or any(abs(position - strong) < typical_height * 1.8 for strong in strong_positions)
    ]
    # Quotes can be much higher than an underscore on the same punctuation row.
    # Merge only nearby bands, and still validate the resulting character counts.
    while len(bands) > len(rows):
        positions = [np.median([c[1][1] - slope * c[1][0] for c in band]) for band in bands]
        distances = np.diff(positions)
        closest = int(np.argmin(distances))
        if distances[closest] > typical_height * 1.8:
            break
        bands[closest].extend(bands.pop(closest + 1))
    assigned = {id(c) for band in bands for c in band}
    band_coordinates = {id(band): [c[1][1] - slope * c[1][0] for c in band] for band in bands}
    for comp in components:
        if id(comp) in assigned or not bands:
            continue
        cy = comp[1][1] - slope * comp[1][0]

        def band_distance(band, cy=cy):
            return min(abs(cy - coordinate) for coordinate in band_coordinates[id(band)])

        nearest = min(bands, key=band_distance)
        distance = band_distance(nearest)
        if distance < typical_height * 1.4:
            nearest.append(comp)
    grouped = []
    for band in bands:
        # Merge pieces that overlap in x or have very small horizontal gaps.
        groups: list[list[int]] = []
        for stats_, _ in sorted(band, key=lambda c: c[0][0]):
            x, y, w, h, _ = stats_
            if groups and x <= groups[-1][2] + max(3, typical_height * 0.09):
                old = groups[-1]
                groups[-1] = [old[0], min(y, old[1]), max(x + w, old[2]), max(y + h, old[3])]
            else:
                groups.append([x, y, x + w, y + h])
        grouped.append(groups)
    return grouped, typical_height


def propose_boxes(mask: np.ndarray, rows: list[str]) -> tuple[list[dict], list[str]]:
    """Label row-ordered glyph groups with the supplied row strings.

    A matching count is not recognition. Callers must review proposals or explicitly
    accept inference.
    """
    height, width = mask.shape
    layout = layout_groups(mask, rows)
    if layout is None:
        return [], ["No ink detected. Try a tighter crop, brighter light, or another ink mode."]
    bands = layout[0]
    errors = []
    if len(bands) != len(rows):
        errors.append(f"Detected {len(bands)} rows; expected {len(rows)}. Supply corrected crops.")
    boxes = []
    for row_index, (groups, characters) in enumerate(zip(bands, rows, strict=False)):
        if len(groups) != len(characters):
            errors.append(
                f"Row {row_index + 1}: found {len(groups)} glyphs for {len(characters)} labels. "
                "Edit the crops; automatic generation is blocked."
            )
        for (x0, y0, x1, y1), char in zip(groups, characters, strict=False):
            pad = 3
            x0, y0, x1, y1 = (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(width, x1 + pad),
                min(height, y1 + pad),
            )
            boxes.append(
                GlyphBox(
                    character=char,
                    x=x0 / width,
                    y=y0 / height,
                    width=(x1 - x0) / width,
                    height=(y1 - y0) / height,
                ).model_dump()
            )
    return boxes, errors


def rows_from_boxes(boxes: list[dict], gap: float = 0.025) -> list[str]:
    """Group the model's labels into text rows by vertical centre, left to right."""
    ordered = sorted(boxes, key=lambda b: b["y"] + b["height"] / 2)
    rows: list[list[dict]] = []
    for box in ordered:
        centre = box["y"] + box["height"] / 2
        if rows and centre - np.mean([b["y"] + b["height"] / 2 for b in rows[-1]]) <= gap:
            rows[-1].append(box)
        else:
            rows.append([box])
    return ["".join(b["character"] for b in sorted(row, key=lambda b: b["x"])) for row in rows]


DUST = 0.08  # On real photos dust is under 0.065 of glyph height and pen dots over 0.10.
SKIP_GROUP = 1.2  # A stray mark is cheaper to ignore than to label wrongly.
MISS_LABEL = 3.0  # A label with no ink nearby is reported rather than forced onto a mark.
MERGE_PART = 0.4  # Extra cost for each additional piece merged into one glyph.
MAX_PARTS = 3


def align_labels(
    boxes: list[dict], groups: list[list[int]], scale: float, size: tuple[int, int]
) -> tuple[list[dict], list[str]]:
    """Match the model's ordered labels to ink groups found in reading order.

    The model reads characters reliably but places boxes loosely. Ink groups give exact
    shapes but no labels. A dynamic program pairs them in order, using the model's box
    centres only as distances: it can skip stray marks, merge split glyphs such as a colon
    or a quote, and leave a label unmatched when no ink is near it.
    """
    width, height = size
    labels, parts = len(boxes), len(groups)
    centres = [
        ((b["x"] + b["width"] / 2) * width, (b["y"] + b["height"] / 2) * height) for b in boxes
    ]

    def union(first: int, count: int) -> list[int]:
        chosen = groups[first : first + count]
        return [
            min(g[0] for g in chosen),
            min(g[1] for g in chosen),
            max(g[2] for g in chosen),
            max(g[3] for g in chosen),
        ]

    def distance(label: int, first: int, count: int) -> float:
        x0, y0, x1, y1 = union(first, count)
        dx, dy = (x0 + x1) / 2 - centres[label][0], (y0 + y1) / 2 - centres[label][1]
        return float(np.hypot(dx, dy)) / scale + MERGE_PART * (count - 1)

    best = np.full((labels + 1, parts + 1), np.inf)
    came: dict[tuple[int, int], tuple[int, int, int]] = {}
    best[0, 0] = 0.0
    for label in range(labels + 1):
        for part in range(parts + 1):
            here = best[label, part]
            moves = []
            if part < parts:
                moves.append((label, part + 1, SKIP_GROUP, 0))
            if label < labels:
                moves.append((label + 1, part, MISS_LABEL, 0))
                for count in range(1, min(MAX_PARTS, parts - part) + 1):
                    moves.append((label + 1, part + count, distance(label, part, count), count))
            for next_label, next_part, cost, used in moves:
                if here + cost < best[next_label, next_part]:
                    best[next_label, next_part] = here + cost
                    came[(next_label, next_part)] = (label, part, used)
    matched: dict[int, tuple[int, int]] = {}
    label, part = labels, parts
    while (label, part) != (0, 0):
        previous_label, previous_part, used = came[(label, part)]
        if used:
            matched[previous_label] = (previous_part, used)
        label, part = previous_label, previous_part
    aligned, errors, pad = [], [], 3
    for index, box in enumerate(boxes):
        if index not in matched:
            errors.append(f"No ink was found for {box['character']!r}. Redraw or remove its crop.")
            aligned.append(box)
            continue
        x0, y0, x1, y1 = union(*matched[index])
        x0, y0, x1, y1 = (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(width, x1 + pad),
            min(height, y1 + pad),
        )
        aligned.append(
            box
            | {
                "x": x0 / width,
                "y": y0 / height,
                "width": (x1 - x0) / width,
                "height": (y1 - y0) / height,
            }
        )
    return aligned, errors


Locator = Callable[[Image.Image], list[dict]]

BASE_WARNINGS = [
    "One sample per character is used. Repeated labels use their first crop.",
    "Stroke smoothing is conservative. Weight and placement remain estimates.",
]


def analyze(
    source: Path,
    output: Path,
    rows: list[str],
    mode: str,
    supplied_boxes: list[dict] | None = None,
    detector: str = "layout",
    locate: Locator | None = None,
) -> dict:
    """Find the glyph boxes. `detector` is "layout", "vision" or "auto" (vision, else layout)."""
    with Image.open(source) as image:
        picture = image.convert("RGB")
    mask, selected_mode = ink_mask(picture, mode)
    Image.fromarray(255 - mask).save(output / "mask.png")
    notes, used, boxes, errors = [], "supplied", supplied_boxes, []
    if not supplied_boxes and detector == "vision" and locate is None:
        raise ImageError("AI character detection is not configured on this server.")
    if not supplied_boxes and detector != "layout" and locate is not None:
        try:
            proposals = locate(picture)
            proposal_mask, _ = ink_mask(picture, mode, for_inference=True)
            layout = layout_groups(proposal_mask, rows_from_boxes(proposals), DUST)
            if layout is None:
                boxes, errors = proposals, ["No ink detected. Try another ink mode or photo."]
            else:
                flat = [group for band in layout[0] for group in band]
                boxes, errors = align_labels(proposals, flat, layout[1], picture.size)
            used = "vision"
        except VisionError as exc:
            if detector == "vision":
                raise ImageError(str(exc)) from exc
            notes.append(f"AI character detection failed ({exc}) Row order was used instead.")
    if not supplied_boxes and used != "vision":
        proposal_mask, _ = ink_mask(picture, mode, for_inference=True)
        boxes, errors = propose_boxes(proposal_mask, rows)
        used = "layout"
    lead = (
        "Characters were located and labelled by a vision model. Review every crop and label."
        if used == "vision"
        else "Labels follow the supplied row order, not handwriting recognition. Review every crop."
    )
    return {
        "glyphs": boxes,
        "mapping_errors": errors,
        "warnings": [*notes, lead, *BASE_WARNINGS],
        "ink_mode": selected_mode,
        "detector": used,
    }

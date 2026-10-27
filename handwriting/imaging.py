"""Conservative ink extraction and layout-based proposals, without OCR."""

import io
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from handwriting.config import Settings
from handwriting.models import GlyphBox


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


def propose_boxes(mask: np.ndarray, rows: list[str]) -> tuple[list[dict], list[str]]:
    """Estimate a common row slope, then group components by reading order.

    This proposes labels from the supplied row strings. A matching count is not
    recognition. Callers must review proposals or explicitly accept inference.
    """
    height, width = mask.shape
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    components = [
        (stats[i].tolist(), centers[i].tolist())
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= 3
    ]
    if not components:
        return [], ["No ink detected. Try a tighter crop, brighter light, or another ink mode."]
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
    errors = []
    if len(bands) != len(rows):
        errors.append(f"Detected {len(bands)} rows; expected {len(rows)}. Supply corrected crops.")
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
    boxes = []
    for row_index, (band, characters) in enumerate(zip(bands, rows, strict=False)):
        # Merge pieces that overlap in x or have very small horizontal gaps.
        groups: list[list[int]] = []
        for stats_, _ in sorted(band, key=lambda c: c[0][0]):
            x, y, w, h, _ = stats_
            if groups and x <= groups[-1][2] + max(3, typical_height * 0.09):
                old = groups[-1]
                groups[-1] = [old[0], min(y, old[1]), max(x + w, old[2]), max(y + h, old[3])]
            else:
                groups.append([x, y, x + w, y + h])
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


def analyze(
    source: Path, output: Path, rows: list[str], mode: str, supplied_boxes: list[dict] | None = None
) -> dict:
    with Image.open(source) as image:
        mask, selected_mode = ink_mask(image.convert("RGB"), mode)
        proposal_mask, _ = ink_mask(image.convert("RGB"), mode, for_inference=True)
    Image.fromarray(255 - mask).save(output / "mask.png")
    boxes, errors = (supplied_boxes, []) if supplied_boxes else propose_boxes(proposal_mask, rows)
    warnings_ = [
        "Labels follow the supplied row order, not handwriting recognition. Review every crop.",
        "One sample per character is used. Repeated labels use their first crop.",
        "Stroke smoothing is conservative. Weight and placement remain estimates.",
    ]
    return {
        "glyphs": boxes,
        "mapping_errors": errors,
        "warnings": warnings_,
        "ink_mode": selected_mode,
    }

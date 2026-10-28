"""Ruled notebook paper must not leak its lines into traced glyphs."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from handwriting.imaging import _line_reach, ink_mask, remove_ruled_lines

BLUE_INK = (14, 20, 70)
BLACK_INK = (20, 20, 20)


def ruled_sheet(width=1200, height=600):
    image = Image.new("RGB", (width, height), (214, 206, 190))
    draw = ImageDraw.Draw(image)
    for y in (120, 260, 400, 540):
        draw.line([(0, y + 6), (width, y)], fill=(34, 30, 24), width=4)
    draw.line([(300, 200), (300, 300)], fill=BLUE_INK, width=9)  # crosses the rule at y~258
    draw.line([(500, 395), (600, 392)], fill=BLUE_INK, width=9)  # lies along the rule at y~397
    draw.line([(800, 200), (800, 300)], fill=BLACK_INK, width=9)  # black pen crosses the rule
    return image


def test_rules_are_erased_across_the_page_including_past_the_detected_ends():
    mask, _ = ink_mask(ruled_sheet(), "auto")
    for x in (3, 100, 650, 1190):
        column = mask[:, x]
        assert not column[100:140].any() and not column[240:280].any(), x


def test_strokes_crossing_or_lying_on_a_rule_survive_intact():
    mask, _ = ink_mask(ruled_sheet(), "auto")
    assert mask[200:301, 300].all(), "blue stroke crossing a rule has a gap"
    assert mask[200:301, 800].all(), "black stroke crossing a rule has a gap"
    assert mask[385:405, 520:580].any(), "blue stroke lying on a rule was erased"


def test_a_page_without_rules_is_returned_untouched():
    mask = np.zeros((300, 400), np.uint8)
    mask[100:200, 150:160] = 255
    rgb = np.full((300, 400, 3), 230, np.uint8)
    assert remove_ruled_lines(mask, rgb) is mask


def test_long_vertical_and_steep_edges_are_not_treated_as_rules():
    mask = np.zeros((600, 800), np.uint8)
    mask[20:580, 40:46] = 255  # a page edge or binding
    for step in range(500):  # a long steep diagonal
        mask[50 + step, 200 + step // 2 : 206 + step // 2] = 255
    rgb = np.full((600, 800, 3), 230, np.uint8)
    assert (remove_ruled_lines(mask, rgb) == mask).all()


def test_line_reach_stops_after_a_gap_and_at_the_image_edge():
    mask = np.zeros((100, 400), np.uint8)
    mask[50, 0:200] = 255
    mask[50, 300:400] = 255  # separate rule after a 100 px gap
    end, _ = _line_reach(mask, 100.0, 50.0, 0.0, 1, reach=20)
    assert 190 <= end <= 200
    start, _ = _line_reach(mask, 150.0, 50.0, 0.0, -1, reach=20)
    assert start <= 4  # ran to the left edge
    far, _ = _line_reach(mask, 300.0, 50.0, 0.0, 1, reach=20)
    assert far >= 390


def test_line_reach_stops_when_a_tilted_rule_leaves_the_image():
    mask = np.zeros((100, 400), np.uint8)
    for x in range(400):
        mask[max(0, 50 - x // 5), x] = 255
    _, y = _line_reach(mask, 10.0, 48.0, -0.2, 1, reach=20)
    assert y < 5


def test_gray_dotted_rules_and_margin_line_are_excluded_in_auto_blue_mode():
    width, height = 1200, 600
    image = Image.new("RGB", (width, height), (185, 187, 185))
    draw = ImageDraw.Draw(image)
    gray_rule = (86, 86, 86)
    draw.line([(95, 0), (85, height)], fill=gray_rule, width=2)  # vertical margin line
    for y in (160, 300, 440):
        for x in range(0, width, 10):
            draw.line([(x, y), (min(width, x + 5), y)], fill=gray_rule, width=2)
    draw.line([(300, 120), (300, 200)], fill=BLUE_INK, width=8)
    mask, selected = ink_mask(image, "auto", for_inference=True)
    assert selected == "blue"
    assert not mask[:, 80:105].any(), "gray vertical margin line leaked into blue-ink mask"
    assert not mask[295:306, :].any(), "gray dotted rule leaked into blue-ink mask"
    assert mask[130:190, 300].all(), "blue glyph stroke was lost"


def test_thin_dotted_rules_on_large_sheet_are_erased_in_dark_mode():
    width, height = 1560, 800
    image = Image.new("RGB", (width, height), (214, 206, 190))
    draw = ImageDraw.Draw(image)
    for y in (120, 260, 400, 540):
        for x in range(0, width, 10):
            y0 = round(y + 6 - 6 * (x / width))
            y1 = round(y + 6 - 6 * (min(width, x + 4) / width))
            draw.line([(x, y0), (min(width, x + 4), y1)], fill=(34, 30, 24), width=1)
    draw.line([(800, 200), (800, 300)], fill=BLACK_INK, width=9)
    mask, selected = ink_mask(image, "dark")
    assert selected == "dark"
    assert not mask[115:135, :].any(), "thin dotted rule survived dark-mode rule removal"
    assert mask[200:301, 800].all(), "black stroke crossing a dotted rule has a gap"


def blue_ring(smear=0):
    """A thin blue 'o'. With `smear`, its colour bleeds as JPEG chroma subsampling does."""
    image = Image.new("RGB", (200, 200), (205, 207, 200))
    ImageDraw.Draw(image).ellipse([86, 86, 114, 114], outline=(10, 20, 110), width=3)
    if not smear:
        return image
    luma, blue_diff, red_diff = image.convert("YCbCr").split()
    blur = ImageFilter.GaussianBlur(smear)
    return Image.merge("YCbCr", (luma, blue_diff.filter(blur), red_diff.filter(blur))).convert(
        "RGB"
    )


def test_blue_bleeding_beside_a_thin_stroke_does_not_thicken_or_fill_it():
    sharp, _ = ink_mask(blue_ring(), "auto")
    bled, mode = ink_mask(blue_ring(smear=3), "auto")
    assert mode == "blue"
    assert np.count_nonzero(bled) <= 1.1 * np.count_nonzero(sharp), "strokes were thickened"
    assert not bled[94:107, 94:107].any(), "the inside of the letter was filled with ink"

"""Ruled notebook paper must not leak its lines into traced glyphs."""

import numpy as np
from PIL import Image, ImageDraw

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

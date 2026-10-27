"""Create a public geometric test sheet without using a person's handwriting."""

import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parent
    boxes = json.loads((root / "glyphs.json").read_text())
    width, height = 1800, 2400
    image = Image.new("RGB", (width, height), (245, 240, 226))
    draw = ImageDraw.Draw(image)
    ink = (20, 40, 120)
    for box in boxes:
        x, y = round(box["x"] * width) + 4, round(box["y"] * height) + 4
        w, h = max(8, round(box["width"] * width) - 8), max(8, round(box["height"] * height) - 8)
        stroke = max(2, min(6, w // 7))
        char = box["character"]
        if char in "ij!":
            draw.line((x + w // 2, y + h // 3, x + w // 2, y + h), fill=ink, width=stroke)
            draw.ellipse((x + w // 2 - stroke, y, x + w // 2 + stroke, y + stroke * 2), fill=ink)
        elif char in ".,;:'\"":
            draw.ellipse((x, y, x + w, y + min(h, w)), fill=ink)
            if char in ';:"':
                draw.ellipse((x, y + max(0, h - w), x + w, y + h), fill=ink)
        elif char in "-_=":
            draw.line((x, y + h // 2, x + w, y + h // 2), fill=ink, width=stroke)
            if char == "=":
                draw.line((x, y + h, x + w, y + h), fill=ink, width=stroke)
        elif ord(char) % 3 == 0:
            draw.line(
                [(x, y + h), (x + w // 2, y), (x + w, y + h), (x, y + h)], fill=ink, width=stroke
            )
        elif ord(char) % 3 == 1:
            draw.ellipse((x, y, x + w, y + h), outline=ink, width=stroke)
        else:
            draw.line([(x + w, y), (x, y), (x, y + h), (x + w, y + h)], fill=ink, width=stroke)
    # A background mark in the first band exercises mandatory count-mismatch review.
    draw.ellipse((1610, 200, 1640, 235), outline=ink, width=4)
    draw.point((700, 1900), fill=ink)
    draw.line((800, 0, 800, 20), fill=ink, width=3)
    draw.ellipse((700, 2050, 703, 2053), fill=ink)
    image.save(root / "reference.png")


if __name__ == "__main__":
    main()

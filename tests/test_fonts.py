import io
import json
from pathlib import Path

import numpy as np
import pytest
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from tests.conftest import upload_sheet, wait_job


def raster_glyph(ttf, char):
    font = ImageFont.truetype(io.BytesIO(ttf), 500)
    image = Image.new("L", (900, 650))
    ImageDraw.Draw(image).text((30, 450), char, fill=255, font=font, anchor="ls")
    return np.asarray(image) > 127


def flood_background(pixels):
    # Independent raster flood fill, without relying on OpenCV contour extraction.
    from collections import deque

    height, width = pixels.shape
    seen = set()
    queue = deque([(0, 0)])
    while queue:
        x, y = queue.popleft()
        if (x, y) in seen or not (0 <= x < width and 0 <= y < height) or pixels[y, x]:
            continue
        seen.add((x, y))
        queue.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    return len(seen) < int(np.count_nonzero(~pixels))


def test_upload_builds_real_outlines_and_matching_webfont(client, sheet):
    job = upload_sheet(client, sheet)
    assert job["status"] == "ready", job
    ttf_response = client.get(job["artifacts"]["font.ttf"])
    woff_response = client.get(job["artifacts"]["font.woff2"])
    assert ttf_response.headers["content-type"] == "font/ttf"
    assert woff_response.headers["content-type"] == "font/woff2"
    ttf = TTFont(io.BytesIO(ttf_response.content))
    woff = TTFont(io.BytesIO(woff_response.content))
    cmap = ttf.getBestCmap()
    assert all(ord(c) in cmap for c in "oiAVlmg. ")
    assert cmap == woff.getBestCmap()
    assert ttf["hmtx"][cmap[32]][0] == 500
    assert ttf["glyf"][cmap[ord("g")]].yMin < 0
    assert ttf["glyf"][cmap[ord(".")]].yMax < 100
    assert ttf["glyf"][cmap[ord("A")]].yMax > 650
    assert ttf["hmtx"][cmap[ord("i")]][0] < ttf["hmtx"][cmap[ord("m")]][0]
    assert ttf["OS/2"].fsSelection & 0x80
    assert ttf["hhea"].lineGap == 100
    for name in ttf.getGlyphOrder():
        assert ttf["glyf"][name].getCoordinates(ttf["glyf"]) == woff["glyf"][name].getCoordinates(
            woff["glyf"]
        )
    assert flood_background(raster_glyph(ttf_response.content, "o"))
    dotted = raster_glyph(ttf_response.content, "i")
    occupied_rows = np.flatnonzero(dotted.any(axis=1))
    assert np.max(np.diff(occupied_rows)) > 10, "The dot and stem must remain separate."
    assert not np.array_equal(
        raster_glyph(ttf_response.content, "A"), raster_glyph(ttf_response.content, "V")
    )
    flags = np.array(ttf["glyf"][cmap[ord("o")]].flags)
    assert np.any(flags & 1 == 0), "Expected quadratic control points"
    assert "GPOS" in ttf and "kern" in ttf and job["kerning_pairs"] > 0
    assert client.get(job["artifacts"]["font.css"]).text.count("font-kerning: normal") == 1
    assert Image.open(io.BytesIO(client.get(job["artifacts"]["sample.png"]).content)).width == 1000
    assert client.get(job["artifacts"]["report.json"]).json()["units_per_em"] == 1000


def test_mapping_correction_invalidates_downloads_and_changes_font(client, sheet):
    first = upload_sheet(client, sheet)
    previous_url = first["artifacts"]["font.ttf"]
    mapping = sheet[1]
    mapping[0]["character"] = "Q"
    mapping[0]["target_height"] = 500
    mapping[0]["baseline"] = -100
    response = client.put(f"/api/fonts/{first['id']}/glyphs", json={"glyphs": mapping})
    assert response.status_code == 200
    assert response.json()["revision"] == first["revision"] + 1
    assert response.json()["artifacts"] == {}
    assert client.get(previous_url).status_code == 404
    assert client.get(response.json()["preview_url"]).status_code == 409
    assert (
        client.post(
            f"/api/fonts/{first['id']}/generate", json={"confirm_mapping": False}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/fonts/{first['id']}/generate", json={"confirm_mapping": True}
        ).status_code
        == 202
    )
    revised = wait_job(client, first["id"])
    assert revised["status"] == "ready", revised
    font = TTFont(io.BytesIO(client.get(revised["artifacts"]["font.ttf"]).content))
    cmap = font.getBestCmap()
    assert ord("Q") in cmap and ord("o") not in cmap
    assert font["glyf"][cmap[ord("Q")]].yMin == -100


def test_preview_limits_missing_characters_and_newlines(client, sheet):
    job = upload_sheet(client, sheet)
    url = job["preview_url"]
    response = client.post(url, json={"text": "oi\nAV ☃", "width": 400, "font_size": 48})
    assert response.status_code == 200
    assert json.loads(response.headers["X-Missing-Characters"]) == "☃"
    image = Image.open(io.BytesIO(response.content))
    assert image.width == 400 and image.height > 170
    assert client.get(url, params={"text": "oi AV"}).status_code == 200
    for body in [{"text": ""}, {"text": "x" * 1001}, {"width": 9000}, {"font_size": 900}]:
        assert client.post(url, json=body).status_code == 422
    assert client.post(url, json={"text": "\n" * 900}).status_code == 422
    wrapped = client.post(url, json={"text": "mmm mmm mmm ", "font_size": 120, "width": 320})
    assert wrapped.status_code == 200
    assert Image.open(io.BytesIO(wrapped.content)).height > 300
    whitespace = client.post(url, json={"text": "o        o", "font_size": 120, "width": 320})
    assert whitespace.status_code == 200
    assert Image.open(io.BytesIO(whitespace.content)).height == 384
    long_word = client.post(url, json={"text": "m" * 40, "font_size": 48, "width": 320})
    assert long_word.status_code == 200
    assert Image.open(io.BytesIO(long_word.content)).height > 300


@pytest.mark.parametrize("name", ["../escape", "<style>", "   "])
def test_font_name_rejects_path_and_css_injection(client, sheet, name):
    response = client.post(
        "/api/fonts", files={"image": ("photo.png", sheet[0])}, data={"name": name}
    )
    assert response.status_code == 422


def test_reference_sheet_font(client):
    root = Path(__file__).resolve().parents[1] / "examples"
    boxes = json.loads((root / "glyphs.json").read_text())
    with (root / "reference.png").open("rb") as photo:
        response = client.post(
            "/api/fonts",
            files={"image": ("photo.jpg", photo, "image/jpeg")},
            data={"glyphs": json.dumps(boxes), "auto_generate": "true"},
        )
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["id"], timeout=30)
    assert job["status"] == "ready", job
    font = TTFont(io.BytesIO(client.get(job["artifacts"]["font.ttf"]).content))
    cmap = font.getBestCmap()
    assert all(
        ord(c) in cmap for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789₹"
    )
    assert cmap[0x2019] == cmap[ord("'")]
    assert cmap[0x00A0] == cmap[32]
    assert cmap[0x2212] == cmap[ord("-")]
    assert job["duplicate_samples"] == ["!"]
    assert len(font["kern"].kernTables[0].kernTable) > 1000
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    raster_font = ImageFont.truetype(
        io.BytesIO(client.get(job["artifacts"]["font.ttf"]).content),
        1000,
        layout_engine=ImageFont.Layout.BASIC,
    )
    profiles = {}
    for character in characters:
        raster = Image.new("L", (3200, 1400))
        ImageDraw.Draw(raster).text((0, 1000), character, font=raster_font, fill=255, anchor="ls")
        pixels = np.asarray(raster) > 32
        profiles[character] = (
            pixels.any(axis=1),
            np.argmax(pixels, axis=1),
            3200 - np.argmax(pixels[:, ::-1], axis=1),
        )
    pairs = font["kern"].kernTables[0].kernTable
    for a in characters:
        for b in characters:
            a_rows, _, a_right = profiles[a]
            b_rows, b_left, _ = profiles[b]
            common = a_rows & b_rows
            if np.any(common):
                gap = font["hmtx"][cmap[ord(a)]][0] + b_left - a_right
                gap += pairs.get((cmap[ord(a)], cmap[ord(b)]), 0)
                assert np.min(gap[common]) >= 50, f"Overlapping or crowded pair: {a}{b}"


def test_explicit_crops_preserve_ink_near_image_edges(client):
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).line((2, 50, 2, 140), fill=(20, 40, 120), width=3)
    data = io.BytesIO()
    image.save(data, format="PNG")
    mapping = [dict(character="l", x=0, y=0.2, width=0.15, height=0.55)]
    response = client.post(
        "/api/fonts",
        files={"image": ("edge.png", data.getvalue())},
        data={"glyphs": json.dumps(mapping), "auto_generate": "true"},
    )
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "ready", job
    assert client.get(job["artifacts"]["font.ttf"]).status_code == 200


def test_extreme_valid_placement_is_not_clipped_in_preview(client, sheet):
    sheet[1][0] |= {"target_height": 800, "baseline": 650}
    job = upload_sheet(client, sheet)
    assert job["status"] == "ready", job
    response = client.post(job["preview_url"], json={"text": "o", "font_size": 120})
    image = np.asarray(Image.open(io.BytesIO(response.content)))
    background = image[0, -1]
    assert np.all(image[:20] == background), "Preview must include a top margin"
    assert np.all(image[-20:] == background), "Preview must include a bottom margin"
    assert np.any(image != background), "The raised glyph must still be visible"

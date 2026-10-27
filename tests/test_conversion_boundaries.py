"""Malformed inputs and artifact failures at their owning conversion boundaries."""

import asyncio
import io
import uuid
from pathlib import Path

import numpy as np
import pytest
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

from handwriting.api import UploadLimit
from handwriting.config import Settings
from handwriting.font import trace_glyph, validate_exports
from handwriting.imaging import ImageError, decode_image, ink_mask
from handwriting.models import GlyphBox
from tests.conftest import upload_sheet, wait_job


def test_default_reference_sheet_requests_review_of_background_marks(client):
    photo = Path(__file__).resolve().parents[1] / "examples/reference.png"
    with photo.open("rb") as source:
        response = client.post(
            "/api/fonts", files={"image": ("photo.jpg", source)}, data={"auto_generate": "true"}
        )
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "review_required"
    assert len(job["glyphs"]) == 83
    assert job["mapping_errors"]
    assert (
        client.post(f"/api/fonts/{job['id']}/generate", json={"confirm_mapping": True}).status_code
        == 409
    )


def test_inference_rejects_extra_separate_rows(client, sheet):
    image = Image.new("RGB", (1200, 900), "white")
    image.paste(Image.open(io.BytesIO(sheet[0])), (0, 0))
    draw = ImageDraw.Draw(image)
    for x in (100, 300, 500):
        draw.ellipse((x, 650, x + 60, 720), outline=(20, 40, 120), width=5)
    data = io.BytesIO()
    image.save(data, format="PNG")
    response = client.post(
        "/api/fonts",
        files={"image": ("rows.png", data.getvalue())},
        data={"rows": '["oiAVlmg."]', "auto_generate": "true"},
    )
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "review_required"
    assert any("rows" in error for error in job["mapping_errors"])


def test_nearby_bands_merge_disconnected_strokes_into_one_review_row(client):
    image = Image.new("RGB", (700, 300), "white")
    draw = ImageDraw.Draw(image)
    for x in (100, 300, 500):
        draw.line((x, 60, x, 80), fill=(20, 40, 120), width=5)
        draw.line((x, 90, x, 110), fill=(20, 40, 120), width=5)
    data = io.BytesIO()
    image.save(data, format="PNG")
    response = client.post(
        "/api/fonts", files={"image": ("marks.png", data.getvalue())}, data={"rows": '["abc"]'}
    )
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "review_required"
    assert not job["mapping_errors"]
    assert len(job["glyphs"]) == 3
    assert all(box["height"] * 300 >= 50 for box in job["glyphs"])


def test_image_orientation_transparency_and_bomb_limits():
    image = Image.new("RGB", (80, 120), "white")
    exif = image.getexif()
    exif[274] = 6
    jpeg = io.BytesIO()
    image.save(jpeg, format="JPEG", exif=exif)
    decoded = decode_image(jpeg.getvalue(), Settings())
    assert decoded.size == (120, 80)
    assert not decoded.getexif()
    transparent = io.BytesIO()
    Image.new("RGBA", (40, 40), (0, 0, 0, 0)).save(transparent, format="PNG")
    assert decode_image(transparent.getvalue(), Settings()).getpixel((10, 10)) == (255, 255, 255)
    maximum = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 8000
        with pytest.raises(ImageError, match="readable"):
            decode_image(jpeg.getvalue(), Settings())
    finally:
        Image.MAX_IMAGE_PIXELS = maximum


def test_large_background_regions_are_only_removed_from_inference():
    image = Image.new("RGB", (1000, 1000), "white")
    ImageDraw.Draw(image).rectangle((100, 100, 900, 900), fill="black")
    inferred, _ = ink_mask(image, "dark", for_inference=True)
    reviewed, _ = ink_mask(image, "dark")
    assert not np.any(inferred)
    assert reviewed[500, 500] == 255


@pytest.mark.parametrize("kind", ["wide", "thin"])
def test_unusable_crop_geometry_has_an_actionable_error(kind):
    mask = (
        np.ones((2, 500), np.uint8) * 255 if kind == "wide" else np.eye(500, dtype=np.uint8) * 255
    )
    box = GlyphBox(character="a", x=0, y=0, width=1, height=1, target_height=20)
    with pytest.raises(ImageError, match=f"too {kind}"):
        trace_glyph(mask, box)


@pytest.mark.parametrize("corrupt", ["mapping", "outline"])
def test_export_validator_rejects_changed_webfonts(client, sheet, tmp_path, corrupt):
    job = upload_sheet(client, sheet)
    ttf_path, woff_path = tmp_path / "font.ttf", tmp_path / "font.woff2"
    ttf_path.write_bytes(client.get(job["artifacts"]["font.ttf"]).content)
    woff_path.write_bytes(client.get(job["artifacts"]["font.woff2"]).content)
    with TTFont(woff_path) as font:
        if corrupt == "mapping":
            for table in font["cmap"].tables:
                if table.isUnicode():
                    table.cmap[ord("o")] = font.getBestCmap()[ord("V")]
        else:
            glyph = font["glyf"][font.getBestCmap()[ord("o")]]
            x, y = glyph.coordinates[0]
            glyph.coordinates[0] = (x + 10, y)
        font.save(tmp_path / "corrupt.woff2")
    with pytest.raises(ImageError, match="validation"):
        validate_exports(ttf_path, tmp_path / "corrupt.woff2")


def test_upload_storage_failure_is_reported_without_retaining_photos(client, sheet, monkeypatch):
    save = Image.Image.save

    def disk_full(image, path, *args, **kwargs):
        if isinstance(path, Path) and path.name == "image.png":
            raise OSError("simulated disk full")
        return save(image, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", disk_full)
    response = client.post("/api/fonts", files={"image": ("photo.png", sheet[0])})
    assert response.status_code == 503


def test_generation_capacity_busy_edits_and_early_mask_are_explicit(client, sheet):
    job = upload_sheet(client, sheet)
    client.app.state.worker.stop()
    store = client.app.state.store
    for _ in range(20):
        pending_id = uuid.uuid4().hex
        store.create(pending_id, "Pending", {"ink_mode": "auto"}, 30)
    edited = client.put(f"/api/fonts/{job['id']}/glyphs", json={"glyphs": sheet[1]})
    assert edited.status_code == 200
    assert (
        client.post(f"/api/fonts/{job['id']}/generate", json={"confirm_mapping": True}).status_code
        == 429
    )
    assert (
        client.put(f"/api/fonts/{pending_id}/glyphs", json={"glyphs": sheet[1]}).status_code == 409
    )
    assert client.get(f"/api/fonts/{pending_id}/mask").status_code == 409


def test_missing_artifact_returns_404(client, sheet, tmp_path):
    job = upload_sheet(client, sheet)
    store = client.app.state.store
    record = store.get(job["id"])
    root = store.path.parent / job["id"] / record["result"]["artifacts"]
    (root / "font.ttf").rename(tmp_path / "removed.ttf")
    assert client.get(job["artifacts"]["font.ttf"]).status_code == 404


def test_chunked_body_limit_and_client_disconnect():
    async def exercise(messages):
        responses = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            responses.append(message)

        async def downstream(scope, receive, send):
            raise AssertionError(
                "Oversized or disconnected requests must not reach the application"
            )

        await UploadLimit(downstream, maximum=10)(
            {"type": "http", "method": "POST", "path": "/api/fonts"}, receive, send
        )
        return responses

    response = asyncio.run(
        exercise(
            [
                {"type": "http.request", "body": b"123456", "more_body": True},
                {"type": "http.request", "body": b"123456", "more_body": False},
            ]
        )
    )
    assert response[0]["status"] == 413
    assert asyncio.run(exercise([{"type": "http.disconnect"}])) == []


def test_buffered_body_is_forwarded_once_then_disconnect_is_preserved():
    async def exercise():
        messages = [
            {"type": "http.request", "body": b"first", "more_body": True},
            {"type": "http.request", "body": b"last", "more_body": False},
            {"type": "http.disconnect"},
        ]

        async def receive():
            return messages.pop(0)

        async def send(message):
            pass

        async def downstream(scope, receive, send):
            first = await receive()
            assert first["body"] == b"firstlast" and not first["more_body"]
            assert await receive() == {"type": "http.disconnect"}

        await UploadLimit(downstream, maximum=20)(
            {"type": "http", "method": "POST", "path": "/api/fonts"}, receive, send
        )
        assert not messages

    asyncio.run(exercise())


def test_small_punctuation_only_font_and_weight_estimation_are_valid(client):
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).ellipse((80, 80, 100, 100), fill=(20, 40, 120))
    data = io.BytesIO()
    image.save(data, format="PNG")
    for char in (".", "o"):
        mapping = [dict(character=char, x=0.3, y=0.3, width=0.3, height=0.3, target_height=20)]
        import json

        response = client.post(
            "/api/fonts",
            files={"image": ("dot.png", data.getvalue())},
            data={"glyphs": json.dumps(mapping), "auto_generate": "true"},
        )
        job = wait_job(client, response.json()["id"])
        assert job["status"] == "ready", job
        font = TTFont(io.BytesIO(client.get(job["artifacts"]["font.ttf"]).content))
        glyph = font["glyf"][font.getBestCmap()[ord(char)]]
        assert glyph.numberOfContours == 1
        if char == ".":
            assert job["kerning_pairs"] == 0
        else:
            assert job["measurements"][char]["measured_stroke"] > 0


@pytest.mark.parametrize("seed,side,density", [(1, 50, 0.08), (358, 40, 0.15)])
def test_disconnected_pixel_noise_cannot_become_an_empty_font_glyph(seed, side, density):
    noise = (np.random.default_rng(seed).random((side, side)) < density).astype(np.uint8) * 255
    box = GlyphBox(character=".", x=0, y=0, width=1, height=1, target_height=20)
    with pytest.raises(ImageError, match="No outline survived"):
        trace_glyph(noise, box)


def test_weight_offset_preserves_a_small_enclosed_counter():
    from handwriting.font import balance_weight, topology

    mask = Image.new("L", (60, 60))
    ImageDraw.Draw(mask).ellipse((20, 20, 40, 40), outline=255, width=6)
    balanced, _ = balance_weight(np.asarray(mask), "o")
    assert balanced[30, 30] == 0
    assert np.count_nonzero(balanced) > 0
    assert topology(np.zeros((10, 10), dtype=np.uint8)) == (0, 0)

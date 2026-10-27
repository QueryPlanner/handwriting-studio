import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from handwriting.api import create_app
from handwriting.config import Settings
from tests.conftest import upload_sheet, wait_job


def test_studio_docs_and_explicit_cors(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/studio.js").status_code == 200
    assert client.get("/static/daisyui.css").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/layouts").json()["default_rows"][0] == "ABCDEFGHIJK"
    assert "/api/fonts/{job_id}/preview" in client.get("/openapi.json").json()["paths"]
    allowed = client.options(
        "/api/fonts",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert (
        client.options(
            "/api/fonts",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        ).status_code
        == 400
    )
    assert client.get("/api/fonts").status_code == 405  # No public upload index.


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", "not-json"),
        ("rows", "{}"),
        ("rows", "[42]"),
        ("rows", "[]"),
        ("rows", '["a b"]'),
        ("rows", '[""]'),
        ("rows", json.dumps(["a"] * 17)),
        ("glyphs", "[]"),
        ("glyphs", "{}"),
        ("glyphs", '[{"character":"a"}]'),
        ("ink_mode", "red"),
    ],
)
def test_malformed_layouts_fail_before_queuing(client, sheet, field, value):
    response = client.post(
        "/api/fonts", files={"image": ("test.png", sheet[0])}, data={field: value}
    )
    assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    "change",
    [
        {"character": " "},
        {"character": "\x00"},
        {"character": "ab"},
        {"x": -1},
        {"x": 0.99, "width": 0.2},
        {"y": 0.99, "height": 0.2},
        {"width": 0},
        {"target_height": 900},
        {"baseline": -900},
        {"unknown": 1},
    ],
)
def test_invalid_crop_edits_fail_without_changing_ready_font(client, sheet, change):
    job = upload_sheet(client, sheet)
    mapping = sheet[1]
    mapping[0] |= change
    response = client.put(f"/api/fonts/{job['id']}/glyphs", json={"glyphs": mapping})
    assert response.status_code == 422
    assert client.get(job["artifacts"]["font.ttf"]).status_code == 200


def test_blank_and_mismatched_images_request_mapping_review(client, sheet):
    blank = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(blank, format="PNG")
    response = client.post(
        "/api/fonts",
        files={"image": ("blank.png", blank.getvalue())},
        data={"auto_generate": "true"},
    )
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "review_required"
    assert job["mapping_errors"] and job["glyphs"] == []
    assert (
        client.post(f"/api/fonts/{job['id']}/generate", json={"confirm_mapping": True}).status_code
        == 409
    )
    assert client.get(job["mask_url"]).status_code == 200
    response = client.post(
        "/api/fonts",
        files={"image": ("sheet.png", sheet[0])},
        data={"rows": '["oiA"]', "auto_generate": "true"},
    )
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "review_required" and job["mapping_errors"]


def test_inferred_mapping_requires_confirmation_or_opt_in(client, sheet):
    for auto in ("false", "true"):
        response = client.post(
            "/api/fonts",
            files={"image": ("sheet.png", sheet[0])},
            data={"rows": '["oiAVlmg."]', "auto_generate": auto},
        )
        job = wait_job(client, response.json()["id"])
        assert job["status"] == ("review_required" if auto == "false" else "ready"), job
        assert not job["mapping_errors"]
        assert "recognition" in job["warnings"][0]


@pytest.mark.parametrize("data", [b"", b"not an image", b"<svg></svg>"])
def test_invalid_upload_is_rejected_even_with_image_mime(client, data):
    response = client.post("/api/fonts", files={"image": ("photo.png", data, "image/png")})
    assert response.status_code == 422


def test_size_pixel_format_and_chunked_limits(tmp_path, sheet):
    settings = Settings(data_dir=tmp_path, max_upload_bytes=5000, max_image_pixels=10000)
    with TestClient(create_app(settings)) as client:
        assert (
            client.post("/api/fonts", files={"image": ("big.png", b"x" * 5001)}).status_code == 413
        )
        assert (
            client.post("/api/fonts", content=iter([b"x" * 100000, b"x" * 100000])).status_code
            == 413
        )
        large = io.BytesIO()
        Image.new("RGB", (150, 150), "white").save(large, format="PNG")
        assert (
            client.post("/api/fonts", files={"image": ("pixels.png", large.getvalue())}).status_code
            == 422
        )
        for size, format_ in [((10, 10), "PNG"), ((50, 50), "GIF")]:
            output = io.BytesIO()
            Image.new("RGB", size, "white").save(output, format=format_)
            assert (
                client.post("/api/fonts", files={"image": ("file", output.getvalue())}).status_code
                == 422
            )


def test_queue_capacity_returns_retry_after(tmp_path, sheet):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_pending_jobs=0))) as client:
        response = client.post("/api/fonts", files={"image": ("photo.png", sheet[0])})
        assert response.status_code == 429
        assert response.headers["retry-after"] == "10"
        assert not list(tmp_path.glob("*/image.png")), "Rejected uploads must not retain photos"


def test_ink_mode_edits_change_the_mask_and_reported_mode(client, sheet):
    image = Image.open(io.BytesIO(sheet[0]))
    from PIL import ImageDraw

    ImageDraw.Draw(image).line((1080, 350, 1120, 350), fill="black", width=5)
    data = io.BytesIO()
    image.save(data, format="PNG")
    response = client.post(
        "/api/fonts",
        files={"image": ("mixed.png", data.getvalue())},
        data={"glyphs": json.dumps(sheet[1]), "ink_mode": "blue"},
    )
    job = wait_job(client, response.json()["id"])
    before = Image.open(io.BytesIO(client.get(job["mask_url"]).content))
    assert before.getpixel((1100, 350)) == 255
    edited = client.put(
        f"/api/fonts/{job['id']}/glyphs", json={"glyphs": sheet[1], "ink_mode": "dark"}
    ).json()
    assert edited["ink_mode"] == "dark"
    after = Image.open(io.BytesIO(client.get(edited["mask_url"]).content))
    assert after.getpixel((1100, 350)) == 0


def test_blank_crop_fails_observably_and_can_be_corrected(client, sheet):
    mapping = [dict(character="a", x=0.8, y=0.8, width=0.1, height=0.1)]
    response = client.post(
        "/api/fonts",
        files={"image": ("photo.png", sheet[0])},
        data={"glyphs": json.dumps(mapping), "auto_generate": "true"},
    )
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "failed" and "No ink" in job["error"]
    assert job["artifacts"] == {}
    assert (
        client.put(f"/api/fonts/{job['id']}/glyphs", json={"glyphs": sheet[1]}).status_code == 200
    )
    assert (
        client.post(f"/api/fonts/{job['id']}/generate", json={"confirm_mapping": True}).status_code
        == 202
    )
    assert wait_job(client, job["id"])["status"] == "ready"


def test_unknown_jobs_and_unsupported_artifacts(client, sheet):
    assert client.get("/api/fonts/unknown").status_code == 404
    assert client.get("/api/fonts/" + "0" * 32).status_code == 404
    job = upload_sheet(client, sheet)
    root = f"/api/fonts/{job['id']}"
    assert client.get(root + f"/versions/{job['revision']}/secrets.txt").status_code == 404
    assert client.post(root + "/generate", json={"confirm_mapping": True}).status_code == 409
    assert client.get(job["image_url"]).headers["content-type"] == "image/png"
    assert client.get(job["mask_url"]).headers["content-type"] == "image/png"
    assert client.get(root).headers["cache-control"] == "no-store"
    assert client.get(job["image_url"]).headers["cache-control"] == "no-store"


def test_storage_capacity_rejects_new_uploads_without_retaining_an_image(tmp_path, sheet):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_data_bytes=1))) as client:
        response = client.post("/api/fonts", files={"image": ("photo.png", sheet[0])})
        assert response.status_code == 503
        assert "Storage capacity" in response.json()["detail"]
        assert not list(tmp_path.glob("*/image.png"))


def test_low_disk_space_rejects_generation(client, sheet, monkeypatch):
    import shutil

    job = upload_sheet(client, sheet)
    client.put(f"/api/fonts/{job['id']}/glyphs", json={"glyphs": sheet[1]})
    usage = shutil.disk_usage(client.app.state.store.path.parent)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=1))
    response = client.post(f"/api/fonts/{job['id']}/generate", json={"confirm_mapping": True})
    assert response.status_code == 503
    assert client.get(f"/api/fonts/{job['id']}").json()["status"] == "review_required"

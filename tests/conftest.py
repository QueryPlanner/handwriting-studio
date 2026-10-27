import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from handwriting.api import create_app
from handwriting.config import Settings


@pytest.fixture(autouse=True)
def no_live_vision(monkeypatch):
    """A developer's exported key must never send test images to a paid service."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)


@pytest.fixture
def sheet():
    """Distinct geometric ink, including a hole, a dot, a descender and thin strokes."""
    image = Image.new("RGB", (1200, 500), "white")
    draw = ImageDraw.Draw(image)
    color = (20, 40, 120)
    draw.ellipse((60, 80, 120, 155), outline=color, width=6)  # o
    draw.line((210, 95, 210, 155), fill=color, width=5)  # i
    draw.ellipse((206, 73, 214, 81), fill=color)
    draw.line([(310, 155), (345, 65), (380, 155)], fill=color, width=5)  # A
    draw.line((325, 120, 365, 120), fill=color, width=5)
    draw.line([(470, 65), (505, 155), (540, 65)], fill=color, width=5)  # V
    draw.line((630, 65, 650, 155), fill=color, width=4)  # l
    draw.line(
        [(735, 155), (735, 100), (760, 125), (785, 100), (805, 155)], fill=color, width=5
    )  # m
    draw.ellipse((890, 95, 940, 150), outline=color, width=5)  # g
    draw.line([(940, 120), (945, 180), (925, 195), (895, 180)], fill=color, width=5)
    draw.ellipse((1030, 145, 1036, 151), fill=color)  # .
    rectangles = [
        (50, 60, 90, 110),
        (195, 60, 35, 110),
        (300, 55, 90, 115),
        (460, 55, 90, 115),
        (620, 55, 45, 115),
        (725, 90, 90, 80),
        (880, 85, 80, 120),
        (1020, 135, 30, 30),
    ]
    boxes = [
        dict(character=c, x=x / 1200, y=y / 500, width=w / 1200, height=h / 500)
        for c, (x, y, w, h) in zip("oiAVlmg.", rectangles, strict=True)
    ]
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), boxes


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        yield client


def wait_job(client, job_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/fonts/{job_id}")
        assert response.status_code == 200, response.text
        value = response.json()
        if value["status"] not in ("queued", "processing"):
            return value
        time.sleep(0.02)
    pytest.fail(f"Job {job_id} did not finish in {timeout} seconds")


def upload_sheet(client, sheet, **options):
    image, boxes = sheet
    response = client.post(
        "/api/fonts",
        files={"image": ("sheet.png", image, "image/png")},
        data={"glyphs": json.dumps(boxes), "auto_generate": "true", **options},
    )
    assert response.status_code == 202, response.text
    return wait_job(client, response.json()["id"])

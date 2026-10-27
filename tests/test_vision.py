"""AI character detection: the OpenRouter client, label alignment, fallbacks, and the API."""

import io
import json
import urllib.error

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from handwriting import vision
from handwriting.api import create_app
from handwriting.config import Settings
from handwriting.imaging import (
    ImageError,
    align_labels,
    analyze,
    layout_groups,
    rows_from_boxes,
)
from handwriting.vision import VisionError, VisionLocator, encode_image, parse_glyphs, post_json
from tests.conftest import wait_job


def answer(glyphs):
    return {"choices": [{"message": {"content": json.dumps({"glyphs": glyphs})}}]}


def entry(character="a", x0=100, y0=200, x1=300, y1=400):
    return {"character": character, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- configuration -----------------------------------------------------------------


def test_api_key_never_appears_in_settings_repr(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-value")
    settings = Settings()
    assert settings.openrouter_api_key == "sk-secret-value"
    assert "sk-secret-value" not in repr(settings)


def test_blank_key_means_detection_is_off(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    assert Settings().openrouter_api_key is None
    with pytest.raises(VisionError, match="not configured"):
        VisionLocator(Settings())


# --- client --------------------------------------------------------------------------


def test_post_json_sends_bearer_token_and_returns_the_decoded_body(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen.update(auth=request.get_header("Authorization"), url=request.full_url, timeout=timeout)
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(vision.urllib.request, "urlopen", fake_urlopen)
    assert post_json({"a": 1}, "key-1", 7.5) == {"ok": True}
    assert seen == {"auth": "Bearer key-1", "url": vision.ENDPOINT, "timeout": 7.5}


def test_post_json_reports_status_without_the_response_body(monkeypatch):
    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 402, "no credit", {}, io.BytesIO(b"secret"))

    monkeypatch.setattr(vision.urllib.request, "urlopen", refuse)
    with pytest.raises(VisionError) as caught:
        post_json({}, "key", 1)
    assert "402" in str(caught.value) and "secret" not in str(caught.value)


@pytest.mark.parametrize("failure", [TimeoutError("slow"), urllib.error.URLError("dns")])
def test_post_json_maps_network_failures(monkeypatch, failure):
    def fail(request, timeout):
        raise failure

    monkeypatch.setattr(vision.urllib.request, "urlopen", fail)
    with pytest.raises(VisionError, match="could not be reached"):
        post_json({}, "key", 1)


def test_post_json_maps_a_non_json_reply(monkeypatch):
    monkeypatch.setattr(
        vision.urllib.request, "urlopen", lambda r, timeout: FakeResponse(b"<html>")
    )
    with pytest.raises(VisionError, match="could not be reached"):
        post_json({}, "key", 1)


def test_image_is_downscaled_and_sent_as_a_jpeg_data_url():
    encoded = encode_image(Image.new("RGB", (3000, 1500), "white"), 1000)
    assert encoded.startswith("data:image/jpeg;base64,")
    import base64

    sent = Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
    assert sent.size == (1000, 500)


def test_locator_requests_structured_output_without_leaking_the_key():
    captured = {}

    def transport(body, key, timeout):
        captured.update(body=body, key=key, timeout=timeout)
        return answer([entry("a")])

    settings = Settings(openrouter_api_key="sk-live", vision_model="vendor/model")
    boxes = VisionLocator(settings, transport)(Image.new("RGB", (400, 300), "white"))
    body = captured["body"]
    assert body["model"] == "vendor/model" and body["provider"] == {"data_collection": "deny"}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "sk-live" not in json.dumps(body)
    assert captured["key"] == "sk-live" and captured["timeout"] == settings.vision_timeout_seconds
    assert boxes[0]["character"] == "a" and boxes[0]["x"] == pytest.approx(0.1)


# --- parsing -------------------------------------------------------------------------


def test_coordinates_are_clamped_and_unusable_entries_skipped():
    boxes = parse_glyphs(
        answer(
            [
                entry("a", -50, 0, 2000, 500),  # clamped to the image
                entry("ab"),  # not one character
                entry("é"),  # decomposed accent
                entry("b", 300, 300, 300, 400),  # no width
                entry("c", 500, 500, 400, 600),  # inverted
                {"character": "d"},  # missing coordinates
                {"character": "f", "x0": "x", "y0": 0, "x1": 1, "y1": 1},  # not a number
                entry(" "),  # whitespace
                entry("z", 10, 20, 30, 40),
            ]
        )
    )
    assert [b["character"] for b in boxes] == ["a", "z"]
    assert boxes[0]["x"] == 0 and boxes[0]["width"] == 1


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": '{"glyphs": "nope"}'}}]},
        {"choices": [{"message": {"content": "[]"}}]},
    ],
)
def test_unreadable_answers_raise(response):
    with pytest.raises(VisionError, match="unreadable"):
        parse_glyphs(response)


def test_an_answer_without_any_usable_character_raises():
    with pytest.raises(VisionError, match="did not find"):
        parse_glyphs(answer([entry("ab")]))
    with pytest.raises(VisionError, match="did not find"):
        parse_glyphs(answer([]))


def test_more_than_the_maximum_glyphs_are_truncated():
    boxes = parse_glyphs(answer([entry(chr(0x4E00 + i)) for i in range(200)]))
    assert len(boxes) == vision.MAX_GLYPHS


# --- alignment -----------------------------------------------------------------------


def box(character, x, y, width=0.05, height=0.05):
    return {"character": character, "x": x, "y": y, "width": width, "height": height}


def test_rows_come_from_vertical_position_and_read_left_to_right():
    boxes = [box("b", 0.30, 0.10), box("a", 0.10, 0.11), box("d", 0.30, 0.40), box("c", 0.10, 0.40)]
    assert rows_from_boxes(boxes) == ["ab", "cd"]


SIZE = (1000, 1000)


def group(x, y, w=40, h=60):
    return [x, y, x + w, y + h]


def test_alignment_recovers_exact_shapes_despite_displaced_boxes():
    groups = [group(100, 100), group(200, 100), group(300, 100)]
    # Every model box is off by roughly a glyph width, but order is right.
    boxes = [box("a", 0.14, 0.105), box("b", 0.24, 0.095), box("c", 0.33, 0.10)]
    aligned, errors = align_labels(boxes, groups, 60.0, SIZE)
    assert errors == []
    assert [b["character"] for b in aligned] == ["a", "b", "c"]
    assert aligned[1]["x"] == pytest.approx((200 - 3) / 1000)
    assert aligned[1]["width"] == pytest.approx((40 + 6) / 1000)


def test_stray_marks_between_glyphs_are_skipped():
    groups = [group(100, 100), group(150, 105, 6, 6), group(200, 100)]
    aligned, errors = align_labels([box("a", 0.12, 0.12), box("b", 0.22, 0.12)], groups, 60, SIZE)
    assert errors == [] and aligned[1]["x"] == pytest.approx((200 - 3) / 1000)


def test_pieces_of_one_glyph_merge_into_a_single_box():
    # A colon is two marks. One label owns both.
    groups = [group(100, 100, 40, 60), group(300, 100, 12, 12), group(302, 140, 12, 12)]
    boxes = [box("a", 0.12, 0.12), box(":", 0.30, 0.12)]
    aligned, errors = align_labels(boxes, groups, 60, SIZE)
    assert errors == []
    assert aligned[1]["y"] == pytest.approx((100 - 3) / 1000)
    assert aligned[1]["height"] == pytest.approx((52 + 6) / 1000)


def test_a_label_with_no_ink_nearby_is_reported_and_keeps_its_box():
    groups = [group(100, 100), group(200, 100)]
    boxes = [box("a", 0.12, 0.12), box("b", 0.22, 0.12), box("x", 0.9, 0.9)]
    aligned, errors = align_labels(boxes, groups, 60, SIZE)
    assert len(errors) == 1 and "'x'" in errors[0]
    assert aligned[2] == boxes[2]


def test_alignment_with_no_ink_reports_every_label():
    aligned, errors = align_labels([box("a", 0.1, 0.1)], [], 60, SIZE)
    assert errors and aligned == [box("a", 0.1, 0.1)]


def test_boxes_are_clipped_to_the_image():
    aligned, _ = align_labels([box("a", 0.0, 0.0)], [group(1, 1, 40, 40)], 60, SIZE)
    assert aligned[0]["x"] == 0 and aligned[0]["y"] == 0


# --- grouping ------------------------------------------------------------------------


def two_rows_with_dust():
    mask = np.zeros((400, 600), np.uint8)
    for row in (60, 220):
        for column in (80, 200, 320):
            mask[row : row + 70, column : column + 40] = 255
    mask[20:23, 330:333] = 255  # dust above the first row
    mask[300:303, 90:93] = 255  # dust below the second row
    return mask


def test_dust_stretches_a_glyph_box_unless_a_speck_threshold_is_given():
    mask = two_rows_with_dust()
    with_dust = layout_groups(mask, ["abc", "def"])
    without = layout_groups(mask, ["abc", "def"], speck=0.1)
    # The dust above the third glyph of row one is absorbed into its group.
    assert min(g[1] for g in with_dust[0][0]) == 20
    assert [len(band) for band in without[0]] == [3, 3]
    assert min(g[1] for g in without[0][0]) == 60


def test_an_empty_mask_has_no_layout():
    assert layout_groups(np.zeros((50, 50), np.uint8), ["a"]) is None


# --- analyze -------------------------------------------------------------------------


@pytest.fixture
def sheet(sheet):
    """The shared sheet, with i and full-stop dots as large as real pen dots, not dust."""
    image = Image.open(io.BytesIO(sheet[0]))
    draw = ImageDraw.Draw(image)
    draw.ellipse((1028, 143, 1042, 157), fill=(20, 40, 120))
    draw.ellipse((204, 71, 216, 83), fill=(20, 40, 120))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), sheet[1]


@pytest.fixture
def sheet_file(tmp_path, sheet):
    path = tmp_path / "image.png"
    path.write_bytes(sheet[0])
    return path


def displaced(sheet, shift=0.012):
    # The model reads the sheet in order but places each box a little off.
    return [b | {"x": b["x"] + shift, "y": b["y"] - shift / 2} for b in sheet[1]]


def test_vision_detection_returns_aligned_boxes_and_marks_the_detector(tmp_path, sheet, sheet_file):
    result = analyze(
        sheet_file, tmp_path, [], "auto", None, "vision", lambda image: displaced(sheet)
    )
    assert result["detector"] == "vision" and result["mapping_errors"] == []
    assert [g["character"] for g in result["glyphs"]] == list("oiAVlmg.")
    assert "vision model" in result["warnings"][0]
    # The dot of the i is part of the i, not left behind.
    i_box = result["glyphs"][1]
    assert i_box["y"] * 500 < 75


def test_auto_falls_back_to_row_order_when_the_service_fails(tmp_path, sheet_file):
    def broken(image):
        raise VisionError("The vision service returned HTTP 500.")

    result = analyze(sheet_file, tmp_path, ["oiAVlmg."], "auto", None, "auto", broken)
    assert result["detector"] == "layout"
    assert "HTTP 500" in result["warnings"][0] and "Row order" in result["warnings"][0]


def test_explicit_vision_fails_loudly_instead_of_falling_back(tmp_path, sheet_file):
    def broken(image):
        raise VisionError("The vision service could not be reached.")

    with pytest.raises(ImageError, match="could not be reached"):
        analyze(sheet_file, tmp_path, ["oiAVlmg."], "auto", None, "vision", broken)


def test_explicit_vision_without_a_locator_is_an_error(tmp_path, sheet_file):
    with pytest.raises(ImageError, match="not configured"):
        analyze(sheet_file, tmp_path, ["a"], "auto", None, "vision", None)


def test_auto_without_a_locator_quietly_uses_row_order(tmp_path, sheet_file):
    result = analyze(sheet_file, tmp_path, ["oiAVlmg."], "auto", None, "auto", None)
    assert result["detector"] == "layout" and "recognition" in result["warnings"][0]


def test_supplied_boxes_skip_detection_entirely(tmp_path, sheet, sheet_file):
    def forbidden(image):
        raise AssertionError("the model must not be called when crops are supplied")

    result = analyze(sheet_file, tmp_path, [], "auto", sheet[1], "vision", forbidden)
    assert result["detector"] == "supplied" and result["glyphs"] == sheet[1]


def test_a_blank_page_returns_the_models_boxes_with_an_error(tmp_path):
    blank = tmp_path / "image.png"
    Image.new("RGB", (300, 300), "white").save(blank)
    result = analyze(
        blank, tmp_path, [], "auto", None, "vision", lambda image: [box("a", 0.1, 0.1)]
    )
    assert result["detector"] == "vision" and "No ink" in result["mapping_errors"][0]
    assert result["glyphs"] == [box("a", 0.1, 0.1)]


# --- API -----------------------------------------------------------------------------


def vision_client(tmp_path, locate):
    return TestClient(create_app(Settings(data_dir=tmp_path), locate=locate))


def post_sheet(client, sheet, **data):
    response = client.post(
        "/api/fonts", files={"image": ("s.png", sheet[0], "image/png")}, data=data
    )
    assert response.status_code == 202, response.text
    return wait_job(client, response.json()["id"])


def test_layouts_report_whether_ai_detection_is_available(tmp_path):
    with vision_client(tmp_path / "off", None) as client:
        (tmp_path / "off").mkdir(exist_ok=True)
        assert client.get("/api/layouts").json()["ai_detection"] is False
    (tmp_path / "on").mkdir()
    with vision_client(tmp_path / "on", lambda image: []) as client:
        assert client.get("/api/layouts").json()["ai_detection"] is True


def test_upload_uses_the_vision_detector_and_generates_when_opted_in(tmp_path, sheet):
    with vision_client(tmp_path, lambda image: displaced(sheet)) as client:
        job = post_sheet(client, sheet, detector="vision", auto_generate="true")
        assert job["status"] == "ready" and job["detector"] == "vision"
        assert client.get(job["artifacts"]["font.ttf"]).status_code == 200


def test_upload_requires_review_after_vision_detection_by_default(tmp_path, sheet):
    with vision_client(tmp_path, lambda image: displaced(sheet)) as client:
        job = post_sheet(client, sheet)
        assert job["status"] == "review_required" and job["detector"] == "vision"


def test_a_configured_key_enables_detection_through_settings(tmp_path):
    settings = Settings(data_dir=tmp_path, openrouter_api_key="sk-test")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/layouts").json()["ai_detection"] is True


def test_explicit_vision_on_a_server_without_a_key_fails_the_job_clearly(tmp_path, sheet):
    with vision_client(tmp_path, None) as client:
        job = post_sheet(client, sheet, detector="vision")
        assert job["status"] == "failed" and "not configured" in job["error"]


def test_unknown_detector_is_rejected(client, sheet):
    response = client.post(
        "/api/fonts", files={"image": ("s.png", sheet[0])}, data={"detector": "magic"}
    )
    assert response.status_code == 422

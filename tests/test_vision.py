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
    align_order,
    align_rows,
    analyze,
    drop_stray_ends,
    layout_groups,
    pair_cost,
)
from handwriting.vision import VisionError, VisionLocator, encode_image, parse_rows, post_json
from tests.conftest import wait_job


def answer(text):
    return {"choices": [{"message": {"content": text}}]}


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


def test_locator_asks_for_plain_text_fast_and_private_without_leaking_the_key():
    captured = {}

    def transport(body, key, timeout):
        captured.update(body=body, key=key, timeout=timeout)
        return answer("ab\ncd")

    settings = Settings(openrouter_api_key="sk-live", vision_model="vendor/model")
    rows = VisionLocator(settings, transport)(Image.new("RGB", (400, 300), "white"))
    body = captured["body"]
    assert rows == ["ab", "cd"]
    assert body["model"] == "vendor/model" and "response_format" not in body
    assert body["provider"] == {"data_collection": "deny", "sort": "throughput"}
    assert "sk-live" not in json.dumps(body)
    assert captured["key"] == "sk-live" and captured["timeout"] == settings.vision_timeout_seconds


def test_a_retry_uses_the_higher_reasoning_effort():
    efforts = []

    def transport(body, key, timeout):
        efforts.append(body["reasoning"]["effort"])
        return answer("a")

    locate = VisionLocator(
        Settings(openrouter_api_key="k", vision_reasoning_effort="low", vision_retry_effort="high"),
        transport,
    )
    image = Image.new("RGB", (50, 50), "white")
    locate(image)
    locate(image, 1)
    assert efforts == ["low", "high"]


# --- parsing -------------------------------------------------------------------------


def test_rows_drop_spaces_fences_blank_lines_and_invisible_characters():
    text = "```\nA B  C\n\n d\u200be f \n```"
    assert parse_rows(answer(text)) == ["ABC", "def"]


def test_text_is_composed_so_one_character_is_one_label():
    assert parse_rows(answer("e\u0301")) == ["é"]


@pytest.mark.parametrize(
    "response",
    [{}, {"choices": []}, {"choices": [{"message": {"content": None}}]}, {"choices": [{}]}],
)
def test_missing_text_raises_an_unreadable_answer_error(response):
    with pytest.raises(VisionError, match="unreadable"):
        parse_rows(response)


def test_an_answer_without_characters_or_with_too_many_raises():
    with pytest.raises(VisionError, match="did not find"):
        parse_rows(answer("  \n \n"))
    with pytest.raises(VisionError, match="more than"):
        parse_rows(answer("a" * 161))


# --- alignment -----------------------------------------------------------------------

SIZE = (1000, 1000)
SCALE = 60.0


def group(x, y=100, w=40, h=60):
    return [x, y, x + w, y + h]


def test_pair_cost_prefers_stacked_then_near_then_forbids_far_marks():
    assert pair_cost(group(100), group(105, 200, 30, 20), SCALE) == pytest.approx(0.1)
    assert pair_cost(group(100), group(150), SCALE) == pytest.approx(0.6)
    assert pair_cost(group(100), group(400), SCALE) == pytest.approx(4.0)


def test_labels_pair_with_groups_in_order():
    groups = [group(100), group(200), group(300)]
    assert align_order("abc", groups, SCALE) == {0: (0, 1), 1: (1, 1), 2: (2, 1)}


def test_a_small_stray_mark_is_skipped_rather_than_a_real_glyph():
    groups = [group(100), group(200, 100, 8, 8), group(300)]
    assert align_order("ab", groups, SCALE) == {0: (0, 1), 1: (2, 1)}


def test_stacked_pieces_merge_into_one_glyph():
    groups = [group(100), group(300, 100, 12, 12), group(302, 140, 12, 12)]
    assert align_order("a:", groups, SCALE) == {0: (0, 1), 1: (1, 2)}


def test_a_label_with_no_mark_left_is_unmatched():
    assert align_order("ab", [group(100)], SCALE) == {0: (0, 1)}
    assert align_order("a", [], SCALE) == {}


def layout(*rows):
    return ([list(band) for band in rows], SCALE)


def test_rows_pair_with_page_rows_and_boxes_are_exact_and_padded():
    page = layout([group(100), group(200)], [group(100, 300)])
    boxes, errors = align_rows(["ab", "c"], page, SIZE)
    assert errors == []
    assert [b["character"] for b in boxes] == ["a", "b", "c"]
    assert boxes[1]["x"] == pytest.approx((200 - 3) / 1000)
    assert boxes[1]["width"] == pytest.approx(46 / 1000)
    assert boxes[2]["y"] == pytest.approx((300 - 3) / 1000)


def test_a_row_with_a_different_number_of_marks_is_reported_with_its_number():
    page = layout([group(100), group(200)], [group(100, 300), group(200, 300), group(300, 300)])
    boxes, errors = align_rows(["ab", "cd"], page, SIZE)
    assert errors == ["Row 2: found 3 marks for 2 characters."]
    assert len(boxes) == 4


def test_unequal_row_counts_fall_back_to_one_sequence_and_are_reported():
    page = layout([group(100), group(200), group(300)])
    boxes, errors = align_rows(["ab", "c"], page, SIZE)
    assert errors == ["Found 1 rows of writing but 2 were transcribed."]
    assert [b["character"] for b in boxes] == ["a", "b", "c"]


def test_a_character_with_no_ink_is_reported_and_has_no_box():
    boxes, errors = align_rows(["ab"], layout([group(100)]), SIZE)
    assert [b["character"] for b in boxes] == ["a"]
    assert "The page: found 1 marks for 2 characters." in errors
    assert "No ink was found for 'b'." in errors


def test_boxes_are_clipped_to_the_image():
    boxes, _ = align_rows(["a"], layout([group(1, 1, 40, 40)]), SIZE)
    assert boxes[0]["x"] == 0 and boxes[0]["y"] == 0


# --- stray marks ---------------------------------------------------------------------


def test_small_isolated_marks_at_either_end_of_a_row_are_dropped():
    row = [group(400 + 60 * i) for i in range(6)]
    assert drop_stray_ends([group(5, 100, 10, 10), *row], SCALE) == row  # leading
    assert drop_stray_ends([*row, group(900, 100, 10, 10)], SCALE) == row  # trailing
    assert drop_stray_ends([group(5, 100, 10, 10), *row, group(900, 100, 10, 10)], SCALE) == row


def test_end_marks_that_are_large_or_close_or_in_short_rows_are_kept():
    row = [group(100 + 60 * i) for i in range(6)]
    far_but_large = [*row, group(900)]
    close_and_small = [*row, group(100 + 60 * 6, 100, 10, 10)]
    assert drop_stray_ends(far_but_large, SCALE) == far_but_large
    assert drop_stray_ends(close_and_small, SCALE) == close_and_small
    short = [group(100), group(200), group(900, 100, 10, 10)]
    assert drop_stray_ends(short, SCALE) == short


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


def reads(*answers):
    """A locator that returns each answer in turn, or raises it when it is an error."""
    calls = []

    def locate(image, attempt=0):
        calls.append(attempt)
        value = answers[min(len(calls), len(answers)) - 1]
        if isinstance(value, Exception):
            raise value
        return value

    locate.calls = calls
    return locate


def test_vision_detection_returns_exact_boxes_for_the_transcribed_rows(tmp_path, sheet_file):
    locate = reads(["oiAVlmg."])
    result = analyze(sheet_file, tmp_path, [], "auto", None, "vision", locate)
    assert result["detector"] == "vision" and result["mapping_errors"] == []
    assert [g["character"] for g in result["glyphs"]] == list("oiAVlmg.")
    assert "vision model" in result["warnings"][0]
    assert result["glyphs"][1]["y"] * 500 < 75  # the dot of the i is part of the i
    assert locate.calls == [0], "an answer that agrees with the ink needs no retry"


def test_a_transcription_that_disagrees_with_the_ink_is_retried_once_and_the_better_kept(
    tmp_path, sheet_file
):
    locate = reads(["oiAVlm"], ["oiAVlmg."])
    result = analyze(sheet_file, tmp_path, [], "auto", None, "vision", locate)
    assert locate.calls == [0, 1] and result["mapping_errors"] == []
    assert len(result["glyphs"]) == 8


def test_a_worse_retry_does_not_replace_the_first_answer(tmp_path, sheet_file):
    locate = reads(["oiAVlm"], ["oiAVlmgxyz"])
    result = analyze(sheet_file, tmp_path, [], "auto", None, "vision", locate)
    assert locate.calls == [0, 1]
    assert [g["character"] for g in result["glyphs"]] == list("oiAVlm")


def test_a_failed_retry_keeps_the_first_answer_for_review(tmp_path, sheet_file):
    locate = reads(["oiAVlm"], VisionError("The vision service could not be reached."))
    result = analyze(sheet_file, tmp_path, [], "auto", None, "vision", locate)
    assert result["detector"] == "vision" and result["mapping_errors"]
    assert len(result["glyphs"]) == 6


def test_auto_falls_back_to_row_order_when_the_service_fails(tmp_path, sheet_file):
    locate = reads(VisionError("The vision service returned HTTP 500."))
    result = analyze(sheet_file, tmp_path, ["oiAVlmg."], "auto", None, "auto", locate)
    assert result["detector"] == "layout"
    assert "HTTP 500" in result["warnings"][0] and "Row order" in result["warnings"][0]


def test_explicit_vision_fails_loudly_instead_of_falling_back(tmp_path, sheet_file):
    locate = reads(VisionError("The vision service could not be reached."))
    with pytest.raises(ImageError, match="could not be reached"):
        analyze(sheet_file, tmp_path, ["oiAVlmg."], "auto", None, "vision", locate)


def test_explicit_vision_without_a_locator_is_an_error(tmp_path, sheet_file):
    with pytest.raises(ImageError, match="not configured"):
        analyze(sheet_file, tmp_path, ["a"], "auto", None, "vision", None)


def test_auto_without_a_locator_quietly_uses_row_order(tmp_path, sheet_file):
    result = analyze(sheet_file, tmp_path, ["oiAVlmg."], "auto", None, "auto", None)
    assert result["detector"] == "layout" and "recognition" in result["warnings"][0]


def test_supplied_boxes_skip_detection_entirely(tmp_path, sheet, sheet_file):
    locate = reads(AssertionError("the model must not be called when crops are supplied"))
    result = analyze(sheet_file, tmp_path, [], "auto", sheet[1], "vision", locate)
    assert result["detector"] == "supplied" and result["glyphs"] == sheet[1]
    assert locate.calls == []


def test_a_blank_page_reports_no_ink_without_a_retry(tmp_path):
    blank = tmp_path / "image.png"
    Image.new("RGB", (300, 300), "white").save(blank)
    locate = reads(["abc"])
    result = analyze(blank, tmp_path, [], "auto", None, "vision", locate)
    assert result["detector"] == "vision" and "No ink" in result["mapping_errors"][0]
    assert result["glyphs"] == [] and locate.calls == [0]


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
    (tmp_path / "off").mkdir()
    with vision_client(tmp_path / "off", None) as client:
        assert client.get("/api/layouts").json()["ai_detection"] is False
    (tmp_path / "on").mkdir()
    with vision_client(tmp_path / "on", reads(["a"])) as client:
        assert client.get("/api/layouts").json()["ai_detection"] is True


def test_upload_uses_the_vision_detector_and_generates_when_opted_in(tmp_path, sheet):
    with vision_client(tmp_path, reads(["oiAVlmg."])) as client:
        job = post_sheet(client, sheet, detector="vision", auto_generate="true")
        assert job["status"] == "ready" and job["detector"] == "vision"
        assert client.get(job["artifacts"]["font.ttf"]).status_code == 200


def test_upload_requires_review_after_vision_detection_by_default(tmp_path, sheet):
    with vision_client(tmp_path, reads(["oiAVlmg."])) as client:
        job = post_sheet(client, sheet)
        assert job["status"] == "review_required" and job["detector"] == "vision"


def test_a_disagreeing_transcription_blocks_automatic_generation(tmp_path, sheet):
    with vision_client(tmp_path, reads(["oiAVlm"])) as client:
        job = post_sheet(client, sheet, detector="vision", auto_generate="true")
        assert job["status"] == "review_required" and job["mapping_errors"]


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

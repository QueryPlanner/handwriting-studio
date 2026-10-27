"""Locate and label handwritten characters with a vision model through OpenRouter.

The model proposes a label and a rough box for every character it can see. Boxes are
untrusted: they are validated here and snapped to the traced ink in `imaging`, and the
studio still lets a person correct them before a font is generated.
"""

import base64
import io
import json
import logging
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable

from PIL import Image
from pydantic import ValidationError

from handwriting.config import Settings
from handwriting.models import DEFAULT_ROWS, GlyphBox

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_GLYPHS = 160
SCALE = 1000  # The model reports integer coordinates on a 0-1000 grid.

PROMPT = (
    "This photograph shows handwriting where each character was written separately, such "
    "as an alphabet sheet. The writer may have added extra letters, digits, punctuation or "
    "symbols (for example the rupee sign ₹), and may have used lined paper. List every "
    "handwritten character you can see in reading order, one entry per character. Give "
    "the character itself and a tight bounding box as integers on a 0-1000 scale: x0, y0 "
    "is the top-left corner and x1, y1 the bottom-right, relative to the image width and "
    "height. The box must include every stroke of that character, such as the dots of i "
    "and j and the tails of g, j, p, q and y. Use the surrounding characters to decide "
    "between look-alikes, for example a rupee sign among symbols. Ignore printed text, "
    "ruled lines, page edges and faint writing showing through from the other side. "
    "Many sheets follow this template, one line per row: "
    + " | ".join(DEFAULT_ROWS)
    + ". Use it to resolve sloppy marks, for example a hurried ? or !, only where the "
    "writing really follows it. Report what is written, including extra or missing "
    "characters, and never invent a character that is not on the page."
)

SCHEMA = {
    "name": "handwritten_characters",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["glyphs"],
        "properties": {
            "glyphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["character", "x0", "y0", "x1", "y1"],
                    "properties": {
                        "character": {"type": "string"},
                        "x0": {"type": "integer"},
                        "y0": {"type": "integer"},
                        "x1": {"type": "integer"},
                        "y1": {"type": "integer"},
                    },
                },
            }
        },
    },
}


class VisionError(RuntimeError):
    """The vision service could not provide usable character boxes."""


Transport = Callable[[dict, str, float], dict]


def post_json(body: dict, api_key: str, timeout: float) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        json.dumps(body).encode(),
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Report the status only. The body may echo prompt or account details.
        raise VisionError(f"The vision service returned HTTP {exc.code}.") from exc
    except (OSError, ValueError) as exc:
        raise VisionError("The vision service could not be reached.") from exc


def encode_image(image: Image.Image, max_side: int) -> str:
    picture = image.copy()
    picture.thumbnail((max_side, max_side))
    output = io.BytesIO()
    picture.save(output, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()


def parse_glyphs(response: dict) -> list[dict]:
    try:
        content = response["choices"][0]["message"]["content"]
        entries = json.loads(content)["glyphs"]
        if not isinstance(entries, list):
            raise TypeError("glyphs is not a list")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise VisionError("The vision service returned an unreadable answer.") from exc
    boxes = []
    for entry in entries[:MAX_GLYPHS]:
        try:
            x0, y0, x1, y1 = (min(max(int(entry[k]), 0), SCALE) for k in ("x0", "y0", "x1", "y1"))
            character = entry["character"]
            if len(character) != 1 or unicodedata.normalize("NFC", character) != character:
                raise ValueError("one composed character per box")
            boxes.append(
                GlyphBox(
                    character=character,
                    x=x0 / SCALE,
                    y=y0 / SCALE,
                    width=(x1 - x0) / SCALE,
                    height=(y1 - y0) / SCALE,
                ).model_dump()
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            logger.info("Skipped an unusable vision box.")
    if not boxes:
        raise VisionError("The vision service did not find any characters.")
    return boxes


class VisionLocator:
    """Callable that turns a photograph into proposed glyph boxes."""

    def __init__(self, settings: Settings, transport: Transport = post_json):
        if not settings.openrouter_api_key:
            raise VisionError("OPENROUTER_API_KEY is not configured.")
        self.settings, self.transport = settings, transport

    def __call__(self, image: Image.Image) -> list[dict]:
        body = {
            "model": self.settings.vision_model,
            "max_tokens": 8000,
            "reasoning": {"effort": self.settings.vision_reasoning_effort},
            # Handwriting photos are personal. Ask OpenRouter to avoid providers that train on them.
            "provider": {"data_collection": "deny"},
            "response_format": {"type": "json_schema", "json_schema": SCHEMA},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(image, self.settings.vision_max_side)
                            },
                        },
                    ],
                }
            ],
        }
        response = self.transport(
            body, self.settings.openrouter_api_key, self.settings.vision_timeout_seconds
        )
        return parse_glyphs(response)

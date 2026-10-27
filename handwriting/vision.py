"""Read the characters on a handwriting sheet with a vision model through OpenRouter.

The model only transcribes: one line of text per row of handwriting, in reading order.
It returns no coordinates, which keeps the answer to a few hundred tokens. The shapes
come from the ink itself in `imaging`, and the studio still lets a person correct them
before a font is generated.
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

from handwriting.config import Settings
from handwriting.models import DEFAULT_ROWS, LOOKALIKES

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_GLYPHS = 160

PROMPT = (
    "This photograph shows handwriting where each character was written separately, such "
    "as an alphabet sheet. The writer may have added extra letters, digits, punctuation or "
    "symbols (for example the rupee sign ₹), and may have used lined paper. Transcribe "
    "every handwritten character in reading order. Answer with plain text only: one line "
    "per visual row of handwriting, the characters of a row joined with no spaces or "
    "separators, and nothing else. Include every mark, even an odd or repeated one. "
    "Ignore printed text, ruled lines, page edges and faint writing showing through from "
    "the other side. Many sheets follow this template, one line per row: "
    + " | ".join(DEFAULT_ROWS)
    + ". Use it to resolve sloppy marks, for example a hurried ? or !, only where the "
    "writing really follows it. Report what is written, including extra or missing "
    "characters, and never invent a character that is not on the page."
)


class VisionError(RuntimeError):
    """The vision service could not provide a usable transcription."""


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


def parse_rows(response: dict) -> list[str]:
    try:
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("the answer has no text")
    except (KeyError, IndexError, TypeError) as exc:
        # A reasoning model can spend its whole budget thinking and return no text.
        raise VisionError("The vision service returned an unreadable answer.") from exc
    rows = []
    for line in content.replace("```", "").splitlines():
        # Drop whitespace the model adds between characters, and any unusable character.
        kept = "".join(
            c
            for c in unicodedata.normalize("NFC", line).translate(LOOKALIKES)
            if not c.isspace() and not unicodedata.category(c).startswith("C")
        )
        if kept:
            rows.append(kept)
    total = sum(map(len, rows))
    if not total:
        raise VisionError("The vision service did not find any characters.")
    if total > MAX_GLYPHS:
        raise VisionError(f"The vision service reported more than {MAX_GLYPHS} characters.")
    return rows


class VisionLocator:
    """Callable that transcribes a photograph into rows of characters.

    `attempt` 0 uses the configured reasoning effort. A retry uses the higher one,
    because a retry means the first answer disagreed with the ink on the page.
    """

    def __init__(self, settings: Settings, transport: Transport = post_json):
        if not settings.openrouter_api_key:
            raise VisionError("OPENROUTER_API_KEY is not configured.")
        self.settings, self.transport = settings, transport

    def __call__(self, image: Image.Image, attempt: int = 0) -> list[str]:
        effort = (
            self.settings.vision_reasoning_effort
            if attempt == 0
            else self.settings.vision_retry_effort
        )
        body = {
            "model": self.settings.vision_model,
            "max_tokens": 4000,
            "reasoning": {"effort": effort},
            # Handwriting photos are personal. Ask OpenRouter to avoid providers that train
            # on them, and to pick the fastest one that qualifies.
            "provider": {"data_collection": "deny", "sort": "throughput"},
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
        return parse_rows(response)

"""Public mapping contracts. Coordinates refer to the EXIF-corrected source image."""

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_ROWS = [
    "ABCDEFGHIJK",
    "LMNOPQRSTUV",
    "WXYZ",
    "abcdefghijk",
    "lmnopqrstuv",
    "wxyz",
    "1234567890!",
    ".,!?:;'\"-_()",
    "/@#&+=%₹",
]


class GlyphBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character: str = Field(min_length=1, max_length=1)
    x: float = Field(ge=0, lt=1, allow_inf_nan=False)
    y: float = Field(ge=0, lt=1, allow_inf_nan=False)
    width: float = Field(gt=0, le=1, allow_inf_nan=False)
    height: float = Field(gt=0, le=1, allow_inf_nan=False)
    # Overrides use font units per em, which is 1000. Width scales with height.
    target_height: int | None = Field(default=None, ge=20, le=800)
    baseline: int | None = Field(default=None, ge=-250, le=650)

    @field_validator("character")
    @classmethod
    def printable_character(cls, value: str) -> str:
        if value.isspace() or unicodedata.category(value).startswith("C"):
            raise ValueError("Use one visible Unicode character. Space is generated automatically.")
        return value

    @model_validator(mode="after")
    def inside_image(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("The crop must be inside the image.")
        return self


class MappingUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    glyphs: list[GlyphBox] = Field(min_length=1, max_length=160)
    ink_mode: Literal["auto", "blue", "dark"] = "auto"


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm_mapping: Literal[True]


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(
        default="The quick brown fox jumps over the lazy dog.", min_length=1, max_length=1000
    )
    font_size: int = Field(default=48, ge=12, le=120)
    width: int = Field(default=1000, ge=320, le=1600)


def validate_rows(rows: list[str]) -> list[str]:
    if not 1 <= len(rows) <= 16 or not 1 <= sum(map(len, rows)) <= 160:
        raise ValueError("Provide 1 to 16 nonempty rows and at most 160 characters.")
    for row in rows:
        if not row or any(c.isspace() or unicodedata.category(c).startswith("C") for c in row):
            raise ValueError("Rows must contain visible characters without spaces.")
    return rows

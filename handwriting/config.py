"""Deployment settings, supplied through environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            value.strip()
            for value in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if value.strip()
        )
    )
    max_upload_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    # JPEGs decode at reduced scale, so camera originals cost far less memory.
    max_jpeg_pixels: int = 64_000_000
    max_image_side: int = 2400
    # Optional AI character detection. The key is read from the environment only and
    # is excluded from repr so it cannot reach logs.
    openrouter_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY") or None, repr=False
    )
    vision_model: str = field(
        default_factory=lambda: os.getenv("VISION_MODEL", "openai/gpt-6-luna")
    )
    vision_timeout_seconds: float = 120.0
    vision_max_side: int = 1568
    vision_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("VISION_REASONING_EFFORT", "medium")
    )
    max_pending_jobs: int = 20
    max_data_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_DATA_BYTES", str(1024**3)))
    )
    minimum_free_bytes: int = 256 * 1024 * 1024

    @property
    def database(self) -> Path:
        return self.data_dir / "jobs.sqlite3"

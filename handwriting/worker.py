"""One bounded worker backed by SQLite, with restart recovery."""

import fcntl
import json
import logging
import shutil
import threading
import uuid
from pathlib import Path

from handwriting.font import build_font
from handwriting.imaging import ImageError, Locator, analyze
from handwriting.store import Store

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, store: Store, data_dir: Path, locate: Locator | None = None):
        self.store, self.data_dir, self.locate = store, data_dir, locate
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self.run, name="font-converter", daemon=True)
        self.lock_file = None

    def start(self):
        self.lock_file = (self.data_dir / ".worker.lock").open("a")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_file.close()
            raise RuntimeError("DATA_DIR is already in use. Run Uvicorn with --workers 1.") from exc
        self.store.initialize()
        self.store.recover()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.wake.set()
        self.thread.join()
        self.lock_file.close()

    def notify(self):
        self.wake.set()

    def run(self):
        while not self.stop_event.is_set():
            job = self.store.claim()
            if job is None:
                self.wake.wait(0.25)
                self.wake.clear()
                continue
            self.process(job)

    def process(self, job: dict):
        root = self.data_dir / job["id"]
        result = job["result"]
        staging = None
        try:
            if job["stage"] == "analyze":
                result = analyze(
                    root / "image.png",
                    root,
                    job["options"]["rows"],
                    job["options"]["ink_mode"],
                    job["options"].get("glyphs"),
                    job["options"].get("detector", "layout"),
                    self.locate,
                )
                if job["options"]["auto_generate"] and not result["mapping_errors"]:
                    self.store.finish(job["id"], "queued", "generate", result)
                else:
                    self.store.finish(job["id"], "review_required", "generate", result)
                return
            # Use a fresh staging directory on each attempt. Never serve partial files.
            staging = root / f".build-{uuid.uuid4().hex}"
            staging.mkdir()
            version = f"v{job['revision']}-{uuid.uuid4().hex[:8]}"
            family = f"{job['name']} {job['id'][:8]} v{job['revision']}"
            font_result = build_font(
                root / "image.png",
                staging,
                result["glyphs"],
                job["options"]["ink_mode"],
                family,
                version,
            )
            (staging / "report.json").write_text(
                json.dumps(font_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # Rename and database publication happen in this order. A crash between
            # them leaves an unreferenced version, never a ready but partial result.
            destination = root / version
            staging.rename(destination)
            result = result | font_result | {"artifacts": version}
            self.store.finish(job["id"], "ready", "complete", result)
        except (ImageError, ValueError) as exc:
            self.store.finish(job["id"], "failed", job["stage"], result, str(exc))
        except Exception as exc:
            # Do not log user text, paths, or image contents.
            logger.error("Font operation failed: %s", type(exc).__name__)
            self.store.finish(
                job["id"],
                "failed",
                job["stage"],
                result,
                "Conversion failed. Review the crops and try again.",
            )
        finally:
            if staging is not None and staging.exists():
                try:
                    shutil.rmtree(staging)
                except OSError as exc:
                    logger.error("Staging cleanup failed: %s", type(exc).__name__)

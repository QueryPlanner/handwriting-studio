"""HTTP contracts and browser studio for the handwriting conversion service."""

import io
import json
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from handwriting.config import Settings
from handwriting.font import preview_bytes
from handwriting.imaging import ImageError, decode_image, ink_mask
from handwriting.models import (
    DEFAULT_ROWS,
    GenerateRequest,
    GlyphBox,
    MappingUpdate,
    PreviewRequest,
    validate_rows,
)
from handwriting.store import QueueFull, Store
from handwriting.worker import Worker

STATIC = Path(__file__).parent / "static"
ARTIFACTS = {
    "font.ttf": "font/ttf",
    "font.woff2": "font/woff2",
    "font.css": "text/css",
    "sample.png": "image/png",
    "report.json": "application/json",
}


class UploadLimit:
    """Bound the multipart body before Starlette spools it, including chunked uploads."""

    def __init__(self, app, maximum: int):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/fonts" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.maximum:
                return await JSONResponse(
                    {"detail": "Upload exceeds the request size limit."}, status_code=413
                )(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.database)
    worker = Worker(store, settings.data_dir)

    @asynccontextmanager
    async def lifespan(app):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        worker.start()
        yield
        await run_in_threadpool(worker.stop)

    app = FastAPI(
        title="Handwriting Studio API",
        version="0.1.0",
        lifespan=lifespan,
        description="Trace labeled handwriting into TTF and WOFF2 fonts. No OCR is used.",
    )
    app.state.store, app.state.worker = store, worker
    app.add_middleware(UploadLimit, maximum=settings.max_upload_bytes + 128 * 1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Missing-Characters", "Content-Disposition"],
    )

    def get_job(job_id: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Font job not found.")
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Font job not found.")
        return job

    def public(job: dict) -> dict:
        root = f"/api/fonts/{job['id']}"
        result = {k: v for k, v in job["result"].items() if k != "artifacts"}
        urls = {}
        if job["status"] == "ready":
            urls = {name: f"{root}/versions/{job['revision']}/{name}" for name in ARTIFACTS}
        return {
            "id": job["id"],
            "name": job["name"],
            "status": job["status"],
            "stage": job["stage"],
            "revision": job["revision"],
            "error": job["error"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "image_url": root + "/image",
            "mask_url": root + "/mask",
            "preview_url": root + "/preview",
            "artifacts": urls,
            **result,
            "ink_mode": job["options"]["ink_mode"],
        }

    def ready_path(job: dict) -> Path:
        if job["status"] != "ready":
            raise HTTPException(409, "The font is not ready. Check the job status.")
        return settings.data_dir / job["id"] / job["result"]["artifacts"]

    def ensure_storage_capacity():
        used = sum(path.stat().st_size for path in settings.data_dir.rglob("*") if path.is_file())
        free = shutil.disk_usage(settings.data_dir).free
        if used >= settings.max_data_bytes or free < settings.minimum_free_bytes:
            raise HTTPException(
                503, "Storage capacity reached. Try again after space is available."
            )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/layouts")
    def layouts():
        return {
            "default_rows": DEFAULT_ROWS,
            "coordinates": "normalized, EXIF-corrected image",
            "ink_modes": ["auto", "blue", "dark"],
            "max_glyphs": 160,
        }

    @app.post("/api/fonts", status_code=202)
    async def upload(
        image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP, at most 15 MiB.")],
        name: Annotated[str, Form(min_length=1, max_length=60)] = "My Handwriting",
        rows: Annotated[str | None, Form(description="JSON array of character rows.")] = None,
        glyphs: Annotated[
            str | None, Form(description="Optional JSON array of labeled crop boxes.")
        ] = None,
        ink_mode: Annotated[Literal["auto", "blue", "dark"], Form()] = "auto",
        auto_generate: Annotated[
            bool, Form(description="Explicitly accept inferred character labels.")
        ] = False,
    ):
        ensure_storage_capacity()
        if not re.fullmatch(r"[\w .-]{1,60}", name, flags=re.UNICODE) or not name.strip():
            raise HTTPException(
                422, "Font name must contain letters, numbers, spaces, dots, or hyphens."
            )
        try:
            parsed_rows = json.loads(rows) if rows is not None else DEFAULT_ROWS
            if not isinstance(parsed_rows, list) or not all(
                isinstance(r, str) for r in parsed_rows
            ):
                raise ValueError("rows must be a JSON array of strings.")
            parsed_rows = validate_rows(parsed_rows)
            parsed_boxes = None
            if glyphs is not None:
                values = json.loads(glyphs)
                if not isinstance(values, list) or not 1 <= len(values) <= 160:
                    raise ValueError("Provide 1 to 160 glyph boxes.")
                parsed_boxes = [GlyphBox.model_validate(box).model_dump() for box in values]
        except (ValueError, ValidationError) as exc:
            raise HTTPException(422, str(exc)) from exc
        data = await image.read(settings.max_upload_bytes + 1)
        await image.close()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "Image exceeds the 15 MiB upload limit.")
        try:
            decoded = await run_in_threadpool(decode_image, data, settings)
        except ImageError as exc:
            raise HTTPException(422, str(exc)) from exc
        job_id = uuid.uuid4().hex
        root = settings.data_dir / job_id
        root.mkdir()
        try:
            await run_in_threadpool(decoded.save, root / "image.png")
            store.create(
                job_id,
                name.strip(),
                {
                    "rows": parsed_rows,
                    "glyphs": parsed_boxes,
                    "ink_mode": ink_mode,
                    "auto_generate": auto_generate,
                },
                settings.max_pending_jobs,
            )
        except QueueFull as exc:
            shutil.rmtree(root)
            raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
        except OSError as exc:
            shutil.rmtree(root)
            raise HTTPException(503, "The upload could not be stored. Try again later.") from exc
        worker.notify()
        return JSONResponse(
            public(get_job(job_id)), status_code=202, headers={"Location": f"/api/fonts/{job_id}"}
        )

    @app.get("/api/fonts/{job_id}")
    def status(job_id: str):
        return JSONResponse(public(get_job(job_id)), headers={"Cache-Control": "no-store"})

    @app.put("/api/fonts/{job_id}/glyphs")
    def update_mapping(job_id: str, update: MappingUpdate):
        get_job(job_id)
        try:
            store.edit(job_id, [box.model_dump() for box in update.glyphs], update.ink_mode)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return public(get_job(job_id))

    @app.post("/api/fonts/{job_id}/generate", status_code=202)
    def generate(job_id: str, request: GenerateRequest):
        get_job(job_id)
        ensure_storage_capacity()
        try:
            store.queue_generation(job_id, settings.max_pending_jobs)
        except QueueFull as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        worker.notify()
        return public(get_job(job_id))

    @app.get("/api/fonts/{job_id}/image")
    def source_image(job_id: str):
        get_job(job_id)
        return FileResponse(
            settings.data_dir / job_id / "image.png",
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/fonts/{job_id}/mask")
    def source_mask(job_id: str):
        job = get_job(job_id)
        if job["stage"] == "analyze":
            raise HTTPException(409, "The mask is not ready.")
        with Image.open(settings.data_dir / job_id / "image.png") as source:
            mask, _ = ink_mask(source.convert("RGB"), job["options"]["ink_mode"])
        output = io.BytesIO()
        Image.fromarray(255 - mask).save(output, format="PNG")
        return Response(
            output.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/fonts/{job_id}/versions/{revision}/{filename}")
    def artifact(job_id: str, revision: int, filename: str):
        job = get_job(job_id)
        if filename not in ARTIFACTS or revision != job["revision"]:
            raise HTTPException(404, "Artifact not found for this font version.")
        path = ready_path(job) / filename
        if not path.is_file():
            raise HTTPException(404, "Artifact file is missing.")
        return FileResponse(
            path,
            media_type=ARTIFACTS[filename],
            filename=filename if filename.endswith((".ttf", ".woff2")) else None,
            headers={"Cache-Control": "private, max-age=0, must-revalidate"},
        )

    def preview_response(job_id: str, request: PreviewRequest):
        job = get_job(job_id)
        root = ready_path(job)
        supported = set(job["result"]["supported_characters"]) | {"\n"}
        missing = "".join(sorted(set(request.text) - supported))
        try:
            output = preview_bytes(root / "font.ttf", request)
        except ImageError as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(
            output,
            media_type="image/png",
            headers={"X-Missing-Characters": json.dumps(missing), "Cache-Control": "no-store"},
        )

    @app.post("/api/fonts/{job_id}/preview")
    def preview_post(job_id: str, request: PreviewRequest):
        return preview_response(job_id, request)

    @app.get("/api/fonts/{job_id}/preview")
    def preview_get(
        job_id: str,
        text: Annotated[str, Query(min_length=1, max_length=1000)] = "Hello, handwriting!",
        font_size: Annotated[int, Query(ge=12, le=120)] = 48,
        width: Annotated[int, Query(ge=320, le=1600)] = 1000,
    ):
        return preview_response(job_id, PreviewRequest(text=text, font_size=font_size, width=width))

    @app.get("/", include_in_schema=False)
    def studio():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()

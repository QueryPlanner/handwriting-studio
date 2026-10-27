"""Exercise a running API with a synthetic sheet or a private reference photo."""

import argparse
import json
import time
from pathlib import Path

import httpx

from handwriting.font import validate_exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, default=Path(__file__).parent / "reference.png")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with httpx.Client(base_url=args.url, timeout=30) as client:
        with args.image.open("rb") as photo:
            response = client.post(
                "/api/fonts",
                files={"image": (args.image.name, photo)},
                data={
                    "name": "Photo Handwriting",
                    "ink_mode": "auto",
                    "auto_generate": "true",
                    "glyphs": (root / "glyphs.json").read_text(),
                },
            )
        response.raise_for_status()
        job = response.json()
        deadline = time.monotonic() + 90
        while job["status"] in {"queued", "processing"}:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Check job {job['id']} for progress.")
            time.sleep(0.2)
            response = client.get(f"/api/fonts/{job['id']}")
            response.raise_for_status()
            job = response.json()
        if job["status"] != "ready":
            raise RuntimeError(json.dumps(job, indent=2))
        output = root / "generated"
        output.mkdir(exist_ok=True)
        for name, url in job["artifacts"].items():
            response = client.get(url)
            response.raise_for_status()
            (output / name).write_bytes(response.content)
        validate_exports(output / "font.ttf", output / "font.woff2")
        print(f"Preview: {args.url}/#{job['id']}")
        print(f"Font artifacts: {output}")
        print(f"Character mappings: {len(job['supported_characters'])}")
        print(f"Kerning pairs: {job['kerning_pairs']}")


if __name__ == "__main__":
    main()

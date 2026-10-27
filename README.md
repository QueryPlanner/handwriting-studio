# Handwriting Studio

A Python FastAPI service that turns photographs of separately written characters into TTF and WOFF2 fonts. SQLite stores jobs, mappings, revisions, warnings, and conversion results. Font files and normalized source images live in a local data directory.

The included browser studio is one step: choose a photo and it reads the letters, builds the font, and offers a try-it box and downloads. The frontend can run on Vercel while the API and SQLite storage run on a persistent Docker server. All studio assets are included and self-hosted: the Lord Handwriting font for headings, and Noto Sans for interface text.

## Run locally

Use Python 3.12 or newer on macOS or Linux. The worker uses a Unix file lock.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
uvicorn handwriting.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000> for the studio. Open <http://127.0.0.1:8000/docs> for interactive API documentation.

## Input contract

Each character must be written separately. Connected cursive paragraphs require transcription and segmentation that this service does not provide. Use plain or ruled neutral or warm paper, blue or black ink, clear lighting, and margins around the page. Perspective, page curvature, touching letters, background patterns, and bleed-through can require crop corrections.

The default row order is:

```text
ABCDEFGHIJK
LMNOPQRSTUV
WXYZ
abcdefghijk
lmnopqrstuv
wxyz
1234567890!
.,!?:;'"-_()
/@#&+=%₹
```

This layout contains two `!` samples. The first labeled crop wins. The font report lists duplicates. Space is generated automatically, with a width of 500 units per em.

Character labels come from these row strings or from explicit labeled boxes. There is no OCR model. The default upload stops at `review_required`. The studio requires users to confirm their crops before generation. Set `auto_generate=true` only when your application accepts inferred labels. Count mismatches still block automatic generation.

For reliable automation, supply reviewed crop boxes. `examples/reference.png` is a synthetic geometric fixture. It exercises mapping and tracing without publishing anyone's handwriting. `examples/glyphs.json` contains its crop boxes. These coordinates are not reused for other uploads. A deliberate background mark exercises the mapping-review state.

## API flow

| Method and path | Result |
| --- | --- |
| `GET /health` | Service health |
| `GET /api/layouts` | Default rows and coordinate rules |
| `POST /api/fonts` | Multipart upload, returns `202` and a job ID |
| `GET /api/fonts/{id}` | Status, crops, errors, warnings, supported characters, versioned artifact URLs |
| `PUT /api/fonts/{id}/glyphs` | Save corrected crops and ink mode; invalidate previous downloads |
| `POST /api/fonts/{id}/generate` | Confirm mapping and enqueue conversion |
| `GET /api/fonts/{id}/image` | Normalized source image |
| `GET /api/fonts/{id}/mask` | Current extracted ink, including changes to ink mode |
| `GET /api/fonts/{id}/versions/{revision}/{filename}` | TTF, WOFF2, CSS, sample PNG, or JSON report |
| `POST /api/fonts/{id}/preview` | PNG preview of supplied text |
| `GET /api/fonts/{id}/preview` | PNG preview through query parameters |

Uploads accept `image`, `name`, `rows`, `glyphs`, `ink_mode`, `detector`, `auto_generate`, and `allow_partial`. With `auto_generate`, `allow_partial=true` builds the font from the characters that were found even when some marks could not be matched; the job's `mapping_errors` still lists them. A job with no usable characters still stops at `review_required`. The studio sets both. `rows` and `glyphs` are JSON strings inside multipart form data. `ink_mode` accepts `auto`, `blue`, or `dark`. Names accept letters, numbers, spaces, dots, and hyphens.

### AI character detection

Row order only works when the sheet matches the supplied rows. With `OPENROUTER_API_KEY` set, the service asks a vision model (default `openai/gpt-6-luna` on OpenRouter) to read the characters on the page, so extra letters and different layouts work. `detector` accepts `auto` (AI when configured, otherwise row order, and row order if the service fails), `vision` (AI only; the job fails if it is unavailable), or `layout`.

The model only transcribes: one line of text per row of writing, with no coordinates, so the answer is a few hundred tokens and detection takes about 10 seconds. The shapes come from the ink itself. Rows of marks found on the page are paired with the transcribed rows, and a small dynamic program pairs characters with marks in order, skipping stray marks and merging split glyphs such as a colon. Any disagreement between the transcription and the ink (a different number of rows or of marks in a row) is reported as a mapping error, which blocks automatic generation and triggers one retry at a higher reasoning effort. Review every crop before generating.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | unset | Enables AI detection. Read from the environment only and never logged. |
| `VISION_MODEL` | `openai/gpt-6-luna` | An OpenRouter vision model. It must accept images and may reason. |
| `VISION_REASONING_EFFORT` | `low` | Reasoning effort of the first attempt. |
| `VISION_RETRY_EFFORT` | `medium` | Reasoning effort of the single retry. |

Measured on two real sheets: `openai/gpt-6-luna` at low effort answered in 7 to 11 seconds and was exact in 5 of 6 runs; the miss dropped a whole row, which the row-count check reports and retries. `z-ai/glm-5.3-flash` answers in 2 to 4 seconds at low effort but drops characters that differ from the default template. `openai/gpt-oss-120b` cannot be used because it does not accept images.

A detection costs a fraction of a cent. Photos are sent to OpenRouter with `data_collection: deny`, which restricts routing to providers that do not store or train on them. Tell your users when you enable this.

```bash
curl -X POST http://127.0.0.1:8000/api/fonts \
  -F 'image=@examples/reference.png' \
  -F 'name=Photo Handwriting' \
  -F 'auto_generate=true' \
  -F 'glyphs=<examples/glyphs.json'
```

Poll the `Location` URL from the response. States are `queued`, `processing`, `review_required`, `ready`, and `failed`. `stage` identifies analysis or generation. A failed job includes a readable `error`. A job in `review_required` includes proposed `glyphs` and `mapping_errors`.

Every glyph uses normalized coordinates between zero and one. Coordinates refer to the image after EXIF orientation correction. Resizing preserves those coordinates. Include all strokes belonging to the glyph, including detached dots.

```json
{
  "glyphs": [
    {"character": "a", "x": 0.13, "y": 0.37, "width": 0.05, "height": 0.04}
  ],
  "ink_mode": "auto"
}
```

Submit that object to `PUT /api/fonts/{id}/glyphs`. Each edit creates a new revision and disables old artifact URLs. An intentionally partial alphabet is allowed. Unsupported characters are reported during preview. Optional `target_height` and `baseline` overrides use font units. The `baseline` value places the glyph's bottom, so descenders use negative values.

After review:

```bash
curl -X POST http://127.0.0.1:8000/api/fonts/JOB_ID/generate \
  -H 'Content-Type: application/json' \
  -d '{"confirm_mapping":true}'
```

When the job reaches `ready`, its `artifacts` object contains URLs for `font.ttf`, `font.woff2`, `font.css`, `sample.png`, and `report.json`. The response also includes `family`, `supported_characters`, `kerning_pairs`, measurements, and warnings.

For PNG previews, send a JSON body to `POST /api/fonts/{id}/preview`:

```json
{"text":"A note in my own handwriting.","font_size":48,"width":1000}
```

The response has `Content-Type: image/png`. `X-Missing-Characters` contains a JSON string listing unsupported characters. PNG previews show a missing-glyph box. The browser studio reports missing characters because browsers can otherwise hide gaps with fallback fonts. Preview text is limited to 1,000 code points, font size to 12–120 pixels, width to 320–1,600 pixels, and computed height to 4,096 pixels.

## Connect another frontend

Upload using `FormData`. Poll the job until it reaches `ready` or requires review. Resolve the relative artifact URLs against the API origin. Use the FontFace API to load the returned WOFF2 URL:

```javascript
const font = new FontFace(job.family, `url("${apiOrigin}${job.artifacts["font.woff2"]}")`);
await font.load();
document.fonts.add(font);
previewElement.style.fontFamily = `"${job.family}", cursive`;
```

Enable `font-kerning: normal`, `letter-spacing: normal`, `word-spacing: normal`, and `font-synthesis: none`. The downloadable CSS supplies these settings and defaults to `1.75rem` text with `1.7` line height. Each revision has a distinct font family and URL.

Configure allowed frontend origins through `CORS_ORIGINS`. The default is `http://localhost:3000`. Set a comma-separated list for your deployment. CORS applies to font downloads and exposes the preview's missing-character header. The studio on the API origin needs no CORS setup.

## Conversion process and limits

1. Validate actual image bytes, correct EXIF rotation, flatten transparency onto white, and resize to at most 2,400 pixels per side. The upload limit is 15 MiB and the decoded limit is 64 megapixels for JPEG (decoded at reduced scale) and 20 megapixels for PNG and WebP. JPEG, PNG, and WebP are supported. Multi-picture camera JPEGs identified as MPO use their primary photograph; auxiliary gain or depth frames are ignored.
2. Separate blue ink by color. In automatic mode, include strong dark strokes under shadows. Dark mode uses local contrast. Erase long, nearly horizontal ruled lines, then restore blue-shifted ink and strokes that cross them. Remove tiny isolated marks. Apply page-background heuristics only when proposing crops, so reviewed edge crops retain their ink.
3. Estimate row slope, group disconnected marks, and assign the supplied labels in reading order. Return count errors for correction.
4. Preserve crop aspect ratio. Estimate lowercase height, capitals, ascenders, descenders, and punctuation placement from the label. Overrides allow corrections.
5. Smooth the raster, estimate stroke thickness along a morphological skeleton, and limit changes to the main connected stroke. Preserve smaller detached marks. Reject weight adjustments that change connected-component or hole counts. Edge offsets are capped at 16 font units. This conservative approximation does not reproduce a professional type designer's optical corrections.
6. Trace all surviving contours and their holes into quadratic TrueType curves. Set proportional advances with a left bearing near 50 units, right padding of 25 units, and a 500-unit space.
7. Measure rendered outlines at one pixel per font unit for all supplied ASCII letter and digit pairs. Use the original silhouettes for a 130-unit reading-band gap target. Expand the safety silhouettes vertically by one pixel to catch narrow endpoint overlaps, then preserve approximately 56 units of common-row clearance. Limit adjustments to −280 through +40 units. Store pairs in GPOS and legacy `kern`. PNG previews apply the legacy pairs once; browsers apply OpenType kerning.
8. Build TTF and WOFF2 files. Reload both and compare their mappings and outlines before publishing the completed version. Add aliases for curly quotes, nonbreaking space, selected hyphens, and minus when their source glyph exists.

The default font metrics use 1,000 units per em, ascent of at least 850, descent of at most −250, and a 100-unit line gap. Extreme placement overrides expand ascent and descent. Weight balancing operates on a working raster before the final vector outline. It is not a post-export vector-offset pass. Page dewarping, OCR, alternates, ligatures, variable fonts, non-Latin shaping, and manual optical kerning are outside this implementation. One sample repeats for every occurrence of a character.

## Jobs and storage

`DATA_DIR` defaults to `data/`. It contains `jobs.sqlite3`, normalized images, and generated version directories. SQLite uses WAL mode, parameterized queries, atomic queue claims, and explicitly closed connections. Upload and generation admission share a pending-job limit of 20. `MAX_DATA_BYTES` defaults to 1 GiB. The service rejects new uploads and generation when stored data reaches that threshold or the filesystem has less than 256 MiB free. Existing downloads remain available. This is an admission limit, so concurrent in-flight writes can temporarily exceed it. A single worker processes conversions. A data-directory lock rejects a second worker. Run with `--workers 1` and one instance per data directory.

On restart, the worker requeues interrupted operations. It creates artifacts in a staging directory, verifies them, renames the completed directory, and publishes the version in SQLite. A crash before publication can leave an unreferenced directory, which the API does not serve. Graceful shutdown finishes the active conversion.

Job IDs are random possession-based access links. There is no public upload listing. This starter does not provide user accounts, quotas for completed jobs, or automatic retention. Put authentication, rate limits, storage quotas, and a retention policy in place before exposing it as a public multi-user service. Keep `DATA_DIR` on persistent local storage and back up its database and artifacts together.

## Verify

```bash
ruff check handwriting tests examples
ruff format --check handwriting tests examples
coverage run -m pytest -q
coverage report -m --fail-under=100
node --check handwriting/static/studio.js && node --check handwriting/static/theme.js
```

With the API running, exercise the supplied photo through HTTP and download the results:

```bash
python examples/run_sample.py
```

The script uses the synthetic `examples/reference.png` and `examples/glyphs.json`, prints a working studio link, and writes the returned artifacts to `examples/generated/`. Use `--image` to select a private image with matching reviewed coordinates. Review crops for every other image. Personal source photos, generated font kits, screenshots, local environments, and `data/` are excluded from Git and deployment uploads.

Tests verify loadable TTF and WOFF2 artifacts, contour holes, detached dots, character mappings, aliases, spacing, crop revisions, invalid inputs, preview bounds, CORS, resource limits, atomic claims, interrupted jobs, persistence, and failed artifact publication. Public tests use synthetic fixtures. Private-photo acceptance checks can run locally without publishing the photo.

The verified local run passed 69 tests with 100% statement and branch coverage across `handwriting/`. Ruff checks, formatting checks, JavaScript syntax checks, and source and wheel builds passed. A separate check with an installed HarfBuzz engine verified that all 3,844 ASCII letter and digit pairs apply kerning exactly once for a private acceptance sample. HarfBuzz is not a runtime dependency of this service.

The private acceptance sample produced 82 distinct handwritten glyphs and 91 character mappings including space and aliases. A separate 2,000-pixel render of every letter and digit pair measured a minimum horizontal clearance of 55 font units. The automated regression checks clearance from the exported font at 1,000 pixels.

## Deploy

`Dockerfile` installs only the pinned runtime dependencies and runs as UID 10001. `compose.yaml` keeps SQLite and font artifacts in the `handwriting-data` volume. The API has a 1 GiB memory limit, one CPU, eight concurrent requests, and a 120-second shutdown grace period. The local gateway binds only `127.0.0.1:8090`. It allows ten uploads and twenty generation requests per minute, with short bursts. Logs rotate and request access logs are disabled.

```bash
docker compose config --quiet
docker compose up -d --build
curl --fail http://127.0.0.1:8090/health
```

The production host routes `/handwriting/*` on its existing HTTPS gateway to port 8090 and strips that prefix. The Vercel website uses the external-origin rewrites in `vercel.json` to forward `/api/`, `/docs`, `/openapi.json`, and `/health` to that gateway. The website's `/api/` is the canonical public API address, so returned relative artifact URLs work unchanged. Static assets stay on Vercel. No Vercel Function processes image bodies.

`node scripts/build-site.mjs` creates the frontend in `site/`. Vercel builds that output with no frontend dependencies. `site/`, `.vercel/`, and environment files are ignored. To use another server, update the destinations in `vercel.json` and configure its HTTPS route. A repository connected to the Vercel project can also be imported into v0 for editing.

The installed Starlette test client emits one deprecation warning about HTTPX. HTTPX remains functional in the pinned environment. The warning does not indicate a failed test.

## Implementation references

- [FastAPI upload files](https://fastapi.tiangolo.com/tutorial/request-files/)
- [FontTools TrueType drawing pen](https://fonttools.readthedocs.io/en/latest/pens/ttGlyphPen.html)
- [FontTools OpenType feature builder](https://fonttools.readthedocs.io/en/latest/feaLib/index.html)
- [FontTools WOFF2 support](https://fonttools.readthedocs.io/en/latest/ttLib/woff2.html)
- [OpenCV contour hierarchy](https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html)
- [Pillow font rendering](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html)

Noto Sans is vendored under the SIL Open Font License in `handwriting/static/fonts/NotoSans-OFL.txt`.

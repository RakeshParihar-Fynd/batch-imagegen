# Batch Image Generation Tool — Design

**Date:** 2026-05-25
**Status:** Approved for implementation planning
**Owner:** rakeshsinghparihar@gofynd.com

## 1. Problem & Goals

The annotation team needs to run image generation/editing jobs across batches of 100+ images using the PixelBin Prediction API. Today this means writing one-off scripts per batch. We want a local Streamlit tool that lets an annotator:

1. Paste their PixelBin API key.
2. Upload local files **or** paste a list of source URLs.
3. Pick a model (Nano Banana Pro or GPT Image 2), set its parameters, and write one prompt for the whole batch.
4. Watch progress live as the worker pool processes images.
5. Download all completed outputs as a single ZIP when done.

**Non-goals (v1):** multi-user auth, hosted deployment, per-image prompts, mid-batch cancellation, webhooks, cost estimation beyond a rough time estimate, image preprocessing.

## 2. Architecture

Single Streamlit process per user, running on localhost. State for each batch lives in one JSON file on disk; a single background thread runs an asyncio worker pool that processes batches and writes status updates to that JSON. The UI polls the JSON on auto-refresh.

```
┌──────────────────┐    submit(batch_id, api_key)   ┌──────────────────────┐
│  Streamlit UI    │ ─────────────────────────────► │  Orchestrator        │
│  (app.py)        │                                │  (background thread, │
│                  │ ◄───── reads JSON every 2s ─── │   asyncio + semaphore)│
└──────────────────┘                                └──────────┬───────────┘
        ▲                                                     │
        │                            uploadAsync / predictAsync│
        │                                                     ▼
        │                                            ┌─────────────────┐
        │            atomic write batch JSON         │  PixelBin SDK   │
        └────────────────────────────────────────────│  (predictions,  │
                                                     │   assets upload)│
                                                     └─────────────────┘
```

**Key properties:**
- **Single process.** `streamlit run app.py` launches everything; the orchestrator thread is spawned lazily on first submission.
- **JSON-per-batch persistence.** `~/.batch-imagegen/batches/<batch_id>.json` is rewritten atomically (tmp file + `os.rename`) after every job status change.
- **Worker pool via `asyncio.Semaphore`.** Concurrency is user-configurable (1–20, default 5).
- **API key is session-only.** Stored in `st.session_state`, never written to disk.
- **Local-only.** No auth, no multi-tenancy.

## 3. Module Layout

```
batch-imagegen/
├── app.py                        # Streamlit UI. No business logic.
├── batch_imagegen/
│   ├── __init__.py
│   ├── models.py                 # Batch, ImageJob, JobStatus dataclasses
│   ├── store.py                  # JSON-per-batch read/write (atomic)
│   ├── uploader.py               # Local file → PixelBin storage → CDN URL
│   ├── predictor.py              # Wraps pixelbin.predictions per model
│   ├── orchestrator.py           # Background thread, worker pool
│   └── zipper.py                 # Output URLs → streaming ZIP
├── tests/
│   ├── test_store.py
│   ├── test_uploader.py
│   ├── test_predictor.py
│   ├── test_orchestrator.py
│   ├── test_zipper.py
│   ├── test_app.py               # Streamlit AppTest
│   └── fake_pixelbin/            # ~50-line FastAPI mock for E2E
├── pyproject.toml
└── README.md
```

Each module has one responsibility and is unit-testable in isolation. `app.py` reads state via `store.py` and never calls the SDK directly.

Runtime data:
```
~/.batch-imagegen/
├── batches/<batch_id>.json
└── downloads/<batch_id>.zip       # built on demand
```

## 4. Data Model (`models.py`)

```python
from dataclasses import dataclass, field
from enum import Enum

class JobStatus(str, Enum):
    PENDING   = "PENDING"      # not yet picked up
    UPLOADING = "UPLOADING"    # local file being uploaded to PixelBin storage
    SUBMITTED = "SUBMITTED"    # predictions.create() called, prediction_id assigned
    RUNNING   = "RUNNING"      # PixelBin reports in-progress
    SUCCESS   = "SUCCESS"
    FAILURE   = "FAILURE"

@dataclass
class ImageJob:
    job_id: str                # local UUID
    source: str                # original filename OR original URL
    source_url: str | None     # PixelBin CDN URL after upload (None until uploaded)
    prediction_id: str | None  # PixelBin _id once submitted
    output_url: str | None     # final result URL
    status: JobStatus
    error: str | None
    attempts: int              # retry counter (max 2)
    updated_at: str            # ISO timestamp

@dataclass
class Batch:
    batch_id: str              # UUID
    name: str                  # user-supplied label
    model: str                 # "nanoBananaPro_generate" | "gpt2_generate"
    prompt: str
    model_params: dict         # aspect_ratio, output_resolution, [quality]
    concurrency: int           # worker pool size
    created_at: str
    completed_at: str | None   # set when all jobs reach terminal status
    jobs: list[ImageJob] = field(default_factory=list)
```

A batch of 500 jobs serializes to under 200 KB — atomic rewrites on every status change are acceptable.

## 5. Orchestrator (`orchestrator.py`)

Singleton per process, started lazily on first submission.

```python
class Orchestrator:
    def __init__(self, store: BatchStore):
        self._store = store
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._thread: threading.Thread | None = None

    def submit(self, batch_id: str, api_key: str) -> None:
        """Non-blocking — enqueues and returns."""
        self._ensure_thread_running()
        self._queue.put((batch_id, api_key))

    def _ensure_thread_running(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._consume_queue())

    async def _consume_queue(self) -> None:
        while True:
            batch_id, api_key = await asyncio.to_thread(self._queue.get)
            await self._process_batch(batch_id, api_key)

    async def _process_batch(self, batch_id: str, api_key: str) -> None:
        batch = self._store.load(batch_id)
        sem = asyncio.Semaphore(batch.concurrency)
        client = make_pixelbin_client(api_key)

        async def run_one(job: ImageJob) -> None:
            async with sem:
                await self._run_single_job(client, batch, job)
                self._store.save(batch)

        await asyncio.gather(*(run_one(j) for j in batch.jobs))
        batch.completed_at = utcnow_iso()
        self._store.save(batch)
```

**Per-job flow (`_run_single_job`):**
1. If `source` is a local path → upload to PixelBin (status `UPLOADING`), set `source_url`.
2. `predictions.createAsync(name=batch.model, input={"prompt": batch.prompt, "images": [source_url], **batch.model_params})` → store `prediction_id`, status `SUBMITTED`.
3. `await pixelbin.predictions.waitAsync(prediction_id)` → on `SUCCESS` store `output_url`; on `FAILURE` store `error`.
4. Up to 2 retries on transient errors (network errors, 5xx). Permanent errors (4xx, invalid prompt) fail-fast.

**Concurrency.** `asyncio.Semaphore(batch.concurrency)` caps in-flight jobs across the upload + predict phases combined. Multiple batches enqueued back-to-back are processed serially (one batch at a time); this is intentional for v1 to keep behavior predictable.

**Lifecycle.** Daemon thread dies with the process — acceptable for a local single-user tool. No cancellation in v1.

## 6. Streamlit UI (`app.py`)

Two pages via `st.navigation` (or a sidebar radio).

### 6.1 Sidebar (always visible)

- `st.text_input("PixelBin API Key", type="password")` → `st.session_state.api_key`. Never written to disk. Empty → submit disabled.
- `st.slider("Concurrent workers", 1, 20, value=5)`.
- Status footer: `Active batches: N · Last update: Xs ago`.

### 6.2 Page 1 — "New batch"

1. `st.text_input("Batch name")` — defaults to `Batch <timestamp>`.
2. `st.radio("Model", ["Nano Banana Pro", "GPT Image 2"])`.
3. Model-specific parameter panel:
   - **Nano Banana Pro:** `aspect_ratio` (selectbox: auto/1:1/2:3/3:2/3:4/4:3/4:5/5:4/9:16/16:9/21:9), `output_resolution` (1K/2K/4K).
   - **GPT Image 2:** `aspect_ratio` (1:1/2:3/3:2/3:4/4:3/4:5/5:4/9:16/16:9/21:9), `output_resolution` (1K/2K/4K), `quality` (low/medium/high).
4. `st.text_area("Prompt", height=150)` — required, with character count.
5. `st.radio("Image source", ["Upload files", "Paste URLs"])`:
   - **Upload:** `st.file_uploader(accept_multiple_files=True, type=["jpg","jpeg","png","webp"])` — shows count + total MB.
   - **URLs:** `st.text_area("One URL per line")` — shows parsed count.
6. Validation summary: `✓ 127 images · ~340 MB · est. ~13 min at 5 workers` (estimate = `images / concurrency × 30s`).
7. `st.button("Start batch", disabled=…)` → creates `Batch` + `ImageJob`s, calls `orchestrator.submit()`, redirects to Page 2.

### 6.3 Page 2 — "Batches"

- **Top:** `st.dataframe` of all batches — columns: name, model, created, status (`47/127 done`), progress bar, actions.
- **Expanded view** (on row click):
  - Big progress bar + counts: `12 pending · 5 running · 105 success · 5 failed`.
  - `st.button("Download ZIP")` — enabled once ≥1 job is `SUCCESS`. See §7.
  - `st.button("Retry failed")` — re-enqueues only `FAILURE` jobs, resets their status and `attempts`.
  - Collapsible jobs table — `st.dataframe`: source, status, output (thumbnail if `SUCCESS`), error (if `FAILURE`).
- `st_autorefresh(interval=2000)` keeps the page live.

### 6.4 Empty / first-run states

- **No API key:** Sidebar shows yellow info box: *"Paste your PixelBin API key to start. Get one at app.pixelbin.io › Settings › API tokens."* "Start batch" disabled with tooltip *"Add your API key in the sidebar."*
- **No batches yet** (Page 2): centered empty state — *"No batches yet. Create your first batch →"* with a button linking to Page 1.
- **No successful jobs yet:** Download ZIP disabled, tooltip *"No completed images yet."*

### 6.5 Error display

| Level | Where | Examples |
|---|---|---|
| Field-level | Below the input | "Prompt is required", "URL list has 3 invalid entries (lines 12, 47, 91)", "Files larger than 25 MB are not supported: photo_03.jpg" |
| Batch-level | Top of expanded view | "Batch failed to start: invalid API key. Check the sidebar." (red banner, dismissible) |
| Per-job | Jobs table "Error" column | Raw `error.message` from PixelBin, truncated to 80 chars, full text in a popover |

### 6.6 Toasts / notifications

- On submit: `st.toast("Batch 'X' started — 127 images queued")`.
- On batch completion: `st.toast("✓ Batch 'X' done — 122 succeeded, 5 failed")`. Fires on the page-render transition where `completed_at` first becomes non-null; tracked in `st.session_state.last_seen_completion[batch_id]`.
- Browser-tab title updates: `Batch (47/127) — Image Gen`.

### 6.7 API key validation

On first key entry: call `pixelbin.predictions.get("nonexistent")`. A 401 means bad key; a 404 means the key works. Cached in `st.session_state.api_key_valid`.

## 7. ZIP Download (`zipper.py`)

```python
def build_zip(batch_id: str, store: BatchStore) -> Path:
    batch = store.load(batch_id)
    zip_path = downloads_dir() / f"{batch_id}.zip"

    # Cache check
    if zip_path.exists() and zip_mtime(zip_path) > batch_updated_at(batch):
        return zip_path

    successes = [j for j in batch.jobs if j.status == JobStatus.SUCCESS]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Parallel fetch with pool of 10
        outputs = fetch_outputs_parallel(successes, max_workers=10)
        for job, content in outputs:
            arcname = unique_arcname(job.source, content_ext(content))
            zf.writestr(arcname, content)
        zf.writestr("manifest.csv", build_manifest_csv(batch))
    return zip_path
```

**Behaviors:**
- **Filename in zip:** original `source` filename, suffixed with `_out` and the output's extension. Collisions → `_2`, `_3`, etc.
- **`manifest.csv`** included: `source,output_url,status,error,download_failed` for *all* jobs (success + failed), so the annotator has a complete record.
- **Streaming fetch:** outputs downloaded via `httpx.AsyncClient` with a pool of 10, written directly into the zip — avoids buffering 200 MB in memory.
- **Caching:** if the zip exists and is newer than the batch JSON's last update, reuse it. Otherwise rebuild (e.g., after "Retry failed").
- **Edge cases:**
  - Output URL 404 → manifest records `download_failed: true` for that row; zip still completes for the rest.
  - Partial batch → only `SUCCESS` jobs zipped; manifest documents the rest.
  - No successes → button disabled (covered in §6.4).
- **Disk space:** zip writes to `~/.batch-imagegen/downloads/`. Not auto-cleaned in v1.

## 8. Testing Strategy

Four layers, ordered by speed and coverage.

### 8.1 Unit tests (pytest)
- `store.py` — round-trip a `Batch`, assert atomic rewrites, schema migration safety.
- `uploader.py` — mock SDK, assert call args, retry on transient failure.
- `predictor.py` — mock `pixelbin.predictions`, assert correct `name`/`input` per model.
- `orchestrator.py` — inject a fake client with scripted results. Assert: semaphore caps in-flight count, retries fire on transient failures only, JSON rewritten after each job, `completed_at` set only when all jobs terminal.
- `zipper.py` — mock fetcher, assert zip contents, filename collision handling, manifest correctness.

### 8.2 Streamlit `AppTest` (UI without browser)
`streamlit.testing.v1.AppTest` runs `app.py` headless and exposes widgets for assertion:
```python
def test_submit_requires_api_key():
    at = AppTest.from_file("app.py").run()
    at.text_area("prompt").set_value("test").run()
    assert at.button("start_batch").disabled
```
Covers form validation, conditional rendering, page wiring.

### 8.3 Fake-PixelBin E2E
A ~50-line FastAPI server mimics `predictions.create` / `predictions.get` with random delays and scripted failure rate. Point the SDK at it via `domain="http://localhost:8080"` and run a 20-image batch end-to-end. Verifies real timing, progress UI, and ZIP download without burning credits.

### 8.4 Real-API smoke (manual, pre-release)
One 5-image batch against the real PixelBin endpoint with a test key.

## 9. Open Risks

- **Streamlit hot-reload kills the orchestrator thread.** In dev only. Acceptable — restart Streamlit to recover. Document in README.
- **`asyncio` event loop sharing.** The orchestrator owns its loop in its own thread; the Streamlit thread never touches it. No cross-thread asyncio calls.
- **PixelBin upload SDK surface.** The two API docs cover `predictions`; the local-upload path depends on `pixelbin.assets.upload` (or equivalent). The implementation plan must verify the exact SDK method before writing `uploader.py`.
- **Disk growth.** Cached zips and JSON state accumulate. v1 punts on cleanup; users delete `~/.batch-imagegen/` manually if needed.

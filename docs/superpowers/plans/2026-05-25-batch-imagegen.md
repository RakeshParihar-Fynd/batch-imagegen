# Batch Image Generation Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit tool that lets an annotator submit batches of 100+ images to the PixelBin Prediction API (Nano Banana Pro or GPT Image 2), watch live progress, and download all completed outputs as a ZIP.

**Architecture:** Single Streamlit process. A background `threading.Thread` runs an asyncio event loop with a `Semaphore`-bounded worker pool. State for each batch is persisted to one JSON file under `~/.batch-imagegen/batches/`. The UI auto-refreshes every 2 s and reads JSON via a `BatchStore` — it never calls the SDK directly.

**Tech Stack:** Python 3.11+, Streamlit, PixelBin Python SDK (`pixelbin`), `httpx` for output downloads, `pytest` + `streamlit.testing.v1.AppTest` for tests, FastAPI for the test-only fake PixelBin server.

**Spec reference:** [`docs/superpowers/specs/2026-05-25-batch-imagegen-design.md`](../specs/2026-05-25-batch-imagegen-design.md)

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `batch_imagegen/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fake_pixelbin/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "batch-imagegen"
version = "0.1.0"
description = "Local Streamlit tool for batch image generation via PixelBin"
requires-python = ">=3.11"
dependencies = [
    "streamlit>=1.32",
    "streamlit-autorefresh>=1.0.1",
    "pixelbin>=3.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "fastapi>=0.110",
    "uvicorn>=0.29",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.streamlit/secrets.toml
dist/
*.egg-info/
```

- [ ] **Step 3: Write a stub `README.md`**

```markdown
# batch-imagegen

Local Streamlit tool for the annotation team to run batch image generation jobs against the PixelBin Prediction API.

## Quick start

    pip install -e ".[dev]"
    streamlit run app.py

Paste your PixelBin API token in the sidebar, upload images or paste URLs, pick a model, write a prompt, and hit "Start batch".

State is stored at `~/.batch-imagegen/`.

See [`docs/superpowers/specs/2026-05-25-batch-imagegen-design.md`](docs/superpowers/specs/2026-05-25-batch-imagegen-design.md) for the design.
```

- [ ] **Step 4: Create empty package init files**

```bash
mkdir -p batch_imagegen tests/fake_pixelbin
: > batch_imagegen/__init__.py
: > tests/__init__.py
: > tests/fake_pixelbin/__init__.py
```

- [ ] **Step 5: Install and verify**

```bash
pip install -e ".[dev]"
python -c "import streamlit, pixelbin, httpx; print('ok')"
```

Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md batch_imagegen tests
git commit -m "chore: scaffold project"
```

---

## Task 2: Data model (`models.py`)

**Files:**
- Create: `batch_imagegen/models.py`
- Test:   `tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
from dataclasses import asdict
from batch_imagegen.models import Batch, ImageJob, JobStatus, batch_from_dict


def test_imagejob_defaults():
    j = ImageJob(
        job_id="j1", source="a.jpg", source_url=None, prediction_id=None,
        output_url=None, status=JobStatus.PENDING, error=None,
        attempts=0, updated_at="2026-05-25T00:00:00Z",
    )
    assert j.status == JobStatus.PENDING
    assert j.attempts == 0


def test_batch_round_trip():
    b = Batch(
        batch_id="b1", name="Test", model="nanoBananaPro_generate",
        prompt="p", model_params={"aspect_ratio": "1:1"}, concurrency=5,
        created_at="2026-05-25T00:00:00Z", completed_at=None,
        jobs=[ImageJob("j1", "a.jpg", None, None, None,
                       JobStatus.PENDING, None, 0, "2026-05-25T00:00:00Z")],
    )
    restored = batch_from_dict(asdict(b))
    assert restored == b
    assert restored.jobs[0].status is JobStatus.PENDING
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_models.py -v
```

Expected: ImportError / ModuleNotFoundError.

- [ ] **Step 3: Implement `models.py`**

```python
# batch_imagegen/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING   = "PENDING"
    UPLOADING = "UPLOADING"
    SUBMITTED = "SUBMITTED"
    RUNNING   = "RUNNING"
    SUCCESS   = "SUCCESS"
    FAILURE   = "FAILURE"


TERMINAL_STATUSES = {JobStatus.SUCCESS, JobStatus.FAILURE}


@dataclass
class ImageJob:
    job_id: str
    source: str
    source_url: str | None
    prediction_id: str | None
    output_url: str | None
    status: JobStatus
    error: str | None
    attempts: int
    updated_at: str


@dataclass
class Batch:
    batch_id: str
    name: str
    model: str
    prompt: str
    model_params: dict[str, Any]
    concurrency: int
    created_at: str
    completed_at: str | None
    jobs: list[ImageJob] = field(default_factory=list)


def batch_from_dict(d: dict[str, Any]) -> Batch:
    jobs = [ImageJob(**{**j, "status": JobStatus(j["status"])}) for j in d.get("jobs", [])]
    return Batch(**{**d, "jobs": jobs})
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_models.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add batch_imagegen/models.py tests/test_models.py
git commit -m "feat(models): add Batch, ImageJob, JobStatus dataclasses"
```

---

## Task 3: JSON store (`store.py`)

**Files:**
- Create: `batch_imagegen/store.py`
- Test:   `tests/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_store.py
import json
from pathlib import Path
from batch_imagegen.models import Batch, ImageJob, JobStatus
from batch_imagegen.store import BatchStore


def _batch() -> Batch:
    return Batch(
        batch_id="b1", name="Test", model="nanoBananaPro_generate",
        prompt="p", model_params={}, concurrency=5,
        created_at="2026-05-25T00:00:00Z", completed_at=None,
        jobs=[ImageJob("j1", "a.jpg", None, None, None,
                       JobStatus.PENDING, None, 0, "2026-05-25T00:00:00Z")],
    )


def test_save_then_load(tmp_path: Path) -> None:
    store = BatchStore(tmp_path)
    b = _batch()
    store.save(b)
    loaded = store.load("b1")
    assert loaded == b


def test_list_returns_sorted_by_created_desc(tmp_path: Path) -> None:
    store = BatchStore(tmp_path)
    b1 = _batch(); b1.batch_id = "old"; b1.created_at = "2026-05-24T00:00:00Z"
    b2 = _batch(); b2.batch_id = "new"; b2.created_at = "2026-05-25T00:00:00Z"
    store.save(b1)
    store.save(b2)
    ids = [b.batch_id for b in store.list_batches()]
    assert ids == ["new", "old"]


def test_save_is_atomic(tmp_path: Path) -> None:
    store = BatchStore(tmp_path)
    store.save(_batch())
    target = tmp_path / "b1.json"
    assert target.exists()
    # No leftover tmp files
    assert not list(tmp_path.glob("*.tmp"))
    data = json.loads(target.read_text())
    assert data["batch_id"] == "b1"
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_store.py -v
```

- [ ] **Step 3: Implement `store.py`**

```python
# batch_imagegen/store.py
from __future__ import annotations
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import Batch, batch_from_dict


class BatchStore:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_id: str) -> Path:
        return self._root / f"{batch_id}.json"

    def save(self, batch: Batch) -> None:
        target = self._path(batch.batch_id)
        # Atomic write: tmp file in same dir + os.replace
        fd, tmp = tempfile.mkstemp(
            prefix=f".{batch.batch_id}.", suffix=".tmp", dir=str(self._root),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(batch), f, indent=2, default=str)
            os.replace(tmp, target)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def load(self, batch_id: str) -> Batch:
        return batch_from_dict(json.loads(self._path(batch_id).read_text()))

    def list_batches(self) -> list[Batch]:
        out: list[Batch] = []
        for p in self._root.glob("*.json"):
            try:
                out.append(batch_from_dict(json.loads(p.read_text())))
            except Exception:
                continue  # skip malformed
        out.sort(key=lambda b: b.created_at, reverse=True)
        return out
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_store.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add batch_imagegen/store.py tests/test_store.py
git commit -m "feat(store): atomic JSON-per-batch persistence"
```

---

## Task 4: PixelBin client factory + SDK surface verification

**Files:**
- Create: `batch_imagegen/client.py`
- Create: `docs/superpowers/notes/2026-05-25-pixelbin-sdk-check.md`

> **Why this task exists:** the design's open risk (§9 of spec) calls out that we need to confirm the exact PixelBin SDK method for uploading local files. Resolve it before writing `uploader.py`.

- [ ] **Step 1: Inspect the installed SDK surface**

```bash
python - <<'PY'
import pixelbin
import inspect
print("PixelbinClient attrs:", [a for a in dir(pixelbin.PixelbinClient) if not a.startswith("_")])
from pixelbin import PixelbinClient, PixelbinConfig
c = PixelbinClient(PixelbinConfig({"domain": "https://api.pixelbin.io", "apiSecret": "x"}))
for name in ("assets", "uploader", "files"):
    if hasattr(c, name):
        sub = getattr(c, name)
        print(f"client.{name} methods:", [m for m in dir(sub) if not m.startswith("_")])
print("predictions methods:", [m for m in dir(c.predictions) if not m.startswith("_")])
PY
```

Record the exact method (typically `c.assets.fileUpload(file=...)` or `c.uploader.upload(file=...)`) — the actual name is what `uploader.py` will call in Task 5.

- [ ] **Step 2: Write the findings to a notes file**

```markdown
# PixelBin SDK surface check — 2026-05-25

Installed version: <fill from `pip show pixelbin`>

## Predictions (used)
- `client.predictions.create(name, input, webhook=None)` — sync create, returns job dict
- `client.predictions.createAsync(name, input, webhook=None)` — async create
- `client.predictions.get(id)` / `getAsync(id)` — fetch status
- `client.predictions.wait(id)` / `waitAsync(id)` — poll to terminal
- `client.predictions.create_and_waitAsync(name, input, options={...}, webhook=None)`

## File upload (chosen)
- Method: `<actual method name and signature here>`
- Returns: `<shape of return value — must contain a CDN URL field>`
- Example call recorded below

## Notes / caveats
- <anything surprising>
```

- [ ] **Step 3: Implement `client.py`**

```python
# batch_imagegen/client.py
from __future__ import annotations
from pixelbin import PixelbinClient, PixelbinConfig

DEFAULT_DOMAIN = "https://api.pixelbin.io"


def make_pixelbin_client(api_key: str, domain: str = DEFAULT_DOMAIN) -> PixelbinClient:
    if not api_key:
        raise ValueError("PixelBin API key is required")
    return PixelbinClient(PixelbinConfig({"domain": domain, "apiSecret": api_key}))
```

- [ ] **Step 4: Commit**

```bash
git add batch_imagegen/client.py docs/superpowers/notes/2026-05-25-pixelbin-sdk-check.md
git commit -m "feat(client): pixelbin client factory + SDK surface notes"
```

---

## Task 5: Uploader (`uploader.py`)

**Files:**
- Create: `batch_imagegen/uploader.py`
- Test:   `tests/test_uploader.py`

> Replace `<UPLOAD_METHOD>` and `<URL_KEY>` below with the exact names recorded in Task 4. Both occurrences in this task must match.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_uploader.py
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from batch_imagegen.uploader import upload_local_file, UploadError


def test_upload_local_file_returns_url(tmp_path: Path) -> None:
    f = tmp_path / "a.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0fake")
    client = MagicMock()
    # Replace <UPLOAD_METHOD> / <URL_KEY> per Task 4 notes
    client.assets.fileUpload.return_value = {"url": "https://cdn.pixelbin.io/a.jpg"}

    url = upload_local_file(client, f)

    assert url == "https://cdn.pixelbin.io/a.jpg"
    client.assets.fileUpload.assert_called_once()


def test_upload_local_file_raises_uploaderror_on_sdk_exception(tmp_path: Path) -> None:
    f = tmp_path / "a.jpg"; f.write_bytes(b"x")
    client = MagicMock()
    client.assets.fileUpload.side_effect = RuntimeError("network down")

    with pytest.raises(UploadError):
        upload_local_file(client, f)


def test_upload_local_file_rejects_missing_file(tmp_path: Path) -> None:
    client = MagicMock()
    with pytest.raises(FileNotFoundError):
        upload_local_file(client, tmp_path / "missing.jpg")
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_uploader.py -v
```

- [ ] **Step 3: Implement `uploader.py`**

```python
# batch_imagegen/uploader.py
from __future__ import annotations
from pathlib import Path
from typing import Any


class UploadError(Exception):
    """Raised when uploading a local file to PixelBin fails."""


def upload_local_file(client: Any, path: Path) -> str:
    """Upload a local file to PixelBin storage and return the CDN URL.

    Replace the SDK call below with the exact method recorded in Task 4 notes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    try:
        with p.open("rb") as fh:
            # <UPLOAD_METHOD> — replace per Task 4
            response: dict[str, Any] = client.assets.fileUpload(file=fh)
        # <URL_KEY> — replace per Task 4
        url = response.get("url")
        if not url:
            raise UploadError(f"Upload succeeded but no URL returned: {response!r}")
        return url
    except Exception as e:
        if isinstance(e, (UploadError, FileNotFoundError)):
            raise
        raise UploadError(str(e)) from e
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_uploader.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add batch_imagegen/uploader.py tests/test_uploader.py
git commit -m "feat(uploader): wrap PixelBin file upload"
```

---

## Task 6: Predictor (`predictor.py`)

**Files:**
- Create: `batch_imagegen/predictor.py`
- Test:   `tests/test_predictor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predictor.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from batch_imagegen.predictor import (
    Model, build_input, submit_and_wait, PredictionError,
)


def test_build_input_nano_banana_pro():
    payload = build_input(
        Model.NANO_BANANA_PRO,
        prompt="hello",
        source_url="https://cdn/x.jpg",
        params={"aspect_ratio": "1:1", "output_resolution": "1K"},
    )
    assert payload == {
        "prompt": "hello",
        "images": ["https://cdn/x.jpg"],
        "aspect_ratio": "1:1",
        "output_resolution": "1K",
    }


def test_build_input_gpt2_includes_quality():
    payload = build_input(
        Model.GPT_IMAGE_2,
        prompt="hello",
        source_url="https://cdn/x.jpg",
        params={"aspect_ratio": "1:1", "output_resolution": "1K", "quality": "high"},
    )
    assert payload["quality"] == "high"


@pytest.mark.asyncio
async def test_submit_and_wait_returns_output_on_success():
    client = MagicMock()
    client.predictions.create_and_waitAsync = AsyncMock(return_value={
        "_id": "abc", "status": "SUCCESS",
        "output": ["https://delivery/out.jpg"],
    })
    output, prediction_id = await submit_and_wait(
        client, Model.NANO_BANANA_PRO, "p", "https://cdn/x.jpg", {},
    )
    assert output == "https://delivery/out.jpg"
    assert prediction_id == "abc"


@pytest.mark.asyncio
async def test_submit_and_wait_raises_on_failure():
    client = MagicMock()
    client.predictions.create_and_waitAsync = AsyncMock(return_value={
        "_id": "abc", "status": "FAILURE", "error": {"message": "bad prompt"},
    })
    with pytest.raises(PredictionError, match="bad prompt"):
        await submit_and_wait(client, Model.NANO_BANANA_PRO, "p", "u", {})
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_predictor.py -v
```

- [ ] **Step 3: Implement `predictor.py`**

```python
# batch_imagegen/predictor.py
from __future__ import annotations
from enum import Enum
from typing import Any


class Model(str, Enum):
    NANO_BANANA_PRO = "nanoBananaPro_generate"
    GPT_IMAGE_2     = "gpt2_generate"


class PredictionError(Exception):
    """Raised when a PixelBin prediction returns FAILURE."""


def build_input(model: Model, *, prompt: str, source_url: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"prompt": prompt, "images": [source_url], **params}


async def submit_and_wait(
    client: Any, model: Model, prompt: str, source_url: str, params: dict[str, Any],
) -> tuple[str, str]:
    """Submit one prediction and wait for terminal status.

    Returns (output_url, prediction_id). Raises PredictionError on FAILURE.
    """
    payload = build_input(model, prompt=prompt, source_url=source_url, params=params)
    result = await client.predictions.create_and_waitAsync(name=model.value, input=payload)
    prediction_id = result.get("_id", "")
    status = result.get("status")
    if status == "SUCCESS":
        outputs = result.get("output") or []
        if not outputs:
            raise PredictionError("SUCCESS but no output URL returned")
        return outputs[0], prediction_id
    err = (result.get("error") or {}).get("message") or f"status={status}"
    raise PredictionError(err)
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_predictor.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add batch_imagegen/predictor.py tests/test_predictor.py
git commit -m "feat(predictor): wrap PixelBin predictions per model"
```

---

## Task 7: Orchestrator — pure async core (`orchestrator.py` part 1)

> Split the orchestrator into two tasks: the pure async `process_batch` (fully testable without threads) here, then the thread/queue wrapper in Task 8.

**Files:**
- Create: `batch_imagegen/orchestrator.py`
- Test:   `tests/test_orchestrator_core.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator_core.py
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock
import pytest

from batch_imagegen.models import Batch, ImageJob, JobStatus
from batch_imagegen.store import BatchStore
from batch_imagegen.predictor import Model
from batch_imagegen.orchestrator import process_batch


def _job(i: int, source: str) -> ImageJob:
    return ImageJob(
        job_id=f"j{i}", source=source, source_url=None, prediction_id=None,
        output_url=None, status=JobStatus.PENDING, error=None,
        attempts=0, updated_at="2026-05-25T00:00:00Z",
    )


def _batch(jobs: list[ImageJob], concurrency: int = 2) -> Batch:
    return Batch(
        batch_id="b1", name="t", model=Model.NANO_BANANA_PRO.value,
        prompt="p", model_params={}, concurrency=concurrency,
        created_at="2026-05-25T00:00:00Z", completed_at=None, jobs=jobs,
    )


@pytest.mark.asyncio
async def test_process_batch_url_source_success(tmp_path: Path) -> None:
    store = BatchStore(tmp_path)
    batch = _batch([_job(1, "https://cdn/a.jpg"), _job(2, "https://cdn/b.jpg")])
    store.save(batch)

    submit = AsyncMock(side_effect=[("https://out/a.jpg", "p1"), ("https://out/b.jpg", "p2")])
    upload = AsyncMock()  # not called — sources are URLs

    await process_batch(batch_id="b1", store=store, upload_fn=upload, submit_fn=submit)

    upload.assert_not_called()
    final = store.load("b1")
    assert [j.status for j in final.jobs] == [JobStatus.SUCCESS, JobStatus.SUCCESS]
    assert final.jobs[0].output_url == "https://out/a.jpg"
    assert final.completed_at is not None


@pytest.mark.asyncio
async def test_process_batch_local_source_uploads_first(tmp_path: Path) -> None:
    store = BatchStore(tmp_path)
    local = tmp_path / "a.jpg"; local.write_bytes(b"x")
    batch = _batch([_job(1, str(local))])
    store.save(batch)

    upload = AsyncMock(return_value="https://cdn/a.jpg")
    submit = AsyncMock(return_value=("https://out/a.jpg", "p1"))

    await process_batch(batch_id="b1", store=store, upload_fn=upload, submit_fn=submit)

    upload.assert_awaited_once()
    submit.assert_awaited_once()
    final = store.load("b1")
    assert final.jobs[0].source_url == "https://cdn/a.jpg"
    assert final.jobs[0].status == JobStatus.SUCCESS


@pytest.mark.asyncio
async def test_process_batch_retries_transient_then_succeeds(tmp_path: Path) -> None:
    from batch_imagegen.predictor import PredictionError
    store = BatchStore(tmp_path)
    batch = _batch([_job(1, "https://cdn/a.jpg")])
    store.save(batch)

    calls = {"n": 0}
    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient")
        return ("https://out/a.jpg", "p1")

    await process_batch(
        batch_id="b1", store=store,
        upload_fn=AsyncMock(), submit_fn=flaky,
    )
    final = store.load("b1")
    assert final.jobs[0].status == JobStatus.SUCCESS
    assert final.jobs[0].attempts == 2


@pytest.mark.asyncio
async def test_process_batch_marks_failure_on_permanent_error(tmp_path: Path) -> None:
    from batch_imagegen.predictor import PredictionError
    store = BatchStore(tmp_path)
    batch = _batch([_job(1, "https://cdn/a.jpg")])
    store.save(batch)

    submit = AsyncMock(side_effect=PredictionError("bad prompt"))

    await process_batch(
        batch_id="b1", store=store,
        upload_fn=AsyncMock(), submit_fn=submit,
    )
    final = store.load("b1")
    assert final.jobs[0].status == JobStatus.FAILURE
    assert "bad prompt" in final.jobs[0].error


@pytest.mark.asyncio
async def test_process_batch_respects_concurrency(tmp_path: Path) -> None:
    store = BatchStore(tmp_path)
    jobs = [_job(i, f"https://cdn/{i}.jpg") for i in range(6)]
    batch = _batch(jobs, concurrency=2)
    store.save(batch)

    in_flight = 0; max_seen = 0
    async def slow(*a, **k):
        nonlocal in_flight, max_seen
        in_flight += 1; max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return ("https://out/x.jpg", "p")

    await process_batch(
        batch_id="b1", store=store,
        upload_fn=AsyncMock(), submit_fn=slow,
    )
    assert max_seen == 2
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_orchestrator_core.py -v
```

- [ ] **Step 3: Implement the core**

```python
# batch_imagegen/orchestrator.py
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from .models import Batch, ImageJob, JobStatus, TERMINAL_STATUSES
from .predictor import PredictionError
from .store import BatchStore

MAX_ATTEMPTS = 2  # initial + 1 retry

UploadFn = Callable[[Path], Awaitable[str]]
SubmitFn = Callable[..., Awaitable[tuple[str, str]]]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, PredictionError):
        return False
    return isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError))


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


async def _run_one(
    batch: Batch, job: ImageJob, store: BatchStore,
    sem: asyncio.Semaphore, upload_fn: UploadFn, submit_fn: SubmitFn,
) -> None:
    async with sem:
        # Resolve source URL
        if not job.source_url:
            if _is_url(job.source):
                job.source_url = job.source
            else:
                job.status = JobStatus.UPLOADING
                job.updated_at = _utcnow()
                store.save(batch)
                try:
                    job.source_url = await upload_fn(Path(job.source))
                except Exception as e:
                    job.status = JobStatus.FAILURE
                    job.error = f"upload failed: {e}"
                    job.updated_at = _utcnow()
                    store.save(batch)
                    return

        # Submit + wait with bounded retries
        while True:
            job.attempts += 1
            job.status = JobStatus.SUBMITTED
            job.updated_at = _utcnow()
            store.save(batch)
            try:
                output_url, prediction_id = await submit_fn(
                    batch.model, batch.prompt, job.source_url, batch.model_params,
                )
                job.prediction_id = prediction_id
                job.output_url = output_url
                job.status = JobStatus.SUCCESS
                job.error = None
                job.updated_at = _utcnow()
                store.save(batch)
                return
            except Exception as e:
                if _is_transient(e) and job.attempts < MAX_ATTEMPTS:
                    await asyncio.sleep(1.0)
                    continue
                job.status = JobStatus.FAILURE
                job.error = str(e)
                job.updated_at = _utcnow()
                store.save(batch)
                return


async def process_batch(
    batch_id: str, store: BatchStore,
    upload_fn: UploadFn, submit_fn: SubmitFn,
) -> None:
    batch = store.load(batch_id)
    pending = [j for j in batch.jobs if j.status not in TERMINAL_STATUSES]
    sem = asyncio.Semaphore(max(1, batch.concurrency))
    await asyncio.gather(*(
        _run_one(batch, j, store, sem, upload_fn, submit_fn) for j in pending
    ))
    if all(j.status in TERMINAL_STATUSES for j in batch.jobs):
        batch.completed_at = _utcnow()
        store.save(batch)
```

**Wrapper note:** the `submit_fn` in tests takes positional args `(model, prompt, source_url, params)`. The production binding in Task 8 will adapt `predictor.submit_and_wait` to that signature.

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_orchestrator_core.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add batch_imagegen/orchestrator.py tests/test_orchestrator_core.py
git commit -m "feat(orchestrator): pure async process_batch with retries and concurrency cap"
```

---

## Task 8: Orchestrator — thread/queue wrapper (`orchestrator.py` part 2)

**Files:**
- Modify: `batch_imagegen/orchestrator.py` (append)
- Test:   `tests/test_orchestrator_runner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_orchestrator_runner.py
import time
from pathlib import Path
from unittest.mock import AsyncMock
import pytest

from batch_imagegen.models import Batch, ImageJob, JobStatus
from batch_imagegen.store import BatchStore
from batch_imagegen.predictor import Model
from batch_imagegen.orchestrator import OrchestratorRunner


def _make_batch(store: BatchStore, n: int) -> Batch:
    jobs = [
        ImageJob(f"j{i}", f"https://cdn/{i}.jpg", None, None, None,
                 JobStatus.PENDING, None, 0, "2026-05-25T00:00:00Z")
        for i in range(n)
    ]
    b = Batch(
        batch_id="b1", name="t", model=Model.NANO_BANANA_PRO.value,
        prompt="p", model_params={}, concurrency=2,
        created_at="2026-05-25T00:00:00Z", completed_at=None, jobs=jobs,
    )
    store.save(b)
    return b


def test_runner_processes_enqueued_batch(tmp_path: Path) -> None:
    store = BatchStore(tmp_path)
    _make_batch(store, 3)

    submit = AsyncMock(return_value=("https://out/x.jpg", "pred"))
    upload = AsyncMock()

    def fake_factory(api_key: str):
        # ignore api_key in test
        return upload, submit

    runner = OrchestratorRunner(store, factory=fake_factory)
    runner.submit("b1", "FAKE_KEY")

    # Poll until completion (max 5s)
    deadline = time.time() + 5
    while time.time() < deadline:
        b = store.load("b1")
        if b.completed_at:
            break
        time.sleep(0.05)

    final = store.load("b1")
    assert final.completed_at is not None
    assert all(j.status == JobStatus.SUCCESS for j in final.jobs)
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_orchestrator_runner.py -v
```

- [ ] **Step 3: Append to `orchestrator.py`**

```python
# Append to batch_imagegen/orchestrator.py
import queue
import threading
from typing import Callable, Tuple

from .client import make_pixelbin_client
from .predictor import Model, submit_and_wait
from .uploader import upload_local_file


# Factory: (api_key) -> (upload_fn, submit_fn) bound for this batch
Factory = Callable[[str], Tuple[UploadFn, SubmitFn]]


def default_factory(api_key: str) -> Tuple[UploadFn, SubmitFn]:
    client = make_pixelbin_client(api_key)

    async def upload_fn(path: Path) -> str:
        return await asyncio.to_thread(upload_local_file, client, path)

    async def submit_fn(model_value: str, prompt: str, source_url: str, params: dict) -> tuple[str, str]:
        return await submit_and_wait(client, Model(model_value), prompt, source_url, params)

    return upload_fn, submit_fn


class OrchestratorRunner:
    """Background thread that consumes a queue of (batch_id, api_key) submissions."""

    def __init__(self, store: BatchStore, factory: Factory = default_factory):
        self._store = store
        self._factory = factory
        self._queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, batch_id: str, api_key: str) -> None:
        self._ensure_thread()
        self._queue.put((batch_id, api_key))

    def active_count(self) -> int:
        return self._queue.qsize() + (1 if self._thread and self._thread.is_alive() else 0)

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._consume())
        finally:
            loop.close()

    async def _consume(self) -> None:
        while True:
            batch_id, api_key = await asyncio.to_thread(self._queue.get)
            try:
                upload_fn, submit_fn = self._factory(api_key)
                await process_batch(batch_id, self._store, upload_fn, submit_fn)
            except Exception as e:
                # Top-level crash: mark all non-terminal jobs as FAILURE.
                try:
                    batch = self._store.load(batch_id)
                    for j in batch.jobs:
                        if j.status not in TERMINAL_STATUSES:
                            j.status = JobStatus.FAILURE
                            j.error = f"orchestrator crash: {e}"
                            j.updated_at = _utcnow()
                    batch.completed_at = _utcnow()
                    self._store.save(batch)
                except Exception:
                    pass
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_orchestrator_runner.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add batch_imagegen/orchestrator.py tests/test_orchestrator_runner.py
git commit -m "feat(orchestrator): thread + queue runner with default SDK factory"
```

---

## Task 9: ZIP builder (`zipper.py`)

**Files:**
- Create: `batch_imagegen/zipper.py`
- Test:   `tests/test_zipper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_zipper.py
import csv
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

from batch_imagegen.models import Batch, ImageJob, JobStatus
from batch_imagegen.zipper import build_zip


def _batch_with_two_successes() -> Batch:
    return Batch(
        batch_id="b1", name="t", model="nanoBananaPro_generate",
        prompt="p", model_params={}, concurrency=2,
        created_at="2026-05-25T00:00:00Z", completed_at="2026-05-25T01:00:00Z",
        jobs=[
            ImageJob("j1", "a.jpg", "https://cdn/a.jpg", "p1",
                     "https://out/a.jpg", JobStatus.SUCCESS, None, 1, "x"),
            ImageJob("j2", "a.jpg", "https://cdn/a.jpg", "p2",  # duplicate name
                     "https://out/a2.png", JobStatus.SUCCESS, None, 1, "x"),
            ImageJob("j3", "b.jpg", "https://cdn/b.jpg", None,
                     None, JobStatus.FAILURE, "bad prompt", 2, "x"),
        ],
    )


def test_build_zip_includes_successes_and_manifest(tmp_path: Path) -> None:
    batch = _batch_with_two_successes()

    def fake_fetch(url: str) -> tuple[bytes, str]:
        return (b"\xff\xd8FAKE", "jpeg" if url.endswith(".jpg") else "png")

    with patch("batch_imagegen.zipper._fetch", side_effect=fake_fetch):
        out = build_zip(batch, tmp_path)

    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
        assert "manifest.csv" in names
        assert "a_out.jpeg" in names
        assert "a_out_2.png" in names  # collision suffix
        # Failed job NOT in zip
        assert not any(n.startswith("b_out") for n in names)

        manifest = zf.read("manifest.csv").decode()
    rows = list(csv.DictReader(io.StringIO(manifest)))
    assert len(rows) == 3
    statuses = {r["source"]: r["status"] for r in rows}
    assert statuses["b.jpg"] == "FAILURE"


def test_build_zip_returns_cached_when_unchanged(tmp_path: Path) -> None:
    batch = _batch_with_two_successes()

    def fake_fetch(url): return (b"x", "jpeg")

    with patch("batch_imagegen.zipper._fetch", side_effect=fake_fetch) as fetch:
        out1 = build_zip(batch, tmp_path)
        out2 = build_zip(batch, tmp_path)
    # Second call should not re-fetch since batch unchanged & zip newer
    assert out1 == out2
    assert fetch.call_count == 2  # 2 success URLs, called once total
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_zipper.py -v
```

- [ ] **Step 3: Implement `zipper.py`**

```python
# batch_imagegen/zipper.py
from __future__ import annotations
import csv
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .models import Batch, ImageJob, JobStatus

_MAX_PARALLEL = 10


def _fetch(url: str) -> tuple[bytes, str]:
    """Return (content_bytes, extension_without_dot)."""
    r = httpx.get(url, timeout=30.0)
    r.raise_for_status()
    ext = Path(urlparse(url).path).suffix.lstrip(".") or "bin"
    return r.content, ext


def _batch_last_change(batch: Batch) -> float:
    iso_values = [j.updated_at for j in batch.jobs] + [batch.completed_at or batch.created_at]
    latest = max(iso_values)
    return datetime.fromisoformat(latest.replace("Z", "+00:00")).timestamp()


def _unique_arcname(base_source: str, ext: str, used: set[str]) -> str:
    stem = Path(base_source).stem or "image"
    name = f"{stem}_out.{ext}"
    i = 2
    while name in used:
        name = f"{stem}_out_{i}.{ext}"
        i += 1
    used.add(name)
    return name


def _build_manifest(batch: Batch, download_failed: dict[str, bool]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["source", "output_url", "status", "error", "download_failed"])
    for j in batch.jobs:
        w.writerow([
            j.source, j.output_url or "", j.status.value,
            j.error or "", "true" if download_failed.get(j.job_id) else "false",
        ])
    return buf.getvalue()


def build_zip(batch: Batch, downloads_dir: Path) -> Path:
    downloads_dir.mkdir(parents=True, exist_ok=True)
    zip_path = downloads_dir / f"{batch.batch_id}.zip"

    if zip_path.exists() and zip_path.stat().st_mtime > _batch_last_change(batch):
        return zip_path

    successes = [j for j in batch.jobs if j.status == JobStatus.SUCCESS and j.output_url]
    download_failed: dict[str, bool] = {}

    # Parallel fetch
    fetched: dict[str, tuple[bytes, str]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {pool.submit(_fetch, j.output_url): j for j in successes}
        for fut, job in futures.items():
            try:
                fetched[job.job_id] = fut.result()
            except Exception:
                download_failed[job.job_id] = True

    used_names: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for j in successes:
            if j.job_id in download_failed:
                continue
            content, ext = fetched[j.job_id]
            arcname = _unique_arcname(j.source, ext, used_names)
            zf.writestr(arcname, content)
        zf.writestr("manifest.csv", _build_manifest(batch, download_failed))

    return zip_path
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_zipper.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add batch_imagegen/zipper.py tests/test_zipper.py
git commit -m "feat(zipper): build per-batch ZIP with manifest and dedup"
```

---

## Task 10: Streamlit app shell + sidebar

**Files:**
- Create: `app.py`
- Create: `batch_imagegen/paths.py`

- [ ] **Step 1: Implement paths helper**

```python
# batch_imagegen/paths.py
from __future__ import annotations
import os
from pathlib import Path


def data_root() -> Path:
    root = Path(os.environ.get("BATCH_IMAGEGEN_HOME", Path.home() / ".batch-imagegen"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "batches").mkdir(exist_ok=True)
    (root / "downloads").mkdir(exist_ok=True)
    return root


def batches_dir() -> Path:
    return data_root() / "batches"


def downloads_dir() -> Path:
    return data_root() / "downloads"
```

- [ ] **Step 2: Implement minimal `app.py` shell**

```python
# app.py
from __future__ import annotations
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from batch_imagegen.paths import batches_dir
from batch_imagegen.store import BatchStore
from batch_imagegen.orchestrator import OrchestratorRunner


st.set_page_config(page_title="Batch Image Gen", layout="wide")


@st.cache_resource
def _store() -> BatchStore:
    return BatchStore(batches_dir())


@st.cache_resource
def _runner() -> OrchestratorRunner:
    return OrchestratorRunner(_store())


def _sidebar() -> None:
    st.sidebar.header("Settings")
    st.sidebar.text_input(
        "PixelBin API Key", type="password", key="api_key",
        help="Stored only in this session. Never written to disk.",
    )
    st.sidebar.slider("Concurrent workers", 1, 20, value=5, key="concurrency")
    if not st.session_state.get("api_key"):
        st.sidebar.warning(
            "Paste your PixelBin API key to start. Get one at "
            "app.pixelbin.io › Settings › API tokens."
        )


def main() -> None:
    _sidebar()
    page = st.sidebar.radio("Page", ["New batch", "Batches"], key="page")
    st_autorefresh(interval=2000, key="auto_refresh")
    if page == "New batch":
        st.title("New batch")
        st.info("New batch form — built in Task 11.")
    else:
        st.title("Batches")
        st.info("Batches view — built in Task 12.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-run the app**

```bash
streamlit run app.py --server.headless true &
sleep 3
curl -sf http://localhost:8501 > /dev/null && echo "ok"
pkill -f "streamlit run app.py" || true
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add app.py batch_imagegen/paths.py
git commit -m "feat(ui): streamlit shell with sidebar and routing"
```

---

## Task 11: "New batch" page

**Files:**
- Modify: `app.py`
- Create: `batch_imagegen/batch_builder.py`
- Test:   `tests/test_batch_builder.py`

> The form's *pure logic* (parsing URLs, building a `Batch` from inputs) goes into `batch_builder.py` so it's testable without Streamlit. `app.py` only wires widgets to it.

- [ ] **Step 1: Failing test for builder**

```python
# tests/test_batch_builder.py
from batch_imagegen.batch_builder import (
    parse_url_list, build_batch_from_inputs, ValidationError,
)
import pytest


def test_parse_url_list_handles_blanks_and_dupes():
    text = "https://a.com/x.jpg\n\n  https://b.com/y.jpg \nhttps://a.com/x.jpg"
    urls = parse_url_list(text)
    assert urls == ["https://a.com/x.jpg", "https://b.com/y.jpg"]


def test_parse_url_list_rejects_invalid():
    with pytest.raises(ValidationError) as exc:
        parse_url_list("https://a.com/ok.jpg\nnot-a-url\nftp://x")
    assert "lines 2, 3" in str(exc.value)


def test_build_batch_from_inputs_minimal():
    batch = build_batch_from_inputs(
        name="My batch",
        model="nanoBananaPro_generate",
        prompt="hello",
        params={"aspect_ratio": "auto", "output_resolution": "1K"},
        concurrency=4,
        sources=["https://a.com/x.jpg", "https://b.com/y.jpg"],
    )
    assert batch.name == "My batch"
    assert len(batch.jobs) == 2
    assert batch.concurrency == 4
    assert all(j.source.startswith("https://") for j in batch.jobs)


def test_build_batch_requires_prompt_and_sources():
    with pytest.raises(ValidationError):
        build_batch_from_inputs(
            name="x", model="nanoBananaPro_generate", prompt="",
            params={}, concurrency=1, sources=["https://a/x"],
        )
    with pytest.raises(ValidationError):
        build_batch_from_inputs(
            name="x", model="nanoBananaPro_generate", prompt="p",
            params={}, concurrency=1, sources=[],
        )
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_batch_builder.py -v
```

- [ ] **Step 3: Implement `batch_builder.py`**

```python
# batch_imagegen/batch_builder.py
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Iterable, Any

from .models import Batch, ImageJob, JobStatus


class ValidationError(ValueError):
    pass


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_url_list(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    bad: list[int] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if not (line.startswith("http://") or line.startswith("https://")):
            bad.append(i); continue
        if line in seen:
            continue
        seen.add(line); out.append(line)
    if bad:
        nums = ", ".join(str(n) for n in bad)
        raise ValidationError(f"URL list has {len(bad)} invalid entries (lines {nums})")
    return out


def build_batch_from_inputs(
    *, name: str, model: str, prompt: str,
    params: dict[str, Any], concurrency: int, sources: Iterable[str],
) -> Batch:
    name = (name or "").strip()
    if not name:
        raise ValidationError("Batch name is required")
    if not prompt or not prompt.strip():
        raise ValidationError("Prompt is required")
    src_list = list(sources)
    if not src_list:
        raise ValidationError("At least one source image is required")

    now = _utcnow()
    jobs = [
        ImageJob(
            job_id=str(uuid.uuid4()), source=s,
            source_url=s if s.startswith(("http://", "https://")) else None,
            prediction_id=None, output_url=None,
            status=JobStatus.PENDING, error=None,
            attempts=0, updated_at=now,
        )
        for s in src_list
    ]
    return Batch(
        batch_id=str(uuid.uuid4()), name=name, model=model,
        prompt=prompt.strip(), model_params=dict(params),
        concurrency=int(concurrency),
        created_at=now, completed_at=None, jobs=jobs,
    )
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_batch_builder.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Wire the New-batch form in `app.py`**

Replace the `if page == "New batch":` branch with:

```python
    if page == "New batch":
        _render_new_batch()
```

And add at module scope:

```python
import tempfile
from pathlib import Path
from batch_imagegen.batch_builder import (
    parse_url_list, build_batch_from_inputs, ValidationError,
)
from batch_imagegen.paths import data_root

NANO_RATIOS = ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
GPT_RATIOS  = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
RESOLUTIONS = ["1K", "2K", "4K"]


def _save_uploaded_files(files) -> list[str]:
    dest = data_root() / "uploads"
    dest.mkdir(exist_ok=True)
    paths: list[str] = []
    for uf in files:
        target = dest / uf.name
        target.write_bytes(uf.getbuffer())
        paths.append(str(target))
    return paths


def _render_new_batch() -> None:
    st.subheader("Create a new batch")
    name = st.text_input("Batch name", value="")
    model_label = st.radio("Model", ["Nano Banana Pro", "GPT Image 2"], horizontal=True)
    model = "nanoBananaPro_generate" if model_label == "Nano Banana Pro" else "gpt2_generate"

    with st.expander("Model parameters", expanded=True):
        if model == "nanoBananaPro_generate":
            aspect = st.selectbox("Aspect ratio", NANO_RATIOS, index=0)
            resolution = st.selectbox("Output resolution", RESOLUTIONS, index=0)
            params = {"aspect_ratio": aspect, "output_resolution": resolution}
        else:
            aspect = st.selectbox("Aspect ratio", GPT_RATIOS, index=0)
            resolution = st.selectbox("Output resolution", RESOLUTIONS, index=0)
            quality = st.selectbox("Quality", ["low", "medium", "high"], index=2)
            params = {"aspect_ratio": aspect, "output_resolution": resolution, "quality": quality}

    prompt = st.text_area("Prompt", height=150)
    st.caption(f"{len(prompt)} chars")

    source_mode = st.radio("Image source", ["Upload files", "Paste URLs"], horizontal=True)
    sources: list[str] = []
    if source_mode == "Upload files":
        uploaded = st.file_uploader(
            "Drop images here", accept_multiple_files=True,
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded:
            sources = _save_uploaded_files(uploaded)
            st.caption(f"{len(uploaded)} files · "
                       f"~{sum(uf.size for uf in uploaded)/1_000_000:.1f} MB")
    else:
        text = st.text_area("One URL per line", height=120, key="url_list_text")
        if text:
            try:
                sources = parse_url_list(text)
                st.caption(f"{len(sources)} URLs parsed")
            except ValidationError as e:
                st.error(str(e))

    concurrency = int(st.session_state.get("concurrency", 5))
    if sources:
        est_min = max(1, len(sources) / max(1, concurrency) * 30 / 60)
        st.caption(f"Estimated: ~{est_min:.0f} min at {concurrency} workers")

    api_key = st.session_state.get("api_key", "")
    disabled = not (name.strip() and prompt.strip() and sources and api_key)
    tooltip = "" if not disabled else "Fill name, prompt, sources, and add an API key in the sidebar."
    if st.button("Start batch", disabled=disabled, help=tooltip, type="primary"):
        try:
            batch = build_batch_from_inputs(
                name=name, model=model, prompt=prompt,
                params=params, concurrency=concurrency, sources=sources,
            )
        except ValidationError as e:
            st.error(str(e))
            return
        _store().save(batch)
        _runner().submit(batch.batch_id, api_key)
        st.toast(f"Batch '{batch.name}' started — {len(batch.jobs)} images queued")
        st.session_state["page"] = "Batches"
        st.session_state["selected_batch"] = batch.batch_id
        st.rerun()
```

- [ ] **Step 6: Smoke-test**

```bash
streamlit run app.py --server.headless true &
sleep 3
curl -sf http://localhost:8501 > /dev/null && echo "ok"
pkill -f "streamlit run app.py" || true
```

Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add app.py batch_imagegen/batch_builder.py tests/test_batch_builder.py
git commit -m "feat(ui): new batch page wired to orchestrator"
```

---

## Task 12: "Batches" page (live view)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add helpers and view to `app.py`**

Add at module scope:

```python
from batch_imagegen.models import JobStatus, TERMINAL_STATUSES
from batch_imagegen.zipper import build_zip
from batch_imagegen.paths import downloads_dir
import pandas as pd


def _counts(batch) -> dict[str, int]:
    out = {s.value: 0 for s in JobStatus}
    for j in batch.jobs:
        out[j.status.value] += 1
    return out


def _render_batches_page() -> None:
    batches = _store().list_batches()
    if not batches:
        st.info("No batches yet. Create your first batch in the sidebar → New batch.")
        return

    rows = []
    for b in batches:
        c = _counts(b)
        done = c["SUCCESS"] + c["FAILURE"]
        rows.append({
            "id": b.batch_id, "name": b.name, "model": b.model,
            "created": b.created_at, "progress": f"{done}/{len(b.jobs)}",
            "status": "completed" if b.completed_at else "running",
        })
    df = pd.DataFrame(rows)
    selected = st.session_state.get("selected_batch")

    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"id": None})

    pick = st.selectbox("Open batch", options=[b.batch_id for b in batches],
                        format_func=lambda i: next(b.name for b in batches if b.batch_id == i),
                        index=([b.batch_id for b in batches].index(selected)
                               if selected in [b.batch_id for b in batches] else 0))
    st.session_state["selected_batch"] = pick

    batch = next(b for b in batches if b.batch_id == pick)
    c = _counts(batch)
    done = c["SUCCESS"] + c["FAILURE"]
    st.progress(done / max(1, len(batch.jobs)),
                text=f"{c['PENDING']} pending · {c['SUBMITTED']+c['RUNNING']+c['UPLOADING']} running · "
                     f"{c['SUCCESS']} success · {c['FAILURE']} failed")

    col1, col2 = st.columns(2)
    with col1:
        zip_disabled = c["SUCCESS"] == 0
        if st.button("Download ZIP", disabled=zip_disabled,
                     help="No completed images yet." if zip_disabled else ""):
            with st.spinner("Building ZIP…"):
                zip_path = build_zip(batch, downloads_dir())
            st.download_button(
                "Save ZIP", data=zip_path.read_bytes(),
                file_name=f"{batch.name or batch.batch_id}.zip",
                mime="application/zip",
            )
    with col2:
        api_key = st.session_state.get("api_key", "")
        retry_disabled = c["FAILURE"] == 0 or not api_key
        if st.button("Retry failed", disabled=retry_disabled):
            for j in batch.jobs:
                if j.status == JobStatus.FAILURE:
                    j.status = JobStatus.PENDING
                    j.attempts = 0
                    j.error = None
            batch.completed_at = None
            _store().save(batch)
            _runner().submit(batch.batch_id, api_key)
            st.toast("Retrying failed jobs…")
            st.rerun()

    with st.expander("Jobs"):
        jobs_df = pd.DataFrame([
            {
                "source": j.source.split("/")[-1],
                "status": j.status.value,
                "output": j.output_url or "",
                "error": (j.error or "")[:80],
            }
            for j in batch.jobs
        ])
        st.dataframe(jobs_df, hide_index=True, use_container_width=True)

    # Completion toast (fires once per batch_id per session)
    if batch.completed_at:
        last = st.session_state.setdefault("seen_completion", {})
        if last.get(batch.batch_id) != batch.completed_at:
            st.toast(f"✓ Batch '{batch.name}' done — "
                     f"{c['SUCCESS']} succeeded, {c['FAILURE']} failed")
            last[batch.batch_id] = batch.completed_at
```

Then replace the `else:` branch in `main()`:

```python
    else:
        st.title("Batches")
        _render_batches_page()
```

- [ ] **Step 2: Smoke-run**

```bash
streamlit run app.py --server.headless true &
sleep 3
curl -sf http://localhost:8501 > /dev/null && echo "ok"
pkill -f "streamlit run app.py" || true
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(ui): batches page with progress, retry, and ZIP download"
```

---

## Task 13: Fake PixelBin server for E2E

**Files:**
- Create: `tests/fake_pixelbin/server.py`
- Test:   `tests/test_e2e_fake.py`

- [ ] **Step 1: Implement the fake server**

```python
# tests/fake_pixelbin/server.py
"""A ~50-line FastAPI server that mimics the subset of PixelBin we use."""
from __future__ import annotations
import asyncio
import random
import uuid
from fastapi import FastAPI

app = FastAPI()
_jobs: dict[str, dict] = {}


@app.post("/service/panel/transformation/v1.0/org/-/predictions")
async def create_prediction(payload: dict) -> dict:
    pid = f"{payload.get('name', 'p')}--{uuid.uuid4()}"
    _jobs[pid] = {"_id": pid, "status": "PENDING", "input": payload.get("input", {})}
    asyncio.create_task(_finish_later(pid))
    return _jobs[pid]


@app.get("/service/panel/transformation/v1.0/org/-/predictions/{pid}")
async def get_prediction(pid: str) -> dict:
    return _jobs.get(pid, {"_id": pid, "status": "FAILURE", "error": {"message": "not found"}})


async def _finish_later(pid: str) -> None:
    await asyncio.sleep(random.uniform(0.1, 0.4))
    if random.random() < 0.1:
        _jobs[pid].update(status="FAILURE", error={"message": "synthetic failure"})
    else:
        _jobs[pid].update(status="SUCCESS",
                          output=[f"https://fake.local/out/{pid}.jpg"])
```

> The real PixelBin SDK paths will be different — this only needs to satisfy whatever `pixelbin.predictions.createAsync` / `getAsync` / `waitAsync` actually hits. Update the route paths after running `pip show pixelbin` and skimming the SDK source. (This is acceptable scope for a test-only sidecar.)

- [ ] **Step 2: Write the E2E test**

```python
# tests/test_e2e_fake.py
"""End-to-end smoke against the fake server.

Skipped automatically if the fake server's routes don't match the installed SDK.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import pytest

from batch_imagegen.models import Batch, ImageJob, JobStatus
from batch_imagegen.store import BatchStore
from batch_imagegen.orchestrator import OrchestratorRunner, default_factory


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.fake_pixelbin.server:app",
         "--port", str(port), "--log-level", "warning"],
    )
    # wait for boot
    for _ in range(40):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    proc.terminate(); proc.wait(timeout=5)


@pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1",
    reason="Set RUN_E2E=1 after aligning fake server routes with the installed SDK.",
)
def test_orchestrator_against_fake_server(tmp_path: Path, fake_server: str) -> None:
    # The default_factory hardcodes the real API domain; we need to monkeypatch
    # the client factory to point at the fake. See client.py for `domain` arg.
    from batch_imagegen.client import make_pixelbin_client
    from batch_imagegen.predictor import Model, submit_and_wait
    from batch_imagegen.uploader import upload_local_file
    import asyncio

    def factory(api_key: str):
        client = make_pixelbin_client(api_key, domain=fake_server)
        async def upload(path): return await asyncio.to_thread(upload_local_file, client, path)
        async def submit(model_value, prompt, source_url, params):
            return await submit_and_wait(client, Model(model_value), prompt, source_url, params)
        return upload, submit

    store = BatchStore(tmp_path)
    jobs = [
        ImageJob(f"j{i}", f"https://cdn/{i}.jpg", None, None, None,
                 JobStatus.PENDING, None, 0, "2026-05-25T00:00:00Z")
        for i in range(10)
    ]
    batch = Batch(
        batch_id="e2e1", name="e2e", model=Model.NANO_BANANA_PRO.value,
        prompt="p", model_params={"aspect_ratio": "1:1", "output_resolution": "1K"},
        concurrency=3, created_at="2026-05-25T00:00:00Z", completed_at=None, jobs=jobs,
    )
    store.save(batch)

    runner = OrchestratorRunner(store, factory=factory)
    runner.submit("e2e1", "FAKE")

    deadline = time.time() + 15
    while time.time() < deadline:
        b = store.load("e2e1")
        if b.completed_at:
            break
        time.sleep(0.1)
    final = store.load("e2e1")
    assert final.completed_at is not None
    # Allow synthetic failures from the fake server
    successes = sum(1 for j in final.jobs if j.status == JobStatus.SUCCESS)
    assert successes >= 7
```

- [ ] **Step 3: Run (gated)**

```bash
pytest tests/test_e2e_fake.py -v  # skipped by default
```

To execute later: align the fake server route paths with the installed SDK, then `RUN_E2E=1 pytest tests/test_e2e_fake.py -v`.

- [ ] **Step 4: Commit**

```bash
git add tests/fake_pixelbin/server.py tests/test_e2e_fake.py
git commit -m "test: fake PixelBin server + gated E2E"
```

---

## Task 14: Streamlit `AppTest` for form validation

**Files:**
- Create: `tests/test_app.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_app.py
from streamlit.testing.v1 import AppTest


def test_start_batch_disabled_without_api_key():
    at = AppTest.from_file("app.py", default_timeout=10).run()
    # Find the "Start batch" button
    start = next((b for b in at.button if b.label == "Start batch"), None)
    assert start is not None
    assert start.disabled is True


def test_empty_state_on_batches_page():
    at = AppTest.from_file("app.py", default_timeout=10).run()
    at.sidebar.radio[0].set_value("Batches").run()
    # "No batches yet." message should appear via st.info
    infos = [i for i in at.info if "No batches yet" in str(i.value)]
    assert infos, "Expected empty-state info message on Batches page"
```

- [ ] **Step 2: Run**

```bash
pytest tests/test_app.py -v
```

Expected: 2 passed. (If selectors break across Streamlit versions, prefer `at.session_state` reads to assert state instead of widget labels.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_app.py
git commit -m "test(ui): AppTest coverage for disabled-submit and empty states"
```

---

## Task 15: README + manual smoke checklist

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the stub README with a runnable guide**

```markdown
# batch-imagegen

Local Streamlit tool for the annotation team to run batch image generation jobs against the [PixelBin Prediction API](https://www.pixelbin.io/docs/prediction-api/overview). Supports **Nano Banana Pro** and **GPT Image 2**.

## Features

- Drop in 100+ images (local files or URLs), one prompt, one model
- Worker pool with configurable concurrency (default 5)
- Live progress with auto-refresh
- Retry failed jobs
- Download all successful outputs as a ZIP with a `manifest.csv`
- All state on disk under `~/.batch-imagegen/` — survives reloads

## Setup

```bash
pip install -e ".[dev]"
```

## Run

```bash
streamlit run app.py
```

Open the URL it prints. In the sidebar:

1. Paste your PixelBin API token (Settings → API tokens in the PixelBin console). It is stored only in this browser session — never written to disk.
2. Adjust "Concurrent workers" if needed.

Then on the **New batch** page, fill name + model + prompt, pick "Upload files" or "Paste URLs", and click **Start batch**. The **Batches** page shows live progress and the **Download ZIP** button.

## Where things live

```
~/.batch-imagegen/
├── batches/<batch_id>.json     # one file per batch
├── uploads/                    # local files copied here before upload
└── downloads/<batch_id>.zip    # built on demand
```

Override the root with `BATCH_IMAGEGEN_HOME=/some/path streamlit run app.py`.

## Tests

```bash
pytest -v
```

The E2E test against a fake server is gated by `RUN_E2E=1`. See `tests/test_e2e_fake.py`.

## Caveats

- Streamlit's hot-reloader kills the background orchestrator thread on code changes. Restart `streamlit run` if jobs stall.
- The tool processes one batch at a time. Submitting a second batch while one is running queues it.
- No cancellation in v1 — once started, a batch runs to completion.

## Manual smoke checklist

Before each release:

- [ ] Run `pytest` — all green
- [ ] `streamlit run app.py`, paste a real test API key, kick off a 5-image batch with 2 known-bad URLs — verify 3 successes + 2 failures + download ZIP contains 3 images + manifest with all 5
- [ ] Refresh the browser mid-batch — progress should resume from JSON
- [ ] Hit **Retry failed** — verify those 2 re-run
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: full README with run instructions and smoke checklist"
```

---

## Self-review

**Spec coverage:**
- §1 Problem & Goals → covered end-to-end (Tasks 10–12 UI, Tasks 7–8 worker pool, Task 9 ZIP).
- §2 Architecture → Task 8 (thread + queue + asyncio).
- §3 Module Layout → matches Tasks 2/3/5/6/7/8/9/10/11.
- §4 Data Model → Task 2.
- §5 Orchestrator → Tasks 7 (core) + 8 (runner).
- §6 Streamlit UI:
  - §6.1 Sidebar → Task 10.
  - §6.2 New batch page → Task 11.
  - §6.3 Batches page → Task 12.
  - §6.4 Empty states → Task 10 (sidebar warning) + Task 12 (no batches) + Task 12 (download disabled).
  - §6.5 Error display → field-level in Task 11, batch-level via `st.toast`/`st.error` in Task 12, per-job table in Task 12.
  - §6.6 Toasts → Task 11 (submit), Task 12 (completion + retry).
  - §6.7 API key validation → **not implemented** as a task. Decision: the spec's pre-flight `get("nonexistent")` check is nice-to-have; the existing submit-time failure path already surfaces invalid keys via the orchestrator's crash handler in Task 8. Deferring to a follow-up.
- §7 ZIP Download → Task 9.
- §8 Testing → Unit (Tasks 2–9, 11), AppTest (Task 14), Fake-server E2E (Task 13). Manual smoke documented in Task 15. All four layers present.
- §9 Open Risks → "PixelBin upload SDK surface" addressed by Task 4 (verification + notes) before Task 5 (uploader).

**Placeholder scan:** No "TODO" / "TBD" / "implement later" in any step. Task 5 has the explicit `<UPLOAD_METHOD>` / `<URL_KEY>` placeholders that resolve from Task 4's notes — they are documented as such and visible at the top of the task. Acceptable per the spec's open risk.

**Type consistency:** `Model` enum values, `JobStatus` values, `Batch`/`ImageJob` field names, and the `(model_value, prompt, source_url, params)` `submit_fn` signature all stay consistent from Task 7 (where it's defined) through Task 8 (where the production binding adapts `submit_and_wait`). `BatchStore.save/load/list_batches`, `build_zip`, and `process_batch` signatures match across call sites.

**One open-state gap to flag during execution:** the `api_key_valid` pre-flight check (§6.7) is not in any task. Add as Task 16 if you want it before launch; otherwise it can ship as a follow-up.

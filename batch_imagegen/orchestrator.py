from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Tuple

from .models import Batch, ImageJob, JobStatus, TERMINAL_STATUSES
from .predictor import PredictionError
from .store import BatchStore

try:
    from pixelbin.common.exceptions import PixelbinServerResponseError
except ImportError:
    PixelbinServerResponseError = None  # type: ignore[assignment]

MAX_ATTEMPTS = 2  # initial + 1 retry

UploadFn = Callable[[Path], Awaitable[str]]
SubmitFn = Callable[..., Awaitable[tuple[str, str]]]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, PredictionError):
        return False
    if PixelbinServerResponseError is not None and isinstance(exc, PixelbinServerResponseError):
        return True
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


# --- thread/queue runner (Task 8) ---
import queue
import threading

from .client import make_pixelbin_client
from .predictor import Model, submit_and_wait
from .uploader import upload_local_file


# Factory: (api_key) -> (upload_fn, submit_fn) bound for this batch
Factory = Callable[[str], Tuple[UploadFn, SubmitFn]]


def default_factory(api_key: str) -> Tuple[UploadFn, SubmitFn]:
    client = make_pixelbin_client(api_key)

    async def upload_fn(path: Path) -> str:
        return await upload_local_file(client, path)

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

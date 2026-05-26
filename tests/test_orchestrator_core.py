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
async def test_process_batch_retries_pixelbin_server_error(tmp_path: Path) -> None:
    pytest.importorskip("pixelbin")
    try:
        from pixelbin.common.exceptions import PixelbinServerResponseError
    except ImportError:
        pytest.skip("PixelbinServerResponseError not available in installed SDK")

    store = BatchStore(tmp_path)
    batch = _batch([_job(1, "https://cdn/a.jpg")])
    store.save(batch)

    calls = {"n": 0}
    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            try:
                raise PixelbinServerResponseError("503 Service Unavailable")
            except TypeError:
                raise PixelbinServerResponseError()
        return ("https://out/a.jpg", "p1")

    await process_batch(
        batch_id="b1", store=store,
        upload_fn=AsyncMock(), submit_fn=flaky,
    )
    final = store.load("b1")
    assert final.jobs[0].status == JobStatus.SUCCESS
    assert final.jobs[0].attempts == 2


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

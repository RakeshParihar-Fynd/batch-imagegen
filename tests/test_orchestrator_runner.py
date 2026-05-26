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

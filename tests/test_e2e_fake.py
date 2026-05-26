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
from batch_imagegen.orchestrator import OrchestratorRunner


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
    from batch_imagegen.client import make_pixelbin_client
    from batch_imagegen.predictor import Model, submit_and_wait
    from batch_imagegen.uploader import upload_local_file
    import asyncio

    def factory(api_key: str):
        client = make_pixelbin_client(api_key, domain=fake_server)
        async def upload(path): return await upload_local_file(client, path)
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
    successes = sum(1 for j in final.jobs if j.status == JobStatus.SUCCESS)
    assert successes >= 7

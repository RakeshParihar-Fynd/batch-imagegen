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

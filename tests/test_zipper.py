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

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

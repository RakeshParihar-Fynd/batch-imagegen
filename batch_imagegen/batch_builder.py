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

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

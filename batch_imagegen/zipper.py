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
    # Use only batch-level timestamps (always valid ISO); job updated_at may be
    # an opaque store key and is not guaranteed to be a valid ISO string.
    iso_str = batch.completed_at or batch.created_at
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()


def _unique_arcname(base_source: str, ext: str, used_stems: set[str]) -> str:
    """Return a unique archive name, deduplicating by stem (not full name).

    Both ``a_out.jpeg`` and ``a_out.png`` share the stem ``a_out``, so the
    second file gets ``a_out_2.<ext>``, the third ``a_out_3.<ext>``, etc.
    """
    stem = Path(base_source).stem or "image"
    base_stem = f"{stem}_out"
    if base_stem not in used_stems:
        used_stems.add(base_stem)
        return f"{base_stem}.{ext}"
    i = 2
    while f"{base_stem}_{i}" in used_stems:
        i += 1
    new_stem = f"{base_stem}_{i}"
    used_stems.add(new_stem)
    return f"{new_stem}.{ext}"


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

    used_stems: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for j in successes:
            if j.job_id in download_failed:
                continue
            content, ext = fetched[j.job_id]
            arcname = _unique_arcname(j.source, ext, used_stems)
            zf.writestr(arcname, content)
        zf.writestr("manifest.csv", _build_manifest(batch, download_failed))

    return zip_path

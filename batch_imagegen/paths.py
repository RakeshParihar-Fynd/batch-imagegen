from __future__ import annotations
import os
from pathlib import Path


def data_root() -> Path:
    root = Path(os.environ.get("BATCH_IMAGEGEN_HOME", Path.home() / ".batch-imagegen"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "batches").mkdir(exist_ok=True)
    (root / "downloads").mkdir(exist_ok=True)
    return root


def batches_dir() -> Path:
    return data_root() / "batches"


def downloads_dir() -> Path:
    return data_root() / "downloads"

from __future__ import annotations
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import Batch, batch_from_dict


class BatchStore:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_id: str) -> Path:
        return self._root / f"{batch_id}.json"

    def save(self, batch: Batch) -> None:
        target = self._path(batch.batch_id)
        # Atomic write: tmp file in same dir + os.replace
        fd, tmp = tempfile.mkstemp(
            prefix=f".{batch.batch_id}.", suffix=".tmp", dir=str(self._root),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(batch), f, indent=2, default=str)
            os.replace(tmp, target)
        except Exception:
            # If os.fdopen never ran (or raised), fd is still open; close it.
            # If fdopen succeeded, the `with` already closed fd — os.close raises OSError.
            try:
                os.close(fd)
            except OSError:
                pass
            Path(tmp).unlink(missing_ok=True)
            raise

    def load(self, batch_id: str) -> Batch:
        return batch_from_dict(json.loads(self._path(batch_id).read_text()))

    def list_batches(self) -> list[Batch]:
        out: list[Batch] = []
        for p in self._root.glob("*.json"):
            try:
                out.append(batch_from_dict(json.loads(p.read_text())))
            except Exception:
                continue  # skip malformed
        out.sort(key=lambda b: b.created_at, reverse=True)
        return out

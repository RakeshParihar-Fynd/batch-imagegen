from __future__ import annotations
from pathlib import Path
from typing import Any


class UploadError(Exception):
    """Raised when uploading a local file to PixelBin fails."""


async def upload_local_file(client: Any, path: Path) -> str:
    """Upload a local file to PixelBin storage and return the CDN URL.

    Uses `client.uploader.uploadAsync(file=<binary>)` per the SDK surface notes
    at docs/superpowers/notes/2026-05-25-pixelbin-sdk-check.md.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    try:
        # Read file bytes — the SDK accepts bytes or BufferedIOBase.
        data = p.read_bytes()
        response: dict[str, Any] = await client.uploader.uploadAsync(file=data, name=p.name)
        url = response.get("url")
        if not url:
            raise UploadError(f"Upload succeeded but no URL returned: {response!r}")
        return url
    except Exception as e:
        if isinstance(e, (UploadError, FileNotFoundError)):
            raise
        raise UploadError(str(e)) from e

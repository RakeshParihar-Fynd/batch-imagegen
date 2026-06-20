from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any


class FalError(Exception):
    """Raised when Fal upload/inference fails."""


class FalConfigError(ValueError):
    """Raised when Fal credentials are missing."""


def _require_key(fal_key: str | None) -> str:
    key = (fal_key or "").strip()
    if not key:
        raise FalConfigError("FAL API key is required for Qwen model")
    return key


def _set_fal_key_env(fal_key: str) -> None:
    # fal_client reads credentials from FAL_KEY.
    os.environ["FAL_KEY"] = fal_key


def _load_fal_client() -> Any:
    try:
        import fal_client  # type: ignore[import-not-found]
    except ImportError as e:
        raise FalConfigError(
            "fal-client is not installed. Install with: pip install fal-client"
        ) from e
    return fal_client


async def upload_local_file_to_fal(path: Path, fal_key: str | None) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    key = _require_key(fal_key)
    _set_fal_key_env(key)
    fal_client = _load_fal_client()

    try:
        return await asyncio.to_thread(fal_client.upload_file, str(p))
    except Exception as e:
        raise FalError(str(e)) from e


async def qwen_edit(
    *,
    model_id: str,
    prompt: str,
    source_url: str,
    params: dict[str, Any],
    fal_key: str | None,
) -> tuple[str, str]:
    key = _require_key(fal_key)
    _set_fal_key_env(key)
    fal_client = _load_fal_client()

    arguments = {"image_urls": [source_url], "prompt": prompt, **params}

    try:
        response = await asyncio.to_thread(
            fal_client.subscribe,
            model_id,
            arguments=arguments,
        )
    except Exception as e:
        raise FalError(str(e)) from e

    payload = response.get("data") if isinstance(response, dict) and isinstance(response.get("data"), dict) else response
    images = payload.get("images") if isinstance(payload, dict) else None
    if not images:
        raise FalError("Fal response missing output images")
    first = images[0] if isinstance(images[0], dict) else {}
    output_url = first.get("url")
    if not output_url:
        raise FalError("Fal response missing image URL")

    request_id = ""
    if isinstance(response, dict):
        request_id = str(response.get("requestId") or response.get("request_id") or "")
    return output_url, request_id

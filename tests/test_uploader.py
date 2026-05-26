from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from batch_imagegen.uploader import upload_local_file, UploadError


@pytest.mark.asyncio
async def test_upload_local_file_returns_url(tmp_path: Path) -> None:
    f = tmp_path / "a.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0fake")
    client = MagicMock()
    client.uploader.uploadAsync = AsyncMock(return_value={"url": "https://cdn.pixelbin.io/a.jpg"})

    url = await upload_local_file(client, f)

    assert url == "https://cdn.pixelbin.io/a.jpg"
    client.uploader.uploadAsync.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_local_file_raises_uploaderror_on_sdk_exception(tmp_path: Path) -> None:
    f = tmp_path / "a.jpg"; f.write_bytes(b"x")
    client = MagicMock()
    client.uploader.uploadAsync = AsyncMock(side_effect=RuntimeError("network down"))

    with pytest.raises(UploadError):
        await upload_local_file(client, f)


@pytest.mark.asyncio
async def test_upload_local_file_rejects_missing_file(tmp_path: Path) -> None:
    client = MagicMock()
    with pytest.raises(FileNotFoundError):
        await upload_local_file(client, tmp_path / "missing.jpg")


@pytest.mark.asyncio
async def test_upload_local_file_raises_uploaderror_when_response_lacks_url(tmp_path: Path) -> None:
    f = tmp_path / "a.jpg"; f.write_bytes(b"x")
    client = MagicMock()
    client.uploader.uploadAsync = AsyncMock(return_value={"id": "abc"})  # no "url"

    with pytest.raises(UploadError):
        await upload_local_file(client, f)

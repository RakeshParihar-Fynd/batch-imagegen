from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from batch_imagegen.orchestrator import default_factory
from batch_imagegen.predictor import Model


@pytest.mark.asyncio
async def test_default_factory_routes_qwen_submit_to_fal(monkeypatch):
    qwen_mock = AsyncMock(return_value=("https://fal/out.jpg", "req_1"))
    pixelbin_mock = AsyncMock(return_value=("https://pb/out.jpg", "pred_1"))

    monkeypatch.setattr("batch_imagegen.orchestrator.qwen_edit", qwen_mock)
    monkeypatch.setattr("batch_imagegen.orchestrator.submit_and_wait", pixelbin_mock)

    upload_fn, submit_fn = default_factory("PIXELBIN_KEY", fal_key="FAL_ID:SECRET")

    out, req_id = await submit_fn(Model.QWEN_IMAGE_EDIT.value, "p", "https://cdn/in.jpg", {})

    assert out == "https://fal/out.jpg"
    assert req_id == "req_1"
    qwen_mock.assert_awaited_once()
    pixelbin_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_factory_routes_non_qwen_upload_to_pixelbin(monkeypatch, tmp_path: Path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")

    fal_upload_mock = AsyncMock(return_value="https://fal/in.jpg")
    pixelbin_upload_mock = AsyncMock(return_value="https://pixelbin/in.jpg")

    monkeypatch.setattr("batch_imagegen.orchestrator.upload_local_file_to_fal", fal_upload_mock)
    monkeypatch.setattr("batch_imagegen.orchestrator.upload_local_file", pixelbin_upload_mock)

    upload_fn, _ = default_factory("PIXELBIN_KEY", fal_key="FAL_ID:SECRET")

    source_url = await upload_fn(Model.NANO_BANANA_PRO.value, f)

    assert source_url == "https://pixelbin/in.jpg"
    pixelbin_upload_mock.assert_awaited_once()
    fal_upload_mock.assert_not_awaited()

from unittest.mock import AsyncMock, MagicMock
import pytest
from batch_imagegen.predictor import (
    Model, build_input, submit_and_wait, PredictionError,
)


def test_build_input_nano_banana_pro():
    payload = build_input(
        Model.NANO_BANANA_PRO,
        prompt="hello",
        source_url="https://cdn/x.jpg",
        params={"aspect_ratio": "1:1", "output_resolution": "1K"},
    )
    assert payload == {
        "prompt": "hello",
        "images": ["https://cdn/x.jpg"],
        "aspect_ratio": "1:1",
        "output_resolution": "1K",
    }


def test_build_input_gpt2_includes_quality():
    payload = build_input(
        Model.GPT_IMAGE_2,
        prompt="hello",
        source_url="https://cdn/x.jpg",
        params={"aspect_ratio": "1:1", "output_resolution": "1K", "quality": "high"},
    )
    assert payload["quality"] == "high"


@pytest.mark.asyncio
async def test_submit_and_wait_returns_output_on_success():
    client = MagicMock()
    client.predictions.create_and_waitAsync = AsyncMock(return_value={
        "_id": "abc", "status": "SUCCESS",
        "output": ["https://delivery/out.jpg"],
    })
    output, prediction_id = await submit_and_wait(
        client, Model.NANO_BANANA_PRO, "p", "https://cdn/x.jpg", {},
    )
    assert output == "https://delivery/out.jpg"
    assert prediction_id == "abc"


@pytest.mark.asyncio
async def test_submit_and_wait_raises_on_failure():
    client = MagicMock()
    client.predictions.create_and_waitAsync = AsyncMock(return_value={
        "_id": "abc", "status": "FAILURE", "error": {"message": "bad prompt"},
    })
    with pytest.raises(PredictionError, match="bad prompt"):
        await submit_and_wait(client, Model.NANO_BANANA_PRO, "p", "u", {})

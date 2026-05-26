from __future__ import annotations
from enum import Enum
from typing import Any


class Model(str, Enum):
    NANO_BANANA_PRO = "nanoBananaPro_generate"
    GPT_IMAGE_2     = "gpt2_generate"


class PredictionError(Exception):
    """Raised when a PixelBin prediction returns FAILURE."""


def build_input(model: Model, *, prompt: str, source_url: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"prompt": prompt, "images": [source_url], **params}


async def submit_and_wait(
    client: Any, model: Model, prompt: str, source_url: str, params: dict[str, Any],
) -> tuple[str, str]:
    """Submit one prediction and wait for terminal status.

    Returns (output_url, prediction_id). Raises PredictionError on FAILURE.
    """
    payload = build_input(model, prompt=prompt, source_url=source_url, params=params)
    result = await client.predictions.create_and_waitAsync(name=model.value, input=payload)
    prediction_id = result.get("_id", "")
    status = result.get("status")
    if status == "SUCCESS":
        outputs = result.get("output") or []
        if not outputs:
            raise PredictionError("SUCCESS but no output URL returned")
        return outputs[0], prediction_id
    err = (result.get("error") or {}).get("message") or f"status={status}"
    raise PredictionError(err)

"""A ~50-line FastAPI server that mimics the subset of PixelBin we use."""
from __future__ import annotations
import asyncio
import random
import uuid
from fastapi import FastAPI

app = FastAPI()
_jobs: dict[str, dict] = {}


@app.post("/service/panel/transformation/v1.0/org/-/predictions")
async def create_prediction(payload: dict) -> dict:
    pid = f"{payload.get('name', 'p')}--{uuid.uuid4()}"
    _jobs[pid] = {"_id": pid, "status": "PENDING", "input": payload.get("input", {})}
    asyncio.create_task(_finish_later(pid))
    return _jobs[pid]


@app.get("/service/panel/transformation/v1.0/org/-/predictions/{pid}")
async def get_prediction(pid: str) -> dict:
    return _jobs.get(pid, {"_id": pid, "status": "FAILURE", "error": {"message": "not found"}})


async def _finish_later(pid: str) -> None:
    await asyncio.sleep(random.uniform(0.1, 0.4))
    if random.random() < 0.1:
        _jobs[pid].update(status="FAILURE", error={"message": "synthetic failure"})
    else:
        _jobs[pid].update(status="SUCCESS",
                          output=[f"https://fake.local/out/{pid}.jpg"])

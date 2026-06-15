from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .schemas import BatchScoreRequest, BatchScoreResponse, EntityProfileResponse, ReviewQueueItem, ScoreRequest, ScoreResponse
from .service import service

app = FastAPI(title="中文金融舆情风险智能预警系统", version="1.0.0")


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> dict:
    return service.score(request.text, text_id=request.text_id, force_llm=request.force_llm)


@app.post("/batch_score", response_model=BatchScoreResponse)
def batch_score(request: BatchScoreRequest) -> dict:
    return {"results": [service.score(item.text, text_id=item.text_id, force_llm=item.force_llm) for item in request.items]}


@app.get("/entity/{name}", response_model=EntityProfileResponse)
def entity(name: str) -> dict:
    profile = service.entity_profile(name)
    if profile is None:
        raise HTTPException(status_code=404, detail="entity profile not found")
    return profile


@app.get("/review_queue", response_model=list[ReviewQueueItem])
def review_queue(status: str = "pending", limit: int = 50) -> list[dict]:
    return service.review_queue(status=status, limit=limit)


@app.get("/metrics")
def metrics() -> dict:
    return service.metrics()

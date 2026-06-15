from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)
    text_id: str | None = None
    force_llm: bool = False


class BatchScoreRequest(BaseModel):
    items: list[ScoreRequest]


class RiskEventResponse(BaseModel):
    entity: str
    risk_type: str
    risk_type_source: str = "keyword_weak_label"
    risk_type_matched: bool = False
    severity: str
    confidence: float
    risk_score: float
    risk_level: str
    evidence: str
    action: str
    source_type: str = "unknown"


class ScoreResponse(BaseModel):
    request_id: str
    text_id: str
    has_negative: bool
    model_has_negative: bool
    entity_missing_review: bool = False
    hallucinated_entity: bool = False
    stage: str
    encoder_confidence: float
    risk_events: list[RiskEventResponse]
    latency_sec: float
    raw: dict[str, Any] = Field(default_factory=dict)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]


class EntityProfileResponse(BaseModel):
    entity: str
    event_count: int
    max_risk_level: str
    avg_risk_score: float
    risk_type_distribution: dict[str, int]
    latest_evidence: str


class ReviewQueueItem(BaseModel):
    id: int
    text_id: str | None
    entity: str | None
    risk_score: float | None
    risk_level: str | None
    reason: str | None
    status: str
    payload: str | None

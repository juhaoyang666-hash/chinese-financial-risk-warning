from __future__ import annotations

from dataclasses import dataclass

from .risk_types import severity_score, source_reliability_score


@dataclass(frozen=True)
class RiskScoreInput:
    model_confidence: float = 0.85
    severity: str = "medium"
    source_type: str = "unknown"
    entity_history_score: float = 0.0
    recency_score: float = 0.5


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def compute_risk_score(score_input: RiskScoreInput) -> float:
    score = (
        0.35 * clamp(score_input.model_confidence)
        + 0.25 * severity_score(score_input.severity)
        + 0.20 * source_reliability_score(score_input.source_type)
        + 0.10 * clamp(score_input.entity_history_score)
        + 0.10 * clamp(score_input.recency_score)
    )
    return round(score * 100, 2)


def risk_level(score: float) -> str:
    if score >= 88:
        return "critical"
    if score >= 70:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def review_action(level: str, confidence: float) -> str:
    if level in {"critical", "high"}:
        return "人工复核"
    if confidence < 0.65:
        return "低置信复核"
    if level == "medium":
        return "观察"
    return "自动放行"


def entity_history_score(event_count: int) -> float:
    return clamp(event_count / 10.0)

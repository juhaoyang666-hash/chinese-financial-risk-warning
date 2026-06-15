from __future__ import annotations

from typing import Any, Iterable

from .candidates import extract_candidate_entities, filter_entities_by_candidates
from .risk_types import extract_evidence, infer_risk_type, infer_source_type
from .scoring import RiskScoreInput, compute_risk_score, entity_history_score, review_action, risk_level
from .schema import dedupe_preserve_order


def build_risk_events(
    text: str,
    entities: Iterable[str],
    *,
    confidence: float = 0.85,
    entity_history: dict[str, int] | None = None,
    candidate_entities: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    entity_history = entity_history or {}
    candidates = list(candidate_entities or extract_candidate_entities(text, entities))
    clean_entities = filter_entities_by_candidates(dedupe_preserve_order(list(entities)), text, candidates)
    risk_match = infer_risk_type(text)
    source_type = infer_source_type(text)
    events: list[dict[str, Any]] = []
    for entity in clean_entities:
        history_score = entity_history_score(entity_history.get(entity, 0))
        score = compute_risk_score(
            RiskScoreInput(
                model_confidence=confidence,
                severity=risk_match.severity,
                source_type=source_type,
                entity_history_score=history_score,
                recency_score=0.5,
            )
        )
        level = risk_level(score)
        events.append(
            {
                "entity": entity,
                "risk_type": risk_match.risk_type,
                "risk_type_source": "keyword_weak_label",
                "risk_type_matched": bool(risk_match.keyword),
                "severity": risk_match.severity,
                "confidence": round(float(confidence), 4),
                "risk_score": score,
                "risk_level": level,
                "evidence": extract_evidence(text, entity, risk_match.keyword),
                "action": review_action(level, float(confidence)),
                "source_type": source_type,
            }
        )
    return events


def record_to_risk_record(record: dict[str, Any], *, use_gold: bool = True) -> dict[str, Any]:
    text = record.get("text", "")
    entities = record.get("entities", []) if use_gold else record.get("pred_entities", [])
    candidate_entities = extract_candidate_entities(text, entities)
    risk_events = build_risk_events(text, entities, candidate_entities=candidate_entities)
    return {
        **record,
        "candidate_entities": candidate_entities,
        "risk_events": risk_events,
        "has_negative": bool(risk_events) if use_gold else bool(record.get("pred_has_negative", risk_events)),
    }

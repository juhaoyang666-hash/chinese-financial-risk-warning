from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def entity_metrics(gold_entities: list[list[str]], pred_entities: list[list[str]]) -> dict[str, float]:
    tp = fp = fn = 0
    for gold, pred in zip(gold_entities, pred_entities):
        gold_set = {item.strip().casefold() for item in gold if item.strip()}
        pred_set = {item.strip().casefold() for item in pred if item.strip()}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "entity_precision": float(precision),
        "entity_recall": float(recall),
        "entity_f1": float(f1),
        "entity_tp": float(tp),
        "entity_fp": float(fp),
        "entity_fn": float(fn),
    }


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, float]:
    y_true = [int(record["gold_has_negative"]) for record in records]
    y_pred = [int(record["pred_has_negative"]) for record in records]
    gold_entities = [record.get("gold_entities", []) for record in records]
    pred_entities = [record.get("pred_entities", []) for record in records]
    metrics = binary_metrics(y_true, y_pred)
    metrics.update(entity_metrics(gold_entities, pred_entities))
    if any("invalid_json" in record for record in records):
        metrics["invalid_json_rate"] = float(np.mean([bool(record.get("invalid_json", False)) for record in records]))
    if any("latency_sec" in record for record in records):
        metrics["avg_latency_sec"] = float(np.mean([float(record.get("latency_sec", 0.0)) for record in records]))
    return metrics


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


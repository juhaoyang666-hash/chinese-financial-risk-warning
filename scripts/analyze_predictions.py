#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.metrics import evaluate_records, save_json
from risk_nlp.schema import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FINNSP prediction errors and entity reliability.")
    parser.add_argument(
        "--predictions",
        default="outputs/qwen3-8b-qlora-main-eval/predictions.jsonl",
    )
    parser.add_argument("--eval_file", default="data/processed/finnsp_eval.jsonl")
    parser.add_argument("--output_dir", default="outputs/analysis/qwen3-8b-qlora-main")
    parser.add_argument("--max_examples", type=int, default=8)
    return parser.parse_args()


def norm(text: str) -> str:
    return "".join(str(text).casefold().split())


def entity_in_text(entity: str, text: str) -> bool:
    entity_key = norm(entity)
    return bool(entity_key) and entity_key in norm(text)


def entity_set(items: list[str]) -> set[str]:
    return {norm(item) for item in items if norm(item)}


def enrich_records(predictions: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_by_id = {record["id"]: record for record in eval_records}
    enriched = []
    for pred in predictions:
        source = eval_by_id.get(pred["id"], {})
        record = dict(pred)
        record["text"] = source.get("text", "")
        record["instruction"] = source.get("instruction", "")
        record["raw_gold_output"] = source.get("raw_output", "")
        enriched.append(record)
    return enriched


def classify_errors(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "false_positive": [],
        "false_negative": [],
        "entity_mismatch": [],
        "multi_entity_cases": [],
        "hallucinated_entity_cases": [],
    }
    for record in records:
        gold_label = bool(record.get("gold_has_negative", False))
        pred_label = bool(record.get("pred_has_negative", False))
        gold_entities = record.get("gold_entities", [])
        pred_entities = record.get("pred_entities", [])

        if pred_label and not gold_label:
            buckets["false_positive"].append(record)
        if gold_label and not pred_label:
            buckets["false_negative"].append(record)
        if gold_label and pred_label and entity_set(gold_entities) != entity_set(pred_entities):
            buckets["entity_mismatch"].append(record)
        if len(gold_entities) > 1 or len(pred_entities) > 1:
            buckets["multi_entity_cases"].append(record)
        if any(not entity_in_text(entity, record.get("text", "")) for entity in pred_entities):
            buckets["hallucinated_entity_cases"].append(record)
    return buckets


def reliability_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    pred_entities_total = 0
    pred_entities_in_text = 0
    hallucinated_entities = 0
    samples_with_pred_entities = 0
    samples_with_hallucinated_entity = 0
    multi_entity_total = 0
    multi_entity_errors = 0

    for record in records:
        text = record.get("text", "")
        pred_entities = record.get("pred_entities", [])
        gold_entities = record.get("gold_entities", [])
        if pred_entities:
            samples_with_pred_entities += 1
        sample_has_hallucination = False
        for entity in pred_entities:
            pred_entities_total += 1
            if entity_in_text(entity, text):
                pred_entities_in_text += 1
            else:
                hallucinated_entities += 1
                sample_has_hallucination = True
        if sample_has_hallucination:
            samples_with_hallucinated_entity += 1

        if len(gold_entities) > 1:
            multi_entity_total += 1
            if entity_set(gold_entities) != entity_set(pred_entities):
                multi_entity_errors += 1

    entity_in_text_rate = pred_entities_in_text / pred_entities_total if pred_entities_total else 0.0
    hallucinated_entity_rate = hallucinated_entities / pred_entities_total if pred_entities_total else 0.0
    sample_hallucination_rate = (
        samples_with_hallucinated_entity / samples_with_pred_entities if samples_with_pred_entities else 0.0
    )
    multi_entity_error_rate = multi_entity_errors / multi_entity_total if multi_entity_total else 0.0
    return {
        "pred_entities_total": float(pred_entities_total),
        "pred_entities_in_text": float(pred_entities_in_text),
        "hallucinated_entities": float(hallucinated_entities),
        "entity_in_text_rate": float(entity_in_text_rate),
        "hallucinated_entity_rate": float(hallucinated_entity_rate),
        "sample_hallucination_rate": float(sample_hallucination_rate),
        "multi_entity_total": float(multi_entity_total),
        "multi_entity_errors": float(multi_entity_errors),
        "multi_entity_error_rate": float(multi_entity_error_rate),
    }


def compact_case(record: dict[str, Any]) -> dict[str, Any]:
    text = record.get("text", "")
    return {
        "id": record.get("id"),
        "text": text[:260],
        "gold": {
            "has_negative": record.get("gold_has_negative"),
            "entities": record.get("gold_entities", []),
        },
        "pred": {
            "has_negative": record.get("pred_has_negative"),
            "entities": record.get("pred_entities", []),
        },
        "raw_output": record.get("raw_output", ""),
    }


def write_summary(
    output_path: Path,
    metrics: dict[str, Any],
    buckets: dict[str, list[dict[str, Any]]],
    max_examples: int,
) -> None:
    lines = [
        "# Qwen3-8B QLoRA 错误分析与实体可信性",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    lines.extend(["", "## 错误类型计数", "", "| 错误类型 | 样本数 |", "|---|---:|"])
    for name, records in buckets.items():
        lines.append(f"| {name} | {len(records)} |")

    for name, records in buckets.items():
        lines.extend(["", f"## {name} 示例", ""])
        for record in records[:max_examples]:
            case = compact_case(record)
            lines.append(f"- `{case['id']}` gold={case['gold']} pred={case['pred']}")
            lines.append(f"  - text: {case['text']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_jsonl(ROOT / args.predictions)
    eval_records = load_jsonl(ROOT / args.eval_file)
    records = enrich_records(predictions, eval_records)
    buckets = classify_errors(records)

    metrics = evaluate_records(records)
    metrics.update(reliability_metrics(records))
    metrics["num_records"] = float(len(records))
    for name, bucket_records in buckets.items():
        metrics[f"{name}_count"] = float(len(bucket_records))

    save_json(metrics, output_dir / "reliability_metrics.json")
    for name, bucket_records in buckets.items():
        write_jsonl([compact_case(record) for record in bucket_records], output_dir / f"{name}.jsonl")
    write_summary(output_dir / "error_summary.md", metrics, buckets, args.max_examples)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()

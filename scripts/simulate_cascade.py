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
    parser = argparse.ArgumentParser(description="Simulate an encoder-to-LLM cascade for FINNSP risk screening.")
    parser.add_argument(
        "--encoder_predictions",
        default="outputs/encoder/valuesimplex-ai-lab__FinBERT2-large/predictions.jsonl",
        help="First-stage encoder predictions.",
    )
    parser.add_argument(
        "--llm_predictions",
        default="outputs/qwen3-8b-qlora-main-eval/predictions.jsonl",
        help="Second-stage LLM predictions.",
    )
    parser.add_argument("--output_dir", default="outputs/analysis/cascade_finbert2_qwen3")
    return parser.parse_args()


def by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in records}


def build_cascade_records(
    encoder_records: list[dict[str, Any]],
    llm_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    llm_by_id = by_id(llm_records)
    cascade_records = []
    llm_calls = 0
    encoder_negative_pass = 0
    encoder_positive_review = 0

    for encoder_record in encoder_records:
        llm_record = llm_by_id[encoder_record["id"]]
        call_llm = bool(encoder_record.get("pred_has_negative", False))
        if call_llm:
            llm_calls += 1
            encoder_positive_review += 1
            pred_has_negative = bool(llm_record.get("pred_has_negative", False))
            pred_entities = llm_record.get("pred_entities", [])
        else:
            encoder_negative_pass += 1
            pred_has_negative = False
            pred_entities = []

        cascade_records.append(
            {
                "id": encoder_record["id"],
                "gold_has_negative": bool(encoder_record.get("gold_has_negative", False)),
                "pred_has_negative": pred_has_negative,
                "gold_entities": encoder_record.get("gold_entities", []),
                "pred_entities": pred_entities,
                "stage": "llm_review" if call_llm else "encoder_pass",
                "encoder_pred_has_negative": bool(encoder_record.get("pred_has_negative", False)),
                "llm_pred_has_negative": bool(llm_record.get("pred_has_negative", False)),
            }
        )

    total = len(cascade_records)
    stats = {
        "num_records": float(total),
        "llm_calls": float(llm_calls),
        "llm_call_rate": float(llm_calls / total if total else 0.0),
        "encoder_negative_pass": float(encoder_negative_pass),
        "encoder_positive_review": float(encoder_positive_review),
    }
    return cascade_records, stats


def write_summary(output_path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 编码器到 LLM 级联风控模拟",
        "",
        "策略：第一阶段编码器判断为无风险时直接放行；判断为有风险时调用 Qwen3-8B QLoRA 复核并抽取主体。",
        "",
        "| 设置 | Accuracy | Recall | F1 | Entity-F1 | LLM 调用率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {accuracy:.4f} | {recall:.4f} | {f1:.4f} | {entity_f1:.4f} | {llm_call_rate:.4f} |".format(
                **row
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_records = load_jsonl(ROOT / args.encoder_predictions)
    llm_records = load_jsonl(ROOT / args.llm_predictions)
    cascade_records, cascade_stats = build_cascade_records(encoder_records, llm_records)

    encoder_metrics = evaluate_records(encoder_records)
    llm_metrics = evaluate_records(llm_records)
    cascade_metrics = evaluate_records(cascade_records)
    cascade_metrics.update(cascade_stats)

    report = {
        "encoder_only": encoder_metrics,
        "llm_all": {**llm_metrics, "llm_call_rate": 1.0, "llm_calls": float(len(llm_records))},
        "finbert2_positive_to_qwen3": cascade_metrics,
    }
    save_json(report, output_dir / "cascade_metrics.json")
    write_jsonl(cascade_records, output_dir / "cascade_predictions.jsonl")

    summary_rows = []
    display_names = {
        "encoder_only": "FinBERT2 only",
        "llm_all": "Qwen3-8B QLoRA all",
        "finbert2_positive_to_qwen3": "FinBERT2 positive -> Qwen3-8B QLoRA",
    }
    for name, metrics in report.items():
        summary_rows.append(
            {
                "name": display_names.get(name, name),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "entity_f1": float(metrics.get("entity_f1", 0.0)),
                "llm_call_rate": float(metrics.get("llm_call_rate", 0.0)),
            }
        )
    write_summary(output_dir / "cascade_summary.md", summary_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved cascade simulation to {output_dir}")


if __name__ == "__main__":
    main()

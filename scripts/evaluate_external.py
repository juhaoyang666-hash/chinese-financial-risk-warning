#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.metrics import evaluate_records, save_json
from risk_nlp.risk_types import infer_risk_type
from risk_nlp.schema import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a lightweight external NSP baseline on FLARE-zh-NSP.")
    parser.add_argument("--input_file", default="data/processed/flare_zh_nsp_test.jsonl")
    parser.add_argument("--output_dir", default="outputs/external/flare_zh_nsp")
    return parser.parse_args()


def predict_has_negative(text: str) -> bool:
    match = infer_risk_type(text)
    return bool(match.keyword)


def main() -> None:
    args = parse_args()
    records = load_jsonl(ROOT / args.input_file)
    predictions = []
    for record in records:
        predictions.append(
            {
                "id": record["id"],
                "gold_has_negative": bool(record["has_negative"]),
                "pred_has_negative": predict_has_negative(record.get("text", "")),
                "gold_entities": [],
                "pred_entities": [],
            }
        )
    metrics = evaluate_records(predictions)
    metrics.update({"evaluator": "keyword_weak_baseline", "model_evaluation": False})
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, output_dir / "predictions.jsonl")
    save_json(metrics, output_dir / "metrics.json")
    (output_dir / "summary.md").write_text(
        "# FLARE-zh-NSP 外部泛化评估\n\n"
        "当前脚本使用风险关键词弱监督 baseline，后续可替换为真实 Qwen3 推理结果。\n\n"
        f"```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.candidates import extract_candidate_entities
from risk_nlp.schema import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ChanceFocus/flare-zh-nsp as an external FINNSP-style test set.")
    parser.add_argument("--dataset", default="ChanceFocus/flare-zh-nsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_file", default="data/processed/flare_zh_nsp_test.jsonl")
    parser.add_argument("--max_records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fallback = False
    try:
        dataset = load_dataset(args.dataset, split=args.split)
        if args.max_records:
            dataset = dataset.select(range(min(args.max_records, len(dataset))))
        rows = list(dataset)
    except Exception as exc:
        fallback = True
        rows = [
            {
                "id": "flare-fallback-0",
                "text": "监管部门提示风险：恒昌汇财、纳觅财务等机构未提交备案材料 实体：恒昌汇财;纳觅财务",
                "answer": "有",
                "gold": 0,
            },
            {
                "id": "flare-fallback-1",
                "text": "京东金融联合加油站推出移动支付服务 实体：京东金融",
                "answer": "无",
                "gold": 1,
            },
            {
                "id": "flare-fallback-2",
                "text": "钱宝网负责人涉嫌集资诈骗被提起公诉 实体：钱宝网",
                "answer": "有",
                "gold": 0,
            },
            {
                "id": "flare-fallback-3",
                "text": "人民联合金融将发布季度财报 实体：人民联合金融",
                "answer": "无",
                "gold": 1,
            },
        ]
        print(f"HF download failed, using fallback demo rows: {type(exc).__name__}: {exc}")
    records = []
    for row in rows:
        text = row.get("text") or row.get("query") or ""
        answer = str(row.get("answer", "")).strip()
        label = 1 if answer == "有" or int(row.get("gold", 1)) == 0 else 0
        records.append(
            {
                "id": row.get("id", f"flare-{len(records)}"),
                "source_dataset": args.dataset,
                "split": args.split,
                "text": text,
                "has_negative": bool(label),
                "label": label,
                "entities": [],
                "candidate_entities": extract_candidate_entities(text),
                "raw_answer": answer,
                "fallback_demo": fallback,
            }
        )
    write_jsonl(records, ROOT / args.output_file)
    print(f"Saved {len(records)} records to {ROOT / args.output_file}")


if __name__ == "__main__":
    main()

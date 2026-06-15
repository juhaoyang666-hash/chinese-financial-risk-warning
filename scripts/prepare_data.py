#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.schema import format_target_json, normalize_finnsp_output, redact_text, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FINNSP data for financial risk NLP experiments.")
    parser.add_argument("--dataset", default="Maciel/FinCUGE-Instruction")
    parser.add_argument("--task", default="FINNSP")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--sample_size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_train", type=int, default=4000)
    parser.add_argument("--expected_eval", type=int, default=500)
    return parser.parse_args()


def label_from_row(row: dict[str, Any]) -> dict[str, Any]:
    if "output" in row:
        return normalize_finnsp_output(row["output"])
    if "answer" in row:
        return normalize_finnsp_output(row["answer"])
    if "gold" in row:
        # ChanceFocus/FLARE uses 0 for "有" and 1 for "无" in the previewed data.
        has_negative = int(row["gold"]) == 0
        return {"has_negative": has_negative, "entities": []}
    raise ValueError(f"Cannot infer label from row keys: {sorted(row.keys())}")


def text_from_row(row: dict[str, Any]) -> str:
    if "input" in row:
        return str(row["input"])
    if "text" in row:
        return str(row["text"])
    if "query" in row:
        return str(row["query"])
    raise ValueError(f"Cannot infer text from row keys: {sorted(row.keys())}")


def instruction_from_row(row: dict[str, Any]) -> str:
    return str(row.get("instruction") or row.get("query") or "")


def process_split(split_name: str, rows: list[dict[str, Any]], task: str, dataset_name: str) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if row.get("task") and row.get("task") != task:
            continue
        text = text_from_row(row)
        instruction = instruction_from_row(row)
        label = label_from_row(row)
        processed.append(
            {
                "id": f"{split_name}-{idx}",
                "source_dataset": dataset_name,
                "split": split_name,
                "task": row.get("task", task),
                "instruction": instruction,
                "text": text,
                "raw_output": row.get("output") or row.get("answer") or row.get("gold"),
                "has_negative": bool(label["has_negative"]),
                "label": int(bool(label["has_negative"])),
                "entities": label["entities"],
                "target_json": format_target_json(bool(label["has_negative"]), label["entities"]),
            }
        )
    return processed


def stats_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(record["label"] for record in records)
    lengths = [len(record["text"]) for record in records]
    return {
        "num_rows": len(records),
        "labels": {"negative_present": labels.get(1, 0), "negative_absent": labels.get(0, 0)},
        "positive_rate": labels.get(1, 0) / len(records) if records else 0,
        "avg_text_chars": sum(lengths) / len(lengths) if lengths else 0,
        "max_text_chars": max(lengths) if lengths else 0,
    }


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    split_map = {"train": "train", "eval": "eval"}
    all_stats: dict[str, Any] = {"dataset": args.dataset, "task": args.task, "splits": {}}

    for split_name, hf_split in split_map.items():
        if hf_split not in dataset:
            continue
        records = process_split(split_name, list(dataset[hf_split]), args.task, args.dataset)
        write_jsonl(records, output_dir / f"finnsp_{split_name}.jsonl")
        all_stats["splits"][split_name] = stats_for(records)

    train_count = all_stats["splits"].get("train", {}).get("num_rows")
    eval_count = all_stats["splits"].get("eval", {}).get("num_rows")
    if args.expected_train and train_count is not None and train_count != args.expected_train:
        raise RuntimeError(f"Expected train={args.expected_train}, got {train_count}")
    if args.expected_eval and eval_count is not None and eval_count != args.expected_eval:
        raise RuntimeError(f"Expected eval={args.expected_eval}, got {eval_count}")

    train_records = []
    train_path = output_dir / "finnsp_train.jsonl"
    if train_path.exists():
        with train_path.open("r", encoding="utf-8") as f:
            train_records = [json.loads(line) for line in f if line.strip()]
    sample_records = []
    for record in train_records[: args.sample_size]:
        item = dict(record)
        item["text"] = redact_text(item["text"])
        item["instruction"] = redact_text(item["instruction"])
        sample_records.append(item)
    write_jsonl(sample_records, output_dir / "finnsp_sample_redacted.jsonl")

    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(all_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


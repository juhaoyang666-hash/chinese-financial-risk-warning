#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "invalid_json_rate",
    "avg_latency_sec",
]

EXCLUDED_MODELS = {
    "qwen2.5-7b-few-shot",
    "qwen2.5-7b-zero-shot",
    "qwen2.5-7b-qlora-eval",
    "qwen3-8b-qlora-4gpu-eval",
    "qwen3.5-9b-zero-shot",
    "qwen3.5-9b-qlora-4gpu-eval",
    "qwen3.5-9b-qlora-4gpu-eval-quick100",
}

MAIN_SWEEP_MODEL = "qwen3-8b-qlora-main-eval"

MODEL_ALIASES = {
    MAIN_SWEEP_MODEL: "qwen3-8b-qlora-main",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect metrics.json files into a paper-ready table.")
    parser.add_argument("--metrics_glob", default="outputs/**/metrics.json")
    parser.add_argument("--output_csv", default="outputs/results_table.csv")
    parser.add_argument("--output_md", default="outputs/results_table.md")
    return parser.parse_args()


def model_name_from_path(path: Path) -> str:
    parent = path.parent
    if parent.name in {"adapter", "checkpoint-final"}:
        parent = parent.parent
    return str(parent).replace("outputs/", "")


def should_skip_model(model_name: str) -> bool:
    if model_name in EXCLUDED_MODELS:
        return True
    if model_name.startswith("tabular_risk/"):
        return True
    if model_name.startswith("lora_sweep/qwen3-8b/") and model_name != MAIN_SWEEP_MODEL:
        return True
    return False


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    def fmt(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(".").glob(args.metrics_glob)):
        model_name = model_name_from_path(path)
        if should_skip_model(model_name):
            continue
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        if metrics.get("model_evaluation") is False:
            continue
        row = {"model": MODEL_ALIASES.get(model_name, model_name)}
        for column in METRIC_COLUMNS[1:]:
            row[column] = metrics.get(column)
        rows.append(row)

    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    markdown = dataframe_to_markdown(df)
    output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved {output_csv} and {output_md}")


if __name__ == "__main__":
    main()

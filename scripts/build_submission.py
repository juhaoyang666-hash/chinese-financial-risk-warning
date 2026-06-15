#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_TITLE = "基于Qwen3-8B_QLoRA的中文金融舆情风险智能预警系统"

INCLUDE_PATTERNS = [
    "README.md",
    "requirements.txt",
    "作业要求.txt",
    "scripts/*.py",
    "app/*.py",
    "src/risk_nlp/*.py",
    "tests/*.py",
    "configs/*.yaml",
    "docs/*.md",
    "docs/*.mmd",
    "paper/*.tex",
    "paper/*.pdf",
    "paper/*.docx",
    "paper/*.md",
    "paper/Makefile",
    "data/processed/dataset_stats.json",
    "data/processed/finnsp_sample_redacted.jsonl",
    "outputs/results_table.csv",
    "outputs/results_table.md",
    "outputs/figures/*.png",
    "outputs/demo_screenshots/*.png",
    "outputs/analysis/**/*.json",
    "outputs/analysis/**/*.md",
    "outputs/risk_profiles/*.json",
    "outputs/external/**/*.json",
    "outputs/external/**/*.md",
    "outputs/robustness/**/*.json",
    "outputs/robustness/**/*.md",
    "outputs/monitoring/*.json",
    "outputs/monitoring/*.md",
    "outputs/tabular_risk/**/*.json",
    "outputs/tabular_risk/**/*.md",
    "outputs/tabular_risk/**/*.csv",
    "outputs/**/metrics.json",
]

BLOCKED_SUFFIXES = {
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".joblib",
    ".parquet",
    ".db",
}

BLOCKED_RELATIVE_PATHS = {
    Path("scripts/collect_lora_sweep_results.py"),
    Path("outputs/qwen2.5-7b-few-shot/metrics.json"),
    Path("outputs/qwen2.5-7b-zero-shot/metrics.json"),
    Path("outputs/qwen2.5-7b-qlora-eval/metrics.json"),
    Path("outputs/qwen3-8b-qlora-4gpu-eval/metrics.json"),
    Path("outputs/qwen3.5-9b-zero-shot/metrics.json"),
    Path("outputs/qwen3.5-9b-qlora-4gpu-eval/metrics.json"),
    Path("outputs/qwen3.5-9b-qlora-4gpu-eval-quick100/metrics.json"),
}

BLOCKED_RELATIVE_PREFIXES = (
    "outputs/analysis/qwen3-8b-qlora-4gpu/",
    "outputs/analysis/qwen3-8b-qlora-s03/",
    "outputs/analysis/cascade_finbert2_qwen3_s03/",
)

MAIN_LORA_SWEEP = "outputs/qwen3-8b-qlora-main-eval/metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean homework submission zip without model weights or full data.")
    parser.add_argument("--student_id", default="学号")
    parser.add_argument("--name", default="姓名")
    parser.add_argument("--kind", default="大作业", choices=["小作业", "大作业"])
    parser.add_argument("--paper_title", default=DEFAULT_PAPER_TITLE)
    parser.add_argument("--output_dir", default="dist")
    return parser.parse_args()


def should_include(path: Path) -> bool:
    if not path.is_file():
        return False
    relative_path = path.relative_to(ROOT)
    relative_text = relative_path.as_posix()
    if relative_path in BLOCKED_RELATIVE_PATHS:
        return False
    if any(relative_text.startswith(prefix) for prefix in BLOCKED_RELATIVE_PREFIXES):
        return False
    if "thinking-1024" in relative_text:
        return False
    if (
        relative_text.startswith("outputs/lora_sweep/qwen3-8b/")
        and relative_text.endswith("/metrics.json")
        and relative_text != MAIN_LORA_SWEEP
    ):
        return False
    if path.suffix in BLOCKED_SUFFIXES:
        return False
    parts = set(path.parts)
    if {"adapter", "checkpoint-final", "checkpoint-4"} & parts:
        return False
    if "__pycache__" in parts:
        return False
    return True


def collect_files() -> list[Path]:
    files: list[Path] = []
    for pattern in INCLUDE_PATTERNS:
        for path in ROOT.glob(pattern):
            if should_include(path) and path not in files:
                files.append(path)
    return sorted(files)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{args.student_id}_{args.name}_{args.kind}.zip"
    files = collect_files()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT))
        paper_prefix = f"{args.student_id}_{args.name}_{args.paper_title}"
        paper_aliases = [
            (ROOT / "paper/main.pdf", f"{paper_prefix}.pdf"),
            (ROOT / "paper/main.docx", f"{paper_prefix}.docx"),
        ]
        for source, alias in paper_aliases:
            if source.exists():
                zf.write(source, alias)
    print(f"Created {zip_path}")
    print("Included files:")
    for path in files:
        print(f"- {path.relative_to(ROOT)}")
    print("Top-level paper files:")
    for suffix in ("pdf", "docx"):
        print(f"- {args.student_id}_{args.name}_{args.paper_title}.{suffix}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.risk_events import record_to_risk_record
from risk_nlp.schema import load_jsonl, write_jsonl


FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
CJK_FONT = font_manager.FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
PALETTE = ["#22A699", "#E58A4D", "#7C8DB5", "#CF7AAD", "#8FBF55", "#E0C340", "#C98B55"]


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text))


def configure_fonts() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback"]
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def apply_cjk_labels(ax) -> None:
    if CJK_FONT is None:
        return
    if contains_cjk(ax.title.get_text()):
        ax.title.set_fontproperties(CJK_FONT)
    if contains_cjk(ax.xaxis.label.get_text()):
        ax.xaxis.label.set_fontproperties(CJK_FONT)
    if contains_cjk(ax.yaxis.label.get_text()):
        ax.yaxis.label.set_fontproperties(CJK_FONT)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        if contains_cjk(tick.get_text()):
            tick.set_fontproperties(CJK_FONT)


def polish_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFD6E0")
    ax.spines["bottom"].set_color("#CFD6E0")
    ax.grid(True, axis="x", color="#E7EBF0", linewidth=0.8)
    ax.grid(False, axis="y")
    ax.tick_params(colors="#253143", labelsize=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weakly labeled FINNSP risk event records.")
    parser.add_argument("--train_file", default="data/processed/finnsp_train.jsonl")
    parser.add_argument("--eval_file", default="data/processed/finnsp_eval.jsonl")
    parser.add_argument("--output_dir", default="outputs/risk_events")
    parser.add_argument("--figure_dir", default="outputs/figures")
    parser.add_argument("--max_records", type=int, default=None)
    return parser.parse_args()


def subset(records: list[dict], limit: int | None) -> list[dict]:
    return records[:limit] if limit else records


def save_distribution(records: list[dict], output_path: Path, title: str) -> None:
    risk_types = [event["risk_type"] for record in records for event in record.get("risk_events", [])]
    counts = Counter(risk_types)
    labels = [label for label, _ in counts.most_common()] or ["无风险事件"]
    values = [counts.get(label, 0) for label in labels] or [0]
    display_labels = [label.replace("/", "、") for label in labels]
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    sns.barplot(x=values, y=display_labels, hue=display_labels, palette=PALETTE[: len(display_labels)], legend=False, ax=ax)
    ax.set_xlabel("样本数")
    ax.set_ylabel("风险类型")
    ax.set_title(title, pad=12, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.12 if values else 1)
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width + max(ax.get_xlim()[1] * 0.01, 1),
            patch.get_y() + patch.get_height() / 2,
            f"{int(width)}",
            va="center",
            ha="left",
            fontsize=9,
            color="#344055",
        )
    polish_axes(ax)
    apply_cjk_labels(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close()


def convert_split(path: Path, output_path: Path, limit: int | None) -> list[dict]:
    records = subset(load_jsonl(path), limit)
    risk_records = [record_to_risk_record(record) for record in records]
    write_jsonl(risk_records, output_path)
    return risk_records


def main() -> None:
    args = parse_args()
    configure_fonts()
    output_dir = ROOT / args.output_dir
    figure_dir = ROOT / args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = convert_split(ROOT / args.train_file, output_dir / "finnsp_train_risk_events.jsonl", args.max_records)
    eval_records = convert_split(ROOT / args.eval_file, output_dir / "finnsp_eval_risk_events.jsonl", args.max_records)
    save_distribution(train_records, figure_dir / "train_risk_type_distribution.png", "训练集风险类型弱标签分布")
    save_distribution(eval_records, figure_dir / "eval_risk_type_distribution.png", "验证集风险类型弱标签分布")
    print(f"Saved risk event records to {output_dir}")


if __name__ == "__main__":
    main()

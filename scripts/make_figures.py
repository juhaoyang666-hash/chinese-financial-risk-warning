#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.schema import load_jsonl


FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
CJK_FONT = font_manager.FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
MODEL_LABELS = {
    "encoder/tfidf_logreg": "TF-IDF + LR",
    "encoder/tfidf_svm": "TF-IDF + SVM",
    "encoder/hfl__chinese-roberta-wwm-ext": "Chinese RoBERTa",
    "encoder/hfl__chinese-macbert-base": "Chinese MacBERT",
    "encoder/valuesimplex-ai-lab__FinBERT2-large": "FinBERT2-large",
    "qwen3-8b-zero-shot": "Qwen3-8B zero-shot",
    "qwen3-8b-few-shot": "Qwen3-8B few-shot",
    "qwen3-8b-qlora-main": "Qwen3-8B QLoRA",
}
MODEL_ORDER = [
    "TF-IDF + LR",
    "TF-IDF + SVM",
    "Chinese RoBERTa",
    "Chinese MacBERT",
    "FinBERT2-large",
    "Qwen3-8B zero-shot",
    "Qwen3-8B few-shot",
    "Qwen3-8B QLoRA",
]
PALETTE = {
    "blue": "#3B6FB6",
    "teal": "#22A699",
    "green": "#5BA86F",
    "orange": "#E58A4D",
    "red": "#CF5C55",
    "purple": "#7C6AC7",
    "gray": "#65758B",
}


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


def polish_axes(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFD6E0")
    ax.spines["bottom"].set_color("#CFD6E0")
    ax.grid(True, axis=grid_axis, color="#E7EBF0", linewidth=0.8)
    other_axis = "x" if grid_axis == "y" else "y"
    ax.grid(False, axis=other_axis)
    ax.tick_params(colors="#253143", labelsize=10)


def annotate_bars(ax, *, horizontal: bool = False, fmt: str = "{:.0f}") -> None:
    for patch in ax.patches:
        if horizontal:
            width = patch.get_width()
            ax.text(
                width + max(ax.get_xlim()[1] * 0.008, 0.01),
                patch.get_y() + patch.get_height() / 2,
                fmt.format(width),
                va="center",
                ha="left",
                fontsize=9,
                color="#344055",
            )
        else:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + max(ax.get_ylim()[1] * 0.015, 0.01),
                fmt.format(height),
                va="bottom",
                ha="center",
                fontsize=9,
                color="#344055",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready figures for FINNSP experiments.")
    parser.add_argument("--train_file", default="data/processed/finnsp_train.jsonl")
    parser.add_argument("--eval_file", default="data/processed/finnsp_eval.jsonl")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--results_csv", default="outputs/results_table.csv")
    parser.add_argument("--output_dir", default="outputs/figures")
    return parser.parse_args()


def save_label_distribution(records: list[dict], output_dir: Path, name: str) -> None:
    labels = ["负面主体" if record["has_negative"] else "无负面主体" for record in records]
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    sns.countplot(
        x=labels,
        hue=labels,
        order=["负面主体", "无负面主体"],
        palette=[PALETTE["red"], PALETTE["teal"]],
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("标签")
    ax.set_ylabel("样本数")
    split_name = "训练集" if name == "train" else "验证集"
    ax.set_title(f"{split_name}标签分布", pad=12, fontsize=13, fontweight="bold")
    polish_axes(ax)
    apply_cjk_labels(ax)
    annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_label_distribution.png", dpi=220)
    plt.close()


def save_length_distribution(records: list[dict], output_dir: Path, name: str) -> None:
    lengths = [len(record["text"]) for record in records]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    sns.histplot(lengths, bins=38, kde=True, color=PALETTE["blue"], edgecolor="white", linewidth=0.5, ax=ax)
    ax.set_xlabel("文本长度（字符数）")
    ax.set_ylabel("样本数")
    split_name = "训练集" if name == "train" else "验证集"
    ax.set_title(f"{split_name}文本长度分布", pad=12, fontsize=13, fontweight="bold")
    polish_axes(ax)
    apply_cjk_labels(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_text_length_distribution.png", dpi=220)
    plt.close()


def save_confusion(prediction_records: list[dict], output_dir: Path) -> None:
    y_true = [int(record["gold_has_negative"]) for record in prediction_records]
    y_pred = [int(record["pred_has_negative"]) for record in prediction_records]
    matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(PALETTE["blue"], as_cmap=True),
        xticklabels=["预测负面", "预测无负面"],
        yticklabels=["真实负面", "真实无负面"],
        cbar=False,
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": 13},
        ax=ax,
    )
    ax.set_title("主模型混淆矩阵", pad=12, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    apply_cjk_labels(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=220)
    plt.close()


def save_model_comparison(results_csv: Path, output_dir: Path) -> None:
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df = df[df["model_label"].isin(MODEL_ORDER)].copy()
    if df.empty:
        return
    df["model_label"] = pd.Categorical(df["model_label"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("model_label")
    plot_df = df.melt(
        id_vars="model_label",
        value_vars=["f1", "entity_f1"],
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map({"f1": "文本分类", "entity_f1": "主体抽取"})

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    sns.barplot(
        data=plot_df,
        x="score",
        y="model_label",
        hue="metric",
        palette=[PALETTE["blue"], PALETTE["orange"]],
        ax=ax,
    )
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("分数")
    ax.set_ylabel("")
    ax.set_title("主任务模型核心指标对比", pad=12, fontsize=13, fontweight="bold")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    polish_axes(ax, grid_axis="x")
    apply_cjk_labels(ax)
    if CJK_FONT is not None:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                if contains_cjk(text.get_text()):
                    text.set_fontproperties(CJK_FONT)
    for container in ax.containers:
        labels = [f"{bar.get_width():.3f}" if bar.get_width() >= 0.01 else "" for bar in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=8, color="#344055")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_dir / "model_comparison_core_metrics.png", dpi=220)
    plt.close()


def draw_box(ax, xy: tuple[float, float], width: float, height: float, title: str, body: str, color: str) -> None:
    rect = Rectangle(
        xy,
        width,
        height,
        linewidth=1.3,
        edgecolor=color,
        facecolor="#FFFFFF",
        zorder=2,
    )
    ax.add_patch(rect)
    x, y = xy
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#1F2A3D",
        fontproperties=CJK_FONT,
    )
    ax.text(
        x + width / 2,
        y + height * 0.35,
        body,
        ha="center",
        va="center",
        fontsize=11,
        color="#42526A",
        linespacing=1.25,
        fontproperties=CJK_FONT,
    )


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#8392A5") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=color,
        zorder=1,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def save_system_framework(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    ax.set_xlim(0.25, 11.25)
    ax.set_ylim(0.35, 5.05)
    ax.axis("off")

    layer_color = "#E8EDF5"
    layer_specs = [(4.25, "数据层"), (2.55, "模型层"), (0.85, "业务层")]
    for y, label in layer_specs:
        ax.add_patch(Rectangle((0.25, y - 0.48), 11.0, 1.3, facecolor="#F7F9FC", edgecolor=layer_color, linewidth=0.9))
        ax.text(0.58, y + 0.15, label, fontsize=14, fontweight="bold", color="#111827", fontproperties=CJK_FONT)

    x0, box_w, box_h = 1.55, 3.12, 0.94
    rows = [
        (
            4.04,
            [
                ("金融文本输入", "新闻、公告、投诉\n社交媒体文本", PALETTE["blue"]),
                ("数据处理", "清洗、脱敏\n标签归一化", PALETTE["teal"]),
                ("实体候选", "原始实体、标签实体\n文本中主体规则", PALETTE["green"]),
            ],
        ),
        (
            2.34,
            [
                ("编码器初筛", "金融领域编码器\n高吞吐二分类", PALETTE["orange"]),
                ("大模型复核", "指令微调大模型\n风险判断与主体抽取", PALETTE["purple"]),
                ("结构化输出", "风险判断\n主体列表与风险事件", PALETTE["blue"]),
            ],
        ),
        (
            0.64,
            [
                ("风控库", "风险事件\n主体画像", PALETTE["teal"]),
                ("审核队列", "高风险、低置信\n主体缺失样本", PALETTE["red"]),
                ("服务与监控", "服务接口与审核看板\n漂移与日志", PALETTE["gray"]),
            ],
        ),
    ]

    for y, row in rows:
        for idx, (title, body, color) in enumerate(row):
            draw_box(ax, (x0 + idx * box_w, y), box_w, box_h, title, body, color)

    for start_y, end_y in [(3.95, 3.34), (2.25, 1.64)]:
        arrow = FancyArrowPatch(
            (6.15, start_y),
            (6.15, end_y),
            arrowstyle="Simple,head_width=22,head_length=18,tail_width=10",
            color="#7C8DA3",
            linewidth=0,
            alpha=0.92,
            zorder=3,
        )
        ax.add_patch(arrow)

    fig.tight_layout()
    fig.savefig(output_dir / "system_framework.png", dpi=240)
    plt.close()


def main() -> None:
    args = parse_args()
    configure_fonts()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_jsonl(ROOT / args.train_file)
    eval_records = load_jsonl(ROOT / args.eval_file)
    save_label_distribution(train_records, output_dir, "train")
    save_label_distribution(eval_records, output_dir, "eval")
    save_length_distribution(train_records, output_dir, "train")
    save_length_distribution(eval_records, output_dir, "eval")
    if args.predictions:
        save_confusion(load_jsonl(ROOT / args.predictions), output_dir)
    save_model_comparison(ROOT / args.results_csv, output_dir)
    save_system_framework(output_dir)
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

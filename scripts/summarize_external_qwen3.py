#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.metrics import binary_metrics, save_json
from risk_nlp.schema import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize real Qwen3 external FLARE-zh-NSP inference results.")
    parser.add_argument("--predictions_file", default="outputs/external/flare_zh_nsp_qwen3_qlora/predictions.jsonl")
    parser.add_argument("--dataset_file", default="data/processed/flare_zh_nsp_test.jsonl")
    parser.add_argument("--keyword_metrics_file", default="outputs/external/flare_zh_nsp/metrics.json")
    parser.add_argument("--output_dir", default="outputs/external/flare_zh_nsp_qwen3_qlora")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    predictions = load_jsonl(ROOT / args.predictions_file)
    dataset = load_jsonl(ROOT / args.dataset_file)
    keyword_metrics = load_json(ROOT / args.keyword_metrics_file)

    y_true = [int(record["gold_has_negative"]) for record in predictions]
    y_pred = [int(record["pred_has_negative"]) for record in predictions]
    counts = Counter(zip(y_true, y_pred))
    metrics = binary_metrics(y_true, y_pred)
    metrics.update(
        {
            "dataset_used": "ChanceFocus/flare-zh-nsp",
            "split": "test",
            "num_samples": len(predictions),
            "num_positive": int(sum(y_true)),
            "num_negative": int(len(y_true) - sum(y_true)),
            "evaluator": "qwen3_8b_qlora_real_inference",
            "model_evaluation": True,
            "evaluation_type": "external_binary_generalization",
            "entity_evaluation": False,
            "entity_metrics_note": "FLARE-zh-NSP only provides binary labels here; predicted entities are retained for qualitative inspection but are not scored.",
            "tp": int(counts[(1, 1)]),
            "fp": int(counts[(0, 1)]),
            "tn": int(counts[(0, 0)]),
            "fn": int(counts[(1, 0)]),
            "invalid_json_rate": float(np.mean([bool(record.get("invalid_json", False)) for record in predictions])),
            "avg_latency_sec": float(np.mean([float(record.get("latency_sec", 0.0)) for record in predictions])),
            "pred_entity_nonempty": int(sum(bool(record.get("pred_entities")) for record in predictions)),
            "enable_thinking": False,
            "max_new_tokens": 128,
            "num_shots": 0,
            "adapter_path": "outputs/qwen3-8b-qlora-main/adapter",
        }
    )
    if keyword_metrics:
        metrics["keyword_baseline_f1"] = float(keyword_metrics.get("f1", 0.0))
        metrics["keyword_baseline_accuracy"] = float(keyword_metrics.get("accuracy", 0.0))
        metrics["f1_improvement_vs_keyword"] = float(metrics["f1"] - metrics["keyword_baseline_f1"])
        metrics["accuracy_improvement_vs_keyword"] = float(metrics["accuracy"] - metrics["keyword_baseline_accuracy"])

    fallback_count = sum(bool(record.get("fallback_demo")) for record in dataset)
    metrics["fallback_demo_records"] = int(fallback_count)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / "metrics.json")
    summary = (
        "# FLARE-zh-NSP Qwen3-8B QLoRA 外部泛化评估\n\n"
        "该实验使用 FINNSP 上微调得到的 Qwen3-8B QLoRA 主模型，对 FLARE-zh-NSP test 集进行真实模型推理。"
        "FLARE-zh-NSP 在当前处理流程中只提供二分类有/无标签，因此本结果只报告二分类外部泛化指标；"
        "模型生成的实体保留在预测文件中用于质检，但不计算 Entity-F1。\n\n"
        f"- 数据集：ChanceFocus/flare-zh-nsp，test，{metrics['num_samples']} 条，正例 {metrics['num_positive']} 条，负例 {metrics['num_negative']} 条。\n"
        f"- 模型：Qwen3-8B QLoRA 主模型，adapter `{metrics['adapter_path']}`。\n"
        f"- 设置：关闭 thinking，max_new_tokens={metrics['max_new_tokens']}，zero-shot 外部推理。\n"
        f"- fallback_demo_records={metrics['fallback_demo_records']}。\n\n"
        "| 方法 | Accuracy | Precision | Recall | F1 | Macro-F1 | invalid JSON rate |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"| 关键词弱监督 baseline | {keyword_metrics.get('accuracy', 0.0):.4f} | {keyword_metrics.get('precision', 0.0):.4f} | {keyword_metrics.get('recall', 0.0):.4f} | {keyword_metrics.get('f1', 0.0):.4f} | {keyword_metrics.get('macro_f1', 0.0):.4f} | - |\n"
        f"| Qwen3-8B QLoRA real inference | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['macro_f1']:.4f} | {metrics['invalid_json_rate']:.4f} |\n\n"
        f"混淆矩阵：TP={metrics['tp']}，FP={metrics['fp']}，TN={metrics['tn']}，FN={metrics['fn']}。"
        f"相对关键词弱监督 baseline，Qwen3-8B QLoRA 的 F1 提升 {metrics.get('f1_improvement_vs_keyword', 0.0):.4f}，"
        f"Accuracy 提升 {metrics.get('accuracy_improvement_vs_keyword', 0.0):.4f}。\n\n"
        f"平均单条推理耗时为 {metrics['avg_latency_sec']:.4f} 秒。"
        f"模型在 {metrics['pred_entity_nonempty']} 条样本中输出了非空实体，但由于外部集缺少实体 gold，该项不作为实体级评估结论。\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

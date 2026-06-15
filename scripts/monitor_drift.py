#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.database import RiskDatabase
from risk_nlp.schema import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create lightweight drift snapshots for risk event records.")
    parser.add_argument("--baseline_file", default="outputs/risk_events/finnsp_train_risk_events.jsonl")
    parser.add_argument("--current_file", default="outputs/risk_events/finnsp_eval_risk_events.jsonl")
    parser.add_argument(
        "--current_reliability_file",
        default="outputs/analysis/qwen3-8b-qlora-main/reliability_metrics.json",
        help="Optional model reliability metrics for the current split.",
    )
    parser.add_argument("--db_path", default="outputs/risk_system/risk_system.db")
    parser.add_argument("--output_dir", default="outputs/monitoring")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(records: list[dict], *, reliability: dict | None = None, source: str) -> dict:
    positive = [record for record in records if record.get("risk_events")]
    risk_levels = Counter(event["risk_level"] for record in records for event in record.get("risk_events", []))
    reliability = reliability or {}
    return {
        "source": source,
        "num_records": len(records),
        "positive_rate": len(positive) / len(records) if records else 0,
        "avg_text_length": sum(len(record.get("text", "")) for record in records) / len(records) if records else 0,
        "avg_entity_count": sum(len(record.get("risk_events", [])) for record in records) / len(records) if records else 0,
        "hallucination_rate": reliability.get("hallucinated_entity_rate"),
        "invalid_json_rate": reliability.get("invalid_json_rate"),
        "risk_level_distribution": dict(risk_levels),
    }


def fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    current_reliability = load_json(ROOT / args.current_reliability_file)
    baseline = snapshot(
        load_jsonl(ROOT / args.baseline_file),
        source="FINNSP train gold weak events",
    )
    current = snapshot(
        load_jsonl(ROOT / args.current_file),
        reliability=current_reliability,
        source="FINNSP eval gold weak events + Qwen3 reliability metrics",
    )
    drift = {
        "baseline": baseline,
        "current": current,
        "positive_rate_delta": current["positive_rate"] - baseline["positive_rate"],
        "avg_text_length_delta": current["avg_text_length"] - baseline["avg_text_length"],
        "avg_entity_count_delta": current["avg_entity_count"] - baseline["avg_entity_count"],
        "retrain_recommendation": abs(current["positive_rate"] - baseline["positive_rate"]) > 0.1,
        "monitoring_scope": (
            "This is an offline train/eval snapshot for the demo risk system, "
            "not a production time-series drift monitor."
        ),
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "drift_report.json").write_text(json.dumps(drift, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = (
        "# 风控文本漂移监控\n\n"
        "| 指标 | baseline | current | delta |\n"
        "|---|---:|---:|---:|\n"
        f"| 样本数 | {fmt(baseline['num_records'])} | {fmt(current['num_records'])} | - |\n"
        f"| 正例率 | {fmt(baseline['positive_rate'])} | {fmt(current['positive_rate'])} | {fmt(drift['positive_rate_delta'])} |\n"
        f"| 平均文本长度 | {fmt(baseline['avg_text_length'])} | {fmt(current['avg_text_length'])} | {fmt(drift['avg_text_length_delta'])} |\n"
        f"| 平均实体数 | {fmt(baseline['avg_entity_count'])} | {fmt(current['avg_entity_count'])} | {fmt(drift['avg_entity_count_delta'])} |\n"
        f"| 幻觉实体率 | {fmt(baseline['hallucination_rate'])} | {fmt(current['hallucination_rate'])} | - |\n"
        f"| invalid JSON rate | {fmt(baseline['invalid_json_rate'])} | {fmt(current['invalid_json_rate'])} | - |\n\n"
        f"- baseline 来源：{baseline['source']}。\n"
        f"- current 来源：{current['source']}。\n"
        f"- 是否建议重训：{str(drift['retrain_recommendation']).lower()}。\n"
        "- 该报告是离线 train/eval 快照监控原型，不等同于生产环境时间序列漂移监控。\n\n"
        "```json\n"
        f"{json.dumps(drift, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )
    (output_dir / "drift_report.md").write_text(
        summary,
        encoding="utf-8",
    )
    db = RiskDatabase(ROOT / args.db_path)
    db.insert_drift_snapshot("baseline", baseline)
    db.insert_drift_snapshot("current", current)
    db.close()
    print(json.dumps(drift, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

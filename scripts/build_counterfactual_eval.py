#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.risk_types import infer_risk_type
from risk_nlp.schema import load_jsonl, write_jsonl


NEGATIVE_CUES = ["诈骗", "非法吸收公众存款", "逾期", "无法提现", "爆雷", "失联", "立案", "处罚"]
DISTRACTOR = "中国人民银行"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate simple FINNSP counterfactual robustness cases.")
    parser.add_argument("--eval_file", default="data/processed/finnsp_eval.jsonl")
    parser.add_argument("--output_dir", default="outputs/robustness/counterfactual")
    parser.add_argument("--num_cases", type=int, default=100)
    return parser.parse_args()


def heuristic_negative(text: str) -> bool:
    return bool(infer_risk_type(text).keyword)


def replace_entity(text: str, entities: list[str]) -> tuple[str, str]:
    if not entities:
        return text, ""
    original = entities[0]
    replacement = "测试金融平台"
    return text.replace(original, replacement), replacement


def remove_negative_cue(text: str) -> str:
    updated = text
    for cue in NEGATIVE_CUES:
        updated = updated.replace(cue, "")
    return updated


def main() -> None:
    args = parse_args()
    source = [record for record in load_jsonl(ROOT / args.eval_file) if record.get("has_negative") and record.get("entities")]
    source = source[: args.num_cases]
    cases = []
    metrics = {
        "entity_switch_total": 0,
        "entity_switch_correct": 0,
        "negative_cue_total": 0,
        "negative_cue_correct": 0,
        "distractor_total": 0,
        "distractor_correct": 0,
        "negation_total": 0,
        "negation_correct": 0,
    }
    for record in source:
        switched_text, replacement = replace_entity(record["text"], record.get("entities", []))
        cases.append({"id": record["id"], "type": "entity_switch", "text": switched_text, "expected_entity": replacement})
        metrics["entity_switch_total"] += 1
        metrics["entity_switch_correct"] += int(bool(replacement) and replacement in switched_text)

        cue_removed = remove_negative_cue(record["text"])
        cases.append({"id": record["id"], "type": "negative_cue_removed", "text": cue_removed, "expected_has_negative": False})
        metrics["negative_cue_total"] += 1
        metrics["negative_cue_correct"] += int(not heuristic_negative(cue_removed))

        distractor_text = record["text"] + f" {DISTRACTOR}表示将持续关注行业风险。"
        cases.append({"id": record["id"], "type": "distractor_inserted", "text": distractor_text, "distractor": DISTRACTOR})
        metrics["distractor_total"] += 1
        metrics["distractor_correct"] += int(DISTRACTOR not in record.get("entities", []))

        negation_text = record["text"] + " 但公司随后发布公告澄清，上述传闻不属实。"
        cases.append({"id": record["id"], "type": "negation_context", "text": negation_text, "expected_review": True})
        metrics["negation_total"] += 1
        metrics["negation_correct"] += int(heuristic_negative(negation_text))

    result = {
        "evaluation_type": "case_generation_and_keyword_baseline",
        "model_evaluation": False,
        "entity_switch_accuracy": metrics["entity_switch_correct"] / metrics["entity_switch_total"] if metrics["entity_switch_total"] else 0,
        "negative_cue_sensitivity": metrics["negative_cue_correct"] / metrics["negative_cue_total"] if metrics["negative_cue_total"] else 0,
        "distractor_resistance": metrics["distractor_correct"] / metrics["distractor_total"] if metrics["distractor_total"] else 0,
        "negation_robustness": metrics["negation_correct"] / metrics["negation_total"] if metrics["negation_total"] else 0,
        **metrics,
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cases, output_dir / "counterfactual_cases.jsonl")
    (output_dir / "counterfactual_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "counterfactual_summary.md").write_text(
        "# 反事实样本构造与规则 baseline 初测\n\n"
        f"生成样本数：{len(cases)}\n\n"
        f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.schema import SYSTEM_PROMPT, build_user_prompt, load_jsonl, parse_model_output, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-8B QLoRA on counterfactual robustness cases.")
    parser.add_argument("--cases_file", default="outputs/robustness/counterfactual/counterfactual_cases.jsonl")
    parser.add_argument("--eval_file", default="data/processed/finnsp_eval.jsonl")
    parser.add_argument("--model_name", default="/data1/yangjuhao/models/Qwen3-8B-ModelScope")
    parser.add_argument("--adapter_path", default="outputs/qwen3-8b-qlora-main/adapter")
    parser.add_argument("--output_dir", default="outputs/robustness/counterfactual_qwen3")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--progress_every", type=int, default=25)
    return parser.parse_args()


def chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{SYSTEM_PROMPT}\n\n用户：{user_prompt}\n\n助手："
    except Exception:
        return f"{SYSTEM_PROMPT}\n\n用户：{user_prompt}\n\n助手："


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(model, str(ROOT / args.adapter_path))
    model.eval()
    return tokenizer, model


def norm_set(items: list[str]) -> set[str]:
    return {str(item).strip().casefold() for item in items if str(item).strip()}


def evaluate_case(case: dict[str, Any], parsed: dict[str, Any], original_by_id: dict[str, dict[str, Any]]) -> bool:
    case_type = case.get("type")
    pred_entities = parsed.get("entities", [])
    pred_entity_set = norm_set(pred_entities)
    pred_has_negative = bool(parsed.get("has_negative"))
    original = original_by_id.get(case.get("id", ""), {})
    original_entities = original.get("entities", [])

    if case_type == "entity_switch":
        expected = str(case.get("expected_entity", "")).strip().casefold()
        replaced_original = str(original_entities[0]).strip().casefold() if original_entities else ""
        return bool(expected) and pred_has_negative and expected in pred_entity_set and (
            not replaced_original or replaced_original not in pred_entity_set
        )
    if case_type == "negative_cue_removed":
        return pred_has_negative == bool(case.get("expected_has_negative", False))
    if case_type == "distractor_inserted":
        distractor = str(case.get("distractor", "")).strip().casefold()
        return pred_has_negative and (not distractor or distractor not in pred_entity_set)
    if case_type == "negation_context":
        return pred_has_negative == bool(case.get("expected_review", True))
    return False


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.unlink(missing_ok=True)

    cases = load_jsonl(ROOT / args.cases_file)
    original_by_id = {record["id"]: record for record in load_jsonl(ROOT / args.eval_file)}
    tokenizer, model = load_model(args)

    predictions: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for idx, case in enumerate(cases, start=1):
        prompt = chat_prompt(tokenizer, build_user_prompt(case["text"]))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        new_tokens = generated[0, inputs.input_ids.shape[-1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed, invalid_json = parse_model_output(raw_output)
        is_correct = evaluate_case(case, parsed, original_by_id)
        record = {
            "id": case.get("id"),
            "type": case.get("type"),
            "text": case.get("text"),
            "expected_entity": case.get("expected_entity"),
            "expected_has_negative": case.get("expected_has_negative"),
            "expected_review": case.get("expected_review"),
            "distractor": case.get("distractor"),
            "pred_has_negative": bool(parsed.get("has_negative")),
            "pred_entities": parsed.get("entities", []),
            "raw_output": raw_output,
            "invalid_json": invalid_json,
            "latency_sec": latency,
            "correct": bool(is_correct),
        }
        predictions.append(record)
        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(cases)):
            elapsed = time.perf_counter() - start_all
            print(
                f"[progress] {idx}/{len(cases)} elapsed={elapsed:.1f}s "
                f"avg={elapsed / idx:.2f}s/sample last_latency={latency:.2f}s "
                f"correct={is_correct} invalid_json={invalid_json}",
                flush=True,
            )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in predictions:
        by_type[str(record.get("type"))].append(record)

    def acc(case_type: str) -> float:
        items = by_type.get(case_type, [])
        return float(np.mean([bool(item["correct"]) for item in items])) if items else 0.0

    metrics = {
        "evaluation_type": "qwen3_qlora_counterfactual_robustness",
        "model_evaluation": True,
        "model": "Qwen3-8B QLoRA",
        "adapter_path": args.adapter_path,
        "num_cases": len(predictions),
        "entity_switch_accuracy": acc("entity_switch"),
        "negative_cue_sensitivity": acc("negative_cue_removed"),
        "distractor_resistance": acc("distractor_inserted"),
        "negation_robustness": acc("negation_context"),
        "overall_accuracy": float(np.mean([bool(record["correct"]) for record in predictions])) if predictions else 0.0,
        "invalid_json_rate": float(np.mean([bool(record["invalid_json"]) for record in predictions])) if predictions else 0.0,
        "avg_latency_sec": float(np.mean([float(record["latency_sec"]) for record in predictions])) if predictions else 0.0,
    }
    for case_type, items in sorted(by_type.items()):
        metrics[f"{case_type}_total"] = len(items)
        metrics[f"{case_type}_correct"] = int(sum(bool(item["correct"]) for item in items))

    (output_dir / "counterfactual_qwen3_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = (
        "# Qwen3-8B QLoRA 反事实鲁棒性实测\n\n"
        "该实验使用 FINNSP 上微调得到的 Qwen3-8B QLoRA 主模型，对构造的反事实样本进行真实推理。"
        "四类扰动分别考察主体替换、负面触发词删除、干扰主体插入和否定/澄清语境。\n\n"
        "| 指标 | 结果 |\n"
        "|---|---:|\n"
        f"| entity_switch_accuracy | {metrics['entity_switch_accuracy']:.4f} |\n"
        f"| negative_cue_sensitivity | {metrics['negative_cue_sensitivity']:.4f} |\n"
        f"| distractor_resistance | {metrics['distractor_resistance']:.4f} |\n"
        f"| negation_robustness | {metrics['negation_robustness']:.4f} |\n"
        f"| overall_accuracy | {metrics['overall_accuracy']:.4f} |\n"
        f"| invalid_json_rate | {metrics['invalid_json_rate']:.4f} |\n"
        f"| avg_latency_sec | {metrics['avg_latency_sec']:.4f} |\n\n"
        f"总样本数：{metrics['num_cases']}。该结果是模型实测，不再是规则 baseline；"
        "但反事实样本由规则自动构造，仍应作为压力测试而非人工标注鲁棒性基准。\n"
    )
    (output_dir / "counterfactual_qwen3_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

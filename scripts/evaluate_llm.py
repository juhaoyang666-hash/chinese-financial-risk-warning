#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.metrics import evaluate_records, save_json
from risk_nlp.schema import SYSTEM_PROMPT, build_user_prompt, format_target_json, load_jsonl, parse_model_output, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot, few-shot, or LoRA-adapted LLM.")
    parser.add_argument("--eval_file", default="data/processed/finnsp_eval.jsonl")
    parser.add_argument("--few_shot_file", default="data/processed/finnsp_train.jsonl")
    parser.add_argument("--model_name", default="/data1/yangjuhao/models/Qwen3-8B-ModelScope")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_dir", default="outputs/llm_eval")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Keep Qwen thinking mode enabled in the chat template instead of forcing JSON-only decoding.",
    )
    return parser.parse_args()


def subset(records: list[dict], limit: int | None) -> list[dict]:
    return records[:limit] if limit else records


def chat_prompt(tokenizer, user_prompt: str, enable_thinking: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{SYSTEM_PROMPT}\n\n用户：{user_prompt}\n\n助手："
    except Exception:
        return f"{SYSTEM_PROMPT}\n\n用户：{user_prompt}\n\n助手："


def build_few_shot_prefix(records: list[dict], num_shots: int) -> str:
    if num_shots <= 0:
        return ""
    shots = []
    used_pos = used_neg = 0
    for record in records:
        if record["has_negative"] and used_pos <= used_neg:
            used_pos += 1
        elif not record["has_negative"] and used_neg <= used_pos:
            used_neg += 1
        else:
            continue
        shots.append(
            "示例：\n"
            f"文本：{record['text']}\n"
            f"输出：{format_target_json(record['has_negative'], record.get('entities', []))}"
        )
        if len(shots) >= num_shots:
            break
    return "\n\n".join(shots) + ("\n\n现在请判断下面的新文本。\n" if shots else "")


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
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter_path))
    model.eval()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_records = subset(load_jsonl(ROOT / args.eval_file), args.max_eval_samples)
    few_shot_records = load_jsonl(ROOT / args.few_shot_file) if args.num_shots else []
    few_shot_prefix = build_few_shot_prefix(few_shot_records, args.num_shots)
    tokenizer, model = load_model(args)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.unlink(missing_ok=True)

    prediction_records = []
    total_start = time.perf_counter()
    for idx, record in enumerate(eval_records, start=1):
        user_prompt = few_shot_prefix + build_user_prompt(record["text"], record.get("instruction"))
        prompt = chat_prompt(tokenizer, user_prompt, args.enable_thinking)
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
        prediction_records.append(
            {
                "id": record["id"],
                "gold_has_negative": bool(record["has_negative"]),
                "pred_has_negative": bool(parsed["has_negative"]),
                "gold_entities": record.get("entities", []),
                "pred_entities": parsed.get("entities", []),
                "raw_output": raw_output,
                "invalid_json": invalid_json,
                "latency_sec": latency,
            }
        )
        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(prediction_records[-1], ensure_ascii=False) + "\n")
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(eval_records)):
            elapsed = time.perf_counter() - total_start
            print(
                f"[progress] {idx}/{len(eval_records)} "
                f"elapsed={elapsed:.1f}s avg={elapsed / idx:.2f}s/sample "
                f"last_latency={latency:.2f}s invalid_json={prediction_records[-1]['invalid_json']}",
                flush=True,
            )

    metrics = evaluate_records(prediction_records)
    metrics["enable_thinking"] = bool(args.enable_thinking)
    metrics["max_new_tokens"] = int(args.max_new_tokens)
    metrics["num_shots"] = int(args.num_shots)
    metrics["load_in_4bit"] = bool(args.load_in_4bit)
    write_jsonl(prediction_records, output_dir / "predictions.jsonl")
    save_json(metrics, output_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()

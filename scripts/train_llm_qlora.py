#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.schema import SYSTEM_PROMPT, build_user_prompt, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for Qwen financial negative entity recognition.")
    parser.add_argument("--train_file", default="data/processed/finnsp_train.jsonl")
    parser.add_argument("--eval_file", default="data/processed/finnsp_eval.jsonl")
    parser.add_argument("--model_name", default="/data1/yangjuhao/models/Qwen3-8B-ModelScope")
    parser.add_argument("--output_dir", default="outputs/qwen3-8b-qlora-main")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def subset(records: list[dict], limit: int | None) -> list[dict]:
    return records[:limit] if limit else records


def chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{SYSTEM_PROMPT}\n\n用户：{user_prompt}\n\n助手："


class SftDataset(TorchDataset):
    def __init__(self, records: list[dict], tokenizer, max_seq_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        user_prompt = build_user_prompt(record["text"], record.get("instruction"))
        prompt = chat_prompt(self.tokenizer, user_prompt)
        answer = record["target_json"] + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        if len(prompt_ids) + len(answer_ids) > self.max_seq_len:
            keep_prompt = max(self.max_seq_len - len(answer_ids), 0)
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt else []
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0)
        labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": str(ROOT / args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": [],
        "seed": args.seed,
        "fp16": True,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "optim": "paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "gradient_checkpointing_kwargs" in signature.parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    device_map = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if torch.cuda.is_available():
            device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = subset(load_jsonl(ROOT / args.train_file), args.max_train_samples)
    eval_records = subset(load_jsonl(ROOT / args.eval_file), args.max_eval_samples)
    train_dataset = SftDataset(train_records, tokenizer, args.max_seq_len)
    eval_dataset = SftDataset(eval_records, tokenizer, args.max_seq_len)

    trainer = Trainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    print(f"Saved LoRA adapter to {output_dir / 'adapter'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from risk_nlp.metrics import binary_metrics, evaluate_records, save_json
from risk_nlp.schema import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF or encoder baselines for FINNSP.")
    parser.add_argument("--train_file", default="data/processed/finnsp_train.jsonl")
    parser.add_argument("--eval_file", default="data/processed/finnsp_eval.jsonl")
    parser.add_argument("--output_dir", default="outputs/encoder")
    parser.add_argument("--backend", choices=["tfidf_logreg", "tfidf_svm", "transformer"], default="tfidf_logreg")
    parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def subset(records: list[dict], limit: int | None) -> list[dict]:
    return records[:limit] if limit else records


def build_prediction_records(eval_records: list[dict], pred_labels: list[int], pred_entities: list[list[str]] | None = None) -> list[dict]:
    pred_entities = pred_entities or [[] for _ in pred_labels]
    output = []
    for record, pred, entities in zip(eval_records, pred_labels, pred_entities):
        output.append(
            {
                "id": record["id"],
                "gold_has_negative": bool(record["has_negative"]),
                "pred_has_negative": bool(pred),
                "gold_entities": record.get("entities", []),
                "pred_entities": entities,
            }
        )
    return output


def train_tfidf(args: argparse.Namespace, train_records: list[dict], eval_records: list[dict]) -> None:
    output_dir = ROOT / args.output_dir / args.backend
    output_dir.mkdir(parents=True, exist_ok=True)
    train_texts = [record["text"] for record in train_records]
    train_labels = [record["label"] for record in train_records]
    eval_texts = [record["text"] for record in eval_records]

    classifier = (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
        if args.backend == "tfidf_logreg"
        else LinearSVC(class_weight="balanced", random_state=args.seed)
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, max_features=80000)),
            ("clf", classifier),
        ]
    )
    model.fit(train_texts, train_labels)
    pred_labels = [int(item) for item in model.predict(eval_texts)]
    records = build_prediction_records(eval_records, pred_labels)
    metrics = evaluate_records(records)
    metrics["classification_report"] = classification_report(
        [record["label"] for record in eval_records],
        pred_labels,
        target_names=["无负面", "有负面"],
        zero_division=0,
    )
    joblib.dump(model, output_dir / "model.joblib")
    write_jsonl(records, output_dir / "predictions.jsonl")
    save_json(metrics, output_dir / "metrics.json")
    print(metrics)


def make_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": 0.01,
        "logging_steps": 20,
        "save_strategy": "epoch",
        "load_best_model_at_end": False,
        "report_to": [],
        "seed": args.seed,
        "fp16": bool(torch.cuda.is_available()),
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def train_transformer(args: argparse.Namespace, train_records: list[dict], eval_records: list[dict]) -> None:
    model_slug = args.model_name.replace("/", "__")
    output_dir = ROOT / args.output_dir / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "无负面", 1: "有负面"},
        label2id={"无负面": 0, "有负面": 1},
        ignore_mismatched_sizes=True,
    )

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        encoded["labels"] = batch["label"]
        return encoded

    train_ds = Dataset.from_list(train_records).map(tokenize, batched=True, remove_columns=list(train_records[0].keys()))
    eval_ds = Dataset.from_list(eval_records).map(tokenize, batched=True, remove_columns=list(eval_records[0].keys()))

    def compute_metrics(pred):
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        return binary_metrics(labels.tolist(), predictions.tolist())

    trainer_kwargs = {
        "model": model,
        "args": make_training_args(args, output_dir),
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    pred_output = trainer.predict(eval_ds)
    pred_labels = np.argmax(pred_output.predictions, axis=-1).astype(int).tolist()
    records = build_prediction_records(eval_records, pred_labels)
    metrics = evaluate_records(records)
    trainer.save_model(str(output_dir / "checkpoint-final"))
    tokenizer.save_pretrained(str(output_dir / "checkpoint-final"))
    write_jsonl(records, output_dir / "predictions.jsonl")
    save_json(metrics, output_dir / "metrics.json")
    print(metrics)


def main() -> None:
    args = parse_args()
    train_records = subset(load_jsonl(ROOT / args.train_file), args.max_train_samples)
    eval_records = subset(load_jsonl(ROOT / args.eval_file), args.max_eval_samples)
    if args.backend.startswith("tfidf"):
        train_tfidf(args, train_records, eval_records)
    else:
        train_transformer(args, train_records, eval_records)


if __name__ == "__main__":
    main()

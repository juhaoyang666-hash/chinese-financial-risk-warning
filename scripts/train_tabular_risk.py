#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "openml_credit_default": {
        "data_id": 42477,
        "name": "default-of-credit-card-clients",
        "target": "y",
        "positive_label": "1",
        "task": "预测信用卡客户下月是否违约",
    },
    "openml_credit_g": {
        "data_id": 31,
        "name": "credit-g",
        "target": "class",
        "positive_label": "bad",
        "task": "预测德国信用数据中的坏账客户",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a real public tabular credit risk model on OpenML.")
    parser.add_argument("--dataset_source", choices=sorted(DATASETS), default="openml_credit_default")
    parser.add_argument("--output_dir", default="outputs/tabular_risk/openml_credit_default")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_true = y_true[order]
    pos = max(y_true.sum(), 1)
    neg = max(len(y_true) - y_true.sum(), 1)
    tpr = np.cumsum(y_true[::-1]) / pos
    fpr = np.cumsum(1 - y_true[::-1]) / neg
    return float(np.max(np.abs(tpr - fpr)))


def best_ks_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0, 1, 501)))
    best_threshold = 0.5
    best_ks = -1.0
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        pos = max((y_true == 1).sum(), 1)
        neg = max((y_true == 0).sum(), 1)
        ks = abs(tp / pos - fp / neg)
        if ks > best_ks:
            best_ks = float(ks)
            best_threshold = float(threshold)
    return best_threshold, best_ks


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    quantiles = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) <= 2:
        return 0.0
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)
    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), 1e-6)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), 1e-6)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def load_openml_frame(dataset_source: str) -> tuple[pd.DataFrame, dict]:
    errors: list[str] = []
    order = [dataset_source]
    if dataset_source != "openml_credit_g":
        order.append("openml_credit_g")

    for source in order:
        meta = DATASETS[source]
        try:
            bundle = fetch_openml(data_id=meta["data_id"], as_frame=True, parser="auto")
            df = bundle.frame.copy()
            if meta["target"] not in df.columns:
                raise ValueError(f"target column {meta['target']} not found")
            meta = dict(meta)
            meta["dataset_used"] = source
            return df, meta
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All real OpenML tabular credit datasets failed: " + " | ".join(errors))


def prepare_xy(df: pd.DataFrame, target: str, positive_label: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    y = (df[target].astype(str) == positive_label).astype(int).to_numpy()
    X = df.drop(columns=[target]).copy()
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = pd.to_numeric(X[column], errors="coerce")
        else:
            codes = X[column].astype("category").cat.codes
            X[column] = codes.replace(-1, np.nan)
    feature_names = list(X.columns)
    return X.fillna(0), y, feature_names


def split_data(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.25, random_state=seed, stratify=y_train_valid
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_model(seed: int):
    if LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(random_state=seed)


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    else:
        importance = np.zeros(len(feature_names), dtype=float)
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )


def main() -> None:
    args = parse_args()
    df, meta = load_openml_frame(args.dataset_source)
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)

    X, y, feature_names = prepare_xy(df, meta["target"], meta["positive_label"])
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, args.seed)

    model = build_model(args.seed)
    model.fit(X_train, y_train)
    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    threshold, valid_ks = best_ks_threshold(y_valid, valid_scores)
    y_pred = (test_scores >= threshold).astype(int)

    label_distribution = {
        "negative": int((y == 0).sum()),
        "positive": int((y == 1).sum()),
        "positive_rate": float(y.mean()),
    }
    metrics = {
        "dataset_source": args.dataset_source,
        "dataset_used": meta["dataset_used"],
        "openml_data_id": meta["data_id"],
        "openml_name": meta["name"],
        "task": meta["task"],
        "model": type(model).__name__,
        "num_samples": int(len(df)),
        "num_features": int(len(feature_names)),
        "label_distribution": label_distribution,
        "auc": float(roc_auc_score(y_test, test_scores)) if len(set(y_test)) > 1 else 0.0,
        "ks": ks_score(y_test, test_scores),
        "valid_ks": valid_ks,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "psi": psi(valid_scores, test_scores),
        "threshold": threshold,
        "split": {"train": int(len(X_train)), "valid": int(len(X_valid)), "test": int(len(X_test))},
    }

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_importance(model, feature_names).to_csv(output_dir / "feature_importance.csv", index=False)
    (output_dir / "summary.md").write_text(
        "# OpenML 真实信用违约 Tabular 风控模型\n\n"
        "使用 OpenML/UCI 真实信用风险数据验证结构化风控分支。\n\n"
        f"```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

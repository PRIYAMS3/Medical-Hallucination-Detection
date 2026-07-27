"""Evaluate a simple retrieval-score hallucination baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="outputs/retrieval_pqa_labeled_top5.csv")
    parser.add_argument("--out", default="outputs/retrieval_baseline_report.json")
    args = parser.parse_args()

    retrieval_df = pd.read_csv(args.retrieval)
    top1 = retrieval_df[retrieval_df["rank"] == 1].copy()
    top1["y_true"] = (top1["answer_label"] == "hallucinated").astype(int)

    train_df, test_df = train_test_split(
        top1,
        test_size=0.2,
        random_state=42,
        stratify=top1["y_true"],
    )

    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in thresholds:
        # Low evidence similarity is treated as more likely hallucinated.
        y_pred = (train_df["evidence_score"] < threshold).astype(int)
        score = f1_score(train_df["y_true"], y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    test_pred = (test_df["evidence_score"] < best_threshold).astype(int)
    report = classification_report(
        test_df["y_true"],
        test_pred,
        target_names=["not_hallucinated", "hallucinated"],
        output_dict=True,
    )
    report["best_threshold"] = best_threshold
    report["train_f1_at_threshold"] = best_f1
    report["mean_top1_score_by_label"] = (
        top1.groupby("answer_label")["evidence_score"].mean().to_dict()
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


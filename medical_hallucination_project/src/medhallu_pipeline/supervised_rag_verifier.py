"""Supervised claim-level verifier using retrieved evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="outputs/retrieval_pqa_labeled_top5.csv")
    parser.add_argument("--out", default="outputs/supervised_rag_verifier_report.json")
    args = parser.parse_args()

    retrieval_df = pd.read_csv(args.retrieval).fillna("")
    top1 = retrieval_df[retrieval_df["rank"] == 1].copy()
    top1["source_row_id"] = top1["claim_id"].str.extract(r"pqa_labeled_(\d+)_")[0]
    top1["label"] = top1["answer_label"]
    if "question" not in top1.columns:
        top1["question"] = ""
    top1["text"] = (
        "Claim: "
        + top1["claim_text"].astype(str)
        + "\nEvidence: "
        + top1["evidence_text"].astype(str)
        + "\nQuestion: "
        + top1["question"].astype(str)
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(
        splitter.split(top1["text"], top1["label"], groups=top1["source_row_id"])
    )
    full_train_df = top1.iloc[train_idx].copy()
    test_df = top1.iloc[test_idx].copy()

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    inner_train_idx, val_idx = next(
        val_splitter.split(
            full_train_df["text"],
            full_train_df["label"],
            groups=full_train_df["source_row_id"],
        )
    )
    train_df = full_train_df.iloc[inner_train_idx].copy()
    val_df = full_train_df.iloc[val_idx].copy()

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=100_000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(train_df["text"], train_df["label"])

    class_labels = list(model.classes_)
    hallucinated_index = class_labels.index("hallucinated")
    val_prob = model.predict_proba(val_df["text"])[:, hallucinated_index]
    val_true = (val_df["label"] == "hallucinated").astype(int)

    best_threshold = 0.5
    best_val_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        val_pred = (val_prob >= threshold).astype(int)
        score = f1_score(val_true, val_pred)
        if score > best_val_f1:
            best_val_f1 = score
            best_threshold = float(threshold)

    # Refit on the full training fold after choosing the threshold.
    model.fit(full_train_df["text"], full_train_df["label"])
    test_prob = model.predict_proba(test_df["text"])[:, hallucinated_index]
    predictions = np.where(test_prob >= best_threshold, "hallucinated", "not_hallucinated")

    report = classification_report(
        test_df["label"],
        predictions,
        labels=["not_hallucinated", "hallucinated"],
        output_dict=True,
        zero_division=0,
    )

    hard_df = test_df[test_df["difficulty"] == "hard"].copy()
    if not hard_df.empty:
        hard_prob = model.predict_proba(hard_df["text"])[:, hallucinated_index]
        hard_predictions = np.where(
            hard_prob >= best_threshold,
            "hallucinated",
            "not_hallucinated",
        )
        report["hard_split"] = classification_report(
        hard_df["label"],
        hard_predictions,
            labels=["not_hallucinated", "hallucinated"],
            output_dict=True,
            zero_division=0,
        )

    report["train_claims"] = int(len(train_df))
    report["validation_claims"] = int(len(val_df))
    report["full_train_claims"] = int(len(full_train_df))
    report["test_claims"] = int(len(test_df))
    report["train_groups"] = int(full_train_df["source_row_id"].nunique())
    report["test_groups"] = int(test_df["source_row_id"].nunique())
    report["best_hallucinated_threshold"] = best_threshold
    report["validation_hallucinated_f1_at_threshold"] = best_val_f1

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

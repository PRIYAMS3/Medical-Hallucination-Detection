"""Fine-grained hallucination type classification baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", default="outputs/claim_level_pqa_labeled.csv")
    parser.add_argument("--out", default="outputs/type_classifier_report.json")
    args = parser.parse_args()

    claim_df = pd.read_csv(args.claims).fillna("")
    hallucinated_df = claim_df[
        (claim_df["answer_label"] == "hallucinated")
        & (claim_df["hallucination_category"].astype(str).str.len() > 0)
    ].copy()

    hallucinated_df["text"] = (
        "Question: "
        + hallucinated_df["question"].astype(str)
        + "\nClaim: "
        + hallucinated_df["claim_text"].astype(str)
        + "\nEvidence: "
        + hallucinated_df["knowledge"].astype(str)
    )

    x_train, x_test, y_train, y_test = train_test_split(
        hallucinated_df["text"],
        hallucinated_df["hallucination_category"],
        test_size=0.2,
        random_state=42,
        stratify=hallucinated_df["hallucination_category"],
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=75_000, ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    labels = sorted(hallucinated_df["hallucination_category"].unique())
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report["labels"] = labels
    report["confusion_matrix"] = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    ).tolist()
    report["train_size"] = len(x_train)
    report["test_size"] = len(x_test)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


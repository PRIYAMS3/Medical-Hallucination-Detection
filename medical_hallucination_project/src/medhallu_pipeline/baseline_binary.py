"""Simple binary baseline for normalized MedHallu records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from medhallu_pipeline.data import load_medhallu


def build_binary_examples(df):
    """Create a binary dataset from MedHallu's paired correct/hallucinated answers."""

    examples = []
    labels = []

    for _, row in df.iterrows():
        question = str(row.get("Question", ""))
        raw_knowledge = row.get("Knowledge", [])
        if isinstance(raw_knowledge, np.ndarray):
            raw_knowledge = raw_knowledge.tolist()
        if isinstance(raw_knowledge, str):
            knowledge = raw_knowledge
        else:
            knowledge = " ".join(str(item) for item in raw_knowledge)
        ground_truth = str(row.get("Ground Truth", ""))
        hallucinated = str(row.get("Hallucinated Answer", ""))

        if ground_truth.strip():
            examples.append(
                f"Question: {question}\nKnowledge: {knowledge}\nAnswer: {ground_truth}"
            )
            labels.append("not_hallucinated")

        if hallucinated.strip():
            examples.append(
                f"Question: {question}\nKnowledge: {knowledge}\nAnswer: {hallucinated}"
            )
            labels.append("hallucinated")

    return examples, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="pqa_labeled")
    parser.add_argument("--out", default="outputs/binary_baseline_report.json")
    args = parser.parse_args()

    df = load_medhallu(split=args.split)
    examples, labels = build_binary_examples(df)

    if len(set(labels)) < 2:
        raise RuntimeError(
            "Could not find at least two label classes. Run inspect_dataset.py and update normalization."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        examples,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    report = classification_report(y_test, predictions, output_dict=True)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

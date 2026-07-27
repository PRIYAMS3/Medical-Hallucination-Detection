"""Collect experiment results into one comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(report: dict, label: str, field: str) -> float | None:
    try:
        return float(report[label][field])
    except KeyError:
        return None


def row(name: str, report: dict) -> dict:
    hard = report.get("hard_split", {})
    return {
        "system": name,
        "accuracy": report.get("accuracy"),
        "macro_f1": metric(report, "macro avg", "f1-score"),
        "hallucinated_f1": metric(report, "hallucinated", "f1-score"),
        "not_hallucinated_f1": metric(report, "not_hallucinated", "f1-score"),
        "hard_accuracy": hard.get("accuracy"),
        "hard_macro_f1": metric(hard, "macro avg", "f1-score") if hard else None,
        "hard_hallucinated_f1": metric(hard, "hallucinated", "f1-score")
        if hard
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/results_comparison.csv")
    args = parser.parse_args()

    candidates = [
        ("Binary TF-IDF", "outputs/binary_baseline_pqa_labeled_report.json"),
        ("Retrieval score", "outputs/retrieval_baseline_report.json"),
        ("Heuristic RAG+NLI", "outputs/heuristic_nli_eval_top1_report.json"),
        ("Supervised RAG verifier", "outputs/supervised_rag_verifier_report.json"),
        (
            "Supervised RAG verifier tuned",
            "outputs/supervised_rag_verifier_tuned_report.json",
        ),
        (
            "Transformer NLI 500-sample",
            "outputs/transformer_nli_eval_0000_0499_report.json",
        ),
    ]

    rows = []
    for name, path in candidates:
        if Path(path).exists():
            rows.append(row(name, load_json(path)))

    df = pd.DataFrame(rows)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))
    print(output_path)


if __name__ == "__main__":
    main()


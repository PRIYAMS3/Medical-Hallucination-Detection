"""Evaluate claim-level NLI outputs as hallucination predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli", default="outputs/heuristic_nli_pqa_labeled.csv")
    parser.add_argument("--out", default="outputs/heuristic_nli_eval_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.nli).fillna("")
    df["y_true"] = (df["answer_label"] == "hallucinated").astype(int)
    df["y_pred"] = df["nli_label"].isin(["contradicted", "unverifiable"]).astype(int)

    report = classification_report(
        df["y_true"],
        df["y_pred"],
        target_names=["not_hallucinated", "hallucinated"],
        output_dict=True,
    )
    report["nli_label_counts"] = df["nli_label"].value_counts().to_dict()
    report["severity_level_counts"] = df["severity_level"].value_counts().to_dict()

    hard_df = df[df["difficulty"] == "hard"]
    if not hard_df.empty:
        hard_report = classification_report(
            hard_df["y_true"],
            hard_df["y_pred"],
            target_names=["not_hallucinated", "hallucinated"],
            output_dict=True,
        )
        report["hard_split"] = hard_report

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


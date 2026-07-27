"""Build claim-level records from MedHallu answer pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from medhallu_pipeline.claims import split_into_claims
from medhallu_pipeline.data import load_medhallu


def normalize_knowledge(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return " ".join(str(item) for item in value)


def build_claim_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for index, row in df.iterrows():
        question = str(row.get("Question", ""))
        knowledge = normalize_knowledge(row.get("Knowledge", ""))
        difficulty = str(row.get("Difficulty Level", ""))
        category = str(row.get("Category of Hallucination", ""))

        answer_pairs = [
            ("ground_truth", str(row.get("Ground Truth", "")), "not_hallucinated", ""),
            ("hallucinated", str(row.get("Hallucinated Answer", "")), "hallucinated", category),
        ]

        for answer_type, answer, answer_label, claim_category in answer_pairs:
            sample_id = f"pqa_labeled_{index}_{answer_type}"
            claims = split_into_claims(answer, sample_id=sample_id)
            if not claims and answer.strip():
                claims = split_into_claims(f"{answer.strip()}.", sample_id=sample_id)

            for claim in claims:
                rows.append(
                    {
                        "source_row_id": str(index),
                        "answer_type": answer_type,
                        "answer_label": answer_label,
                        "claim_id": claim.claim_id,
                        "claim_text": claim.text,
                        "question": question,
                        "knowledge": knowledge,
                        "difficulty": difficulty,
                        "hallucination_category": claim_category,
                    }
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="pqa_labeled")
    parser.add_argument("--out", default="outputs/claim_level_pqa_labeled.csv")
    args = parser.parse_args()

    df = load_medhallu(split=args.split)
    claim_df = pd.DataFrame(build_claim_rows(df))

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    claim_df.to_csv(output_path, index=False, encoding="utf-8")

    summary = {
        "rows": len(claim_df),
        "answer_label_counts": claim_df["answer_label"].value_counts().to_dict(),
        "difficulty_counts": claim_df["difficulty"].value_counts().to_dict(),
        "hallucination_category_counts": claim_df[
            claim_df["answer_label"] == "hallucinated"
        ]["hallucination_category"].value_counts().to_dict(),
    }
    print(summary)
    print(output_path)


if __name__ == "__main__":
    main()


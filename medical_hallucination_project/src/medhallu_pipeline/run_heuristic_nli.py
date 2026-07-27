"""Run heuristic NLI verification over retrieved claim-evidence pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from medhallu_pipeline.heuristic_nli import verify_claim
from medhallu_pipeline.severity import infer_risk_tier, score_severity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="outputs/retrieval_pqa_labeled_top5.csv")
    parser.add_argument("--out", default="outputs/heuristic_nli_pqa_labeled.csv")
    parser.add_argument("--max-rank", type=int, default=1)
    args = parser.parse_args()

    retrieval_df = pd.read_csv(args.retrieval).fillna("")
    retrieval_df = retrieval_df[retrieval_df["rank"] <= args.max_rank].copy()

    rows = []
    for _, row in tqdm(retrieval_df.iterrows(), total=len(retrieval_df)):
        prediction = verify_claim(str(row["evidence_text"]), str(row["claim_text"]))
        risk_tier = infer_risk_tier(str(row["claim_text"]))
        severity = score_severity(
            hallucination_type=str(row["hallucination_category"]),
            contradiction_strength=prediction.contradiction_score,
            clinical_risk_tier=risk_tier,
        )

        rows.append(
            {
                **row.to_dict(),
                "nli_label": prediction.label,
                "support_score": prediction.support_score,
                "contradiction_score": prediction.contradiction_score,
                "unverifiable_score": prediction.unverifiable_score,
                "token_overlap": prediction.token_overlap,
                "negation_mismatch": prediction.negation_mismatch,
                "clinical_risk_tier": risk_tier,
                "severity_score": severity.score,
                "severity_level": severity.level,
            }
        )

    output_df = pd.DataFrame(rows)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8")

    print(
        {
            "rows": len(output_df),
            "nli_label_counts": output_df["nli_label"].value_counts().to_dict(),
            "severity_counts": output_df["severity_level"].value_counts().to_dict(),
        }
    )
    print(output_path)


if __name__ == "__main__":
    main()


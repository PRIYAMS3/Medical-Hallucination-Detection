"""Run claim-level evidence retrieval over MedHallu knowledge passages."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from medhallu_pipeline.retrieval import TfidfEvidenceRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", default="outputs/claim_level_pqa_labeled.csv")
    parser.add_argument("--out", default="outputs/retrieval_pqa_labeled_top5.csv")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    claim_df = pd.read_csv(args.claims).fillna("")
    documents = claim_df["knowledge"].drop_duplicates().tolist()
    doc_ids = [f"knowledge_{i}" for i in range(len(documents))]

    retriever = TfidfEvidenceRetriever()
    retriever.fit(documents, doc_ids=doc_ids)

    rows = []
    for _, row in tqdm(claim_df.iterrows(), total=len(claim_df)):
        hits = retriever.search(str(row["claim_text"]), top_k=args.top_k)
        for rank, hit in enumerate(hits, start=1):
            rows.append(
                {
                    "claim_id": row["claim_id"],
                    "answer_label": row["answer_label"],
                    "difficulty": row["difficulty"],
                    "hallucination_category": row["hallucination_category"],
                    "claim_text": row["claim_text"],
                    "rank": rank,
                    "evidence_doc_id": hit.doc_id,
                    "evidence_score": hit.score,
                    "evidence_text": hit.text,
                }
            )

    retrieval_df = pd.DataFrame(rows)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_df.to_csv(output_path, index=False, encoding="utf-8")

    top1 = retrieval_df[retrieval_df["rank"] == 1]
    print(
        {
            "claims": int(claim_df["claim_id"].nunique()),
            "retrieval_rows": len(retrieval_df),
            "mean_top1_score": float(top1["evidence_score"].mean()),
            "median_top1_score": float(top1["evidence_score"].median()),
        }
    )
    print(output_path)


if __name__ == "__main__":
    main()


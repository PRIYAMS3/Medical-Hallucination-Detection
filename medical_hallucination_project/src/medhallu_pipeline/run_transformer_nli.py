"""Run transformer NLI verification over retrieved claim-evidence pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from medhallu_pipeline.severity import infer_risk_tier, score_severity


def normalize_label(label: str) -> str:
    label = label.lower()
    if "contradiction" in label:
        return "contradicted"
    if "entail" in label:
        return "supported"
    if "neutral" in label:
        return "unverifiable"
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="outputs/retrieval_pqa_labeled_top5.csv")
    parser.add_argument("--out", default="outputs/transformer_nli_pqa_labeled_top1.csv")
    parser.add_argument("--model", default="typeform/distilbert-base-uncased-mnli")
    parser.add_argument("--max-rank", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    retrieval_df = pd.read_csv(args.retrieval).fillna("")
    retrieval_df = retrieval_df[retrieval_df["rank"] <= args.max_rank].copy()
    if args.start > 0:
        retrieval_df = retrieval_df.iloc[args.start:].copy()
    if args.limit > 0:
        retrieval_df = retrieval_df.head(args.limit).copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    id_to_label = model.config.id2label
    rows = []

    for start in tqdm(range(0, len(retrieval_df), args.batch_size)):
        batch = retrieval_df.iloc[start : start + args.batch_size]
        encoded = tokenizer(
            batch["evidence_text"].astype(str).tolist(),
            batch["claim_text"].astype(str).tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        for row, probs in zip(batch.to_dict("records"), probabilities):
            label_scores = {
                normalize_label(id_to_label[i]): float(prob)
                for i, prob in enumerate(probs)
            }
            supported = label_scores.get("supported", 0.0)
            contradicted = label_scores.get("contradicted", 0.0)
            unverifiable = label_scores.get("unverifiable", 0.0)
            nli_label = max(
                [
                    ("supported", supported),
                    ("contradicted", contradicted),
                    ("unverifiable", unverifiable),
                ],
                key=lambda item: item[1],
            )[0]

            risk_tier = infer_risk_tier(str(row["claim_text"]))
            severity = score_severity(
                hallucination_type=str(row["hallucination_category"]),
                contradiction_strength=contradicted,
                clinical_risk_tier=risk_tier,
            )

            rows.append(
                {
                    **row,
                    "nli_label": nli_label,
                    "support_score": supported,
                    "contradiction_score": contradicted,
                    "unverifiable_score": unverifiable,
                    "clinical_risk_tier": risk_tier,
                    "severity_score": severity.score,
                    "severity_level": severity.level,
                    "nli_model": args.model,
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
            "model": args.model,
        }
    )
    print(output_path)


if __name__ == "__main__":
    main()

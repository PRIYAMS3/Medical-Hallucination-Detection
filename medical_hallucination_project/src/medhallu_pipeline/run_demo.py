"""Small offline demo for the prototype pipeline."""

from __future__ import annotations

from medhallu_pipeline.claims import split_into_claims
from medhallu_pipeline.retrieval import TfidfEvidenceRetriever
from medhallu_pipeline.severity import infer_risk_tier, score_severity


def main() -> None:
    evidence_docs = [
        "Metformin is commonly used to improve glycemic control in type 2 diabetes.",
        "Metformin is not a cure for type 1 diabetes and insulin remains required for type 1 diabetes management.",
        "Metformin use may be contraindicated or require caution in severe renal impairment.",
    ]
    answer = (
        "Metformin cures type 1 diabetes. "
        "Metformin should be given to all patients with kidney failure."
    )

    retriever = TfidfEvidenceRetriever()
    retriever.fit(evidence_docs)

    for claim in split_into_claims(answer, sample_id="demo"):
        hits = retriever.search(claim.text, top_k=2)
        risk = infer_risk_tier(claim.text)
        severity = score_severity(
            hallucination_type="drug/treatment entity error",
            contradiction_strength=0.85,
            clinical_risk_tier=risk,
        )
        print(f"Claim: {claim.text}")
        print(f"Top evidence: {hits[0].text if hits else 'none'}")
        print(f"Risk tier: {risk}")
        print(f"Severity: {severity.level} ({severity.score:.2f})")
        print()


if __name__ == "__main__":
    main()


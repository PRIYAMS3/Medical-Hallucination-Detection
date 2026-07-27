"""Fast heuristic NLI-style verifier for claim-evidence pairs.

This is not the final NLI model. It is a transparent baseline that lets us
exercise the RAG + verification pipeline before downloading a transformer
cross-encoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "without",
    "lack",
    "lacks",
    "lacking",
    "absence",
    "absent",
    "contraindicated",
    "contraindication",
    "failed",
    "fails",
    "cannot",
    "did not",
    "does not",
    "do not",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(frozen=True)
class HeuristicNliPrediction:
    label: str
    support_score: float
    contradiction_score: float
    unverifiable_score: float
    token_overlap: float
    negation_mismatch: bool


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in NEGATION_TERMS)


def verify_claim(evidence: str, claim: str) -> HeuristicNliPrediction:
    claim_tokens = set(tokenize(claim))
    evidence_tokens = set(tokenize(evidence))

    if not claim_tokens:
        return HeuristicNliPrediction("unverifiable", 0.0, 0.0, 1.0, 0.0, False)

    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    negation_mismatch = has_negation(evidence) != has_negation(claim)

    contradiction_score = 0.0
    support_score = overlap

    if overlap >= 0.35 and negation_mismatch:
        contradiction_score = min(1.0, 0.45 + overlap)
        support_score = max(0.0, overlap - 0.35)

    unverifiable_score = max(0.0, 1.0 - max(support_score, contradiction_score))

    if contradiction_score >= 0.60:
        label = "contradicted"
    elif support_score >= 0.45:
        label = "supported"
    else:
        label = "unverifiable"

    return HeuristicNliPrediction(
        label=label,
        support_score=float(support_score),
        contradiction_score=float(contradiction_score),
        unverifiable_score=float(unverifiable_score),
        token_overlap=float(overlap),
        negation_mismatch=negation_mismatch,
    )


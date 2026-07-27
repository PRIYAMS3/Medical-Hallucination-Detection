"""Claim decomposition utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """A verifiable atomic claim extracted from a generated answer."""

    sample_id: str
    claim_id: str
    text: str


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(
    r"\s+(?:and|but|while|whereas|although|however|therefore|because)\s+",
    flags=re.IGNORECASE,
)


def split_into_claims(answer: str, sample_id: str = "sample") -> list[Claim]:
    """Split a medical answer into simple sentence/clause-level claims.

    This is intentionally conservative for the first prototype. It avoids
    introducing an LLM dependency before we have the dataset and baselines
    working.
    """

    if not answer or not answer.strip():
        return []

    claims: list[Claim] = []
    sentences = _SENTENCE_SPLIT_RE.split(answer.strip())

    for sentence in sentences:
        sentence = sentence.strip(" \n\t;:")
        if not sentence:
            continue

        clauses = _CLAUSE_SPLIT_RE.split(sentence)
        for clause in clauses:
            text = clause.strip(" \n\t;:")
            if len(text.split()) < 3:
                continue
            claim_id = f"{sample_id}_c{len(claims) + 1}"
            claims.append(Claim(sample_id=sample_id, claim_id=claim_id, text=text))

    return claims


def aggregate_claim_labels(labels: list[str]) -> str:
    """Aggregate claim-level labels into an answer-level hallucination label."""

    normalized = {label.lower() for label in labels}
    if "contradicted" in normalized:
        return "hallucinated"
    if "unverifiable" in normalized:
        return "possibly_hallucinated"
    if normalized == {"supported"}:
        return "supported"
    return "unknown"


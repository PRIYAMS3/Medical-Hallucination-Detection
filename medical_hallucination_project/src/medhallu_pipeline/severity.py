"""Clinical severity scoring."""

from __future__ import annotations

from dataclasses import dataclass


TYPE_WEIGHTS = {
    "numeric/dosage error": 1.00,
    "dosage error": 1.00,
    "drug/treatment entity error": 0.95,
    "entity error": 0.85,
    "diagnosis error": 0.90,
    "causal/mechanistic error": 0.80,
    "relation error": 0.75,
    "relational error": 0.75,
    "contextual error": 0.60,
    "unsupported factual error": 0.50,
    "factual error": 0.50,
    "misinterpretation of #question#": 0.70,
    "incomplete information": 0.55,
    "mechanism and pathway misattribution": 0.85,
    "methodological and evidence fabrication": 0.90,
}

RISK_WEIGHTS = {
    "high": 1.00,
    "medium": 0.70,
    "low": 0.40,
}


@dataclass(frozen=True)
class SeverityResult:
    """Severity score and category."""

    score: float
    level: str


def score_severity(
    hallucination_type: str | None,
    contradiction_strength: float,
    clinical_risk_tier: str = "medium",
) -> SeverityResult:
    """Compute severity from type, contradiction strength, and risk tier."""

    type_weight = TYPE_WEIGHTS.get((hallucination_type or "").lower(), 0.50)
    risk_weight = RISK_WEIGHTS.get(clinical_risk_tier.lower(), 0.70)
    score = max(0.0, min(1.0, type_weight * contradiction_strength * risk_weight))

    if score <= 0.30:
        level = "low"
    elif score <= 0.60:
        level = "moderate"
    else:
        level = "high"

    return SeverityResult(score=score, level=level)


def infer_risk_tier(claim_text: str) -> str:
    """Simple keyword risk tier heuristic for the first prototype."""

    high_terms = [
        "dose",
        "dosage",
        "mg",
        "contraindication",
        "contraindicated",
        "pregnant",
        "kidney failure",
        "renal failure",
        "emergency",
        "mortality",
        "cure",
        "cures",
        "treat",
        "treatment",
        "diagnosis",
        "diabetes",
    ]
    medium_terms = [
        "risk",
        "prognosis",
        "screening",
        "test",
        "biomarker",
        "symptom",
    ]

    text = claim_text.lower()
    if any(term in text for term in high_terms):
        return "high"
    if any(term in text for term in medium_terms):
        return "medium"
    return "low"

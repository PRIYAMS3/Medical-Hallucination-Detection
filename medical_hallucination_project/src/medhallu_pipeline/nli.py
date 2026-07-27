"""NLI verification wrapper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NliResult:
    """NLI prediction for a claim-evidence pair."""

    label: str
    entailment: float
    contradiction: float
    neutral: float


class ZeroShotNliVerifier:
    """Thin Hugging Face wrapper for NLI verification."""

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The `transformers` package is required. Install requirements.txt first."
            ) from exc

        self._classifier = pipeline("text-classification", model=model_name, top_k=None)

    def verify(self, evidence: str, claim: str) -> NliResult:
        text = f"{evidence}</s></s>{claim}"
        raw = self._classifier(text)[0]
        scores = {item["label"].lower(): float(item["score"]) for item in raw}

        entailment = _score_for(scores, ["entailment", "entails", "label_0"])
        contradiction = _score_for(scores, ["contradiction", "contradicts", "label_2"])
        neutral = _score_for(scores, ["neutral", "neither", "label_1"])

        best = max(
            [
                ("supported", entailment),
                ("contradicted", contradiction),
                ("unverifiable", neutral),
            ],
            key=lambda item: item[1],
        )[0]

        return NliResult(
            label=best,
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
        )


def _score_for(scores: dict[str, float], keys: list[str]) -> float:
    for key in keys:
        if key in scores:
            return scores[key]
    return 0.0


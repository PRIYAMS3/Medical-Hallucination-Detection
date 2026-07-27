from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from phishing_hybrid.config import RuleConfig


@dataclass
class HybridDecision:
    prediction: str
    confidence: float
    decision_type: str
    model_label: int
    final_label: int
    rules_triggered: list[str]


def _check_rule(value: float, op: str, expected: float) -> bool:
    if op == "==":
        return float(value) == float(expected)
    if op == "!=":
        return float(value) != float(expected)
    if op == ">":
        return float(value) > float(expected)
    if op == ">=":
        return float(value) >= float(expected)
    if op == "<":
        return float(value) < float(expected)
    if op == "<=":
        return float(value) <= float(expected)
    raise ValueError(f"Unsupported operator in rule: {op}")


class HybridPhishingSystem:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: list[str],
        rules: list[RuleConfig],
        threshold: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.feature_to_idx = {name: i for i, name in enumerate(feature_names)}
        self.rules = rules
        self.threshold = threshold
        self.device = device

    def _predict_with_confidence(self, x: np.ndarray) -> tuple[int, float]:
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
        return int(pred.item()), float(confidence.item())

    def _apply_rules(self, x: np.ndarray) -> list[str]:
        triggered: list[str] = []
        for rule in self.rules:
            if rule.feature not in self.feature_to_idx:
                continue
            idx = self.feature_to_idx[rule.feature]
            if _check_rule(x[idx], rule.op, rule.value):
                triggered.append(rule.reason)
        return triggered

    def predict(self, x: np.ndarray) -> HybridDecision:
        model_label, confidence = self._predict_with_confidence(x)
        triggered = self._apply_rules(x)

        if confidence < self.threshold and triggered:
            final_label = 1
            decision_type = "rule_override"
        else:
            final_label = model_label
            decision_type = "model_only"

        prediction = "Phishing" if final_label == 1 else "Legitimate"
        return HybridDecision(
            prediction=prediction,
            confidence=round(confidence, 4),
            decision_type=decision_type,
            model_label=model_label,
            final_label=final_label,
            rules_triggered=triggered,
        )

    def predict_as_dict(self, x: np.ndarray) -> dict[str, Any]:
        decision = self.predict(x)
        return {
            "Prediction": decision.prediction,
            "Confidence": decision.confidence,
            "Decision Type": decision.decision_type,
            "Model Label": decision.model_label,
            "Final Label": decision.final_label,
            "Rules Triggered": decision.rules_triggered,
        }

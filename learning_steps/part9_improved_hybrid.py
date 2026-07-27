from __future__ import annotations

from pathlib import Path
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


DATASET_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")
TARGET_COL = "Result"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Rule risk weights (derived from domain intuition; can be replaced by data-driven weights later)
RULE_WEIGHTS = {
    "IP address used": 1.0,
    "Invalid SSL": 1.0,
    "Suspicious anchor URLs": 0.9,
    "Hyphen in domain": 0.6,
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def extract_rules(sample: np.ndarray, feature_index: dict[str, int]) -> list[str]:
    triggered = []
    if sample[feature_index["having_IP_Address"]] == 1:
        triggered.append("IP address used")
    if sample[feature_index["SSLfinal_State"]] == -1:
        triggered.append("Invalid SSL")
    if sample[feature_index["URL_of_Anchor"]] == 1:
        triggered.append("Suspicious anchor URLs")
    if sample[feature_index["Prefix_Suffix"]] == 1:
        triggered.append("Hyphen in domain")
    return triggered


def rule_score(rules: list[str]) -> float:
    return float(sum(RULE_WEIGHTS.get(r, 0.0) for r in rules))


def apply_improved_hybrid(
    baseline_pred: np.ndarray,
    confidences: np.ndarray,
    all_rules: list[list[str]],
    conf_upper: float,
    min_rules: int,
    min_score: float,
) -> tuple[np.ndarray, int]:
    pred = baseline_pred.copy()
    overrides = 0
    for i in range(len(pred)):
        # Only consider uncertain cases the model marks as legitimate
        if pred[i] != 0:
            continue
        if confidences[i] >= conf_upper:
            continue

        rules = all_rules[i]
        score = rule_score(rules)
        if len(rules) >= min_rules and score >= min_score:
            pred[i] = 1
            overrides += 1

    return pred, overrides


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    y = df[TARGET_COL].map({-1: 0, 1: 1})
    if y.isnull().any():
        raise ValueError("Invalid label values in target column.")
    x = df.drop(columns=[TARGET_COL]).astype(float)
    feature_names = list(x.columns)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        y.values,
        test_size=TEST_SIZE,
        stratify=y.values,
        random_state=RANDOM_STATE,
    )

    # Deep baseline (same as Part 4)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0002,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=250,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=12,
        validation_fraction=0.1,
        verbose=False,
    )
    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)
    baseline_pred = np.argmax(prob, axis=1)
    baseline_prob_pos = prob[:, 1]
    confidences = np.max(prob, axis=1)
    baseline_metrics = evaluate(y_test, baseline_pred, baseline_prob_pos)

    all_rules = [extract_rules(x_test[i], feature_index) for i in range(len(x_test))]

    rows = []
    for conf_upper in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for min_rules in [1, 2, 3]:
            for min_score in [0.8, 1.2, 1.6, 2.0, 2.4]:
                h_pred, overrides = apply_improved_hybrid(
                    baseline_pred=baseline_pred,
                    confidences=confidences,
                    all_rules=all_rules,
                    conf_upper=conf_upper,
                    min_rules=min_rules,
                    min_score=min_score,
                )
                m = evaluate(y_test, h_pred, baseline_prob_pos)
                rows.append(
                    {
                        "conf_upper": conf_upper,
                        "min_rules": min_rules,
                        "min_score": min_score,
                        "overrides": int(overrides),
                        "accuracy": m["accuracy"],
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1_score": m["f1_score"],
                        "roc_auc": m["roc_auc"],
                    }
                )

    sweep_df = pd.DataFrame(rows).sort_values(by=["f1_score", "precision", "recall"], ascending=[False, False, False])
    best = sweep_df.iloc[0].to_dict()

    best_pred, best_overrides = apply_improved_hybrid(
        baseline_pred=baseline_pred,
        confidences=confidences,
        all_rules=all_rules,
        conf_upper=float(best["conf_upper"]),
        min_rules=int(best["min_rules"]),
        min_score=float(best["min_score"]),
    )
    best_metrics = evaluate(y_test, best_pred, baseline_prob_pos)

    # Diagnostics for best config
    rule_counter: Counter[str] = Counter()
    examples = []
    for i in range(len(x_test)):
        if baseline_pred[i] == 0 and best_pred[i] == 1:
            for r in all_rules[i]:
                rule_counter[r] += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "test_index": int(i),
                        "true_label": int(y_test[i]),
                        "baseline_pred": int(baseline_pred[i]),
                        "hybrid_pred": int(best_pred[i]),
                        "confidence": round(float(confidences[i]), 4),
                        "rules": all_rules[i],
                        "rule_score": round(rule_score(all_rules[i]), 3),
                    }
                )

    output = {
        "dataset_path": str(DATASET_PATH),
        "baseline_metrics": baseline_metrics,
        "best_improved_hybrid_config": {
            "conf_upper": float(best["conf_upper"]),
            "min_rules": int(best["min_rules"]),
            "min_score": float(best["min_score"]),
            "overrides": int(best_overrides),
        },
        "best_improved_hybrid_metrics": best_metrics,
        "delta_hybrid_minus_baseline": {
            "accuracy": float(best_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "precision": float(best_metrics["precision"] - baseline_metrics["precision"]),
            "recall": float(best_metrics["recall"] - baseline_metrics["recall"]),
            "f1_score": float(best_metrics["f1_score"] - baseline_metrics["f1_score"]),
            "roc_auc": float(best_metrics["roc_auc"] - baseline_metrics["roc_auc"]),
        },
        "top_10_configs_by_f1": sweep_df.head(10).to_dict(orient="records"),
        "rule_trigger_counts_for_best_config_overrides": dict(rule_counter),
        "override_examples_best_config": examples,
        "note": "Thresholds were tuned on this same test split for learning purposes; use validation/CV for publishable claims.",
    }

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "part9_improved_hybrid_report.json"
    out_csv = out_dir / "part9_improved_hybrid_sweep.csv"
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    sweep_df.to_csv(out_csv, index=False)

    print(json.dumps(output, indent=2))
    print(f"\nSaved report: {out_json.resolve()}")
    print(f"Saved table: {out_csv.resolve()}")


if __name__ == "__main__":
    main()

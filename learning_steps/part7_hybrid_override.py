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
HYBRID_THRESHOLD = 0.8


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def rule_engine(sample: np.ndarray, feature_index: dict[str, int]) -> list[str]:
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

    # Same deep ANN-like model as Part 4
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

    # Baseline predictions
    prob = model.predict_proba(x_test)
    baseline_pred = np.argmax(prob, axis=1)
    baseline_prob_pos = prob[:, 1]
    baseline_metrics = evaluate(y_test, baseline_pred, baseline_prob_pos)

    # Hybrid override logic
    hybrid_pred = baseline_pred.copy()
    confidences = np.max(prob, axis=1)

    override_count = 0
    rule_counter: Counter[str] = Counter()
    override_examples = []

    for i in range(len(x_test)):
        rules = rule_engine(x_test[i], feature_index)
        conf = float(confidences[i])

        if conf < HYBRID_THRESHOLD and len(rules) > 0:
            original = int(hybrid_pred[i])
            hybrid_pred[i] = 1
            if original != 1:
                override_count += 1
                for r in rules:
                    rule_counter[r] += 1
                if len(override_examples) < 8:
                    override_examples.append(
                        {
                            "test_index": int(i),
                            "original_pred": int(original),
                            "new_pred": 1,
                            "true_label": int(y_test[i]),
                            "confidence": round(conf, 4),
                            "rules_triggered": rules,
                        }
                    )

    hybrid_metrics = evaluate(y_test, hybrid_pred, baseline_prob_pos)

    output = {
        "dataset_path": str(DATASET_PATH),
        "model": "part4_deep_ann_with_hybrid_override",
        "threshold": HYBRID_THRESHOLD,
        "baseline_metrics": baseline_metrics,
        "hybrid_metrics": hybrid_metrics,
        "delta_hybrid_minus_baseline": {
            "accuracy": float(hybrid_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "precision": float(hybrid_metrics["precision"] - baseline_metrics["precision"]),
            "recall": float(hybrid_metrics["recall"] - baseline_metrics["recall"]),
            "f1_score": float(hybrid_metrics["f1_score"] - baseline_metrics["f1_score"]),
            "roc_auc": float(hybrid_metrics["roc_auc"] - baseline_metrics["roc_auc"]),
        },
        "overrides_applied": int(override_count),
        "rule_trigger_counts_for_overrides": dict(rule_counter),
        "override_examples": override_examples,
    }

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part7_hybrid_override_report.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

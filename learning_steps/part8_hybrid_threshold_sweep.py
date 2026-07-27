from __future__ import annotations

from pathlib import Path
import json

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


def apply_hybrid_override(
    baseline_pred: np.ndarray,
    confidences: np.ndarray,
    x_test: np.ndarray,
    threshold: float,
    feature_index: dict[str, int],
) -> tuple[np.ndarray, int]:
    pred = baseline_pred.copy()
    overrides = 0
    for i in range(len(x_test)):
        if confidences[i] < threshold:
            rules = rule_engine(x_test[i], feature_index)
            if rules and pred[i] != 1:
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

    # Same deep baseline used in Part 4/7
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

    rows = []
    thresholds = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
    for t in thresholds:
        h_pred, count = apply_hybrid_override(
            baseline_pred=baseline_pred,
            confidences=confidences,
            x_test=x_test,
            threshold=t,
            feature_index=feature_index,
        )
        m = evaluate(y_test, h_pred, baseline_prob_pos)
        rows.append(
            {
                "threshold": t,
                "overrides": int(count),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"],
            }
        )

    sweep_df = pd.DataFrame(rows)
    best_by_f1 = sweep_df.sort_values(by=["f1_score", "recall"], ascending=[False, False]).iloc[0].to_dict()

    output = {
        "dataset_path": str(DATASET_PATH),
        "baseline_metrics": baseline_metrics,
        "threshold_sweep": rows,
        "best_threshold_by_f1": best_by_f1,
    }

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "part8_hybrid_threshold_sweep_report.json"
    out_csv = out_dir / "part8_hybrid_threshold_sweep_table.csv"
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    sweep_df.to_csv(out_csv, index=False)

    print(json.dumps(output, indent=2))
    print(f"\nSaved report: {out_json.resolve()}")
    print(f"Saved table: {out_csv.resolve()}")


if __name__ == "__main__":
    main()

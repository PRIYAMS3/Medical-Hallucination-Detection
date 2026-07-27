from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


DATASET_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")
TARGET_COL = "Result"
RANDOM_STATE = 42
TEST_SIZE = 0.2

PART3_REPORT_PATH = Path("learning_steps") / "outputs" / "part3_simple_ann_report.json"


def _load_part3_metrics() -> dict[str, float] | None:
    if not PART3_REPORT_PATH.exists():
        return None
    payload = json.loads(PART3_REPORT_PATH.read_text(encoding="utf-8"))
    return payload.get("metrics")


def main() -> None:
    # 1) Load data
    df = pd.read_csv(DATASET_PATH)

    # 2) Single-pass label mapping
    y = df[TARGET_COL].map({-1: 0, 1: 1})
    if y.isnull().any():
        raise ValueError("Invalid label values found in target column.")

    x = df.drop(columns=[TARGET_COL]).astype(float)

    # 3) Same split as Part 3 for fair comparison
    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        y.values,
        test_size=TEST_SIZE,
        stratify=y.values,
        random_state=RANDOM_STATE,
    )

    # 4) Deeper ANN-like baseline
    # Mirrors notebook deep model idea (more capacity than Part 3).
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

    # 5) Evaluate
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "iterations_used": int(model.n_iter_),
        "final_loss": float(model.loss_),
    }

    cls_report = classification_report(y_test, y_pred, digits=4)

    # 6) Compare against Part 3 if available
    part3 = _load_part3_metrics()
    delta = None
    if part3 is not None:
        delta = {
            "accuracy_delta_vs_part3": float(metrics["accuracy"] - part3["accuracy"]),
            "precision_delta_vs_part3": float(metrics["precision"] - part3["precision"]),
            "recall_delta_vs_part3": float(metrics["recall"] - part3["recall"]),
            "f1_delta_vs_part3": float(metrics["f1_score"] - part3["f1_score"]),
            "roc_auc_delta_vs_part3": float(metrics["roc_auc"] - part3["roc_auc"]),
        }

    output = {
        "dataset_path": str(DATASET_PATH),
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "model": "DeepANN_MLPClassifier",
        "hyperparameters": {
            "hidden_layer_sizes": [128, 64, 32],
            "batch_size": 64,
            "learning_rate_init": 0.001,
            "max_iter": 250,
            "early_stopping": True,
            "alpha": 0.0002,
        },
        "metrics": metrics,
        "classification_report_text": cls_report,
        "comparison_vs_part3": delta,
    }

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part4_deep_ann_report.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

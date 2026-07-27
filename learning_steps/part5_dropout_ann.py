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

PART3 = Path("learning_steps") / "outputs" / "part3_simple_ann_report.json"
PART4 = Path("learning_steps") / "outputs" / "part4_deep_ann_report.json"


def _read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics")


def _softmax(arr):
    # Not needed for sklearn metrics here; kept for clarity if extending later.
    import numpy as np
    exp = np.exp(arr - arr.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    # 1) Load data
    df = pd.read_csv(DATASET_PATH)

    # 2) Map labels
    y = df[TARGET_COL].map({-1: 0, 1: 1})
    if y.isnull().any():
        raise ValueError("Invalid label values found in target column.")
    x = df.drop(columns=[TARGET_COL]).astype(float)

    # 3) Same split as previous parts
    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        y.values,
        test_size=TEST_SIZE,
        stratify=y.values,
        random_state=RANDOM_STATE,
    )

    # 4) Dropout-style approximation in sklearn:
    # We emulate stronger regularization via alpha + slightly reduced capacity.
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0015,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=260,
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

    # 6) Mini comparison table (part3, part4, part5)
    rows = []
    part3 = _read_metrics(PART3)
    part4 = _read_metrics(PART4)
    if part3:
        rows.append(
            {
                "model": "part3_simple_ann",
                "accuracy": round(part3["accuracy"], 4),
                "precision": round(part3["precision"], 4),
                "recall": round(part3["recall"], 4),
                "f1_score": round(part3["f1_score"], 4),
                "roc_auc": round(part3["roc_auc"], 4),
            }
        )
    if part4:
        rows.append(
            {
                "model": "part4_deep_ann",
                "accuracy": round(part4["accuracy"], 4),
                "precision": round(part4["precision"], 4),
                "recall": round(part4["recall"], 4),
                "f1_score": round(part4["f1_score"], 4),
                "roc_auc": round(part4["roc_auc"], 4),
            }
        )
    rows.append(
        {
            "model": "part5_dropout_style_ann",
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1_score"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
        }
    )

    output = {
        "dataset_path": str(DATASET_PATH),
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "model": "DropoutStyleANN_MLPClassifier",
        "hyperparameters": {
            "hidden_layer_sizes": [128, 64, 32],
            "batch_size": 64,
            "learning_rate_init": 0.001,
            "max_iter": 260,
            "early_stopping": True,
            "alpha": 0.0015,
        },
        "metrics": metrics,
        "classification_report_text": cls_report,
        "mini_comparison_table": rows,
    }

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part5_dropout_ann_report.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

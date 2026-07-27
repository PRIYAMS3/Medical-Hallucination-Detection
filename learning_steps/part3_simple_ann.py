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


def main() -> None:
    # 1) Load dataset
    df = pd.read_csv(DATASET_PATH)

    # 2) Clean target mapping exactly once: -1 => 0, 1 => 1
    y = df[TARGET_COL].map({-1: 0, 1: 1})
    if y.isnull().any():
        raise ValueError("Invalid label values found in target column.")

    x = df.drop(columns=[TARGET_COL]).astype(float)

    # 3) Train-test split (stratified)
    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        y.values,
        test_size=TEST_SIZE,
        stratify=y.values,
        random_state=RANDOM_STATE,
    )

    # 4) Simple ANN baseline using MLPClassifier:
    # architecture mirrors notebook idea: input -> 64 -> 32 -> output(2)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=200,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=10,
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

    output = {
        "dataset_path": str(DATASET_PATH),
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "model": "SimpleANN_MLPClassifier",
        "hyperparameters": {
            "hidden_layer_sizes": [64, 32],
            "batch_size": 64,
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "early_stopping": True,
        },
        "metrics": metrics,
        "classification_report_text": cls_report,
    }

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part3_simple_ann_report.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

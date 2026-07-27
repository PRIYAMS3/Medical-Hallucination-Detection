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


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    y = df[TARGET_COL].map({-1: 0, 1: 1})
    if y.isnull().any():
        raise ValueError("Invalid label values found in target column.")
    x = df.drop(columns=[TARGET_COL]).astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        y.values,
        test_size=TEST_SIZE,
        stratify=y.values,
        random_state=RANDOM_STATE,
    )

    configs = {
        "part3_simple_ann": dict(
            hidden_layer_sizes=(64, 32),
            alpha=0.0001,
            max_iter=200,
            n_iter_no_change=10,
        ),
        "part4_deep_ann": dict(
            hidden_layer_sizes=(128, 64, 32),
            alpha=0.0002,
            max_iter=250,
            n_iter_no_change=12,
        ),
        "part5_dropout_style_ann": dict(
            hidden_layer_sizes=(128, 64, 32),
            alpha=0.0015,
            max_iter=260,
            n_iter_no_change=12,
        ),
    }

    per_model = {}
    probs = []

    for name, cfg in configs.items():
        model = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=cfg["alpha"],
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=cfg["max_iter"],
            random_state=RANDOM_STATE,
            early_stopping=True,
            n_iter_no_change=cfg["n_iter_no_change"],
            validation_fraction=0.1,
            verbose=False,
        )
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        probs.append(prob)

        m = evaluate(y_test, prob)
        m["iterations_used"] = int(model.n_iter_)
        m["final_loss"] = float(model.loss_)
        per_model[name] = m

    ensemble_prob = np.mean(np.vstack(probs), axis=0)
    ensemble_metrics = evaluate(y_test, ensemble_prob)

    best_single_name = max(per_model.keys(), key=lambda n: per_model[n]["f1_score"])
    best_single_f1 = float(per_model[best_single_name]["f1_score"])
    ensemble_f1 = float(ensemble_metrics["f1_score"])

    output = {
        "dataset_path": str(DATASET_PATH),
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "members": list(configs.keys()),
        "individual_metrics": per_model,
        "ensemble_metrics": ensemble_metrics,
        "comparison": {
            "best_single_model": best_single_name,
            "best_single_f1": best_single_f1,
            "ensemble_f1": ensemble_f1,
            "ensemble_minus_best_single_f1": float(ensemble_f1 - best_single_f1),
        },
    }

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part6_ensemble_report.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

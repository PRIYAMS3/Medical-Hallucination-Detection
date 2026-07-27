from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


DATASET_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")
TARGET_COL = "Result"
SEED = 42


def build_model(name: str) -> MLPClassifier:
    if name == "simple_ann":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=200,
            random_state=SEED,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            verbose=False,
        )
    if name == "deep_ann":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0002,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=250,
            random_state=SEED,
            early_stopping=True,
            n_iter_no_change=12,
            validation_fraction=0.1,
            verbose=False,
        )
    if name == "dropout_style_ann":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0015,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=260,
            random_state=SEED,
            early_stopping=True,
            n_iter_no_change=12,
            validation_fraction=0.1,
            verbose=False,
        )
    raise ValueError(name)


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 1e-6, 1.0 - 1e-6)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    y = df[TARGET_COL].map({-1: 0, 1: 1}).astype(int).values
    x = df.drop(columns=[TARGET_COL]).astype(float).values

    # Split once: train pool / test.
    x_train_pool, x_test, y_train_pool, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    # Calibration split from train pool.
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train_pool, y_train_pool, test_size=0.2, stratify=y_train_pool, random_state=SEED
    )

    # Train 3 members and build ensemble score.
    probs_cal = []
    probs_test = []
    for name in ["simple_ann", "deep_ann", "dropout_style_ann"]:
        m = build_model(name)
        m.fit(x_train, y_train)
        probs_cal.append(m.predict_proba(x_cal)[:, 1])
        probs_test.append(m.predict_proba(x_test)[:, 1])

    ens_cal = np.mean(np.vstack(probs_cal), axis=0)
    ens_test = np.mean(np.vstack(probs_test), axis=0)

    # Platt scaling
    platt = LogisticRegression(random_state=SEED)
    platt.fit(ens_cal.reshape(-1, 1), y_cal)
    platt_test = platt.predict_proba(ens_test.reshape(-1, 1))[:, 1]

    # Isotonic scaling
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ens_cal, y_cal)
    iso_test = iso.predict(ens_test)

    out = {
        "dataset": str(DATASET_PATH),
        "splits": {
            "train": int(len(y_train)),
            "calibration": int(len(y_cal)),
            "test": int(len(y_test)),
        },
        "uncalibrated_ensemble": metrics(y_test, clip01(ens_test)),
        "platt_calibrated": metrics(y_test, clip01(platt_test)),
        "isotonic_calibrated": metrics(y_test, clip01(iso_test)),
    }

    # Pick by lowest Brier (calibration quality), keep F1 impact visible.
    candidates = {
        "uncalibrated_ensemble": out["uncalibrated_ensemble"]["brier"],
        "platt_calibrated": out["platt_calibrated"]["brier"],
        "isotonic_calibrated": out["isotonic_calibrated"]["brier"],
    }
    best = min(candidates, key=candidates.get)
    out["best_by_brier"] = best
    out["notes"] = [
        "Calibration mainly improves probability quality (Brier), not always classification F1.",
        "Use calibrated probabilities for threshold tuning and risk-aware hybrid decisions.",
    ]

    out_path = Path("learning_steps") / "outputs" / "part19_quick_calibration_report.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

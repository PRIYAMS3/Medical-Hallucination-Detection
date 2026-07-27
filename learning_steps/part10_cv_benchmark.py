from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


DATASET_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")
TARGET_COL = "Result"
RANDOM_STATE = 42
N_SPLITS = 5
HYBRID_THRESHOLD = 0.8


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
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
            random_state=RANDOM_STATE,
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
            random_state=RANDOM_STATE,
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
            random_state=RANDOM_STATE,
            early_stopping=True,
            n_iter_no_change=12,
            validation_fraction=0.1,
            verbose=False,
        )
    raise ValueError(f"Unknown model name: {name}")


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    y = df[TARGET_COL].map({-1: 0, 1: 1})
    if y.isnull().any():
        raise ValueError("Invalid target values. Expected {-1,1}.")
    x = df.drop(columns=[TARGET_COL]).astype(float)

    x_values = x.values
    y_values = y.values
    feature_index = {col: i for i, col in enumerate(x.columns)}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    per_fold_rows: list[dict[str, float | int | str]] = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(x_values, y_values), start=1):
        x_train, x_test = x_values[train_idx], x_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]

        probs_map: dict[str, np.ndarray] = {}
        for model_name in ["simple_ann", "deep_ann", "dropout_style_ann"]:
            model = build_model(model_name)
            model.fit(x_train, y_train)
            probs_map[model_name] = model.predict_proba(x_test)[:, 1]
            m = evaluate(y_test, probs_map[model_name])
            per_fold_rows.append({"fold": fold_id, "model": model_name, **m})

        # Ensemble: average probability across the three models
        ensemble_prob = np.mean(
            np.vstack(
                [
                    probs_map["simple_ann"],
                    probs_map["deep_ann"],
                    probs_map["dropout_style_ann"],
                ]
            ),
            axis=0,
        )
        ensemble_metrics = evaluate(y_test, ensemble_prob)
        per_fold_rows.append({"fold": fold_id, "model": "ensemble_softvote", **ensemble_metrics})

        # Hybrid on deep_ann (notebook-style override)
        deep_prob = probs_map["deep_ann"]
        deep_pred = (deep_prob >= 0.5).astype(int)
        deep_conf = np.maximum(deep_prob, 1 - deep_prob)
        hybrid_pred = deep_pred.copy()
        overrides = 0
        for i in range(len(x_test)):
            rules = rule_engine(x_test[i], feature_index)
            if deep_conf[i] < HYBRID_THRESHOLD and len(rules) > 0 and hybrid_pred[i] != 1:
                hybrid_pred[i] = 1
                overrides += 1

        hybrid_metrics = evaluate(y_test, deep_prob)  # keep AUC from deep probabilities
        # Replace classification metrics to reflect hybrid labels
        hybrid_metrics["accuracy"] = float(accuracy_score(y_test, hybrid_pred))
        hybrid_metrics["precision"] = float(precision_score(y_test, hybrid_pred, zero_division=0))
        hybrid_metrics["recall"] = float(recall_score(y_test, hybrid_pred, zero_division=0))
        hybrid_metrics["f1_score"] = float(f1_score(y_test, hybrid_pred, zero_division=0))
        per_fold_rows.append(
            {
                "fold": fold_id,
                "model": "hybrid_override_t0.8",
                **hybrid_metrics,
                "overrides": overrides,
            }
        )

    per_fold_df = pd.DataFrame(per_fold_rows)
    summary_df = (
        per_fold_df.groupby("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]
    summary_df = summary_df.sort_values(by=["f1_mean", "roc_auc_mean"], ascending=[False, False])

    output_dir = Path("learning_steps") / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_fold_path = output_dir / "part10_cv_per_fold.csv"
    summary_path = output_dir / "part10_cv_summary.csv"
    json_path = output_dir / "part10_cv_report.json"

    per_fold_df.to_csv(per_fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report = {
        "dataset_path": str(DATASET_PATH),
        "n_splits": N_SPLITS,
        "hybrid_threshold": HYBRID_THRESHOLD,
        "summary_sorted_by_f1": summary_df.to_dict(orient="records"),
        "best_model_by_f1": summary_df.iloc[0]["model"] if len(summary_df) > 0 else None,
        "note": "This is 5-fold CV on the provided dataset only. No external data mixed.",
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved per-fold metrics: {per_fold_path.resolve()}")
    print(f"Saved summary metrics: {summary_path.resolve()}")
    print(f"Saved report: {json_path.resolve()}")


if __name__ == "__main__":
    main()

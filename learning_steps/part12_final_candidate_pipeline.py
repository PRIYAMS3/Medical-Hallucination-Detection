from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


DATASET_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")
TARGET_COL = "Result"
RANDOM_STATE = 42
TEST_SIZE = 0.2
HYBRID_THRESHOLD = 0.8


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
    raise ValueError(f"Unknown model: {name}")


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
        raise ValueError("Invalid target labels. Expected {-1, 1}.")
    x = df.drop(columns=[TARGET_COL]).astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x.values,
        y.values,
        test_size=TEST_SIZE,
        stratify=y.values,
        random_state=RANDOM_STATE,
    )

    # Train candidate ensemble members
    member_names = ["simple_ann", "deep_ann", "dropout_style_ann"]
    trained = {}
    probs = []
    for name in member_names:
        model = build_model(name)
        model.fit(x_train, y_train)
        trained[name] = model
        probs.append(model.predict_proba(x_test)[:, 1])

    # Ensemble baseline (soft voting)
    ensemble_prob = np.mean(np.vstack(probs), axis=0)
    baseline_metrics = evaluate(y_test, ensemble_prob)

    # Hybrid mode on top of ensemble:
    # if confidence low + rules triggered, force phishing class
    feature_index = {col: idx for idx, col in enumerate(x.columns)}
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    confidence = np.maximum(ensemble_prob, 1 - ensemble_prob)
    hybrid_pred = ensemble_pred.copy()

    overrides = 0
    examples = []
    for i in range(len(x_test)):
        rules = rule_engine(x_test[i], feature_index)
        if confidence[i] < HYBRID_THRESHOLD and len(rules) > 0 and hybrid_pred[i] != 1:
            hybrid_pred[i] = 1
            overrides += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "test_index": int(i),
                        "true_label": int(y_test[i]),
                        "baseline_pred": int(ensemble_pred[i]),
                        "hybrid_pred": int(hybrid_pred[i]),
                        "confidence": round(float(confidence[i]), 4),
                        "rules_triggered": rules,
                    }
                )

    hybrid_metrics = {
        "accuracy": float(accuracy_score(y_test, hybrid_pred)),
        "precision": float(precision_score(y_test, hybrid_pred, zero_division=0)),
        "recall": float(recall_score(y_test, hybrid_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, hybrid_pred, zero_division=0)),
        # keep ranking metric from probability model
        "roc_auc": float(roc_auc_score(y_test, ensemble_prob)),
        "confusion_matrix": confusion_matrix(y_test, hybrid_pred).tolist(),
    }

    # Persist artifacts for later VS Code / API usage
    out_dir = Path("learning_steps") / "outputs" / "part12_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in trained.items():
        joblib.dump(model, out_dir / f"{name}.joblib")

    config = {
        "dataset_path": str(DATASET_PATH),
        "target_column": TARGET_COL,
        "member_models": member_names,
        "hybrid_threshold": HYBRID_THRESHOLD,
        "feature_order": list(x.columns),
    }
    (out_dir / "inference_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = {
        "dataset_path": str(DATASET_PATH),
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "ensemble_members": member_names,
        "baseline_ensemble_metrics": baseline_metrics,
        "hybrid_metrics": hybrid_metrics,
        "delta_hybrid_minus_baseline": {
            "accuracy": float(hybrid_metrics["accuracy"] - baseline_metrics["accuracy"]),
            "precision": float(hybrid_metrics["precision"] - baseline_metrics["precision"]),
            "recall": float(hybrid_metrics["recall"] - baseline_metrics["recall"]),
            "f1_score": float(hybrid_metrics["f1_score"] - baseline_metrics["f1_score"]),
            "roc_auc": float(hybrid_metrics["roc_auc"] - baseline_metrics["roc_auc"]),
        },
        "overrides_applied": int(overrides),
        "override_examples": examples,
        "artifacts_dir": str(out_dir.resolve()),
        "note": "Hybrid mode is included and switchable, but baseline ensemble is the recommended default based on current results.",
    }

    out_report = Path("learning_steps") / "outputs" / "part12_final_candidate_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {out_report.resolve()}")
    print(f"Saved artifacts in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

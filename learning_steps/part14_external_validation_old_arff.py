from __future__ import annotations

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


OLD_ARFF_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\phishing+websites\.old.arff")
TRAIN_REFERENCE_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")
ARTIFACT_DIR = Path("learning_steps") / "outputs" / "part12_artifacts"


def parse_arff(path: Path) -> pd.DataFrame:
    attribute_pattern = re.compile(r"^@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))", re.IGNORECASE)
    attributes: list[str] = []
    rows: list[list[str]] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            low = line.lower()
            if not in_data:
                if low.startswith("@attribute"):
                    match = attribute_pattern.match(line)
                    if match:
                        name = next(group for group in match.groups() if group is not None)
                        attributes.append(name)
                elif low.startswith("@data"):
                    in_data = True
                continue

            rows.append([token.strip() for token in line.split(",")])

    df = pd.DataFrame(rows, columns=attributes)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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


def run_one_scenario(
    y_values: np.ndarray,
    x_values: np.ndarray,
    feature_order: list[str],
    threshold: float,
    model_names: list[str],
) -> dict[str, object]:
    probs = []
    per_model = {}
    for name in model_names:
        model = joblib.load(ARTIFACT_DIR / f"{name}.joblib")
        prob = model.predict_proba(x_values)[:, 1]
        probs.append(prob)
        per_model[name] = evaluate(y_values, prob)

    ensemble_prob = np.mean(np.vstack(probs), axis=0)
    ensemble_metrics = evaluate(y_values, ensemble_prob)

    feature_index = {c: i for i, c in enumerate(feature_order)}
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    confidence = np.maximum(ensemble_prob, 1 - ensemble_prob)
    hybrid_pred = ensemble_pred.copy()
    overrides = 0
    for i in range(len(x_values)):
        rules = rule_engine(x_values[i], feature_index)
        if confidence[i] < threshold and len(rules) > 0 and hybrid_pred[i] != 1:
            hybrid_pred[i] = 1
            overrides += 1

    hybrid_metrics = {
        "accuracy": float(accuracy_score(y_values, hybrid_pred)),
        "precision": float(precision_score(y_values, hybrid_pred, zero_division=0)),
        "recall": float(recall_score(y_values, hybrid_pred, zero_division=0)),
        "f1_score": float(f1_score(y_values, hybrid_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_values, ensemble_prob)),
        "confusion_matrix": confusion_matrix(y_values, hybrid_pred).tolist(),
    }

    return {
        "label_distribution": {
            "0": int((y_values == 0).sum()),
            "1": int((y_values == 1).sum()),
        },
        "per_model_metrics": per_model,
        "ensemble_metrics": ensemble_metrics,
        "hybrid_metrics": hybrid_metrics,
        "delta_hybrid_minus_ensemble": {
            "accuracy": float(hybrid_metrics["accuracy"] - ensemble_metrics["accuracy"]),
            "precision": float(hybrid_metrics["precision"] - ensemble_metrics["precision"]),
            "recall": float(hybrid_metrics["recall"] - ensemble_metrics["recall"]),
            "f1_score": float(hybrid_metrics["f1_score"] - ensemble_metrics["f1_score"]),
            "roc_auc": float(hybrid_metrics["roc_auc"] - ensemble_metrics["roc_auc"]),
        },
        "hybrid_overrides_applied": int(overrides),
    }


def main() -> None:
    cfg = json.loads((ARTIFACT_DIR / "inference_config.json").read_text(encoding="utf-8"))
    feature_order: list[str] = cfg["feature_order"]
    threshold = float(cfg.get("hybrid_threshold", 0.8))
    model_names = cfg["member_models"]

    df_old = parse_arff(OLD_ARFF_PATH)
    if "Result" not in df_old.columns:
        raise ValueError("Expected target column 'Result' in .old.arff")

    x_old = df_old.drop(columns=["Result"]).copy()
    missing = [c for c in feature_order if c not in x_old.columns]
    if missing:
        raise ValueError(f".old.arff is missing expected features: {missing}")
    x_old = x_old[feature_order].fillna(0).astype(float)

    # Harmonize binary encoding drift:
    # if train used {-1,1} and old uses {0,1}, map 0 -> -1.
    train_ref = pd.read_csv(TRAIN_REFERENCE_PATH)
    mapped_cols = []
    for col in feature_order:
        train_vals = set(train_ref[col].dropna().astype(int).unique().tolist())
        old_vals = set(x_old[col].dropna().astype(int).unique().tolist())
        if train_vals == {-1, 1} and old_vals.issubset({0, 1}):
            x_old[col] = x_old[col].replace({0: -1, 1: 1})
            mapped_cols.append(col)

    x_values = x_old.values

    # Scenario A: legacy mapping used in earlier notebook steps.
    y_legacy = df_old["Result"].map({-1: 0, 1: 1})
    if y_legacy.isnull().any():
        raise ValueError("Unexpected target values in .old.arff for legacy mapping")

    # Scenario B: phishing-positive mapping (recommended).
    # Here -1 is treated as phishing class (1), 1 as legitimate (0).
    y_phishing_positive = df_old["Result"].map({-1: 1, 1: 0})
    if y_phishing_positive.isnull().any():
        raise ValueError("Unexpected target values in .old.arff for phishing-positive mapping")

    scenario_legacy = run_one_scenario(
        y_values=y_legacy.values.astype(int),
        x_values=x_values,
        feature_order=feature_order,
        threshold=threshold,
        model_names=model_names,
    )
    scenario_phishing_positive = run_one_scenario(
        y_values=y_phishing_positive.values.astype(int),
        x_values=x_values,
        feature_order=feature_order,
        threshold=threshold,
        model_names=model_names,
    )

    output = {
        "external_dataset": str(OLD_ARFF_PATH),
        "external_shape": [int(x_values.shape[0]), int(x_values.shape[1])],
        "encoding_harmonization": {
            "mapped_0_to_minus1_columns_count": len(mapped_cols),
            "mapped_columns": mapped_cols,
        },
        "label_scenarios": {
            "legacy_minus1_to_0_plus1_to_1": scenario_legacy,
            "phishing_positive_minus1_to_1_plus1_to_0": scenario_phishing_positive,
        },
        "recommended_for_phishing_detection": {
            "label_scenario": "phishing_positive_minus1_to_1_plus1_to_0",
            "reason": "Treats phishing as positive class and aligns external performance with model ranking behavior.",
        },
        "hybrid_threshold": threshold,
    }

    out_path = Path("learning_steps") / "outputs" / "part14_external_validation_report.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

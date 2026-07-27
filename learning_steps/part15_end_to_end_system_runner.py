from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


ARTIFACT_DIR = Path("learning_steps") / "outputs" / "part12_artifacts"
TRAIN_REFERENCE_PATH = Path(r"C:\Users\PRIYAMVADA NAMBIAR\Downloads\output.csv")


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


def read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".arff":
        return parse_arff(path)
    return pd.read_csv(path)


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


def harmonize_features(x: pd.DataFrame, feature_order: list[str]) -> tuple[pd.DataFrame, list[str]]:
    train_ref = pd.read_csv(TRAIN_REFERENCE_PATH)
    x = x.copy()
    mapped_cols: list[str] = []

    for col in feature_order:
        train_vals = set(train_ref[col].dropna().astype(int).unique().tolist())
        old_vals = set(x[col].dropna().astype(int).unique().tolist())
        if train_vals == {-1, 1} and old_vals.issubset({0, 1}):
            x[col] = x[col].replace({0: -1, 1: 1})
            mapped_cols.append(col)

    return x, mapped_cols


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end phishing detection runner (baseline + hybrid).")
    parser.add_argument("--input-path", required=True, help="Path to input CSV or ARFF.")
    parser.add_argument(
        "--mode",
        default="ensemble",
        choices=["ensemble", "hybrid"],
        help="Prediction mode: ensemble or hybrid.",
    )
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path for per-row predictions.")
    parser.add_argument(
        "--label-scenario",
        default="phishing_positive",
        choices=["phishing_positive", "legacy"],
        help="Only used if input contains 'Result'.",
    )
    args = parser.parse_args()

    cfg = json.loads((ARTIFACT_DIR / "inference_config.json").read_text(encoding="utf-8"))
    feature_order: list[str] = cfg["feature_order"]
    model_names: list[str] = cfg["member_models"]
    threshold = float(cfg.get("hybrid_threshold", 0.8))

    input_path = Path(args.input_path)
    df = read_dataset(input_path)

    has_target = "Result" in df.columns
    x = df.drop(columns=["Result"]) if has_target else df.copy()
    missing = [c for c in feature_order if c not in x.columns]
    if missing:
        raise ValueError(f"Input missing required features: {missing}")

    x = x[feature_order].fillna(0).astype(float)
    x, mapped_cols = harmonize_features(x, feature_order)
    x_values = x.values

    # Ensemble probabilities
    probs = []
    for name in model_names:
        model = joblib.load(ARTIFACT_DIR / f"{name}.joblib")
        probs.append(model.predict_proba(x_values)[:, 1])
    ensemble_prob = np.mean(np.vstack(probs), axis=0)

    feature_index = {c: i for i, c in enumerate(feature_order)}
    pred_prob = ensemble_prob.copy()
    pred_label = (pred_prob >= 0.5).astype(int)
    decision_type = np.array(["model_only"] * len(pred_label), dtype=object)
    rules_triggered_all: list[list[str]] = [[] for _ in range(len(pred_label))]

    if args.mode == "hybrid":
        conf = np.maximum(pred_prob, 1 - pred_prob)
        for i in range(len(x_values)):
            rules = rule_engine(x_values[i], feature_index)
            rules_triggered_all[i] = rules
            if conf[i] < threshold and len(rules) > 0 and pred_label[i] != 1:
                pred_label[i] = 1
                decision_type[i] = "rule_override"
            else:
                decision_type[i] = "model_only"

    result_df = x.copy()
    result_df["prob_positive_class"] = pred_prob
    result_df["pred_label"] = pred_label
    result_df["decision_type"] = decision_type
    result_df["rules_triggered"] = [",".join(r) if r else "" for r in rules_triggered_all]

    summary: dict[str, object] = {
        "input_path": str(input_path),
        "rows": int(len(result_df)),
        "mode": args.mode,
        "feature_harmonization_mapped_columns_count": len(mapped_cols),
        "feature_harmonization_mapped_columns": mapped_cols,
    }

    if has_target:
        if args.label_scenario == "phishing_positive":
            y = df["Result"].map({-1: 1, 1: 0})
        else:
            y = df["Result"].map({-1: 0, 1: 1})
        if y.isnull().any():
            raise ValueError("Input contains unsupported target values in 'Result'.")
        y_values = y.values.astype(int)
        # For hybrid mode, keep AUC from probabilities and classification from final labels
        metrics = compute_metrics(y_values, pred_prob)
        metrics["accuracy"] = float(accuracy_score(y_values, pred_label))
        metrics["precision"] = float(precision_score(y_values, pred_label, zero_division=0))
        metrics["recall"] = float(recall_score(y_values, pred_label, zero_division=0))
        metrics["f1_score"] = float(f1_score(y_values, pred_label, zero_division=0))
        summary["metrics"] = metrics

    out_dir = Path("learning_steps") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else (out_dir / "part15_predictions.csv")
    result_df.to_csv(output_csv, index=False)

    summary_path = out_dir / "part15_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved predictions: {output_csv.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()

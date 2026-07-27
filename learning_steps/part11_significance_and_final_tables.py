from __future__ import annotations

from pathlib import Path
import itertools
import json
from math import erf, sqrt

import numpy as np
import pandas as pd


PER_FOLD_PATH = Path("learning_steps") / "outputs" / "part10_cv_per_fold.csv"
SUMMARY_PATH = Path("learning_steps") / "outputs" / "part10_cv_summary.csv"
OUT_DIR = Path("learning_steps") / "outputs"


def mean_std_text(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def paired_ttest_approx_pvalue(diffs: np.ndarray) -> float:
    # Normal approximation for small n (reported with caution in final note).
    n = len(diffs)
    if n < 2:
        return 1.0
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    if std_d == 0:
        return 1.0
    t_stat = mean_d / (std_d / sqrt(n))
    p_two_sided = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return float(max(0.0, min(1.0, p_two_sided)))


def exact_sign_flip_pvalue(diffs: np.ndarray) -> float:
    # Exact paired randomization test under H0 (sign-flip symmetry).
    n = len(diffs)
    if n == 0:
        return 1.0
    observed = abs(float(np.mean(diffs)))
    count_extreme = 0
    total = 0
    for signs in itertools.product([-1, 1], repeat=n):
        flipped = diffs * np.array(signs, dtype=float)
        stat = abs(float(np.mean(flipped)))
        if stat >= observed - 1e-12:
            count_extreme += 1
        total += 1
    return float(count_extreme / total)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        val = (m - rank + 1) * p
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def main() -> None:
    if not PER_FOLD_PATH.exists():
        raise FileNotFoundError(f"Missing per-fold file: {PER_FOLD_PATH}")
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    per_fold = pd.read_csv(PER_FOLD_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    # 1) Paper-ready summary table
    summary_table = pd.DataFrame(
        {
            "Model": summary["model"],
            "Accuracy (mean±std)": [
                mean_std_text(m, s) for m, s in zip(summary["accuracy_mean"], summary["accuracy_std"])
            ],
            "Precision (mean±std)": [
                mean_std_text(m, s) for m, s in zip(summary["precision_mean"], summary["precision_std"])
            ],
            "Recall (mean±std)": [
                mean_std_text(m, s) for m, s in zip(summary["recall_mean"], summary["recall_std"])
            ],
            "F1 (mean±std)": [mean_std_text(m, s) for m, s in zip(summary["f1_mean"], summary["f1_std"])],
            "ROC-AUC (mean±std)": [
                mean_std_text(m, s) for m, s in zip(summary["roc_auc_mean"], summary["roc_auc_std"])
            ],
        }
    )

    # 2) Significance tests on fold-level F1
    models = sorted(per_fold["model"].unique().tolist())
    rows = []
    pvals_for_holm = []

    # Focus primary claims against best model by mean F1
    best_model = summary.sort_values(by="f1_mean", ascending=False).iloc[0]["model"]
    best_f1 = summary.sort_values(by="f1_mean", ascending=False).iloc[0]["f1_mean"]

    for model in models:
        if model == best_model:
            continue
        a = per_fold[per_fold["model"] == best_model].sort_values("fold")
        b = per_fold[per_fold["model"] == model].sort_values("fold")
        diffs = (a["f1_score"].values - b["f1_score"].values).astype(float)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        dz = float(mean_diff / std_diff) if std_diff > 0 else 0.0
        p_exact = exact_sign_flip_pvalue(diffs)
        p_t_approx = paired_ttest_approx_pvalue(diffs)
        pvals_for_holm.append(p_exact)
        rows.append(
            {
                "comparison": f"{best_model} vs {model}",
                "metric": "f1_score",
                "best_model_f1_mean": float(best_f1),
                "other_model_f1_mean": float(summary[summary["model"] == model]["f1_mean"].iloc[0]),
                "mean_fold_diff_best_minus_other": mean_diff,
                "std_fold_diff": std_diff,
                "effect_size_dz": dz,
                "p_value_exact_sign_flip": p_exact,
                "p_value_t_approx": p_t_approx,
                "n_folds": int(len(diffs)),
            }
        )

    # Holm correction across primary comparisons
    if rows:
        adjusted = holm_bonferroni(pvals_for_holm)
        for i, adj in enumerate(adjusted):
            rows[i]["p_value_exact_sign_flip_holm"] = float(adj)

    significance_df = pd.DataFrame(rows).sort_values(by="p_value_exact_sign_flip", ascending=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table1_path = OUT_DIR / "part11_paper_ready_summary_table.csv"
    table2_path = OUT_DIR / "part11_significance_table.csv"
    json_path = OUT_DIR / "part11_significance_report.json"

    summary_table.to_csv(table1_path, index=False)
    significance_df.to_csv(table2_path, index=False)

    report = {
        "best_model_by_cv_f1": best_model,
        "best_model_cv_f1_mean": float(best_f1),
        "summary_table_path": str(table1_path.resolve()),
        "significance_table_path": str(table2_path.resolve()),
        "notes": [
            "Exact sign-flip p-values are primary due to small number of folds (n=5).",
            "t-test p-values are approximate and provided only as secondary reference.",
            "Non-significant p-values do not mean models are equal; they indicate limited evidence with n=5.",
        ],
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved summary table: {table1_path.resolve()}")
    print(f"Saved significance table: {table2_path.resolve()}")
    print(f"Saved report: {json_path.resolve()}")


if __name__ == "__main__":
    main()

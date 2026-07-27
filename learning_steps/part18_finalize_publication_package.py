from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


OUT = Path("learning_steps") / "outputs"


def j(path: str) -> dict:
    p = OUT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def c(path: str) -> pd.DataFrame:
    p = OUT / path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def f(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    cv = j("part10_cv_report.json")
    ext = j("part14_external_validation_report.json")
    p12 = j("part12_final_candidate_report.json")
    sig = c("part11_significance_table.csv")

    best = cv.get("best_model_by_f1", "ensemble_softvote")
    cv_rows = cv.get("summary_sorted_by_f1", [])
    top = cv_rows[0] if cv_rows else {}

    ext_s = ext.get("label_scenarios", {}).get("phishing_positive_minus1_to_1_plus1_to_0", {})
    ext_ens = ext_s.get("ensemble_metrics", {})
    ext_hyb = ext_s.get("hybrid_metrics", {})
    ext_delta = ext_s.get("delta_hybrid_minus_ensemble", {})

    p_best = "N/A"
    if not sig.empty:
        p_best = str(sig.iloc[0].get("p_value_exact_sign_flip", "N/A"))

    lines = []
    lines.append("# Final Publication Package (Ready Draft)")
    lines.append("")
    lines.append("## A. Core Claim")
    lines.append(
        "We propose an ensemble-hybrid phishing detection framework and evaluate it with cross-validation, "
        "ablation, and external-dataset validation."
    )
    lines.append("")
    lines.append("## B. Main Internal Result (5-fold CV)")
    lines.append(f"- Best model by F1: **{best}**")
    if top:
        lines.append(
            f"- F1: {f(top['f1_mean'])} ± {f(top['f1_std'])}, "
            f"ROC-AUC: {f(top['roc_auc_mean'])} ± {f(top['roc_auc_std'])}"
        )
    lines.append("")
    lines.append("## C. External Validation (.old.arff, phishing-positive scenario)")
    if ext_ens and ext_hyb:
        lines.append(
            f"- Ensemble F1: {f(ext_ens['f1_score'])}, Hybrid F1: {f(ext_hyb['f1_score'])}, "
            f"Delta: {f(ext_delta['f1_score'])}"
        )
        lines.append(
            f"- Ensemble Recall: {f(ext_ens['recall'])}, Hybrid Recall: {f(ext_hyb['recall'])}"
        )
    lines.append("")
    lines.append("## D. Statistical Evidence")
    lines.append(f"- Primary exact sign-flip p-value (best comparison in table): {p_best}")
    lines.append("- With 5 folds, report effect sizes and confidence intervals alongside p-values.")
    lines.append("")
    lines.append("## E. System Availability")
    lines.append("- Batch runner (ensemble/hybrid): `part15_end_to_end_system_runner.py`")
    lines.append("- Lightweight API (CPU-friendly): `part17_api_lite.py`")
    lines.append("- Saved artifacts: `part12_artifacts/`")
    lines.append("")
    lines.append("## F. Reproducibility Block")
    lines.append("- Dataset used: `output.csv` (+ external `.old.arff` validation)")
    lines.append("- Deterministic random seed: 42")
    lines.append("- Evaluation: single-run + 5-fold CV + external validation")
    lines.append("- Outputs to cite: `part10`, `part11`, `part14`, `part15` reports")
    lines.append("")
    lines.append("## G. Conservative Conclusion Text")
    lines.append(
        "The proposed framework demonstrates robust internal performance through cross-validation, "
        "with ensemble soft-voting as the strongest internal performer. "
        "External evaluation under harmonized feature encoding further indicates that hybrid rule augmentation "
        "can improve phishing-focused F1/recall in shifted data conditions."
    )
    lines.append("")
    lines.append("## H. What To Submit")
    lines.append("1. Main manuscript using this file as Results/Discussion backbone.")
    lines.append("2. CSV tables from `part11` and `part10` as supplementary material.")
    lines.append("3. Inference scripts (`part15`, `part17`) and `part12_artifacts` for reproducibility.")

    out_md = OUT / "part18_final_publication_package.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    status = {
        "final_package": str(out_md.resolve()),
        "next_steps_remaining": [
            "Optional calibration experiment",
            "Industry hardening (FastAPI tests/logging/CI)",
        ],
    }
    (OUT / "part18_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    print(f"\nSaved: {out_md.resolve()}")


if __name__ == "__main__":
    main()

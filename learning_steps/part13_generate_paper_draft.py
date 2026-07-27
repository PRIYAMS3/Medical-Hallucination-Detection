from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


OUT_DIR = Path("learning_steps") / "outputs"


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def main() -> None:
    p10 = _safe_read_json(OUT_DIR / "part10_cv_report.json")
    p11_sig = _safe_read_csv(OUT_DIR / "part11_significance_table.csv")
    p11_sum = _safe_read_csv(OUT_DIR / "part11_paper_ready_summary_table.csv")
    p12 = _safe_read_json(OUT_DIR / "part12_final_candidate_report.json")

    best_model = p10.get("best_model_by_f1", "N/A")
    summary_rows = p10.get("summary_sorted_by_f1", [])

    baseline_ens = p12.get("baseline_ensemble_metrics", {})
    hybrid = p12.get("hybrid_metrics", {})
    delta = p12.get("delta_hybrid_minus_baseline", {})

    lines = []
    lines.append("# Draft Results and Methodology (Phishing Detection)")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "Build an intelligent phishing detection system with deep model benchmarking, ensemble learning, "
        "and an explainable rule-based hybrid decision layer."
    )
    lines.append("")
    lines.append("## 2. Dataset and Preprocessing")
    lines.append("- Dataset: `output.csv` converted from `Training Dataset.arff` (exact match verified).")
    lines.append("- Size: 11,055 samples, 30 features + 1 target (`Result`).")
    lines.append("- Target mapping: `-1 -> 0` (legitimate), `1 -> 1` (phishing), applied once in preprocessing.")
    lines.append("- Split strategy used in single-run experiments: stratified 80/20.")
    lines.append("- Cross-validation strategy: 5-fold Stratified CV.")
    lines.append("")
    lines.append("## 3. Models Evaluated")
    lines.append("- Simple ANN")
    lines.append("- Deep ANN")
    lines.append("- Dropout-style ANN")
    lines.append("- Soft-voting ensemble (Simple + Deep + Dropout-style)")
    lines.append("- Hybrid override system (confidence threshold + domain rules)")
    lines.append("")
    lines.append("## 4. Main Cross-Validation Findings")
    lines.append(f"- Best model by mean CV F1: **{best_model}**.")
    if summary_rows:
        top = summary_rows[0]
        lines.append(
            f"- Top score (mean ± std): F1 = {_fmt(top['f1_mean'])} ± {_fmt(top['f1_std'])}, "
            f"ROC-AUC = {_fmt(top['roc_auc_mean'])} ± {_fmt(top['roc_auc_std'])}."
        )
    lines.append("- Full model table is exported in `part11_paper_ready_summary_table.csv`.")
    lines.append("")
    lines.append("## 5. Hybrid System Findings")
    if baseline_ens and hybrid:
        lines.append(
            f"- Baseline ensemble F1: {_fmt(baseline_ens.get('f1_score', 0.0))}, "
            f"Hybrid F1: {_fmt(hybrid.get('f1_score', 0.0))}."
        )
        lines.append(
            f"- Recall changed from {_fmt(baseline_ens.get('recall', 0.0))} to {_fmt(hybrid.get('recall', 0.0))}, "
            f"while precision changed from {_fmt(baseline_ens.get('precision', 0.0))} to {_fmt(hybrid.get('precision', 0.0))}."
        )
        lines.append(
            f"- Net delta (Hybrid - Baseline): "
            f"F1 {_fmt(delta.get('f1_score', 0.0))}, "
            f"Accuracy {_fmt(delta.get('accuracy', 0.0))}."
        )
        lines.append(
            "- Interpretation: current hybrid configuration improves phishing recall but introduces additional "
            "false positives, reducing overall F1."
        )
    else:
        lines.append("- Hybrid report not found.")
    lines.append("")
    lines.append("## 6. Statistical Testing")
    if not p11_sig.empty:
        best_comp = p11_sig.iloc[0].to_dict()
        lines.append(
            "- Pairwise significance was evaluated using exact sign-flip tests across CV folds "
            "(with Holm correction)."
        )
        lines.append(
            f"- Strongest comparison in table: `{best_comp.get('comparison', 'N/A')}`, "
            f"exact p = {_fmt(float(best_comp.get('p_value_exact_sign_flip', 1.0)))}."
        )
        lines.append(
            "- With 5 folds, p-values are conservative; effect sizes and mean±std should be reported together."
        )
    else:
        lines.append("- Significance table not found.")
    lines.append("")
    lines.append("## 7. Deployable Candidate")
    lines.append(
        "- Final candidate pipeline artifacts are saved under `part12_artifacts/` "
        "(trained models + inference config)."
    )
    lines.append("- Recommended default mode: ensemble baseline.")
    lines.append("- Optional mode: hybrid override for high-recall alerting scenarios.")
    lines.append("")
    lines.append("## 8. Novelty Positioning")
    lines.append(
        "Potential novelty claim: an evidence-driven evaluation of when rule-augmented hybrid phishing "
        "decision logic helps recall but may harm balanced performance, supported by CV and ablation."
    )
    lines.append("")
    lines.append("## 9. Immediate Next Steps")
    lines.append("1. Add probability calibration (Platt/Isotonic) and re-test hybrid thresholds.")
    lines.append("2. Add external evaluation using `.old.arff` as a generalization check.")
    lines.append("3. Package inference endpoint (API) and include reproducibility checklist in appendix.")

    draft_path = OUT_DIR / "part13_draft_results_methodology.md"
    draft_path.write_text("\n".join(lines), encoding="utf-8")

    status = {
        "draft_path": str(draft_path.resolve()),
        "uses_outputs_from_parts": [10, 11, 12],
    }
    (OUT_DIR / "part13_draft_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(json.dumps(status, indent=2))
    print(f"\nSaved draft: {draft_path.resolve()}")


if __name__ == "__main__":
    main()

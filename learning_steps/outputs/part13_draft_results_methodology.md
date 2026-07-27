# Draft Results and Methodology (Phishing Detection)

## 1. Objective
Build an intelligent phishing detection system with deep model benchmarking, ensemble learning, and an explainable rule-based hybrid decision layer.

## 2. Dataset and Preprocessing
- Dataset: `output.csv` converted from `Training Dataset.arff` (exact match verified).
- Size: 11,055 samples, 30 features + 1 target (`Result`).
- Target mapping: `-1 -> 0` (legitimate), `1 -> 1` (phishing), applied once in preprocessing.
- Split strategy used in single-run experiments: stratified 80/20.
- Cross-validation strategy: 5-fold Stratified CV.

## 3. Models Evaluated
- Simple ANN
- Deep ANN
- Dropout-style ANN
- Soft-voting ensemble (Simple + Deep + Dropout-style)
- Hybrid override system (confidence threshold + domain rules)

## 4. Main Cross-Validation Findings
- Best model by mean CV F1: **ensemble_softvote**.
- Top score (mean ± std): F1 = 0.9704 ± 0.0026, ROC-AUC = 0.9954 ± 0.0006.
- Full model table is exported in `part11_paper_ready_summary_table.csv`.

## 5. Hybrid System Findings
- Baseline ensemble F1: 0.9731, Hybrid F1: 0.9628.
- Recall changed from 0.9862 to 0.9984, while precision changed from 0.9604 to 0.9297.
- Net delta (Hybrid - Baseline): F1 -0.0104, Accuracy -0.0127.
- Interpretation: current hybrid configuration improves phishing recall but introduces additional false positives, reducing overall F1.

## 6. Statistical Testing
- Pairwise significance was evaluated using exact sign-flip tests across CV folds (with Holm correction).
- Strongest comparison in table: `ensemble_softvote vs deep_ann`, exact p = 0.0625.
- With 5 folds, p-values are conservative; effect sizes and mean±std should be reported together.

## 7. Deployable Candidate
- Final candidate pipeline artifacts are saved under `part12_artifacts/` (trained models + inference config).
- Recommended default mode: ensemble baseline.
- Optional mode: hybrid override for high-recall alerting scenarios.

## 8. Novelty Positioning
Potential novelty claim: an evidence-driven evaluation of when rule-augmented hybrid phishing decision logic helps recall but may harm balanced performance, supported by CV and ablation.

## 9. Immediate Next Steps
1. Add probability calibration (Platt/Isotonic) and re-test hybrid thresholds.
2. Add external evaluation using `.old.arff` as a generalization check.
3. Package inference endpoint (API) and include reproducibility checklist in appendix.
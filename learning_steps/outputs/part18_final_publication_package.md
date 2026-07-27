# Final Publication Package (Ready Draft)

## A. Core Claim
We propose an ensemble-hybrid phishing detection framework and evaluate it with cross-validation, ablation, and external-dataset validation.

## B. Main Internal Result (5-fold CV)
- Best model by F1: **ensemble_softvote**
- F1: 0.9704 ± 0.0026, ROC-AUC: 0.9954 ± 0.0006

## C. External Validation (.old.arff, phishing-positive scenario)
- Ensemble F1: 0.8820, Hybrid F1: 0.8956, Delta: 0.0137
- Ensemble Recall: 0.8201, Hybrid Recall: 0.8473

## D. Statistical Evidence
- Primary exact sign-flip p-value (best comparison in table): 0.0625
- With 5 folds, report effect sizes and confidence intervals alongside p-values.

## E. System Availability
- Batch runner (ensemble/hybrid): `part15_end_to_end_system_runner.py`
- Lightweight API (CPU-friendly): `part17_api_lite.py`
- Saved artifacts: `part12_artifacts/`

## F. Reproducibility Block
- Dataset used: `output.csv` (+ external `.old.arff` validation)
- Deterministic random seed: 42
- Evaluation: single-run + 5-fold CV + external validation
- Outputs to cite: `part10`, `part11`, `part14`, `part15` reports

## G. Conservative Conclusion Text
The proposed framework demonstrates robust internal performance through cross-validation, with ensemble soft-voting as the strongest internal performer. External evaluation under harmonized feature encoding further indicates that hybrid rule augmentation can improve phishing-focused F1/recall in shifted data conditions.

## H. What To Submit
1. Main manuscript using this file as Results/Discussion backbone.
2. CSV tables from `part11` and `part10` as supplementary material.
3. Inference scripts (`part15`, `part17`) and `part12_artifacts` for reproducibility.
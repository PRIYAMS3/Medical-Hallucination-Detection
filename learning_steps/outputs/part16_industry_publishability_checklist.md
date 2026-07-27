# Part 16: Industry + Publishability Readiness Checklist

## A. What Is Already Done

- [x] Verified data integrity (`output.csv` exactly matches `Training Dataset.arff`)
- [x] Clean preprocessing with one-time label mapping and stratified splits
- [x] Multi-model ANN benchmarking
- [x] Ensemble (soft voting) benchmarking
- [x] Hybrid (confidence + rules) design and ablation
- [x] Threshold sweep for hybrid behavior
- [x] 5-fold CV benchmarking with mean ± std tables
- [x] Fold-level statistical significance analysis
- [x] External validation on `.old.arff` with encoding harmonization
- [x] End-to-end batch runner (`ensemble` and `hybrid`) with prediction exports

## B. Current Best Empirical Position

- Internal CV (provided dataset): `ensemble_softvote` is best by mean F1.
- External dataset (`.old.arff`, phishing-positive mapping): hybrid mode on top of ensemble improves F1 over ensemble baseline.
- This supports a strong narrative: ensemble robustness + context-aware hybrid gains under distribution shift.

## C. To Be Fully Industry-Grade

- [ ] Convert runner to a maintained service layer (FastAPI + versioned API contract)
- [ ] Add unit/integration tests (preprocessing, inference, hybrid override logic)
- [ ] Add model/version registry metadata and deterministic run IDs
- [ ] Add input schema validation + error handling for malformed payloads
- [ ] Add monitoring hooks (prediction drift, confidence drift, rule-trigger rates)
- [ ] Add security hardening (rate limit, auth, structured logging, PII-safe logs)
- [ ] Add CI pipeline (lint, tests, build verification)

## D. To Be Strongly Publishable

- [ ] Compare against selected SOTA baselines from target paper list using same metrics
- [ ] Add calibration experiment (Platt/Isotonic) and evaluate hybrid impact
- [ ] Add proper train/val/test or nested CV protocol for hyperparameter tuning
- [ ] Add external-dataset protocol section (encoding harmonization + label semantics handling)
- [ ] Add error analysis section with representative false-positive/false-negative cases
- [ ] Add reproducibility appendix (seed, versions, scripts, artifact hashes)

## E. Recommended Claim Style

Use a conservative claim:
"The proposed ensemble-hybrid phishing framework achieves robust cross-validated performance and demonstrates improved phishing recall/F1 in external evaluation under a harmonized feature-space protocol."

Avoid over-claiming "best of all papers" until direct apples-to-apples benchmark comparisons are completed.

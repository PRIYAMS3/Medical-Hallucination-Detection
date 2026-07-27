# Intelligent Hybrid Phishing Detection System

This folder contains a clean, reproducible Python project that upgrades the Colab notebook into an industry-ready baseline.

## What Is Included

- Multi-model deep learning benchmarking (7 architectures)
- Weighted training for class imbalance
- Ensemble inference (soft voting)
- Hybrid decision layer (model confidence + rule-based override)
- Optional SHAP explainability
- Config-driven runs

## Project Layout

```text
phishing_hybrid/
|-- configs/
|-- data/
|-- outputs/
|-- src/phishing_hybrid/
|-- tests/
|-- requirements.txt
|-- run.py
`-- README.md
```

## Setup

```bash
cd phishing_hybrid
pip install -r requirements.txt
```

## Run Baseline Experiment

```bash
python run.py run-baseline --config configs/baseline.yaml
```

Outputs are saved under `outputs/`:

- `outputs/results/results.csv`
- `outputs/results/metrics_detailed.json`
- `outputs/models/*.pt`

## Run SHAP (Optional)

```bash
python run.py explain --config configs/baseline.yaml --checkpoint outputs/models/deep_ann.pt
```

## Run Hybrid Inference On One Sample

```bash
python run.py predict-one --config configs/baseline.yaml --checkpoint outputs/models/deep_ann.pt --sample-index 3
```

## Notes

- `output.csv` and `Training Dataset.arff` are both supported.
- Label handling is fixed to avoid repeated remapping bugs.
- `results.csv` is generated from actual evaluations, not hardcoded values.

# AI-Powered Warehouse Inventory Optimization System

This project implements a rubric-aligned machine learning pipeline for warehouse inventory optimization using the M5 retail sales dataset structure. Because the provided data contains historical sales and calendar context but not explicit warehouse stock fields, this implementation derives inventory proxy features from past demand in a deterministic and documented way.

The system predicts:

- `future_7_day_demand`
- `stockout_risk`
- `days_to_stockout`
- `reorder_quantity`
- `product_movement_class`

## Repository Structure

```text
project-name/
|-- README.md
|-- requirements.txt
|-- Dockerfile
|-- config.yaml
|-- data/
|-- src/
|-- pipeline/
|-- models/
|-- app/
|-- logs/
`-- notebooks/
```

## Problem Framing

- Regression task: forecast next 7-day demand.
- Classification task: predict stockout risk.
- Additional business rules: infer reorder quantity and classify product movement as `fast_moving`, `slow_moving`, or `dead_stock`.

## Feature Engineering

Two core meaningful transformations are implemented in `src/features.py`:

1. Lag and rolling demand features
2. Inventory coverage features

## No Data Leakage

- Features use only historical values via lagged and shifted rolling windows.
- Targets use future demand.
- Train, validation, and test sets are split chronologically by date.

## Models Compared

Regression:

- Random Forest Regressor
- XGBoost Regressor

Classification:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

## Setup

```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
uvicorn app.app:app --reload
```

## Important Assumption

The provided CSV files do not include explicit stock levels, cost prices, or selling prices. To keep the project executable and consistent with your objective, those fields are generated deterministically from historical demand and stable item-level hashes. This can later be replaced with real warehouse ERP data without changing the pipeline structure.

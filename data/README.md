# Data Notes

This project uses the M5 retail demand dataset format supplied by the user:

- `sales_train_validation.csv`
- `calendar.csv`

The raw CSV files are not copied into this repository because they are large. Instead, the pipeline reads them from the local paths defined in `config.yaml`.

## Source

- Kaggle M5 / retail demand style data
- See `data/dataset_link.txt`

## Processing Summary

The original sales file is wide format with one column per day (`d_1`, `d_2`, ...). The project converts it into a long daily transaction-style dataset and then creates:

- date features
- lag features
- rolling demand features
- proxy stock and reorder fields
- future 7-day demand target
- stockout risk target

## Missing Value Handling

- Calendar event columns (`event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`) are filled with `Unknown` before feature creation.
- Model input preprocessing uses:
  - `SimpleImputer(strategy="median")` for numeric fields
  - `SimpleImputer(strategy="most_frequent")` for categorical fields
- Lag/rolling rows without enough history are dropped only for training/evaluation target creation.

## Feature Justification

- **Lag + rolling demand features (`lag_7`, `rolling_mean_28`, `rolling_std_28`)**: capture short-term and medium-term demand trend/volatility needed for forecasting and stockout risk detection.
- **Inventory coverage features (`stock_level`, `reorder_level`, `inventory_turnover_proxy`)**: convert demand signals into stock control indicators directly aligned with reorder decisions.

## Leakage Prevention

- Only past demand is used to build features
- Targets are created from future demand windows
- Data is split chronologically

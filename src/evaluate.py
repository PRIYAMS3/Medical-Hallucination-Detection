from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_percentage_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import build_long_sales_frame
from src.features import create_model_dataset, get_feature_columns
from src.utils import PROJECT_ROOT, load_config, save_json, setup_logging


def evaluate() -> dict:
    config = load_config()
    logger = setup_logging(PROJECT_ROOT / config["paths"]["app_log"])
    artifact = joblib.load(PROJECT_ROOT / config["paths"]["model_output"])

    long_df = build_long_sales_frame(config)
    dataset = create_model_dataset(
        long_df=long_df,
        forecast_horizon=config["data"]["forecast_horizon_days"],
        stock_cover_days=config["data"]["stock_cover_days"],
        reorder_cover_days=config["data"]["reorder_cover_days"],
    )

    validation_end = pd.Timestamp(artifact["date_splits"]["validation_end_date"])
    test_df = dataset[dataset["date"] > validation_end].copy()

    features = get_feature_columns()
    X_test = test_df[features]

    y_test_reg = test_df["future_7_day_demand"]
    y_pred_reg = artifact["regression_pipeline"].predict(X_test)
    rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
    mape = mean_absolute_percentage_error(y_test_reg + 1e-3, y_pred_reg + 1e-3)

    y_test_clf = test_df["stockout_risk"]
    y_pred_clf = artifact["classification_pipeline"].predict(X_test)
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    cm = confusion_matrix(y_test_clf, y_pred_clf).tolist()

    error_df = test_df[["item_id", "store_id", "date", "future_7_day_demand"]].copy()
    error_df["predicted_future_7_day_demand"] = y_pred_reg
    error_df["absolute_error"] = (
        error_df["future_7_day_demand"] - error_df["predicted_future_7_day_demand"]
    ).abs()
    error_df = error_df.sort_values("absolute_error", ascending=False)
    error_df.head(100).to_csv(PROJECT_ROOT / config["paths"]["error_analysis"], index=False)

    summary = {
        "best_regression_model": artifact["regression_model_name"],
        "best_classification_model": artifact["classification_model_name"],
        "regression_metrics": {"rmse": rmse, "mape": mape},
        "classification_metrics": {"accuracy": accuracy, "confusion_matrix": cm},
        "basic_error_analysis": {
            "largest_error_rows_saved_to": config["paths"]["error_analysis"],
            "mean_actual_demand": float(y_test_reg.mean()),
            "mean_predicted_demand": float(y_pred_reg.mean()),
        },
    }
    save_json(summary, PROJECT_ROOT / config["paths"]["evaluation_summary"])
    logger.info("Evaluation complete | RMSE=%.4f | MAPE=%.4f | ACC=%.4f", rmse, mape, accuracy)
    return summary


if __name__ == "__main__":
    result = evaluate()
    print(result)

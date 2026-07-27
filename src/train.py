from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.data_loader import build_long_sales_frame
from src.features import create_model_dataset, get_feature_columns
from src.preprocess import build_preprocessor
from src.utils import PROJECT_ROOT, load_config, save_json, setup_logging


def time_split(
    df: pd.DataFrame, train_ratio: float, validation_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(df["date"].unique()))
    train_end = int(len(unique_dates) * train_ratio)
    valid_end = int(len(unique_dates) * (train_ratio + validation_ratio))

    train_dates = unique_dates[:train_end]
    valid_dates = unique_dates[train_end:valid_end]
    test_dates = unique_dates[valid_end:]

    train_df = df[df["date"].isin(train_dates)].copy()
    valid_df = df[df["date"].isin(valid_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, valid_df, test_df


def build_regression_models(random_state: int) -> dict[str, object]:
    return {
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state,
        ),
        "xgboost_regressor": XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=random_state,
        ),
    }


def build_classification_models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=800,
            random_state=random_state,
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state,
        ),
        "xgboost_classifier": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        ),
    }


def train() -> dict:
    config = load_config()
    logger = setup_logging(PROJECT_ROOT / config["paths"]["app_log"])
    logger.info("Starting training run.")

    long_df = build_long_sales_frame(config)
    dataset = create_model_dataset(
        long_df=long_df,
        forecast_horizon=config["data"]["forecast_horizon_days"],
        stock_cover_days=config["data"]["stock_cover_days"],
        reorder_cover_days=config["data"]["reorder_cover_days"],
    )

    dataset.head(500).to_csv(
        PROJECT_ROOT / config["paths"]["sample_processed_data"], index=False
    )
    train_df, valid_df, test_df = time_split(
        dataset,
        train_ratio=config["training"]["train_ratio"],
        validation_ratio=config["training"]["validation_ratio"],
    )

    features = get_feature_columns()
    X_train, X_valid = train_df[features], valid_df[features]
    y_train_reg, y_valid_reg = train_df["future_7_day_demand"], valid_df["future_7_day_demand"]
    y_train_clf, y_valid_clf = train_df["stockout_risk"], valid_df["stockout_risk"]

    regression_results = {}
    best_reg_name = None
    best_reg_mape = float("inf")
    best_reg_pipeline = None
    for name, model in build_regression_models(config["project"]["random_state"]).items():
        pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train_reg)
        predictions = pipeline.predict(X_valid)
        rmse = mean_squared_error(y_valid_reg, predictions, squared=False)
        mape = mean_absolute_percentage_error(y_valid_reg + 1e-3, predictions + 1e-3)
        regression_results[name] = {"rmse": rmse, "mape": mape}
        logger.info("Regression model %s | RMSE=%.4f | MAPE=%.4f", name, rmse, mape)
        if mape < best_reg_mape:
            best_reg_mape = mape
            best_reg_name = name
            best_reg_pipeline = pipeline

    classification_results = {}
    best_clf_name = None
    best_clf_accuracy = float("-inf")
    best_clf_pipeline = None
    for name, model in build_classification_models(config["project"]["random_state"]).items():
        pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train_clf)
        predictions = pipeline.predict(X_valid)
        accuracy = accuracy_score(y_valid_clf, predictions)
        classification_results[name] = {"accuracy": accuracy}
        logger.info("Classification model %s | ACC=%.4f", name, accuracy)
        if accuracy > best_clf_accuracy:
            best_clf_accuracy = accuracy
            best_clf_name = name
            best_clf_pipeline = pipeline

    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    X_train_valid = train_valid_df[features]
    y_train_valid_reg = train_valid_df["future_7_day_demand"]
    y_train_valid_clf = train_valid_df["stockout_risk"]

    best_reg_pipeline.fit(X_train_valid, y_train_valid_reg)
    best_clf_pipeline.fit(X_train_valid, y_train_valid_clf)

    date_splits = {
        "train_end_date": str(train_df["date"].max().date()),
        "validation_end_date": str(valid_df["date"].max().date()),
        "test_end_date": str(test_df["date"].max().date()),
    }

    artifact = {
        "regression_model_name": best_reg_name,
        "classification_model_name": best_clf_name,
        "regression_pipeline": best_reg_pipeline,
        "classification_pipeline": best_clf_pipeline,
        "feature_columns": features,
        "date_splits": date_splits,
        "config": config,
    }

    model_path = PROJECT_ROOT / config["paths"]["model_output"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    summary = {
        "regression_results": regression_results,
        "classification_results": classification_results,
        "best_regression_model": best_reg_name,
        "best_classification_model": best_clf_name,
        "dataset_rows": len(dataset),
        "train_rows": len(train_df),
        "validation_rows": len(valid_df),
        "test_rows": len(test_df),
        "date_splits": date_splits,
    }
    save_json(summary, PROJECT_ROOT / config["paths"]["training_summary"])
    logger.info(
        "Training complete. Best regression=%s, best classification=%s",
        best_reg_name,
        best_clf_name,
    )
    return summary


if __name__ == "__main__":
    result = train()
    print(result)

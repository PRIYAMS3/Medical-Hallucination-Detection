from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.utils import movement_class_from_demand


FEATURE_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "snap_flag",
    "is_weekend",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "cost_price",
    "selling_price",
    "gross_margin",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "sales_momentum",
    "stock_level",
    "reorder_level",
    "inventory_gap",
    "inventory_turnover_proxy",
]


def _future_sum(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]


def create_model_dataset(
    long_df: pd.DataFrame,
    forecast_horizon: int,
    stock_cover_days: int,
    reorder_cover_days: int,
) -> pd.DataFrame:
    df = long_df.copy().sort_values(["id", "date"]).reset_index(drop=True)
    grouped_sales = df.groupby("id")["quantity_sold"]

    df["lag_1"] = grouped_sales.shift(1)
    df["lag_7"] = grouped_sales.shift(7)
    df["lag_28"] = grouped_sales.shift(28)
    lagged_sales = grouped_sales.shift(1)

    # Rolling windows must be computed per item to avoid cross-item leakage.
    df["rolling_mean_7"] = (
        lagged_sales.groupby(df["id"]).rolling(7, min_periods=7).mean().reset_index(level=0, drop=True)
    )
    df["rolling_mean_28"] = (
        lagged_sales.groupby(df["id"]).rolling(28, min_periods=28).mean().reset_index(level=0, drop=True)
    )
    df["rolling_std_28"] = (
        lagged_sales.groupby(df["id"]).rolling(28, min_periods=28).std().reset_index(level=0, drop=True)
    )

    df["rolling_std_28"] = df["rolling_std_28"].fillna(0.0)
    df["gross_margin"] = df["selling_price"] - df["cost_price"]
    df["sales_momentum"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-3)

    safety_stock = df["rolling_std_28"].fillna(0) * 2
    df["stock_level"] = np.ceil(
        np.maximum(
            df["rolling_mean_28"].fillna(0) * stock_cover_days + safety_stock,
            df["lag_7"].fillna(0) * 1.5 + 5,
        )
    )
    df["reorder_level"] = np.ceil(
        np.maximum(
            df["rolling_mean_7"].fillna(0) * reorder_cover_days + safety_stock,
            3,
        )
    )
    df["inventory_gap"] = df["stock_level"] - df["reorder_level"]
    df["inventory_turnover_proxy"] = df["rolling_mean_28"] / (df["stock_level"] + 1e-3)

    df["future_7_day_demand"] = grouped_sales.transform(
        lambda series: _future_sum(series, forecast_horizon)
    )
    df["future_daily_demand"] = df["future_7_day_demand"] / forecast_horizon
    df["days_to_stockout_target"] = df["stock_level"] / (df["future_daily_demand"] + 0.1)
    df["stockout_risk"] = (df["days_to_stockout_target"] <= reorder_cover_days).astype(int)
    df["product_movement_class"] = df["rolling_mean_28"].apply(movement_class_from_demand)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["lag_28", "rolling_mean_28", "future_7_day_demand"]).reset_index(drop=True)
    return df


def get_feature_columns() -> List[str]:
    return FEATURE_COLUMNS.copy()


def build_realtime_feature_row(payload: dict) -> pd.DataFrame:
    sales_history = payload["last_28_days_sales"]
    last_7 = sales_history[-7:]
    rolling_mean_7 = float(np.mean(last_7))
    rolling_mean_28 = float(np.mean(sales_history))
    rolling_std_28 = float(np.std(sales_history, ddof=1)) if len(sales_history) > 1 else 0.0
    lag_1 = float(sales_history[-1])
    lag_7 = float(sales_history[-7])
    lag_28 = float(sales_history[0])
    gross_margin = payload["selling_price"] - payload["cost_price"]
    sales_momentum = rolling_mean_7 / (rolling_mean_28 + 1e-3)
    inventory_gap = payload["current_stock_level"] - payload["reorder_level"]
    turnover = rolling_mean_28 / (payload["current_stock_level"] + 1e-3)
    forecast_date = pd.Timestamp(payload["forecast_date"])
    is_weekend = int(forecast_date.day_name() in {"Saturday", "Sunday"})

    row = {
        "item_id": payload["item_id"],
        "dept_id": payload["dept_id"],
        "cat_id": payload["cat_id"],
        "store_id": payload["store_id"],
        "state_id": payload["state_id"],
        "wm_yr_wk": payload.get("wm_yr_wk", 0),
        "wday": forecast_date.dayofweek + 1,
        "month": forecast_date.month,
        "year": forecast_date.year,
        "snap_flag": payload.get("snap_flag", 0),
        "is_weekend": is_weekend,
        "event_name_1": payload.get("event_name_1", "Unknown"),
        "event_type_1": payload.get("event_type_1", "Unknown"),
        "event_name_2": payload.get("event_name_2", "Unknown"),
        "event_type_2": payload.get("event_type_2", "Unknown"),
        "cost_price": payload["cost_price"],
        "selling_price": payload["selling_price"],
        "gross_margin": gross_margin,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_28": lag_28,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_28": rolling_mean_28,
        "rolling_std_28": rolling_std_28,
        "sales_momentum": sales_momentum,
        "stock_level": payload["current_stock_level"],
        "reorder_level": payload["reorder_level"],
        "inventory_gap": inventory_gap,
        "inventory_turnover_proxy": turnover,
    }
    return pd.DataFrame([row], columns=get_feature_columns())

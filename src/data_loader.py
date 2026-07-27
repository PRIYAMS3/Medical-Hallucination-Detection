from __future__ import annotations

from typing import Dict

import pandas as pd


META_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_raw_data(config: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_df = pd.read_csv(config["paths"]["sales_data"])
    calendar_df = pd.read_csv(config["paths"]["calendar_data"])
    return sales_df, calendar_df


def _sample_items(sales_df: pd.DataFrame, max_items: int, random_state: int) -> pd.DataFrame:
    if len(sales_df) <= max_items:
        return sales_df.copy()

    sampled = (
        sales_df.groupby("cat_id", group_keys=False)
        .apply(
            lambda part: part.sample(
                n=max(1, int(round(max_items * len(part) / len(sales_df)))),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )

    if len(sampled) > max_items:
        sampled = sampled.sample(n=max_items, random_state=random_state)
    elif len(sampled) < max_items:
        missing = max_items - len(sampled)
        remaining = sales_df.loc[~sales_df["id"].isin(sampled["id"])]
        if not remaining.empty:
            extra = remaining.sample(
                n=min(missing, len(remaining)),
                random_state=random_state,
            )
            sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.reset_index(drop=True)


def _assign_deterministic_prices(df: pd.DataFrame) -> pd.DataFrame:
    item_hash = pd.util.hash_pandas_object(df["item_id"], index=False).astype("uint64")
    cost_price = 5.0 + (item_hash % 2500).astype(float) / 100.0
    markup = 1.15 + (item_hash % 40).astype(float) / 100.0
    df["cost_price"] = cost_price.round(2)
    df["selling_price"] = (df["cost_price"] * markup).round(2)
    return df


def build_long_sales_frame(config: Dict) -> pd.DataFrame:
    sales_df, calendar_df = load_raw_data(config)
    random_state = config["project"]["random_state"]
    max_items = config["data"]["max_items"]
    history_days = config["data"]["history_days"]

    sampled_sales = _sample_items(sales_df, max_items=max_items, random_state=random_state)
    day_columns = [column for column in sampled_sales.columns if column.startswith("d_")]
    selected_days = day_columns[-history_days:]

    long_df = sampled_sales.melt(
        id_vars=META_COLUMNS,
        value_vars=selected_days,
        var_name="d",
        value_name="quantity_sold",
    )
    long_df = long_df.merge(calendar_df, on="d", how="left")
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df["quantity_sold"] = long_df["quantity_sold"].astype(float)

    long_df = _assign_deterministic_prices(long_df)

    snap_columns = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    long_df["snap_flag"] = long_df.apply(
        lambda row: row.get(snap_columns.get(row["state_id"], ""), 0), axis=1
    )

    for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        long_df[column] = long_df[column].fillna("Unknown")

    long_df["is_weekend"] = long_df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    long_df = long_df.sort_values(["id", "date"]).reset_index(drop=True)
    return long_df

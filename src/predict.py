from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import build_realtime_feature_row
from src.utils import PROJECT_ROOT, load_config, movement_class_from_demand


def predict_from_payload(payload: dict) -> dict:
    config = load_config()
    artifact = joblib.load(PROJECT_ROOT / config["paths"]["model_output"])
    feature_frame = build_realtime_feature_row(payload)

    predicted_demand = float(artifact["regression_pipeline"].predict(feature_frame)[0])
    stockout_risk = int(artifact["classification_pipeline"].predict(feature_frame)[0])
    current_stock = float(payload["current_stock_level"])
    days_to_stockout = current_stock / max(predicted_demand / 7.0, 0.1)
    reorder_quantity = max(
        round(predicted_demand + payload["reorder_level"] - current_stock, 2), 0.0
    )
    movement_class = movement_class_from_demand(feature_frame.loc[0, "rolling_mean_28"])

    return {
        "predicted_future_7_day_demand": round(predicted_demand, 2),
        "stockout_risk": stockout_risk,
        "days_to_stockout": round(days_to_stockout, 2),
        "reorder_quantity": reorder_quantity,
        "product_movement_class": movement_class,
        "recommended_reorder_now": bool(
            stockout_risk or current_stock <= payload["reorder_level"]
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inventory prediction from a JSON payload.")
    parser.add_argument("--input_json", required=True, help="Path to JSON payload.")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    print(json.dumps(predict_from_payload(payload), indent=2))

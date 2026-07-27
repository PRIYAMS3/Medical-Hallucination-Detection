from __future__ import annotations

import joblib

from src.features import build_realtime_feature_row
from src.utils import PROJECT_ROOT, load_config, movement_class_from_demand


class InventoryOptimizationPipeline:
    def __init__(self, model_path: str | None = None) -> None:
        config = load_config()
        resolved_path = model_path or str(PROJECT_ROOT / config["paths"]["model_output"])
        self.artifact = joblib.load(resolved_path)

    def predict(self, payload: dict) -> dict:
        feature_frame = build_realtime_feature_row(payload)
        predicted_demand = float(self.artifact["regression_pipeline"].predict(feature_frame)[0])
        stockout_risk = int(self.artifact["classification_pipeline"].predict(feature_frame)[0])
        current_stock = float(payload["current_stock_level"])
        reorder_quantity = max(
            round(predicted_demand + payload["reorder_level"] - current_stock, 2),
            0.0,
        )
        days_to_stockout = current_stock / max(predicted_demand / 7.0, 0.1)

        return {
            "predicted_future_7_day_demand": round(predicted_demand, 2),
            "stockout_risk": stockout_risk,
            "days_to_stockout": round(days_to_stockout, 2),
            "reorder_quantity": reorder_quantity,
            "product_movement_class": movement_class_from_demand(
                feature_frame.loc[0, "rolling_mean_28"]
            ),
            "recommended_reorder_now": bool(
                stockout_risk or current_stock <= payload["reorder_level"]
            ),
        }

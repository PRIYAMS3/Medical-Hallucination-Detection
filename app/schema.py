from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str
    forecast_date: str
    current_stock_level: float = Field(ge=0)
    reorder_level: float = Field(ge=0)
    cost_price: float = Field(gt=0)
    selling_price: float = Field(gt=0)
    last_28_days_sales: List[float]
    snap_flag: int = 0
    wm_yr_wk: Optional[int] = 0
    event_name_1: Optional[str] = "Unknown"
    event_type_1: Optional[str] = "Unknown"
    event_name_2: Optional[str] = "Unknown"
    event_type_2: Optional[str] = "Unknown"

    @field_validator("last_28_days_sales")
    @classmethod
    def validate_sales_history(cls, value: List[float]) -> List[float]:
        if len(value) != 28:
            raise ValueError("last_28_days_sales must contain exactly 28 values.")
        return value


class PredictionResponse(BaseModel):
    predicted_future_7_day_demand: float
    stockout_risk: int
    days_to_stockout: float
    reorder_quantity: float
    product_movement_class: str
    recommended_reorder_now: bool

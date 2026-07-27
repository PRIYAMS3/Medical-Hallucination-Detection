from __future__ import annotations

from fastapi import FastAPI

from app.schema import PredictionRequest, PredictionResponse
from pipeline.pipeline import InventoryOptimizationPipeline


app = FastAPI(title="Warehouse Inventory Optimization API", version="1.0.0")
model_pipeline = InventoryOptimizationPipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    prediction = model_pipeline.predict(request.model_dump())
    return PredictionResponse(**prediction)

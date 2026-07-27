from __future__ import annotations

import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from phishing_hybrid.config import load_config
from phishing_hybrid.hybrid import HybridPhishingSystem
from phishing_hybrid.training import load_checkpoint, resolve_device


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature map with names matching training columns.")


def create_app() -> FastAPI:
    config_path = os.getenv("PHISHING_CONFIG", "configs/baseline.yaml")
    checkpoint_path = os.getenv("PHISHING_CHECKPOINT", "outputs/models/deep_ann.pt")

    config = load_config(config_path)
    device = resolve_device(config.training.device)
    model, metadata = load_checkpoint(checkpoint_path, device)

    feature_names = metadata.get("feature_names", [])
    if not feature_names:
        raise RuntimeError("Checkpoint does not contain feature names. Re-train with new pipeline.")

    system = HybridPhishingSystem(
        model=model,
        feature_names=feature_names,
        rules=config.hybrid.rules,
        threshold=config.hybrid.threshold,
        device=device,
    )

    app = FastAPI(title="Hybrid Phishing Detection API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        missing = [col for col in feature_names if col not in req.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

        vector = np.array([float(req.features[col]) for col in feature_names], dtype=np.float32)
        return system.predict_as_dict(vector)

    return app


app = create_app()

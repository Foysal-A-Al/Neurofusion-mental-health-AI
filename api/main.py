from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest
from neurofusion.inference import NeuroFusionPredictor

app = FastAPI(
    title="NeuroFusion API",
    version="1.0.0",
    description="Research-only multimodal mental-health AI prototype",
)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = NeuroFusionPredictor()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Models are not trained. Run scripts/generate_data.py and scripts/train.py",
            ) from exc
    return _predictor


@app.get("/health")
def health():
    directory = Path(os.environ.get("NEUROFUSION_MODEL_DIR", "artifacts"))
    return {
        "status": "ok",
        "models_available": all(
            (directory / name).is_file() for name in ["state_model.joblib", "risk_model.joblib"]
        ),
    }


@app.post("/api/v1/predict")
def predict(request: PredictionRequest):
    records = [o.model_dump(mode="json") for o in request.observations]
    try:
        return get_predictor().predict(records)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

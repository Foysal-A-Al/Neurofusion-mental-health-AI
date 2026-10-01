from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd

from .data.validation import validate_longitudinal_frame
from .evaluation import uncertainty_interval
from .explainability import human_explanation
from .features import FEATURE_COLUMNS, aggregate_patient_features


class NeuroFusionPredictor:
    def __init__(self, artifact_dir: str | Path | None = None, min_confidence: float | None = None):
        artifact_dir = Path(artifact_dir or os.environ.get("NEUROFUSION_MODEL_DIR", "artifacts"))
        min_confidence = (
            float(os.environ.get("NEUROFUSION_ABSTAIN_THRESHOLD", "0.55"))
            if min_confidence is None
            else min_confidence
        )
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between zero and one")
        self.state_model = joblib.load(artifact_dir / "state_model.joblib")
        self.risk_model = joblib.load(artifact_dir / "risk_model.joblib")
        self.min_confidence = min_confidence

    def predict(self, records: list[dict]) -> dict:
        if not 3 <= len(records) <= 60:
            raise ValueError("Provide 3 to 60 observations for one patient")
        raw = pd.DataFrame(records)
        required_defaults = {"baseline_sleep": 7.0, "age": 30, "sex": "unknown"}
        for col, value in required_defaults.items():
            if col not in raw:
                raw[col] = value
        if "patient_id" not in raw:
            raw["patient_id"] = "API-PATIENT"
        if "date" not in raw:
            raw["date"] = pd.date_range("2026-01-01", periods=len(raw))
        # Labels are placeholders because aggregation expects a training-compatible schema.
        raw["mood_state"] = raw.get("mood_state", "stable")
        raw["deterioration_7d"] = raw.get("deterioration_7d", 0)
        validate_longitudinal_frame(raw)
        if raw.patient_id.nunique() != 1:
            raise ValueError("Each prediction request must contain exactly one patient")

        feat = aggregate_patient_features(raw).iloc[0]
        X = pd.DataFrame([{c: feat[c] for c in FEATURE_COLUMNS}])
        state_probs = self.state_model.predict_proba(X)[0]
        state_classes = self.state_model.classes_
        state_idx = int(state_probs.argmax())
        state = str(state_classes[state_idx])
        confidence = float(state_probs[state_idx])
        risk = float(self.risk_model.predict_proba(X)[0, 1])
        lo, hi = uncertainty_interval(risk)
        return {
            "predicted_state": state if confidence >= self.min_confidence else "uncertain",
            "state_confidence": confidence,
            "state_probabilities": {str(c): float(p) for c, p in zip(state_classes, state_probs)},
            "seven_day_deterioration_risk": risk,
            "illustrative_risk_interval": [lo, hi],
            "uncertainty_note": "Heuristic probability band using assumed effective_n=30; not a validated individual confidence interval.",
            "prediction_status": "review_recommended"
            if confidence < self.min_confidence or risk >= 0.65
            else "model_confident",
            "top_factors": human_explanation(feat),
            "explanation_method": "Reference-based observations; not model-derived feature attribution.",
            "disclaimer": "Research prototype only; not for diagnosis, treatment, or emergency use.",
        }

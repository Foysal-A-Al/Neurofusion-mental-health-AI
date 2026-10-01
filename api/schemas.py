from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Observation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    patient_id: str = Field(default="API-PATIENT", min_length=1)
    date: date
    age: int = Field(ge=18, le=100)
    sex: str = Field(min_length=1)
    baseline_sleep: float = Field(gt=0, le=16)
    mood_score: float = Field(ge=0, le=10)
    sleep_hours: float = Field(ge=0, le=24)
    stress_level: float = Field(ge=0, le=10)
    activity_minutes: float = Field(ge=0, le=500)
    medication_adherence: float = Field(ge=0, le=1)
    speech_rate: float = Field(ge=40, le=300)
    pause_ratio: float = Field(ge=0, le=1)
    sentiment_score: float = Field(ge=-1, le=1)


class PredictionRequest(BaseModel):
    observations: list[Observation] = Field(min_length=3, max_length=60)

    @model_validator(mode="after")
    def validate_patient_window(self):
        if len({o.patient_id for o in self.observations}) != 1:
            raise ValueError("Exactly one patient is allowed per request")
        dates = [o.date for o in self.observations]
        if len(set(dates)) != len(dates):
            raise ValueError("Duplicate observation dates are not allowed")
        for field in ["age", "sex", "baseline_sleep"]:
            if len({getattr(o, field) for o in self.observations}) != 1:
                raise ValueError(f"{field} must be consistent within a patient window")
        return self

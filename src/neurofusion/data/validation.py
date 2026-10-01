from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {
    "patient_id", "date", "age", "sex", "mood_score", "sleep_hours",
    "stress_level", "activity_minutes", "medication_adherence",
    "speech_rate", "pause_ratio", "sentiment_score", "mood_state",
    "deterioration_7d", "baseline_sleep"
}


def validate_longitudinal_frame(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("date contains invalid or missing dates")
    if df.assign(date=dates).duplicated(["patient_id", "date"]).any():
        raise ValueError("Duplicate patient/date observations are not allowed")
    numeric = ["age", "baseline_sleep", "mood_score", "sleep_hours", "stress_level",
               "activity_minutes", "medication_adherence", "speech_rate", "pause_ratio", "sentiment_score"]
    try:
        values = df[numeric].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Observation features must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Observation features must be finite and nonmissing")
    if not df["mood_state"].isin(["depressive", "stable", "elevated"]).all():
        raise ValueError("mood_state contains invalid labels")
    if not df["deterioration_7d"].isin([0, 1]).all():
        raise ValueError("deterioration_7d must contain binary labels")
    bounded = {
        "mood_score": (0, 10), "sleep_hours": (0, 24), "stress_level": (0, 10),
        "medication_adherence": (0, 1), "pause_ratio": (0, 1), "sentiment_score": (-1, 1),
        "age": (18, 100), "baseline_sleep": (0.01, 16), "activity_minutes": (0, 500), "speech_rate": (40, 300)
    }
    for col, (lo, hi) in bounded.items():
        if not df[col].between(lo, hi).all():
            raise ValueError(f"{col} contains values outside [{lo}, {hi}]")
    if df["sex"].isna().any() or df["sex"].astype(str).str.strip().eq("").any():
        raise ValueError("sex must contain nonempty categories")
    for column in ["age", "sex", "baseline_sleep"]:
        if df.groupby("patient_id")[column].nunique().gt(1).any():
            raise ValueError(f"{column} must be consistent within each patient window")

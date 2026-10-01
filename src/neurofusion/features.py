from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "age",
    "mood_score_mean",
    "mood_score_std",
    "mood_score_last",
    "sleep_hours_mean",
    "sleep_hours_std",
    "sleep_deviation_mean",
    "stress_level_mean",
    "stress_level_last",
    "activity_minutes_mean",
    "medication_adherence_mean",
    "speech_rate_mean",
    "pause_ratio_mean",
    "sentiment_score_mean",
    "mood_trend",
    "stress_trend",
]
CATEGORICAL_FEATURES = ["sex"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _slope(s: pd.Series) -> float:
    if len(s) < 2:
        return 0.0
    return float((s.iloc[-1] - s.iloc[0]) / (len(s) - 1))


def aggregate_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["patient_id", "date"])
    data["sleep_deviation"] = (data["sleep_hours"] - data["baseline_sleep"]).abs()

    records = []
    for pid, g in data.groupby("patient_id", sort=False):
        tail = g.tail(7)
        records.append(
            {
                "patient_id": pid,
                "age": int(g["age"].iloc[0]),
                "sex": str(g["sex"].iloc[0]),
                "mood_score_mean": tail["mood_score"].mean(),
                "mood_score_std": tail["mood_score"].std(ddof=0),
                "mood_score_last": tail["mood_score"].iloc[-1],
                "sleep_hours_mean": tail["sleep_hours"].mean(),
                "sleep_hours_std": tail["sleep_hours"].std(ddof=0),
                "sleep_deviation_mean": tail["sleep_deviation"].mean(),
                "stress_level_mean": tail["stress_level"].mean(),
                "stress_level_last": tail["stress_level"].iloc[-1],
                "activity_minutes_mean": tail["activity_minutes"].mean(),
                "medication_adherence_mean": tail["medication_adherence"].mean(),
                "speech_rate_mean": tail["speech_rate"].mean(),
                "pause_ratio_mean": tail["pause_ratio"].mean(),
                "sentiment_score_mean": tail["sentiment_score"].mean(),
                "mood_trend": _slope(tail["mood_score"]),
                "stress_trend": _slope(tail["stress_level"]),
                "mood_state": str(tail["mood_state"].iloc[-1]),
                "deterioration_7d": int(tail["deterioration_7d"].iloc[-1]),
            }
        )
    return pd.DataFrame(records)

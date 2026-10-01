from __future__ import annotations

import pandas as pd

REFERENCE = {
    "sleep_deviation_mean": 0.7,
    "stress_level_mean": 5.0,
    "medication_adherence_mean": 0.8,
    "activity_minutes_mean": 35.0,
    "sentiment_score_mean": 0.0,
}


def human_explanation(row: pd.Series | dict) -> list[str]:
    r = dict(row)
    factors: list[tuple[float, str]] = []
    factors.append(
        (
            max(0, r.get("sleep_deviation_mean", 0) - REFERENCE["sleep_deviation_mean"]),
            "sleep differed substantially from the personal baseline",
        )
    )
    factors.append(
        (
            max(0, r.get("stress_level_mean", 0) - REFERENCE["stress_level_mean"]),
            "recent stress was elevated",
        )
    )
    factors.append(
        (
            max(0, REFERENCE["medication_adherence_mean"] - r.get("medication_adherence_mean", 1))
            * 4,
            "medication adherence was below the reference level",
        )
    )
    factors.append(
        (
            max(0, REFERENCE["activity_minutes_mean"] - r.get("activity_minutes_mean", 100)) / 20,
            "recent physical activity was reduced",
        )
    )
    factors.append(
        (max(0, -r.get("sentiment_score_mean", 0)), "speech-derived sentiment was more negative")
    )
    selected = [text for score, text in sorted(factors, reverse=True) if score > 0.05][:3]
    return selected or ["no observed pattern exceeded the illustrative reference thresholds"]

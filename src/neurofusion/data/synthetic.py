from __future__ import annotations
import numpy as np
import pandas as pd

STATE_NAMES = np.array(["depressive", "stable", "elevated"])


def generate_longitudinal_data(n_patients: int = 800, days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate non-trivial synthetic longitudinal mental-health research data."""
    if n_patients < 30:
        raise ValueError("n_patients must be at least 30")
    if days < 7:
        raise ValueError("days must be at least 7")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    start = pd.Timestamp("2026-01-01")

    for patient_idx in range(n_patients):
        patient_id = f"SYN-{patient_idx:05d}"
        age = int(rng.integers(18, 76))
        sex = rng.choice(["female", "male"], p=[0.52, 0.48])
        baseline_sleep = float(np.clip(rng.normal(7.1, 0.8), 4.5, 9.5))
        baseline_stress = float(np.clip(rng.normal(4.5, 1.4), 1, 9))
        vulnerability = float(rng.beta(2.2, 3.5))
        latent = float(rng.normal(0, 0.45))

        for day in range(days):
            shock = rng.normal(0, 0.22) + (rng.random() < 0.04) * rng.normal(0, 1.2)
            latent = 0.78 * latent + shock + 0.05 * (baseline_stress - 4.5)
            stress = float(np.clip(baseline_stress + 1.45 * abs(latent) + rng.normal(0, 0.7), 0, 10))
            adherence = float(np.clip(0.93 - 0.16 * abs(latent) - 0.03 * stress + rng.normal(0, 0.08), 0, 1))
            sleep = float(np.clip(baseline_sleep - 0.85 * max(latent, 0) - 0.55 * max(-latent, 0) + rng.normal(0, 0.45), 2.5, 11))
            activity = float(np.clip(48 + 22 * latent - 17 * max(-latent, 0) - 2.8 * stress + rng.normal(0, 10), 0, 150))
            speech_rate = float(np.clip(125 + 34 * latent + rng.normal(0, 9), 65, 220))
            pause_ratio = float(np.clip(0.18 - 0.055 * latent + 0.035 * max(-latent, 0) + rng.normal(0, 0.025), 0.03, 0.45))
            sentiment = float(np.clip(0.05 + 0.55 * latent - 0.1 * vulnerability + rng.normal(0, 0.16), -1, 1))
            mood_score = float(np.clip(5 + 2.0 * latent - 0.22 * stress + 0.8 * adherence + rng.normal(0, 0.55), 0, 10))

            if latent < -0.43:
                state = "depressive"
            elif latent > 0.52:
                state = "elevated"
            else:
                state = "stable"

            # Future deterioration is probabilistic and depends on recent state drivers.
            logit = -4.0 + 1.4 * vulnerability + 0.32 * stress + 1.4 * (1 - adherence) + 0.32 * abs(sleep - baseline_sleep) + 0.75 * abs(latent)
            deterioration_prob = 1 / (1 + np.exp(-logit))
            risk_7d = int(rng.random() < deterioration_prob)

            rows.append({
                "patient_id": patient_id,
                "date": start + pd.Timedelta(days=day),
                "age": age,
                "sex": sex,
                "baseline_sleep": round(baseline_sleep, 3),
                "vulnerability": round(vulnerability, 4),
                "mood_score": round(mood_score, 3),
                "sleep_hours": round(sleep, 3),
                "stress_level": round(stress, 3),
                "activity_minutes": round(activity, 3),
                "medication_adherence": round(adherence, 4),
                "speech_rate": round(speech_rate, 3),
                "pause_ratio": round(pause_ratio, 4),
                "sentiment_score": round(sentiment, 4),
                "mood_state": state,
                "deterioration_7d": risk_7d,
            })

    return pd.DataFrame(rows)

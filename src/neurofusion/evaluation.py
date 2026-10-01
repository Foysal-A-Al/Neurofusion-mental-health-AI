from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values, n_boot: int = 500, confidence: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("values must be a nonempty finite one-dimensional array")
    if n_boot < 1 or not 0 < confidence < 1:
        raise ValueError("n_boot must be positive and confidence strictly between zero and one")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - confidence) / 2
    return float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def uncertainty_interval(probability: float, effective_n: int = 30) -> tuple[float, float]:
    """Approximate interval around a probability for transparent demo use."""
    if not np.isfinite(probability) or not 0 <= probability <= 1 or effective_n < 1:
        raise ValueError("probability must be in [0,1] and effective_n must be positive")
    p = float(probability)
    se = np.sqrt(max(p * (1 - p), 1e-6) / effective_n)
    return float(max(0, p - 1.96 * se)), float(min(1, p + 1.96 * se))

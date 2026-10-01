# Data Card

The included generator creates synthetic patient-level longitudinal observations. It models relationships among sleep, stress, activity, adherence, speech-derived features, mood state, and a probabilistic deterioration label. No real patient records are distributed.

For each day, `deterioration_7d` is a Bernoulli draw whose logit is `-4 + 1.4*vulnerability + 0.32*stress + 1.4*(1-adherence) + 0.32*abs(sleep-baseline_sleep) + 0.75*abs(latent)`. There is no simulated subsequent seven-day event trajectory behind this label. Mood-state thresholds and the full generator are inspectable in [synthetic.py](../src/neurofusion/data/synthetic.py).

Feature aggregation uses the last seven available observations per patient, then predicts the final observation's state and synthetic risk label. Holdout splitting is patient-level. Features must be finite, dates valid and unique per patient, and demographics/baseline sleep consistent within the window.

Synthetic generation can introduce unrealistic assumptions and hidden shortcuts. Results must not be generalized to clinical populations.

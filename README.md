# NeuroFusion — Mental-Health ML Research Prototype

[![Tests](https://github.com/Foysal-A-Al/Neurofusion-mental-health-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/Foysal-A-Al/Neurofusion-mental-health-AI/actions/workflows/tests.yml)
[![Docker](https://github.com/Foysal-A-Al/Neurofusion-mental-health-AI/actions/workflows/docker-image.yml/badge.svg)](https://github.com/Foysal-A-Al/Neurofusion-mental-health-AI/actions/workflows/docker-image.yml)

A reproducible synthetic-data workflow for patient-level mood-state classification and an illustrative deterioration-risk score, with calibrated logistic-regression baselines, a FastAPI service, and a Streamlit dashboard.

**Research and education only.** Software tests and synthetic metrics do not establish diagnostic accuracy, clinical utility, or reliable future-event forecasting. This is not a medical device or an emergency service.

## What is implemented

| Component | Behavior |
|---|---|
| Synthetic data | Deterministic sleep, mood, stress, adherence, activity, and speech-derived numerical features |
| Features | Last seven available observations aggregated into one row per patient |
| Models | Mood-state classifier (`depressive`, `stable`, `elevated`) and binary synthetic-risk classifier |
| Evaluation | Patient-level holdout, accuracy, macro F1, ROC AUC, and Brier score |
| Calibration | Sigmoid calibration with preprocessing fitted inside each internal fold |
| Inference | Probabilities, confidence threshold, observed-pattern summaries, illustrative probability band |
| Interfaces | REST prediction endpoint and CSV/example dashboard |

The simulator samples `deterioration_7d` from a Bernoulli probability derived from same-day drivers. It **does not simulate or measure a subsequent seven-day event window**. The API field `seven_day_deterioration_risk` is retained for compatibility but describes that synthetic label, not a validated forecast. Speech/wearable inputs are numerical proxies, not raw audio or sensor processing.

## Quick start

Use Python 3.10+ and run setup from the repository root:

```bash
git clone https://github.com/Foysal-A-Al/Neurofusion-mental-health-AI.git
cd Neurofusion-mental-health-AI
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS:

```bash
python -m pip install -e ".[dev]"
python scripts/run_demo.py
```

The demo generates 300 patients × 21 days, trains both models, saves artifacts, and prints evaluation and an example prediction. It rebuilds local demo data/models on each run. Runtime depends on hardware. The source and tests are ordinary repository files; no ZIP extraction is required.

For a larger dataset or separate steps:

```bash
python scripts/generate_data.py --patients 800 --days 30 --seed 42
python scripts/train.py
```

## API and dashboard

After training, start these in two terminals with the same environment activated:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```bash
streamlit run dashboard/app.py
```

- API documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/health
- Dashboard: http://localhost:8501

`/health` reports process liveness and whether both model files exist. Without trained artifacts, predictions return HTTP 503. Restart the API after retraining to reload its cached models.

Portable Python API example:

```python
import json
from pathlib import Path
import httpx

payload = json.loads(Path("examples/prediction_request.json").read_text())
response = httpx.post("http://127.0.0.1:8000/api/v1/predict", json=payload)
response.raise_for_status()
print(response.json())
```

Or use `curl` (PowerShell users may need `curl.exe`):

```bash
curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d @examples/prediction_request.json
```

## Observation schema

Provide **3–60 observations for exactly one patient**, with unique dates. Age, sex, and baseline sleep must be consistent in the window. The API and dashboard reject mixed-patient input instead of silently selecting the first patient. Invalid request payloads return HTTP 422.

| Field | Type/range |
|---|---|
| `patient_id` | Nonempty string; API default is `API-PATIENT` |
| `date` | ISO date, e.g. `2026-06-01` |
| `age`, `sex`, `baseline_sleep` | Age 18–100; nonempty category; baseline sleep `(0,16]` hours |
| `mood_score`, `stress_level` | Finite numbers in `[0,10]` |
| `sleep_hours`, `activity_minutes` | `[0,24]` hours; `[0,500]` minutes |
| `medication_adherence`, `pause_ratio` | `[0,1]` |
| `speech_rate`, `sentiment_score` | `[40,300]`; `[-1,1]` |

Dashboard CSV columns use the same field names. Feature aggregation sorts dates and uses the last seven available records; these need not represent seven consecutive days.

## Interpreting output

- `predicted_state` becomes `uncertain` below the configured confidence threshold (default 0.55).
- `state_probabilities` are calibrated baseline model outputs, not clinical probabilities validated on patients.
- `illustrative_risk_interval` is a heuristic band assuming effective sample size 30. It is **not a validated individual 95% confidence interval**. The inaccurate former field name `risk_interval_95` has been removed.
- `top_factors` are reference-based observed patterns, not SHAP values, causal explanations, or model-derived attribution.

See [model card](reports/model_card.md), [data card](reports/data_card.md), and [ethical considerations](reports/ethical_considerations.md).

## Docker

```bash
docker compose up --build
```

Compose first runs a trainer, then starts the API and dashboard when training succeeds. Data and artifacts persist in local bind mounts. The default services bind only to localhost. Re-running Compose retrains the synthetic demo; do not use this demo setup with irreplaceable custom artifacts.

```bash
docker compose down
```

For details and troubleshooting, see [docs/operations.md](docs/operations.md).

## Configuration and reproducibility

`configs/config.yaml` controls generator defaults, holdout size, seed, and training artifact directory. Both generation and training accept `--config FILE`; their data/output paths are relative to the working directory. Run them from the repository root.

Serving recognizes `NEUROFUSION_MODEL_DIR` and `NEUROFUSION_ABSTAIN_THRESHOLD`. The `.env.example` file documents these variables; Python does not automatically load an `.env` file. If you change the training artifact directory, set the serving variable to that same path. Only load artifacts you generated or trust: joblib/pickle files can execute code.

## Tests and contributions

```bash
python -m pytest
ruff check .
```

CI checks tests, lint, a training/prediction demo, and Docker startup/prediction. See [CONTRIBUTING.md](CONTRIBUTING.md). The project is licensed under [MIT](LICENSE).

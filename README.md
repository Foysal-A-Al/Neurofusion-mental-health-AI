# NeuroFusion

NeuroFusion is a research-grade, uncertainty-aware multimodal machine-learning platform for longitudinal mental-health state monitoring and seven-day risk forecasting.

> **Important:** This repository is an educational and research prototype. It is not a medical device, does not provide diagnosis or treatment advice, and must not be used for emergencies or clinical decision-making without appropriate validation and governance.

## Features

- Reproducible synthetic longitudinal patient-data generator
- Leakage-aware patient-level train/test split
- Classical baseline models and calibrated probabilities
- Mood-state classification: depressive, stable, elevated
- Seven-day deterioration-risk prediction
- Bootstrap uncertainty intervals and abstention logic
- Global feature importance and patient-level explanations
- FastAPI REST service
- Streamlit dashboard
- Data-quality checks, model card, data card, ethics documentation
- Unit tests, lint configuration, Docker, and GitHub Actions

## Architecture

```text
Synthetic longitudinal records
        ↓
Validation and feature engineering
        ↓
Patient-level aggregation and splitting
        ↓
Calibrated classification pipelines
        ↓
Metrics, calibration, bootstrap uncertainty
        ↓
FastAPI + Streamlit dashboard
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -e .[dev]
```

### 2. Generate data and train models

```bash
python scripts/generate_data.py --patients 800 --days 30
python scripts/train.py
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs`.

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

## One-command demo

```bash
python scripts/run_demo.py
```

This creates synthetic data, trains the models, prints evaluation results, and generates one example prediction.

## Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## API example

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @examples/prediction_request.json
```

## Repository structure

```text
api/                 FastAPI application
dashboard/           Streamlit interface
src/neurofusion/     Data, model, evaluation, and inference code
scripts/             Reproducible CLI workflows
tests/               Unit and integration tests
reports/             Model, data, and ethics documentation
examples/            Example API payloads
artifacts/            Trained model outputs (generated)
data/                 Synthetic and processed data (generated)
```

## Reproducibility

The default random seed is `42`. Configuration is stored in `configs/config.yaml`. Generated datasets and trained model artifacts are intentionally excluded from Git by default, but their directories are retained.

## Limitations

- Data are synthetic and cannot establish clinical validity.
- The model may encode assumptions from the simulator.
- Fairness analysis is illustrative rather than evidentiary.
- Speech and wearable inputs are represented by derived numerical features, not raw signals.
- Performance should not be extrapolated to real patient populations.



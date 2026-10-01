import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api.main as api_module
from api.schemas import PredictionRequest
from neurofusion.data.synthetic import generate_longitudinal_data
from neurofusion.data.validation import validate_longitudinal_frame
from neurofusion.features import aggregate_patient_features
from neurofusion.inference import NeuroFusionPredictor
from neurofusion.modeling import train_models


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    raw = generate_longitudinal_data(120, 12, seed=42)
    features = aggregate_patient_features(raw)
    result = train_models(features)
    directory = tmp_path_factory.mktemp("trained")
    joblib.dump(result.state_model, directory / "state_model.joblib")
    joblib.dump(result.risk_model, directory / "risk_model.joblib")
    return directory, result


@pytest.fixture
def example():
    return json.loads(
        (Path(__file__).resolve().parents[1] / "examples/prediction_request.json").read_text()
    )


def test_calibration_preprocessing_is_fitted_within_each_fold(trained):
    _, result = trained
    for calibrated in result.state_model.calibrated_classifiers_:
        estimator = calibrated.estimator
        scaler = estimator.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
        assert scaler.n_samples_seen_ < result.metrics["n_train"]


def test_full_artifact_and_api_prediction(trained, example, monkeypatch):
    directory, _ = trained
    monkeypatch.setenv("NEUROFUSION_MODEL_DIR", str(directory))
    monkeypatch.setattr(api_module, "_predictor", None)
    client = TestClient(api_module.app)
    assert client.get("/health").json()["models_available"] is True
    response = client.post("/api/v1/predict", json=example)
    assert response.status_code == 200
    output = response.json()
    assert np.isclose(sum(output["state_probabilities"].values()), 1)
    assert 0 <= output["seven_day_deterioration_risk"] <= 1
    assert "not a validated" in output["uncertainty_note"]
    assert "risk_interval_95" not in output


def test_untrained_api_is_alive_but_prediction_returns_503(tmp_path, example, monkeypatch):
    monkeypatch.setenv("NEUROFUSION_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(api_module, "_predictor", None)
    client = TestClient(api_module.app)
    assert client.get("/health").json()["models_available"] is False
    assert client.post("/api/v1/predict", json=example).status_code == 503


@pytest.mark.parametrize("change", ["patient", "date", "age", "infinite"])
def test_invalid_api_windows_are_rejected(example, change):
    observations = example["observations"]
    if change == "patient":
        observations[0]["patient_id"] = "OTHER"
    if change == "date":
        observations[0]["date"] = observations[1]["date"]
    if change == "age":
        observations[0]["age"] = 70
    if change == "infinite":
        observations[0]["speech_rate"] = float("inf")
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(example)


def test_predictor_rejects_multiple_patients_directly(trained, example):
    example["observations"][0]["patient_id"] = "OTHER"
    with pytest.raises(ValueError, match="exactly one patient"):
        NeuroFusionPredictor(trained[0]).predict(example["observations"])


def test_validation_requires_baseline_and_unique_dates():
    frame = generate_longitudinal_data(30, 7)
    with pytest.raises(ValueError, match="baseline_sleep"):
        validate_longitudinal_frame(frame.drop(columns="baseline_sleep"))
    frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_longitudinal_frame(frame)


def test_dashboard_can_load_and_run_analysis(trained, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEUROFUSION_MODEL_DIR", str(trained[0]))
    path = Path(__file__).resolve().parents[1] / "dashboard/app.py"
    app = AppTest.from_file(path).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3

from neurofusion.data.synthetic import generate_longitudinal_data
from neurofusion.features import aggregate_patient_features
from neurofusion.modeling import train_models


def test_training_pipeline_runs():
    raw = generate_longitudinal_data(90, 12, seed=3)
    features = aggregate_patient_features(raw)
    result = train_models(features, test_size=0.2, seed=3)
    assert 0 <= result.metrics["state_macro_f1"] <= 1
    assert 0 <= result.metrics["risk_roc_auc"] <= 1

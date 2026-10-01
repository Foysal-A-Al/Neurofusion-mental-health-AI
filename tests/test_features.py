from neurofusion.data.synthetic import generate_longitudinal_data
from neurofusion.features import FEATURE_COLUMNS, aggregate_patient_features


def test_feature_aggregation():
    raw = generate_longitudinal_data(35, 10, seed=2)
    features = aggregate_patient_features(raw)
    assert len(features) == 35
    assert all(c in features.columns for c in FEATURE_COLUMNS)
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0

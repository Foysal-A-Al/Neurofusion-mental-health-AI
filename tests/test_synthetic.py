from neurofusion.data.synthetic import generate_longitudinal_data
from neurofusion.data.validation import validate_longitudinal_frame


def test_generator_shape_and_validation():
    df = generate_longitudinal_data(40, 10, seed=1)
    assert len(df) == 400
    assert df.patient_id.nunique() == 40
    validate_longitudinal_frame(df)

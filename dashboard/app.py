from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import ValidationError

from api.schemas import PredictionRequest
from neurofusion.inference import NeuroFusionPredictor

st.set_page_config(page_title="NeuroFusion", page_icon="🧠", layout="wide")
st.title("🧠 NeuroFusion")
st.caption("Uncertainty-aware mental-health AI research dashboard — not a medical device")

artifact_dir = Path(os.environ.get("NEUROFUSION_MODEL_DIR", "artifacts"))
if not all((artifact_dir / name).is_file() for name in ["state_model.joblib", "risk_model.joblib"]):
    st.error(
        "Models are not available. Run `python scripts/generate_data.py` and `python scripts/train.py`."
    )
    st.stop()


@st.cache_resource
def load_predictor(directory, state_mtime, risk_mtime):
    return NeuroFusionPredictor(directory)


predictor = load_predictor(
    str(artifact_dir),
    (artifact_dir / "state_model.joblib").stat().st_mtime,
    (artifact_dir / "risk_model.joblib").stat().st_mtime,
)
example_path = Path("examples/prediction_request.json")
example = json.loads(example_path.read_text())

uploaded = st.file_uploader("Upload longitudinal observations as CSV", type=["csv"])
try:
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.DataFrame(example["observations"])
    request = PredictionRequest.model_validate({"observations": df.to_dict(orient="records")})
    df = pd.DataFrame([row.model_dump(mode="json") for row in request.observations]).sort_values(
        "date"
    )
except (ValueError, ValidationError) as exc:
    st.error(f"Invalid observation file: {exc}")
    st.stop()

st.subheader("Longitudinal input")
st.dataframe(df, use_container_width=True)

if st.button("Run analysis", type="primary"):
    result = predictor.predict(df.to_dict(orient="records"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted state", result["predicted_state"])
    c2.metric("State confidence", f"{result['state_confidence']:.1%}")
    c3.metric("7-day risk", f"{result['seven_day_deterioration_risk']:.1%}")
    st.info(result["prediction_status"].replace("_", " ").title())
    st.write("**Reference-based observed patterns**")
    st.caption(result["explanation_method"])
    for factor in result["top_factors"]:
        st.write(f"- {factor}")
    probs = pd.DataFrame(
        {
            "state": list(result["state_probabilities"].keys()),
            "probability": list(result["state_probabilities"].values()),
        }
    )
    st.plotly_chart(
        px.bar(probs, x="state", y="probability", range_y=[0, 1], title="Mood-state probabilities"),
        use_container_width=True,
    )

st.subheader("Patient timeline")
metric = st.selectbox(
    "Variable",
    ["mood_score", "sleep_hours", "stress_level", "activity_minutes", "medication_adherence"],
)
fig = px.line(
    df, x="date", y=metric, markers=True, title=f"{metric.replace('_', ' ').title()} over time"
)
st.plotly_chart(fig, use_container_width=True)

if (artifact_dir / "metrics.json").exists():
    st.subheader("Model evaluation")
    st.json(json.loads((artifact_dir / "metrics.json").read_text()))

st.warning(
    "The results are produced from synthetic-data models and must not be interpreted as clinical conclusions."
)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from neurofusion.config import load_config
from neurofusion.data.validation import validate_longitudinal_frame
from neurofusion.features import aggregate_patient_features
from neurofusion.modeling import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train both calibrated synthetic-data models")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data_path = Path(config["data"]["output"])
    if not data_path.exists():
        raise FileNotFoundError("Run python scripts/generate_data.py first")
    raw = pd.read_csv(data_path)
    validate_longitudinal_frame(raw)
    features = aggregate_patient_features(raw)
    result = train_models(
        features, test_size=config["training"]["test_size"], seed=config["training"]["random_state"]
    )
    artifact_dir = Path(config["artifacts"]["directory"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.state_model, artifact_dir / "state_model.joblib")
    joblib.dump(result.risk_model, artifact_dir / "risk_model.joblib")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    (artifact_dir / "test_patients.json").write_text(
        json.dumps(result.test_patient_ids, indent=2), encoding="utf-8"
    )
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    features.to_csv("data/processed/patient_features.csv", index=False)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


@dataclass
class TrainingResult:
    state_model: CalibratedClassifierCV
    risk_model: CalibratedClassifierCV
    metrics: dict
    test_patient_ids: list[str]


def build_pipeline(multiclass: bool = False) -> CalibratedClassifierCV:
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    base = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)
    # Calibrate the entire pipeline so each calibration fold fits its own preprocessing.
    estimator = Pipeline([("preprocess", preprocessor), ("model", base)])
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=3)


def train_models(features: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> TrainingResult:
    if features.empty or features.patient_id.isna().any() or features.patient_id.duplicated().any():
        raise ValueError("Training requires one nonempty feature row per unique patient.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between zero and one.")
    if features.mood_state.nunique() < 2 or not features.deterioration_7d.isin([0, 1]).all():
        raise ValueError("Training requires multiple mood states and binary deterioration labels.")
    idx_train, idx_test = train_test_split(
        np.arange(len(features)),
        test_size=test_size,
        random_state=seed,
        stratify=features["mood_state"],
    )
    train, test = features.iloc[idx_train], features.iloc[idx_test]
    for target in ["mood_state", "deterioration_7d"]:
        counts = train[target].value_counts()
        if len(counts) < 2 or counts.min() < 3:
            raise ValueError(
                f"Training target {target} needs at least three examples per class for calibration."
            )
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]

    state_model = build_pipeline(multiclass=True)
    risk_model = build_pipeline(multiclass=False)
    state_model.fit(X_train, train["mood_state"])
    risk_model.fit(X_train, train["deterioration_7d"])

    state_pred = state_model.predict(X_test)
    risk_pred = risk_model.predict(X_test)
    risk_prob = risk_model.predict_proba(X_test)[:, 1]
    metrics = {
        "state_accuracy": float(accuracy_score(test["mood_state"], state_pred)),
        "state_balanced_accuracy": float(balanced_accuracy_score(test["mood_state"], state_pred)),
        "state_macro_f1": float(f1_score(test["mood_state"], state_pred, average="macro")),
        "risk_accuracy": float(accuracy_score(test["deterioration_7d"], risk_pred)),
        "risk_f1": float(f1_score(test["deterioration_7d"], risk_pred, zero_division=0)),
        "risk_roc_auc": float(roc_auc_score(test["deterioration_7d"], risk_prob))
        if test["deterioration_7d"].nunique() == 2
        else None,
        "risk_brier": float(brier_score_loss(test["deterioration_7d"], risk_prob)),
        "n_train": len(train),
        "n_test": len(test),
    }
    return TrainingResult(state_model, risk_model, metrics, test["patient_id"].tolist())

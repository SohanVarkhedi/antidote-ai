"""
Ensemble Models — Antidote AI
Trains RF, LogisticRegression, and GradientBoosting with soft-voting ensemble.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_ensemble(cleaned_df: pd.DataFrame, target_column: str = "target") -> dict:
    """
    Train an ensemble of three classifiers using soft voting.

    Models
    ------
    - RandomForestClassifier
    - LogisticRegression
    - GradientBoostingClassifier

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned dataset with features and a target column.
    target_column : str
        Name of the label column.

    Returns
    -------
    dict with keys:
        accuracy       – ensemble test accuracy
        n_samples      – training sample count
        n_features     – feature count
        model_path     – path to saved ensemble
        individual     – dict of per-model accuracies
    """
    if target_column not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = cleaned_df.drop(columns=[target_column]).select_dtypes(include=[np.number])
    y = cleaned_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for LogisticRegression stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Individual estimators ─────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
    )
    lr = LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=120, max_depth=6, random_state=42
    )

    # ── Soft-voting ensemble ──────────────────────────────────
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("lr", lr), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )

    ensemble.fit(X_train_scaled, y_train)

    y_pred = ensemble.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))

    # Individual accuracies
    individual = {}
    for name, estimator in ensemble.named_estimators_.items():
        ind_pred = estimator.predict(X_test_scaled)
        individual[name] = round(float(accuracy_score(y_test, ind_pred)), 4)

    # ── Save artefacts ────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    joblib.dump(ensemble, model_path)

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    meta_path = os.path.join(MODEL_DIR, "ensemble_meta.pkl")
    meta = {
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_column": target_column,
    }
    joblib.dump(meta, meta_path)

    return {
        "accuracy": round(acc, 4),
        "n_samples": int(len(X_train)),
        "n_features": int(X.shape[1]),
        "model_path": model_path,
        "individual": individual,
    }

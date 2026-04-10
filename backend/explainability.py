"""
Explainability Module — Antidote AI
Provides feature-level deviation explanations for inference decisions.
"""

import numpy as np
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def explain(
    x: list | np.ndarray,
    training_data: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    threshold_std: float = 2.0,
) -> list[str]:
    """
    Explain which features deviate significantly from training distribution.

    For each feature, checks if the input value is beyond mean ± threshold_std * std.

    Parameters
    ----------
    x : array-like of shape (n_features,)
        Single input sample.
    training_data : ndarray or None
        Training features. If None, attempts to load from saved drift model.
    feature_names : list[str] or None
        Optional human-readable feature names.
    threshold_std : float
        Number of standard deviations to consider as "high deviation".

    Returns
    -------
    list[str]  – Explanation strings like "Feature 2 deviation high".
    """
    x = np.array(x, dtype=float).flatten()

    # Try to load training data if not provided
    if training_data is None:
        drift_path = os.path.join(MODEL_DIR, "drift_model.pkl")
        if os.path.exists(drift_path):
            training_data = joblib.load(drift_path)
        else:
            return ["No training distribution available for explanation."]

    explanations = []
    n_features = min(len(x), training_data.shape[1])

    for i in range(n_features):
        col = training_data[:, i]
        mean = float(np.mean(col))
        std = float(np.std(col))

        if std == 0:
            # Constant feature — flag if different
            if x[i] != mean:
                name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}"
                explanations.append(f"{name} deviation high (constant feature altered)")
            continue

        z_score = abs((x[i] - mean) / std)

        if z_score > threshold_std:
            name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}"
            direction = "high" if x[i] > mean else "low"
            explanations.append(f"{name} deviation {direction}")

    if not explanations:
        explanations.append("All features within expected range")

    return explanations

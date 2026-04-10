"""
Drift Detector — Antidote AI
Compares training distribution vs new input using the Kolmogorov-Smirnov test.
"""

import numpy as np
from scipy.stats import ks_2samp
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class DriftDetector:
    """Detects distribution drift between training data and new inputs."""

    def __init__(self):
        self.training_data = None  # ndarray (n_samples, n_features)
        self._fitted = False

    # ── Fit on clean training data ────────────────────────────────
    def fit(self, X_train: np.ndarray):
        """
        Store the training data distribution for future comparisons.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
        """
        self.training_data = np.array(X_train, dtype=float)
        self._fitted = True

    # ── Detect drift for a single input ──────────────────────────
    def detect(self, x: list | np.ndarray, p_threshold: float = 0.05) -> dict:
        """
        Run KS test per feature comparing the input against training distribution.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Single input sample.
        p_threshold : float
            P-value threshold below which drift is flagged.

        Returns
        -------
        dict  {"drift_flag": bool, "drift_score": float, "drifted_features": list[int]}
        """
        if not self._fitted:
            return {"drift_flag": False, "drift_score": 0.0, "drifted_features": []}

        x = np.array(x, dtype=float).flatten()
        n_features = min(len(x), self.training_data.shape[1])

        drifted_features = []
        p_values = []

        for i in range(n_features):
            train_col = self.training_data[:, i]
            # Create a small sample centred on the input value for KS comparison
            synthetic = np.full(max(10, len(train_col) // 10), x[i])
            stat, p_value = ks_2samp(train_col, synthetic)
            p_values.append(p_value)
            if p_value < p_threshold:
                drifted_features.append(i)

        # Drift score: average (1 - p_value) across features, scaled to 0–100
        drift_score = float(np.mean([1.0 - p for p in p_values]) * 100) if p_values else 0.0

        return {
            "drift_flag": len(drifted_features) > 0,
            "drift_score": round(drift_score, 2),
            "drifted_features": drifted_features,
        }

    # ── Persistence ──────────────────────────────────────────────
    def save(self, path: str | None = None):
        path = path or os.path.join(MODEL_DIR, "drift_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.training_data, path)

    def load(self, path: str | None = None) -> bool:
        path = path or os.path.join(MODEL_DIR, "drift_model.pkl")
        if os.path.exists(path):
            self.training_data = joblib.load(path)
            self._fitted = True
            return True
        return False

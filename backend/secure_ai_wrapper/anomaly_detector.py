import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Detects anomalous inputs using Isolation Forest.
    """

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.fitted = False

    def fit(self, X):
        X = np.array(X)
        self.model.fit(X)
        self.fitted = True

    def check(self, X):
        if not self.fitted:
            # MVP behaviour: fit on first batch
            self.fit(X)

        preds = self.model.predict(X)

        # -1 = anomaly, 1 = normal
        return preds[0] == -1
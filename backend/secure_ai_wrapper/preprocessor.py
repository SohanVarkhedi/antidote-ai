import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Handles input normalization and simple noise clipping.
    """

    def __init__(self, clip_range: float = 5.0):
        self.scaler = StandardScaler()
        self.clip_range = clip_range
        self.fitted = False

    def fit(self, X):
        X = np.array(X)
        self.scaler.fit(X)
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            # fallback: fit on first batch (MVP behaviour)
            self.fit(X)

        X_scaled = self.scaler.transform(X)

        # clip extreme noise
        X_clipped = np.clip(
            X_scaled,
            -self.clip_range,
            self.clip_range
        )

        return X_clipped
import numpy as np


class InputValidator:
    """
    Validates model input before inference.
    """

    def __init__(self, expected_features: int = None):
        self.expected_features = expected_features

    def validate(self, X):
        # Convert to numpy
        X = np.array(X)

        # Check NaN
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")

        # Check Inf
        if np.isinf(X).any():
            raise ValueError("Input contains Inf values")

        # Reshape single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Feature count validation
        if self.expected_features is not None:
            if X.shape[1] != self.expected_features:
                raise ValueError(
                    f"Feature count mismatch. Expected {self.expected_features}, got {X.shape[1]}"
                )

        return X
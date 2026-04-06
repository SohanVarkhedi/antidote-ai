from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from secure_ai_wrapper.wrapper import SecureAIWrapper
from fastapi.middleware.cors import CORSMiddleware

# ---------- create app ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------- load demo model ----------
X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

secure_model = SecureAIWrapper(model, X_train=X)


# ---------- request schema ----------
class PredictRequest(BaseModel):
    input: list


# ---------- prediction route ----------
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # convert input to numpy 2D array
        X = np.array(data.input).reshape(1, -1)

        result = secure_model.predict(X)

        return {
        "prediction": int(result.prediction),
        "confidence": float(result.confidence) if result.confidence is not None else None,
            "is_anomaly": bool(result.is_anomaly)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
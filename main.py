from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib

app = FastAPI(
    title="Iris Classifier API",
    description="FastAPI serving an Iris classifier (LogisticRegression + StandardScaler)",
    version="1.0.0"
)

_model = None
_meta = None

class PredictionInput(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float] = None

@app.on_event("startup")
def load_model():
    global _model, _meta
    try:
        artifact = joblib.load("model.pkl")
        _model = artifact["pipeline"]
        _meta = artifact["metadata"]
    except Exception as e:
        # Defer failure to request-time with a clear message
        _model = None
        _meta = None

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train the model and ensure model.pkl is present.")
    try:
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(features)[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
        else:
            idx = int(_model.predict(features)[0])
            conf = None
        label = _meta["classes"][idx]
        return PredictionOutput(prediction=label, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    if _meta is None:
        raise HTTPException(status_code=500, detail="Model metadata not available")
    return _meta
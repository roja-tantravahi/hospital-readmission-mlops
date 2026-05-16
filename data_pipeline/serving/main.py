from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Predicts 30-day hospital readmission risk for patients",
    version="1.0.0"
)

# Load model and scaler at startup
MODEL_PATH = os.getenv("MODEL_PATH", "training/models/random_forest.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "training/models/scaler.pkl")

model = None
scaler = None


@app.on_event("startup")
async def load_model():
    """Load model and scaler on startup."""
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise RuntimeError("Model file not found")


class PatientFeatures(BaseModel):
    """Input schema for patient features."""
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender_encoded: int = Field(..., ge=0, le=1, description="Gender: 1=Male, 0=Female")
    length_of_stay: int = Field(..., ge=0, description="Length of hospital stay in days")
    num_diagnoses: int = Field(..., ge=0, description="Number of diagnoses")
    num_prescriptions: int = Field(..., ge=0, description="Number of prescriptions")
    is_emergency: int = Field(..., ge=0, le=1, description="Emergency admission: 1=Yes, 0=No")
    prev_admissions: int = Field(..., ge=0, description="Number of previous admissions")


class PredictionResponse(BaseModel):
    """Output schema for prediction."""
    readmission_risk: str
    probability: float
    message: str


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Hospital Readmission API is running"}


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    """Predict 30-day readmission risk."""
    start_time = time.time()

    if model is None:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare input
        features = np.array([[
            patient.age,
            patient.gender_encoded,
            patient.length_of_stay,
            patient.num_diagnoses,
            patient.num_prescriptions,
            patient.is_emergency,
            patient.prev_admissions
        ]])

        # Predict
        probability = model.predict_proba(features)[0][1]
        risk_label = "HIGH" if probability >= 0.5 else "LOW"

        duration = time.time() - start_time
        logger.info(f"Prediction completed in {duration:.3f}s — Risk: {risk_label} ({probability:.3f})")

        return PredictionResponse(
            readmission_risk=risk_label,
            probability=round(float(probability), 3),
            message=f"Patient has {risk_label} risk of 30-day readmission"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

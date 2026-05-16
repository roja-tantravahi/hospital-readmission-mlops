from fastapi.testclient import TestClient
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch
from serving.main import app

client = TestClient(app)


def create_mock_model():
    """Create a simple mock model for testing."""
    X = np.random.rand(100, 7)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    os.makedirs("training/models", exist_ok=True)
    joblib.dump(model, "training/models/random_forest.pkl")


def test_root():
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    """Test detailed health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_valid():
    """Test prediction with valid input."""
    create_mock_model()
    payload = {
        "age": 65,
        "gender_encoded": 1,
        "length_of_stay": 5,
        "num_diagnoses": 3,
        "num_prescriptions": 4,
        "is_emergency": 1,
        "prev_admissions": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "readmission_risk" in data
    assert data["readmission_risk"] in ["HIGH", "LOW"]
    assert "probability" in data


def test_predict_invalid_age():
    """Test prediction with invalid age."""
    payload = {
        "age": 200,
        "gender_encoded": 1,
        "length_of_stay": 5,
        "num_diagnoses": 3,
        "num_prescriptions": 4,
        "is_emergency": 1,
        "prev_admissions": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field():
    """Test prediction with missing field."""
    payload = {
        "age": 65,
        "gender_encoded": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

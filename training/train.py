import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def load_features(path: str) -> pd.DataFrame:
    """Load processed features."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded features: {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Features file not found: {e}")
        raise


def prepare_data(df: pd.DataFrame):
    """Split features and target."""
    feature_cols = [
        "age", "gender_encoded", "length_of_stay",
        "num_diagnoses", "num_prescriptions",
        "is_emergency", "prev_admissions"
    ]
    X = df[feature_cols]
    y = df["readmitted_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    return metrics


def train_model(X_train, X_test, y_train, y_test, model_name: str, model, params: dict):
    """Train a model and log everything to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        # Save model locally
        os.makedirs("training/models", exist_ok=True)
        model_path = f"training/models/{model_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        return metrics


if __name__ == "__main__":
    # Load data
    df = load_features("data/processed/features.csv")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    os.makedirs("training/models", exist_ok=True)
    joblib.dump(scaler, "training/models/scaler.pkl")

    # Set MLflow experiment
    mlflow.set_experiment("hospital-readmission")

    # Experiment 1 — Logistic Regression
    lr_params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    lr_model = LogisticRegression(**lr_params)
    logger.info("Training Logistic Regression...")
    train_model(X_train_scaled, X_test_scaled, y_train, y_test,
                "logistic_regression", lr_model, lr_params)

    # Experiment 2 — Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    rf_model = RandomForestClassifier(**rf_params)
    logger.info("Training Random Forest...")
    train_model(X_train, X_test, y_train, y_test,
                "random_forest", rf_model, rf_params)

    logger.info("All experiments complete. Check MLflow for results.")

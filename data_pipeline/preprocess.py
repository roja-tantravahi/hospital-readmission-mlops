import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def load_raw_data(data_dir: str) -> dict:
    """Load raw MIMIC CSV files."""
    try:
        logger.info("Loading raw MIMIC data...")
        admissions = pd.read_csv(os.path.join(data_dir, "ADMISSIONS.csv"))
        patients = pd.read_csv(os.path.join(data_dir, "PATIENTS.csv"))
        diagnoses = pd.read_csv(os.path.join(data_dir, "DIAGNOSES_ICD.csv"))
        prescriptions = pd.read_csv(os.path.join(data_dir, "PRESCRIPTIONS.csv"))
        logger.info("Raw data loaded successfully.")
        return {
            "admissions": admissions,
            "patients": patients,
            "diagnoses": diagnoses,
            "prescriptions": prescriptions
        }
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise


def engineer_features(data: dict) -> pd.DataFrame:
    """Engineer features for readmission prediction."""
    logger.info("Starting feature engineering...")

    admissions = data["admissions"].copy()
    patients = data["patients"].copy()
    diagnoses = data["diagnoses"].copy()
    prescriptions = data["prescriptions"].copy()

    # Parse dates
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")

    # Length of stay
    admissions["length_of_stay"] = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.days

    # 30-day readmission label
    admissions = admissions.sort_values(["subject_id", "admittime"])
    admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)
    admissions["days_to_readmission"] = (
        admissions["next_admittime"] - admissions["dischtime"]
    ).dt.days
    admissions["readmitted_30d"] = (
        admissions["days_to_readmission"] <= 30
    ).astype(int)

    # Merge patient info
    df = admissions.merge(patients[["subject_id", "dob", "gender"]], on="subject_id", how="left")

    # Safe age calculation — MIMIC uses shifted DOB for privacy
    df["age"] = df["admittime"].dt.year - df["dob"].dt.year
    df["age"] = df["age"].clip(0, 100)

    df["gender_encoded"] = (df["gender"] == "M").astype(int)

    # Number of diagnoses per admission
    diag_count = diagnoses.groupby("hadm_id")["icd9_code"].count().reset_index()
    diag_count.columns = ["hadm_id", "num_diagnoses"]
    df = df.merge(diag_count, on="hadm_id", how="left")
    df["num_diagnoses"] = df["num_diagnoses"].fillna(0)

    # Number of prescriptions per admission
    rx_count = prescriptions.groupby("hadm_id")["drug"].count().reset_index()
    rx_count.columns = ["hadm_id", "num_prescriptions"]
    df = df.merge(rx_count, on="hadm_id", how="left")
    df["num_prescriptions"] = df["num_prescriptions"].fillna(0)

    # Emergency admission flag
    df["is_emergency"] = (df["admission_type"] == "EMERGENCY").astype(int)

    # Previous admissions count
    df["prev_admissions"] = df.groupby("subject_id").cumcount()

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select final features for model."""
    feature_cols = [
        "hadm_id", "subject_id",
        "age", "gender_encoded", "length_of_stay",
        "num_diagnoses", "num_prescriptions",
        "is_emergency", "prev_admissions",
        "readmitted_30d"
    ]
    df_model = df[feature_cols].dropna()
    logger.info(f"Final dataset shape: {df_model.shape}")
    return df_model


if __name__ == "__main__":
    data = load_raw_data("data/raw")
    df = engineer_features(data)
    final_df = get_model_features(df)
    os.makedirs("data/processed", exist_ok=True)
    final_df.to_csv("data/processed/features.csv", index=False)
    logger.info("Saved to data/processed/features.csv")

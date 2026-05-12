from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def create_spark_session():
    """Create Spark session."""
    spark = SparkSession.builder \
        .appName("HospitalReadmission") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    logger.info("Spark session created.")
    return spark


def load_raw_data(spark, data_dir: str) -> dict:
    """Load raw MIMIC CSV files into Spark dataframes."""
    try:
        logger.info("Loading raw data into Spark...")
        admissions = spark.read.csv(
            os.path.join(data_dir, "ADMISSIONS.csv"), header=True, inferSchema=True
        )
        patients = spark.read.csv(
            os.path.join(data_dir, "PATIENTS.csv"), header=True, inferSchema=True
        )
        diagnoses = spark.read.csv(
            os.path.join(data_dir, "DIAGNOSES_ICD.csv"), header=True, inferSchema=True
        )
        prescriptions = spark.read.csv(
            os.path.join(data_dir, "PRESCRIPTIONS.csv"), header=True, inferSchema=True
        )
        logger.info("Raw data loaded into Spark successfully.")
        return {
            "admissions": admissions,
            "patients": patients,
            "diagnoses": diagnoses,
            "prescriptions": prescriptions
        }
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def engineer_features(data: dict):
    """Engineer features using PySpark."""
    logger.info("Starting PySpark feature engineering...")

    admissions = data["admissions"]
    patients = data["patients"]
    diagnoses = data["diagnoses"]
    prescriptions = data["prescriptions"]

    # Parse timestamps
    admissions = admissions.withColumn("admittime", F.to_timestamp("admittime"))
    admissions = admissions.withColumn("dischtime", F.to_timestamp("dischtime"))

    # Length of stay
    admissions = admissions.withColumn(
        "length_of_stay",
        F.datediff(F.col("dischtime"), F.col("admittime"))
    )

    # Emergency flag
    admissions = admissions.withColumn(
        "is_emergency",
        (F.col("admission_type") == "EMERGENCY").cast("int")
    )

    # Number of diagnoses per admission
    diag_count = diagnoses.groupBy("hadm_id").agg(
        F.count("icd9_code").alias("num_diagnoses")
    )

    # Number of prescriptions per admission
    rx_count = prescriptions.groupBy("hadm_id").agg(
        F.count("drug").alias("num_prescriptions")
    )

    # Merge everything
    df = admissions.join(diag_count, on="hadm_id", how="left")
    df = df.join(rx_count, on="hadm_id", how="left")

    # Merge patients
    patients = patients.withColumn("dob", F.to_timestamp("dob"))
    df = df.join(patients.select("subject_id", "dob", "gender"), on="subject_id", how="left")

    # Age calculation
    df = df.withColumn(
        "age",
        (F.year("admittime") - F.year("dob"))
    )
    df = df.withColumn("age", F.when(F.col("age") > 100, 90).otherwise(F.col("age")))

    # Gender encoding
    df = df.withColumn(
        "gender_encoded",
        (F.col("gender") == "M").cast("int")
    )

    # Fill nulls
    df = df.fillna(0, subset=["num_diagnoses", "num_prescriptions"])

    # Select final features
    feature_cols = [
        "hadm_id", "subject_id",
        "age", "gender_encoded", "length_of_stay",
        "num_diagnoses", "num_prescriptions",
        "is_emergency"
    ]
    df = df.select(feature_cols).dropna()

    logger.info(f"PySpark feature engineering complete.")
    return df


if __name__ == "__main__":
    spark = create_spark_session()
    data = load_raw_data(spark, "data/raw")
    df = engineer_features(data)
    df.coalesce(1).write.csv("data/processed/spark_features", header=True, mode="overwrite")
    logger.info("Saved PySpark features to data/processed/spark_features/")
    spark.stop()

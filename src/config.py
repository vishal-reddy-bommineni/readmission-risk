"""Paths and feature lists in one place, so every script agrees on them."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# RAW_OVERRIDE is only used by `make smoke` to point at fake test data
RAW_CSV = ROOT / os.environ.get("RAW_OVERRIDE", "data/raw/diabetic_data.csv")
DB_PATH = ROOT / "data" / "processed" / "readmission.duckdb"
FEATURES_PARQUET = ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DASHBOARD_DIR = ROOT / "dashboard"

SEED = 42
TARGET = "label_readmit_30d"
GROUP_COL = "patient_nbr"          # split by patient, never by encounter
AUDIT_COLS = ["race", "gender", "age_band"]   # used for fairness checks, NOT as model inputs

NUMERIC_FEATURES = [
    "age_mid", "length_of_stay", "num_lab_procedures", "num_procedures", "num_medications",
    "number_diagnoses", "number_outpatient", "number_emergency", "number_inpatient",
    "total_prior_visits", "any_prior_inpatient",
    "cm_heart_failure", "cm_renal", "cm_copd", "cm_ischemic_heart", "cm_cancer",
    "cm_diabetes_complicated", "a1c_tested", "a1c_high", "glucose_tested",
    "any_med_change", "on_diabetes_med", "n_meds_changed", "n_diabetes_meds",
]
CATEGORICAL_FEATURES = [
    "admission_type", "discharge_disposition", "admission_source", "payer_code",
    "medical_specialty", "primary_dx_group", "secondary_dx_group", "insulin_status",
]
# The "simple baseline" a reviewer will ask about: can 3 utilization counts do as well?
BASELINE_FEATURES = ["number_inpatient", "number_emergency", "number_outpatient"]

TOP_K_FRACTION = 0.10   # "if the care team can call the riskiest 10% of discharges..."

"""Generate a FAKE file with the same columns as diabetic_data.csv, only to check that the
pipeline runs end to end. Numbers from this file mean nothing - never report them.

Usage:  python scripts/make_smoke_test_data.py data/raw/diabetic_data.csv
"""
import sys

import numpy as np
import pandas as pd

MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
        "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]


def main(out, n=8000, seed=0):
    r = np.random.default_rng(seed)
    pats = r.integers(1, int(n * 0.75), n)
    inpatient = r.poisson(0.6, n)
    los = r.integers(1, 15, n)
    dx_pool = ["428", "250.83", "414", "486", "584", "V57", "786", "250", "401", "599", "715", "174"]
    logit = -2.6 + 0.35 * inpatient + 0.05 * los + r.normal(0, 0.5, n)
    y = r.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "encounter_id": np.arange(n) + 1000, "patient_nbr": pats,
        "race": r.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Other", "?"], n, p=[.7, .18, .03, .04, .05]),
        "gender": r.choice(["Female", "Male"], n),
        "age": r.choice([f"[{a}-{a + 10})" for a in range(0, 100, 10)], n),
        "weight": "?", "admission_type_id": r.integers(1, 9, n),
        "discharge_disposition_id": r.choice([1, 1, 1, 3, 6, 11, 18], n),
        "admission_source_id": r.choice([1, 7, 7, 4], n), "time_in_hospital": los,
        "payer_code": r.choice(["?", "MC", "HM", "SP"], n),
        "medical_specialty": r.choice(["?", "InternalMedicine", "Cardiology", "Emergency/Trauma"], n),
        "num_lab_procedures": r.integers(1, 100, n), "num_procedures": r.integers(0, 6, n),
        "num_medications": r.integers(1, 60, n), "number_outpatient": r.poisson(0.4, n),
        "number_emergency": r.poisson(0.2, n), "number_inpatient": inpatient,
        "diag_1": r.choice(dx_pool, n), "diag_2": r.choice(dx_pool + ["?"], n), "diag_3": r.choice(dx_pool, n),
        "number_diagnoses": r.integers(1, 16, n),
        "max_glu_serum": r.choice(["None", ">200", ">300", "Norm"], n, p=[.9, .04, .03, .03]),
        "A1Cresult": r.choice(["None", ">7", ">8", "Norm"], n, p=[.8, .05, .1, .05]),
    })
    for m in MEDS:
        df[m] = r.choice(["No", "Steady", "Up", "Down"], n, p=[.8, .14, .03, .03])
    df["change"] = r.choice(["No", "Ch"], n)
    df["diabetesMed"] = r.choice(["Yes", "No"], n, p=[.77, .23])
    df["readmitted"] = np.where(y, "<30", r.choice([">30", "NO"], n))
    df.to_csv(out, index=False)
    print(f"wrote FAKE smoke-test data ({n} rows) to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/diabetic_data.csv")

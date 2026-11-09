# Data

**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository, id 296)
**Link:** https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
**Paper:** Strack et al. (2014), "Impact of HbA1c Measurement on Hospital Readmission Rates," *BioMed Research International*.
**License:** CC BY 4.0

## Steps
1. Download the zip from the link above.
2. Unzip it and copy **both** `diabetic_data.csv` and `IDS_mapping.csv` into `data/raw/`.

The raw files are not committed to git (see `.gitignore`). Anyone cloning the repo downloads them the same way.

## Check you have the right file
101,766 rows, 50 columns, 71,518 unique `patient_nbr`, and `readmitted` counts of
`NO` 54,864 / `>30` 35,545 / `<30` 11,357. `reports/eda.md` prints these after a run.

## What is in it
Inpatient encounters of diabetic patients from 130 US hospitals: demographics, admission/discharge
codes, length of stay, lab/procedure/medication counts, prior-year visit counts, three ICD-9
diagnoses, A1C and glucose results, 23 diabetes medications, and the outcome `readmitted`.
`IDS_mapping.csv` explains the admission type, discharge disposition and admission source codes.

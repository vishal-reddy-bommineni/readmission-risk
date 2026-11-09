# Data

**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository, id 296)
**Link:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
**Paper:** Strack et al. (2014), "Impact of HbA1c Measurement on Hospital Readmission Rates," *BioMed Research International*.
**License:** CC BY 4.0

## Steps
1. Download the zip from the link above.
2. Unzip it and copy `diabetic_data.csv` into `data/raw/`.
3. (`IDS_mapping.csv` in the same zip explains what the admission/discharge ID codes mean. Read it - you will be asked.)

The raw file is not committed to git (see `.gitignore`). Anyone cloning the repo downloads it the same way.

## What is in it
About 100k inpatient encounters of diabetic patients from 130 US hospitals, 50 columns: demographics,
admission/discharge codes, length of stay, lab/procedure/medication counts, prior-year visit counts,
three ICD-9 diagnoses, A1C and glucose results, 23 diabetes medications, and the outcome
`readmitted` (`<30`, `>30`, `NO`). The exact row and patient counts for your copy are written to
`reports/cohort_audit.csv` when you run the pipeline - quote those, not this README.

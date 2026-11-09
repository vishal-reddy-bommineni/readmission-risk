-- 02_features.sql
-- Feature engineering in SQL. One row per encounter, ready for modeling.
-- ICD-9 grouping follows Strack et al. (2014), the paper that released this dataset.

CREATE OR REPLACE MACRO icd9_group(code) AS
    CASE
        WHEN code IS NULL THEN 'Missing'
        WHEN code LIKE 'V%' OR code LIKE 'E%' THEN 'External/Supplementary'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 250 AND 250.99 THEN 'Diabetes'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 390 AND 459
          OR TRY_CAST(code AS DOUBLE) BETWEEN 785 AND 785.99 THEN 'Circulatory'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 460 AND 519
          OR TRY_CAST(code AS DOUBLE) BETWEEN 786 AND 786.99 THEN 'Respiratory'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 520 AND 579
          OR TRY_CAST(code AS DOUBLE) BETWEEN 787 AND 787.99 THEN 'Digestive'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 800 AND 999 THEN 'Injury'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 710 AND 739 THEN 'Musculoskeletal'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 580 AND 629
          OR TRY_CAST(code AS DOUBLE) BETWEEN 788 AND 788.99 THEN 'Genitourinary'
        WHEN TRY_CAST(code AS DOUBLE) BETWEEN 140 AND 239 THEN 'Neoplasms'
        ELSE 'Other'
    END;

-- Comorbidity flag: TRUE if any of the three diagnosis slots falls in [lo, hi]
CREATE OR REPLACE MACRO any_dx(d1, d2, d3, lo, hi) AS
    (COALESCE(TRY_CAST(d1 AS DOUBLE) BETWEEN lo AND hi, FALSE)
  OR COALESCE(TRY_CAST(d2 AS DOUBLE) BETWEEN lo AND hi, FALSE)
  OR COALESCE(TRY_CAST(d3 AS DOUBLE) BETWEEN lo AND hi, FALSE));

-- Medical specialty: keep the 10 most common, bucket the rest
CREATE OR REPLACE TABLE top_specialties AS
SELECT medical_specialty
FROM cohort
WHERE medical_specialty IS NOT NULL
GROUP BY medical_specialty
ORDER BY COUNT(*) DESC
LIMIT 10;

CREATE OR REPLACE TABLE features AS
WITH meds AS (
    -- 23 diabetes medication columns; each is No / Steady / Up / Down
    SELECT encounter_id,
        list_count(list_filter(
            [metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide,
             glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol,
             troglitazone, tolazamide, examide, citoglipton, insulin,
             "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
             "metformin-rosiglitazone", "metformin-pioglitazone"],
            lambda x: x IN ('Up', 'Down'))) AS n_meds_changed,
        list_count(list_filter(
            [metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide,
             glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol,
             troglitazone, tolazamide, examide, citoglipton, insulin,
             "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
             "metformin-rosiglitazone", "metformin-pioglitazone"],
            lambda x: x <> 'No')) AS n_diabetes_meds
    FROM cohort
)
SELECT
    c.encounter_id,
    c.patient_nbr,                                    -- used only for the grouped split, never as a feature
    c.label_readmit_30d,

    -- Audit-only columns (kept out of the model, used for subgroup fairness checks)
    COALESCE(c.race, 'Unknown')                       AS race,
    c.gender,
    c.age                                             AS age_band,

    -- Demographics used as features
    CAST(regexp_extract(c.age, '\[(\d+)-', 1) AS INTEGER) + 5 AS age_mid,

    -- Encounter intensity
    c.time_in_hospital                                AS length_of_stay,
    c.num_lab_procedures,
    c.num_procedures,
    c.num_medications,
    c.number_diagnoses,

    -- Prior utilization in the year before the encounter (strongest known signal)
    c.number_outpatient,
    c.number_emergency,
    c.number_inpatient,
    c.number_outpatient + c.number_emergency + c.number_inpatient AS total_prior_visits,
    CASE WHEN c.number_inpatient > 0 THEN 1 ELSE 0 END             AS any_prior_inpatient,

    -- Admission / discharge context (categorical codes)
    CAST(c.admission_type_id AS VARCHAR)              AS admission_type,
    CAST(c.discharge_disposition_id AS VARCHAR)       AS discharge_disposition,
    CAST(c.admission_source_id AS VARCHAR)            AS admission_source,
    CASE WHEN c.payer_code IS NULL THEN 'Missing' ELSE c.payer_code END AS payer_code,
    CASE
        WHEN c.medical_specialty IS NULL THEN 'Missing'
        WHEN c.medical_specialty IN (SELECT medical_specialty FROM top_specialties) THEN c.medical_specialty
        ELSE 'Other'
    END                                               AS medical_specialty,

    -- Diagnoses
    icd9_group(c.diag_1)                              AS primary_dx_group,
    icd9_group(c.diag_2)                              AS secondary_dx_group,

    -- Comorbidity flags across all three diagnosis slots (Charlson/Elixhauser-inspired subset)
    CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 428, 428.99) AS INTEGER) AS cm_heart_failure,
    CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 584, 586.99)
      OR any_dx(c.diag_1, c.diag_2, c.diag_3, 403, 404.99) AS INTEGER)  AS cm_renal,
    CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 490, 496.99) AS INTEGER)  AS cm_copd,
    CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 410, 414.99) AS INTEGER)  AS cm_ischemic_heart,
    CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 140, 239.99) AS INTEGER)  AS cm_cancer,
    CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 250.4, 250.99) AS INTEGER) AS cm_diabetes_complicated,

    -- Diabetes management
    CASE WHEN c.A1Cresult IS NULL OR c.A1Cresult = 'None' THEN 0 ELSE 1 END     AS a1c_tested,
    CASE WHEN c.A1Cresult IN ('>7', '>8') THEN 1 ELSE 0 END                     AS a1c_high,
    CASE WHEN c.max_glu_serum IS NULL OR c.max_glu_serum = 'None' THEN 0 ELSE 1 END AS glucose_tested,
    c.insulin                                         AS insulin_status,
    CASE WHEN c.change = 'Ch' THEN 1 ELSE 0 END       AS any_med_change,
    CASE WHEN c.diabetesMed = 'Yes' THEN 1 ELSE 0 END AS on_diabetes_med,
    m.n_meds_changed,
    m.n_diabetes_meds,

    -- ---------------------------------------------------------------- experiment features
    -- Added in the feature experiment (reports/feature_experiment.md). Whether the model uses
    -- them is decided in src/config.py, based on VALIDATION results only.

    -- Discharge destination grouped using IDS_mapping.csv (raw codes are sparse: 26 values)
    CASE
        WHEN c.discharge_disposition_id = 1 THEN 'Home'
        WHEN c.discharge_disposition_id IN (6, 8) THEN 'Home with health service'
        WHEN c.discharge_disposition_id IN (3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30) THEN 'Post-acute facility'
        WHEN c.discharge_disposition_id = 2 THEN 'Transfer to short-term hospital'
        WHEN c.discharge_disposition_id = 7 THEN 'Left against medical advice'
        ELSE 'Other / unknown'
    END                                               AS discharge_group,

    -- How many of the six tracked comorbidities the patient has
    CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 428, 428.99) AS INTEGER)
      + CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 584, 586.99)
             OR any_dx(c.diag_1, c.diag_2, c.diag_3, 403, 404.99) AS INTEGER)
      + CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 490, 496.99) AS INTEGER)
      + CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 410, 414.99) AS INTEGER)
      + CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 140, 239.99) AS INTEGER)
      + CAST(any_dx(c.diag_1, c.diag_2, c.diag_3, 250.4, 250.99) AS INTEGER) AS comorbidity_count,

    -- "Frequent flyer": 3+ inpatient stays in the prior year
    CASE WHEN c.number_inpatient >= 3 THEN 1 ELSE 0 END                     AS frequent_flyer,

    -- Came in through the emergency route AND has used the ER before
    CASE WHEN c.admission_type_id = 1 AND c.number_emergency > 0 THEN 1 ELSE 0 END AS emergency_repeat
FROM cohort c
JOIN meds m USING (encounter_id);

-- 01_load_and_clean.sql
-- Load the raw UCI "Diabetes 130-US Hospitals (1999-2008)" extract and define the study cohort.
-- Raw file uses '?' for missing values. Diagnosis codes mix numbers and letters (V57, E880), so force VARCHAR.

CREATE OR REPLACE TABLE raw_encounters AS
SELECT *
FROM read_csv(
    '__RAW_PATH__',
    header = true,
    nullstr = '?',
    sample_size = -1,
    types = {'diag_1': 'VARCHAR', 'diag_2': 'VARCHAR', 'diag_3': 'VARCHAR',
             'payer_code': 'VARCHAR', 'medical_specialty': 'VARCHAR'}
);

-- Cohort rules (document every one of these in the README):
--  1. Drop encounters where the patient died or went to hospice: they cannot be readmitted,
--     so keeping them would make the model look better than it is.
--     discharge_disposition_id: 11 = expired, 13/14 = hospice, 19/20/21 = expired (various)
--  2. Drop the handful of rows with gender 'Unknown/Invalid'.
--  3. Label = 1 if readmitted within 30 days ('<30'), else 0 ('>30' or 'NO').
CREATE OR REPLACE TABLE cohort AS
SELECT
    *,
    CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END AS label_readmit_30d
FROM raw_encounters
WHERE discharge_disposition_id NOT IN (11, 13, 14, 19, 20, 21)
  AND gender <> 'Unknown/Invalid';

-- Audit table: how many rows each rule removed (goes straight into the README)
CREATE OR REPLACE TABLE cohort_audit AS
SELECT 'raw encounters' AS step, COUNT(*) AS n FROM raw_encounters
UNION ALL
SELECT 'after removing expired/hospice', COUNT(*) FROM raw_encounters
    WHERE discharge_disposition_id NOT IN (11, 13, 14, 19, 20, 21)
UNION ALL
SELECT 'final cohort', COUNT(*) FROM cohort
UNION ALL
SELECT 'unique patients in cohort', COUNT(DISTINCT patient_nbr) FROM cohort
UNION ALL
SELECT 'readmitted <30d in cohort', CAST(SUM(label_readmit_30d) AS BIGINT) FROM cohort;

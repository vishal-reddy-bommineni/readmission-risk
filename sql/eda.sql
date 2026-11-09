-- eda.sql - exploratory questions answered before modeling.
-- Run with: python -m scripts.run_eda   (writes reports/eda.md)
-- Each query is preceded by a "name" comment line so the runner can label it.

-- name: Dataset size and repeat patients
SELECT
    (SELECT COUNT(*) FROM raw_encounters)                                    AS raw_encounters,
    (SELECT COUNT(DISTINCT patient_nbr) FROM raw_encounters)                 AS raw_patients,
    (SELECT COUNT(*) FROM (SELECT patient_nbr FROM raw_encounters
                           GROUP BY 1 HAVING COUNT(*) >= 2))                AS patients_with_2plus_encounters,
    (SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM raw_encounters GROUP BY patient_nbr)) AS max_encounters_one_patient;

-- name: Outcome distribution (raw file)
SELECT readmitted, COUNT(*) AS n, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM raw_encounters GROUP BY 1 ORDER BY n DESC;

-- name: Readmission rate by discharge destination (cohort)
SELECT c.discharge_disposition_id AS code, d.description, COUNT(*) AS n,
       ROUND(AVG(label_readmit_30d), 3) AS readmit_30d_rate
FROM cohort c LEFT JOIN discharge_map d ON d.id = c.discharge_disposition_id
GROUP BY ALL HAVING COUNT(*) >= 100 ORDER BY readmit_30d_rate DESC;

-- name: Readmission rate by prior-year inpatient stays
SELECT CASE WHEN number_inpatient >= 5 THEN '5+' ELSE CAST(number_inpatient AS VARCHAR) END AS prior_inpatient,
       COUNT(*) AS n, ROUND(AVG(label_readmit_30d), 3) AS readmit_30d_rate
FROM cohort GROUP BY 1 ORDER BY MIN(number_inpatient);

-- name: Readmission rate by age band
SELECT age AS age_band, COUNT(*) AS n, ROUND(AVG(label_readmit_30d), 3) AS readmit_30d_rate
FROM cohort GROUP BY 1 ORDER BY 1;

-- name: Readmission rate by primary diagnosis group
SELECT primary_dx_group, COUNT(*) AS n, ROUND(AVG(label_readmit_30d), 3) AS readmit_30d_rate
FROM features GROUP BY 1 ORDER BY n DESC;

-- name: HbA1c testing practice
SELECT COALESCE(A1Cresult, 'None') AS a1c_result, COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_encounters,
       ROUND(AVG(label_readmit_30d), 3) AS readmit_30d_rate
FROM cohort GROUP BY 1 ORDER BY n DESC;

-- name: Is missing payer code informative?
SELECT CASE WHEN payer_code IS NULL THEN 'Missing' ELSE 'Recorded' END AS payer_code,
       COUNT(*) AS n, ROUND(AVG(label_readmit_30d), 3) AS readmit_30d_rate,
       ROUND(AVG(number_inpatient), 2) AS avg_prior_inpatient
FROM cohort GROUP BY 1;

-- name: Missing values in the raw file (columns with any)
SELECT * FROM (
    UNPIVOT (SELECT
        ROUND(100.0 * COUNT(*) FILTER (WHERE race IS NULL) / COUNT(*), 1)              AS race,
        ROUND(100.0 * COUNT(*) FILTER (WHERE weight IS NULL) / COUNT(*), 1)            AS weight,
        ROUND(100.0 * COUNT(*) FILTER (WHERE payer_code IS NULL) / COUNT(*), 1)        AS payer_code,
        ROUND(100.0 * COUNT(*) FILTER (WHERE medical_specialty IS NULL) / COUNT(*), 1) AS medical_specialty,
        ROUND(100.0 * COUNT(*) FILTER (WHERE diag_1 IS NULL) / COUNT(*), 1)            AS diag_1,
        ROUND(100.0 * COUNT(*) FILTER (WHERE diag_2 IS NULL) / COUNT(*), 1)            AS diag_2,
        ROUND(100.0 * COUNT(*) FILTER (WHERE diag_3 IS NULL) / COUNT(*), 1)            AS diag_3
     FROM raw_encounters)
    ON COLUMNS(*) INTO NAME column_name VALUE pct_missing)
ORDER BY pct_missing DESC;

-- name: Constant columns (carry no information)
SELECT 'examide' AS column_name, COUNT(DISTINCT examide) AS distinct_values FROM raw_encounters
UNION ALL SELECT 'citoglipton', COUNT(DISTINCT citoglipton) FROM raw_encounters;

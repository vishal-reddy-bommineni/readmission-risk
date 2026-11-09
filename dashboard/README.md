# Dashboard build notes (Tableau Public or Power BI Desktop - both free)

Source: `scored_test_encounters.csv` (written by `src/train.py`). One row = one held-out encounter.

## Page 1 - Outreach list (the "so what")
- KPI tiles: encounters scored, observed 30-day readmission rate, % flagged High, % of readmissions
  captured in the High tier (= recall at top 10%).
- Table of High-tier encounters sorted by `predicted_risk`, with `driver_1..3` - this is what a care
  coordinator would actually use.

## Page 2 - Where risk concentrates
- Bar: observed readmission rate by `primary_dx_group`, `discharge_disposition`, `age_band`.
- Bar: mean `predicted_risk` vs observed rate by the same fields (calibration by segment).
- Heatmap: `number_inpatient` (0,1,2,3+) x `length_of_stay` bucket, colored by observed rate.

## Page 3 - Model honesty
- Calibration chart: bin `predicted_risk` into deciles, plot mean predicted vs observed.
- Fairness: observed vs predicted rate and recall by `race`, `gender`, `age_band`.

## Rules
- Title says "Held-out test set, UCI Diabetes 130-US Hospitals 1999-2008".
- Publish to Tableau Public (free link) or export a PDF from Power BI; put a screenshot in the main README.

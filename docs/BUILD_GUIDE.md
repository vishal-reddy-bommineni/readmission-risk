# Build guide - Project 1 (about 2 weeks part-time)

The code is a working starting point. **It only becomes your project when you have run it on the
real data, read every line, changed things, and can explain each decision without notes.**
Interviewers test exactly that. Work through these stages in order and commit to git after each one.

## Stage 0 - Setup (day 1)
- [ ] Create a public GitHub repo `readmission-risk`, push this folder.
- [ ] `python -m venv .venv`, activate, `pip install -r requirements.txt`.
- [ ] `make smoke` (or run the three smoke commands by hand on Windows). It should finish without errors.
- [ ] `make clean`, then download the real data (`data/README.md`).

## Stage 1 - Know the data before modeling (days 2-3)
- [ ] Open `diabetic_data.csv` and `IDS_mapping.csv`. Write down what discharge codes 1, 3, 6, 11, 13, 14 mean.
- [ ] In DuckDB (Python or the `duckdb` CLI), answer these yourself and save the queries in
      `sql/eda.sql`:
  - How many encounters, how many unique patients, how many patients have 2+ encounters?
  - 30-day readmission rate overall and by discharge disposition. Which disposition is highest? Why might that be?
  - What share of `A1Cresult` is "None"? What does that tell you about testing practice?
- [ ] Run `python -m src.build_features`. Check `reports/cohort_audit.csv` against your own counts.

## Stage 2 - Make one feature change of your own (days 4-5)
The SQL is a starting point. Pick at least one and measure whether it helps (rerun train, compare RESULTS.md):
- A "frequent flyer" flag (e.g. `number_inpatient >= 3`).
- Group `discharge_disposition` into home / home health / facility / AMA using IDS_mapping.
- A comorbidity count (sum of the `cm_*` flags).
Write in the README what you tried and what happened, **including ideas that didn't help.**

## Stage 3 - Train and read the results (days 6-8)
- [ ] `python -m src.train --n-boot 1000`.
- [ ] Open every figure. Can you explain the calibration plot to a non-technical manager?
- [ ] Read `subgroup_audit.csv`. Is the model worse for any group? Write it down.
- [ ] Read the leakage check. This is your best "what broke" story.

## Stage 4 - Product layer (days 9-11)
- [ ] `streamlit run src/app.py` and record a 60-second screen capture (GIF in README).
- [ ] Build the dashboard from `dashboard/README.md`, publish, screenshot into README.

## Stage 5 - Finish (days 12-14)
- [ ] Fill every `[ ]` in README from `reports/RESULTS.md`. Delete the italic instructions.
- [ ] Ask a friend to clone and run it from the README alone. Fix whatever confused them.
- [ ] Send me your RESULTS.md and I will write your resume bullets from it.

## Questions you should be able to answer (practice out loud)
1. Why split by patient? What would happen otherwise, in numbers?
2. Why remove hospice and expired discharges?
3. Your AUROC is ~0.6x. Is that a bad model? (Answer with the top-10% recall/lift and the baseline comparison.)
4. Why did you not use SMOTE? (Resampling distorts predicted probabilities, and you care about calibrated risk.
   Class imbalance does not hurt AUROC/PR-AUC ranking; you handle it with the threshold instead.)
5. What does isotonic calibration do, and why fit it on validation rather than test?
6. Why exclude race from the inputs but keep it for the audit? What are the arguments for including it?
7. What is a SHAP value? Why is `number_inpatient` at the top, and is that useful clinically?
8. How would you deploy this in a hospital? (Batch score at discharge, write to EHR worklist, monitor
   calibration drift monthly, retrain when it drifts.)
9. What would you need to make this a real clinical tool? (Prospective validation, recent data,
   ICD-10, readmissions to other hospitals, clinician input on the threshold.)

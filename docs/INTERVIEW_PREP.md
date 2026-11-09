# Interview prep - 30-Day Readmission Risk

All numbers below come from `reports/`. If you rerun anything and the numbers change, update this file.

## The 2-minute walkthrough (practice out loud until it's natural)
1. **Problem.** Hospitals are penalized for excess 30-day readmissions, and care teams can only
   follow up with so many patients. I built a model that ranks diabetic discharges by readmission
   risk so a team knows who to call first.
2. **Data.** 101,766 encounters from 130 US hospitals (UCI). I removed patients who died or went
   to hospice since they can't be readmitted, leaving 99,340 encounters, 11.4% readmitted in 30 days.
3. **Features in SQL.** In DuckDB: ICD-9 diagnosis groups, six comorbidity flags, prior-year
   utilization, medication changes, A1C testing.
4. **Evaluation.** I split by patient, so the test set only has people the model never saw. I
   compared against a 3-feature baseline and reported patient-level bootstrap confidence intervals.
5. **Result.** XGBoost with Platt calibration: AUROC 0.689 vs 0.619 for the baseline. More
   usefully, calling the top 10% reaches 26.5% of readmissions, 2.65x random, 64 more than the baseline.
6. **What I learned.** Repeat patients are readmitted 4x as often, because a readmission is itself a
   second encounter, so I avoided features built from a patient's other encounters. And the
   fairness audit showed the model is weak for patients over 80 and has lower recall for African
   American patients.

## Likely questions, with answers

**1. Why split by patient instead of randomly?**
71,518 patients have 101,766 encounters, so many patients appear more than once. A random split
puts the same person in train and test (42% of test encounters, in my check), which is not how the
model would be used: in practice it scores new discharges. Interestingly, the naive split did not
inflate the overall AUROC here (0.688 vs 0.689), but splitting that test set showed returning
patients have a 20.3% readmission rate vs 4.6% for new ones, because a readmission creates a
second encounter. That is why grouping matters and why I avoided a "number of encounters in the
dataset" feature: it would be nearly the label.

**2. Why remove expired and hospice discharges?**
They cannot be readmitted, so they are guaranteed negatives. Keeping them would make the model look
better without making it more useful. 2,423 encounters removed.

**3. An AUROC of 0.69 seems low. Is this a bad model?**
For 30-day readmission from administrative data, high 0.6s is typical; the outcome depends on
things not in the data (social support, follow-up care). The useful question is operational: with
capacity to call 10% of discharges, the model reaches 26.5% of readmissions vs 10% at random, and
its high-risk tier has an observed 30% readmission rate vs 6.1% in the low tier.

**4. Why didn't you use SMOTE for the class imbalance?**
Resampling distorts predicted probabilities, and I needed calibrated risks people can read as
percentages. Imbalance does not hurt ranking metrics like AUROC or PR-AUC; I handle it at decision
time by choosing a threshold that matches outreach capacity.

**5. What does calibration do here, and why Platt instead of isotonic?**
Calibration makes "30% risk" mean about 30% of such patients are readmitted. I compared Platt and
isotonic by cross-fitting inside the validation set; Platt had lower log loss (0.3300 vs 0.3317).
Platt is monotonic so it keeps the ranking, while isotonic produced tied scores that cost about
0.012 PR-AUC in my first run. After calibration, predicted vs observed rates match within ~1.5
points in each tier.

**6. Why fit calibration and pick the threshold on validation, not test?**
Anything tuned on the test set makes the test score optimistic. Test was used once, for the final table.

**7. How did you get the confidence intervals?**
1,000 bootstrap resamples of patients, not encounters, because one patient's encounters are
correlated. Resampling encounters would make the intervals too narrow.

**8. Why exclude race from the model but keep it for the audit?**
I didn't want the model to use group membership directly, but I still needed to check outcomes by
group. The audit found similar AUROC across race but lower recall for African American patients
(23.0% vs 26.9%) despite slight over-prediction. The argument for including race is that it can
sometimes improve calibration within groups; either way, you have to measure it.

**9. What is a SHAP value? What did it show?**
For each prediction, SHAP splits the model's output into contributions from each feature, so they
add up to the prediction. Prior inpatient stays and discharge destination dominate. Missing payer
code ranks 4th though its raw rate barely differs, likely a proxy for which hospital recorded the
data, which is a deployment risk.

**10. You tried feature engineering. What happened?**
Four new features (comorbidity count, frequent flyer, emergency repeat, grouped discharge) gave no
gain in 5-fold patient-grouped CV (AUROC -0.0001). Grouping discharge codes made it worse in all 5
folds because my grouping merged rehab (27.7% readmitted) with nursing facilities (14.7%).

**11. Why XGBoost over logistic regression?**
I chose on validation AUROC (0.681 vs 0.663). Logistic regression is close (test 0.680 vs 0.689)
and easier to explain, so in a real hospital I'd discuss that trade-off with clinicians.

**12. How would you deploy this?**
Score each patient at discharge in a nightly batch, write the high-risk list to the care team's
worklist, track calibration and recall by subgroup monthly, and retrain when calibration drifts.
Before real use: recent data, ICD-10 codes, readmissions to other hospitals, and a prospective pilot.

## Know these files well enough to explain line by line
`sql/02_features.sql`, `src/train.py` (functions `grouped_split`, `choose_calibrator`,
`top_k_stats`, `cluster_bootstrap_ci`, `leakage_check`). Interviewers often ask you to open a file
and walk through it.

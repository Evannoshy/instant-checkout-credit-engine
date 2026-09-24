# Tabular Constant Base-Rate Baseline Report

**Document status:** Evidence record for Week 1 review
**Version:** 0.1
**Date:** 18 September 2026
**Accountable owner:** Evan (Tabular Track Tech Lead)
**Primary author:** Tabular Analyst
**Reviewers:** Tabular Track Tech Lead
**Experiment ID:** `TAB-20260918-001-constant-baseline`
**Dataset version:** `kaggle-adarshsng-local-2026-09-09`
**Split version:** `split-v1`
**Cohort:** `real_text_matured_v1`
**Baseline version:** `tabular-baseline-constant-v1`

## 1. Purpose

This report records the first rung of the tabular track's "minimum experiment ladder"
(`docs/tabular-analyst-handbook.md` §12.3): a constant/prevalence baseline that assigns every
validation loan the training-set default rate. It exists to prove the evaluation framework and
data pipeline work end to end on the real, frozen `data/split_manifest.csv`, and to establish the
minimum floor that the upcoming logistic-regression baseline must beat.

## 2. What was built

- `src/tabular/evaluate.py` — a shared, model-agnostic manifest loader/validator, split-target
  extractor, prediction-frame builder/validator, and core evaluation-metric function. Every future
  tabular model (logistic regression, then XGBoost) is expected to import these same functions.
- `src/tabular/baseline.py` — the constant base-rate model itself: computes the training default
  rate, predicts it uniformly for every validation loan, validates and evaluates the result, and
  writes the two artifacts below.

## 3. Test-label isolation

The source `split_manifest.csv` is the repository's shared split registry and contains metadata
for all three splits. The model-development loader now returns **only train and validation rows**.
It validates targets only for those two splits and removes test outcome aggregates from the
statistics object before returning it to model code.

This boundary is verified three ways:

1. **Structurally:** `load_manifest()` returns no row whose split is `test`;
   `get_split_targets()` also refuses a direct `"test"` request.
2. **Tested:** the synthetic fixture places non-binary sentinel strings in the test target cells.
   Loading and running the baseline succeed, while tests assert that neither the test rows nor the
   sentinel values reach model code. Tests also assert that returned test statistics contain only
   the mechanical row count.
3. **Measured:** every run records `"test_rows_available_to_model": 0`, calculated from the
   returned frame. The baseline fails immediately if that count is non-zero; it is not a hard-coded
   statement of compliance.

The baseline does not evaluate, tune on, or make any decision from test outcomes. A future
data-contract revision should physically separate final-test labels from the shared manifest;
until then, this loader is the enforced model-development boundary.

## 4. Method

The baseline computes `rate = mean(target)` over the 86,293 training rows and predicts that same
scalar for all 21,784 validation loans — no features, no learned parameters. This is intentionally
non-discriminating: it exists to give every later model a floor to beat, per the handbook's
explicit instruction not to skip it "because the pitch already names XGBoost."

## 5. Results (validation split, real data)

| Metric | Value |
|---|---:|
| Training rows / positives | 86,293 / 13,197 |
| Training default rate (the predicted constant) | 0.1529324510678734 |
| Validation rows / positives | 21,784 / 3,298 |
| Validation prevalence | 0.15139551964744766 |
| ROC-AUC (baseline: 0.5) | 0.5 |
| PR-AUC (baseline: prevalence) | 0.15139551964744766 |
| Brier score | 0.12847727843631807 |
| Log loss | 0.42513128226810804 |
| F1 @ 0.5 threshold | 0.0 |
| Best-threshold F1 | 0.2629774340164261 |
| Best F1 threshold | 0.1529324510678734 |

**Reading these numbers:** ROC-AUC of exactly 0.5 and PR-AUC exactly equal to prevalence are the
mathematically guaranteed result of scoring every row identically — this is a correct outcome for
a model with zero discrimination, not a bug. Likewise, F1 @ 0.5 is 0.0 because the constant score
(0.153) never reaches the fixed 0.5 threshold, so the model never predicts the positive class at
that cutoff; the "best" F1 (0.263) occurs only because the *threshold itself* was swept down to
meet the constant score — it collapses to a single point equal to the constant value
(`f1_best_threshold == 0.1529324510678734`) and should not be read as evidence of genuine ranking
ability. Any future model is expected to materially exceed PR-AUC 0.151 and ROC-AUC 0.5 to justify
its added complexity.

## 6. Artifacts

- `reports/tabular/constant_baseline_predictions.csv` — 21,784 rows, columns exactly
  `loan_id,split,p_default_tabular,model_name,model_version`; every `p_default_tabular` equals
  0.1529324510678734; every `loan_id` is unique and covers the validation split exactly.
- `reports/tabular/constant_baseline_metrics.json` — the full metrics object shown in §5, plus
  `dataset_version`, `split_version`, `cohort`, `test_rows_available_to_model`, and `environment`
  metadata.
- Both were verified to be byte-identical across repeated runs (no timestamps or randomness in the
  constant model), so committing them is reproducible.

## 7. Verification

From the repository root:

```
python -m pip install -r src/tabular/requirements.txt
python -m pytest src/tabular/test_evaluate.py src/tabular/test_baseline.py -v -s
python -m src.tabular.baseline
```

44 tests pass (35 in `test_evaluate.py`, 9 in `test_baseline.py`), including one real-data
regression check in each file that reads the real `data/split_manifest.csv` and
`data/split_statistics.json` directly (never `data/raw/loan.csv`) and asserts the training default
rate equals `13197 / 86293` exactly.

## 8. Known limitations and rollback

- **Metric scope:** this baseline reports only core ranking/probability-quality metrics
  (ROC-AUC, PR-AUC, Brier score, log loss, F1 sweep). Bootstrap confidence intervals, the KS
  statistic, Expected Calibration Error, calibration slope/intercept, and policy metrics (approval
  rate, bad rate, expected loss) are deferred to the logistic-regression baseline PR.
- **Schema reconciliation deferred:** the flat prediction schema here
  (`loan_id, split, p_default_tabular, model_name, model_version`) is a research/evidence artifact
  for the experiment ladder. It does not yet match the `TabularRiskOutputV1` JSON contract in
  `docs/architecture.md` §7 or the proposed manifest fields in decision D-005 — both remain status
  "Proposed," not "Accepted." Reconciling these is future work, not solved here.
- **Rollback:** this baseline has no model artifact to roll back — it is a pure function of the
  training default rate. Reverting means deleting `src/tabular/` and the two report artifacts; no
  other component depends on them yet.

## 9. Next decision

**Update, 24 September 2026:** Done. The logistic-regression baseline (handbook §12.3, step 3) is
complete — see `reports/tabular/logistic_baseline_report.md` (experiment
`TAB-20260924-001-logistic-baseline`). It reused `src/tabular/evaluate.py`'s `load_manifest`,
`get_split_targets`, `build_prediction_frame`, `validate_prediction_frame`, and
`evaluate_predictions` unchanged, as planned, and beat this floor on ROC-AUC (0.668 vs. 0.5) and
PR-AUC (0.259 vs. 0.151), while landing worse on Brier score and log loss — an explained trade-off
from `class_weight="balanced"`, not a defect. Next: an XGBoost challenger on the same rows, splits
and metric functions (handbook §12.3, step 4).

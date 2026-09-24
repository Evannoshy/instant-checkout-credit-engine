# Tabular Logistic-Regression Baseline Report

**Document status:** Evidence record for Week 2 review
**Version:** 0.1
**Date:** 24 September 2026
**Accountable owner:** Evan (Tabular Track Tech Lead)
**Primary author:** Tabular Analyst
**Reviewers:** Tabular Track Tech Lead
**Experiment ID:** `TAB-20260924-001-logistic-baseline`
**Dataset version:** `kaggle-adarshsng-local-2026-09-09`
**Split version:** `split-v1`
**Cohort:** `real_text_matured_v1`
**Feature set version:** `tabular_features_v1` (D-016, Accepted)
**Baseline version:** `tabular-logistic-v1`

## 1. Purpose

This report records the second rung of the tabular track's minimum experiment ladder
(`docs/tabular-analyst-handbook.md` §12.3): a regularized logistic regression fit on the nine
approved `tabular_features_v1` features (D-016), evaluated against the constant base-rate floor
from `docs/tabular-baseline-report.md`. Per the task brief for this experiment: "The logistic model
does not need to achieve a particular score. A correctly produced and honestly reported result is
the Week 2 goal." The numbers below are reported plainly, including where the logistic model does
*not* uniformly beat the constant floor.

## 2. What was built

- `configs/tabular_logistic_v1.toml` — the four fixed hyperparameters (`C=1.0`,
  `class_weight="balanced"`, `max_iter=1000`, `random_state=42`). No grid search was run, per the
  task brief.
- `src/tabular/logistic_baseline.py` — builds an sklearn `Pipeline`/`ColumnTransformer` over the
  approved feature set, fits it on the 86,293 training loans only, scores the 21,784 validation
  loans, and writes every artifact below. It reuses two already-approved modules **unchanged**:
  `src/tabular/preprocess.py` (Brandon's `load_tabular_split`, merged in PR #8) for all feature
  engineering, and `src/tabular/evaluate.py` (merged in PR #5) for manifest loading, the shared
  prediction schema, and every ranking/probability-quality metric. No new data loader and no
  reimplementation of an existing metric were written, per the task brief.

## 3. Method

**Preprocessing (fit on training rows only):**
- Numeric features (`loan_amnt`, `annual_inc_log`, `dti`, `revol_util`, `delinq_2yrs`,
  `inq_last_6mths`, `credit_history_months`): median imputation, then standard scaling.
- Categorical features (`term`, `home_ownership`): a separate `"missing"` category, then one-hot
  encoding with `handle_unknown="ignore"`, so a category level absent from training (verified in
  testing with a synthetic `home_ownership="OWN"` row that only appears in validation) does not
  crash prediction.

**Model:** `LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)`.

**No trained model file was saved.** Fitting is fast (~20 seconds end to end, including two full
scans of the 1.19 GB raw CSV) and fully deterministic, so every run reproduces the identical
pipeline in memory rather than needing a serialized artifact — verified directly: two consecutive
runs against the real data produced byte-identical `logistic_baseline_predictions.csv` files.

## 4. Test-set non-access — written confirmation

**The locked `test` split was not loaded, evaluated, or used for any decision in this experiment.**
Guaranteed the same three ways as the constant baseline, extended to this module:

1. **Structurally:** `preprocess.load_tabular_split` independently rejects `split="test"` before
   any raw-CSV row is read, and `evaluate.load_manifest` already returns a manifest with test rows
   filtered out. `logistic_baseline.run()` only ever calls `load_tabular_split` with `"train"` and
   `"validation"`.
2. **Tested:** `test_run_only_requests_train_and_validation_splits` spies on every split value
   `run()` requests and asserts `"test"` never appears.
3. **Written:** `logistic_baseline_metrics.json` records `"test_rows_available_to_model": 0`,
   computed from the manifest at run time, not a hard-coded literal.

## 5. Results (validation split, real data)

| Metric | Value |
|---|---:|
| Training rows / positives | 86,293 / 13,197 |
| Validation rows / positives | 21,784 / 3,298 |
| Validation prevalence | 0.15139551964744766 |
| ROC-AUC (baseline: 0.5) | **0.6679297469765033** |
| PR-AUC (baseline: prevalence, 0.1514) | **0.2594412053841364** |
| Brier score | 0.23213831609216826 |
| Log loss | 0.657697664433428 |
| F1 @ 0.5 threshold | 0.3378491386519485 |
| Best-threshold F1 | 0.34256978653530373 (at threshold 0.548) |
| Confusion matrix @ 0.5 | TN 11,846 / FP 6,640 / FN 1,278 / TP 2,020 |

## 6. Comparison against the constant baseline

Numbers below are read directly from the committed `constant_baseline_metrics.json` and the
freshly generated `logistic_baseline_metrics.json` — not retyped from memory.

| Model | ROC-AUC | PR-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Constant baseline (existing result) | 0.5 | 0.1514 | **0.1285** | **0.4251** |
| Logistic regression (new result) | **0.6679** | **0.2594** | 0.2321 | 0.6577 |

**Reading of this table:** logistic regression clearly improves *discrimination*. It ranks riskier
loans above safer ones more effectively than the constant baseline (ROC-AUC 0.668 vs. 0.5 and
PR-AUC 0.259 vs. 0.151). However, its outputs are not calibrated estimates of a loan's probability
of default. The model's mean validation score is 0.477 while the observed validation default rate is
0.151. The Brier score and log loss are therefore worse than the constant model.

This result is an expected consequence of `class_weight="balanced"`. The setting gives additional
weight to defaulted loans during training, which helps the model identify more of them but changes
the meaning of the output values. For this experiment, `p_default_tabular` must be treated as an
uncalibrated risk score rather than a probability of default. Before the score is used for model
fusion, credit decisions, or customer-facing risk estimates, a later experiment must compare an
unweighted model or fit a probability-calibration step using training data only. F1 at the 0.5
threshold and the confusion matrix are included as diagnostic results, not as an approved operating
threshold.

## 7. Calibration

See `reports/tabular/logistic_calibration.png` (a 10-bin reliability diagram via scikit-learn's
`CalibrationDisplay`). Consistent with the Brier/log-loss numbers above, the curve sits below the
diagonal in the low-probability region — the model over-predicts risk there, the direct visual
signature of `class_weight="balanced"`'s rebalancing.

## 8. Feature associations

`reports/tabular/logistic_coefficients.csv` contains **all 13 fitted coefficients** (7 numeric
features + 2 categorical features one-hot-expanded into 6 levels). This is fewer than the 20 rows
implied by "ten most positive, ten most negative" — with only 13 coefficients total, any top-10 and
bottom-10 selection would overlap by 7 rows, which would misrepresent the model as having more
distinct signal than it does. All 13 are reported instead, ranked, so nothing is hidden or padded:

| Feature | Coefficient | Reading |
|---|---:|---|
| `term_60` | +0.486 | Higher fitted score than `term_36` within the term feature |
| `home_ownership_OTHER` | +0.425 | Highest fitted score among the encoded home-ownership levels |
| `revol_util` | +0.239 | Higher revolving-credit utilization associates with higher risk |
| `inq_last_6mths` | +0.237 | More recent credit inquiries associate with higher risk |
| `loan_amnt` | +0.169 | Larger loans associate with higher risk |
| `delinq_2yrs` | +0.070 | More recent delinquencies associate with higher risk |
| `dti` | +0.057 | Higher debt-to-income associates with higher risk |
| `home_ownership_RENT` | -0.017 | Second-highest fitted score among the encoded home-ownership levels |
| `credit_history_months` | -0.041 | Longer credit history associates with lower risk |
| `home_ownership_OWN` | -0.071 | Third-highest fitted score among the encoded home-ownership levels |
| `home_ownership_MORTGAGE` | -0.193 | Lowest fitted score among the encoded home-ownership levels |
| `annual_inc_log` | -0.329 | Higher income associates with lower risk |
| `term_36` | -0.342 | Lower fitted score than `term_60` within the term feature |

**These are associations learned by the fitted model, not causal effects.** Numeric coefficients
describe changes in the fitted score after standardisation. The categorical encoder retains every
level, so an individual categorical coefficient is not a comparison against an omitted reference
category. Its sign depends on the chosen encoding and regularisation. The categorical rows should
therefore be read as contrasts between levels of the same feature, not as independent effects. For
example, the difference between `term_60` and `term_36` is meaningful within this fitted model, but
neither coefficient alone is evidence that changing a loan's term would change its default risk.
This follows the repository's existing non-causal explanation convention in decision D-009.

## 9. Verification

From the repository root, with the raw LendingClub CSV at `data/raw/loan.csv` (gitignored, not
committed):

```
python -m pip install -r src/tabular/requirements.txt
python -m pytest src/tabular/ -v
LENDINGCLUB_RAW_CSV="$(pwd)/data/raw/loan.csv" python -m pytest src/tabular/test_logistic_baseline.py -v
python -m src.tabular.logistic_baseline
```

85 tests pass without the raw CSV present (2 real-data checks correctly skip); with
`LENDINGCLUB_RAW_CSV` set, all 87 pass, including one asserting the real pipeline covers all 21,784
validation loans and never sees a test row.

## 10. Known limitations and rollback

- **Calibration is worse than the constant baseline.** This is an expected consequence of
  `class_weight="balanced"`, but it remains a material limitation. The output must be treated as an
  uncalibrated risk score until a later experiment evaluates an unweighted or calibrated model.
- **No hyperparameter search** — per the task brief, Week 2 intentionally used one fixed
  configuration rather than tuning `C` or trying alternate solvers/penalties.
- **13 coefficients, not 20** — the "ten highest / ten lowest" framing doesn't cleanly apply to a
  9-feature, one-hot-expanded model; §8 reports all 13 instead of forcing an overlapping split.
- **Schema reconciliation still deferred** — the flat prediction schema
  (`loan_id, split, p_default_tabular, model_name, model_version`) remains unreconciled with the
  `TabularRiskOutputV1` contract in `docs/architecture.md` (D-006, still "Proposed") — unchanged
  situation from the constant-baseline report.
- **Rollback:** no trained model artifact exists to roll back — refitting is deterministic and
  takes seconds. Reverting means deleting `src/tabular/logistic_baseline.py`,
  `configs/tabular_logistic_v1.toml`, and the four `reports/tabular/logistic_*` artifacts; nothing
  else in the repository depends on them yet.

## 11. Next decision

Compare against an XGBoost challenger (handbook §12.3, step 4) on the identical rows, splits and
`evaluate.py` metric functions used here, and revisit whether `class_weight="balanced"` or a
different rebalancing/calibration strategy should be standardized across future tabular models
given the discrimination/calibration trade-off observed in §6.

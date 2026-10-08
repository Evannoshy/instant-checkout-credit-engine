# Tabular XGBoost challenger: results and selected configuration

**Analyst:** Allan · **Config:** `configs/tabular_xgboost_v1.toml` (`tabular-xgboost-v1`) · **Feature set:** `tabular_features_v1` (9 features) · **Seed:** 42

## Environment correction, 8 October 2026

The first results used pandas 2.3.3, NumPy 2.3.5 and scikit-learn 1.7.2, which did not match the repository pins. Rerunning with the pinned packages changed individual XGBoost probabilities by up to 0.0722. LR probabilities matched to numerical precision. The package mismatch was therefore not just a reporting issue.

All experiment results and the six handoff files have been regenerated together. The reference runtime is Python 3.13.15 on Windows AMD64, with pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1 and XGBoost 3.4.1. The remaining numerical package versions are recorded in each metadata file and pinned in `src/tabular/requirements.txt`.

Every runner checks the modelling package versions before reading data or fitting. Prediction export also requires the same recorded environment as the search. A changed runtime requires a fresh search and fresh predictions. A real-data regression test refits all three candidates and compares every saved probability. Seeds alone do not guarantee identical predictions across environments.

Use the refreshed files for calibration. Do not mix them with copies downloaded before this correction. This update does not change the cohort, split IDs, features, fold boundaries, parameter trials or selection rule.

## How every number here was produced
- **Rows:** all models were fitted on the `model_fit` role only (59,353 loans, 2007-06 to 2012-12). The calibration and validation rows were not used to choose anything. The test split was never loaded: `test_rows_available_to_model = 0` in every output.
- **Cross-validation:** expanding-window folds by `issue_month` inside `model_fit`. The first 24,119 loans are training-only. The remaining 35,234 loans form five contiguous holdout blocks from November 2011 to December 2012. No month is split between blocks. Each fold trains only on earlier months.
- **Uncertainty:** model differences use paired bootstrap 95% CIs with 1,000 resamples on the pooled out-of-fold predictions. Fold means and pooled differences are different estimates, so they are labelled separately below.

## Task 1: default XGBoost vs logistic regression

Scores below are fold means; ranking scores include the sample standard deviation across folds.

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|
| Logistic regression (approved settings) | 0.663 ± 0.011 | 0.273 ± 0.017 | 0.237 | 0.667 |
| XGBoost default (300 trees, depth 3) | 0.6599 ± 0.0101 | 0.2685 ± 0.0139 | 0.1305 | 0.4233 |

| Metric | Pooled XGBoost minus LR | Paired 95% CI |
|---|---:|---|
| ROC-AUC | -0.00326 | (-0.00619, -0.00009) |
| PR-AUC | -0.00384 | (-0.00830, +0.00036) |

Default XGBoost does not improve ranking over LR. Its ROC-AUC interval is narrowly below zero in this row-level bootstrap; the time-clustering limitation below still applies. LR uses `class_weight="balanced"`, while XGBoost is unweighted. The raw Brier and log-loss gaps do not isolate the effect of the algorithm. Compare calibrated candidates before choosing a champion.

## Task 2: feature-family ablation

Each variant removes one family from default XGBoost. Negative differences mean removal reduced performance. CIs refer to the pooled difference, not to the fold-mean difference.

| Removed family | Mean ROC-AUC change | Pooled ROC-AUC change (95% CI) | Mean PR-AUC change | Pooled PR-AUC change (95% CI) |
|---|---:|---|---:|---|
| loan_information | -0.0592 | -0.0608 (-0.0669, -0.0542) | -0.0567 | -0.0568 (-0.0643, -0.0495) |
| income_affordability | -0.0234 | -0.0238 (-0.0285, -0.0190) | -0.0148 | -0.0151 (-0.0197, -0.0107) |
| credit_behaviour | -0.0249 | -0.0244 (-0.0290, -0.0193) | -0.0149 | -0.0135 (-0.0184, -0.0085) |
| credit_history | -0.0005 | -0.0007 (-0.0025, +0.0010) | +0.0010 | +0.0003 (-0.0017, +0.0023) |

Loan information has the largest effect in this experiment. Income/affordability and credit behaviour also help. Credit history has no detectable additional effect with the other eight features present. These are model-specific predictive contributions, not causal effects or evidence that a family is useless in other models.

## Task 3: bounded search and selected configuration

All 20 predeclared trials ran (0 failed, 0 unstable). Full log: `reports/tabular/xgboost_trials.csv`. Decision: `reports/tabular/xgboost_selection.json`.

**Selected: trial 1, with `n_estimators=200, max_depth=2`.** Everything else is unchanged from the default: learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, min_child_weight 5, reg_lambda 1, reg_alpha 0.

Why it was selected:
- **Best stable mean:** CV ROC-AUC 0.6617 ± 0.0129. It is also the simplest eligible model, with complexity 800 = 200 × 2².
- **Small differences:** 17 of 20 trials meet the selection rule's margin of 0.00579, including the default at 0.6599. This is a practical selection tolerance, not a statistical proof of equivalence.
- **Deeper trials:** trials 18 and 19 scored 0.6444 and 0.6368. Increasing complexity did not help in this search.
- **Comparison with LR:** selected XGBoost averages 0.6617, while LR averages 0.6629. There is no ranking improvement here. Calibration and model complexity remain relevant to the final choice.

## Task 4: lexical features in logistic regression

Both variants use the same 59,353 model-fit loans and folds; 35,234 loans receive holdout predictions. Mean scores and pooled differences are reported separately.

| Metric | 9-feature fold mean | 15-feature fold mean | Pooled change (95% CI) | Clear improvement? |
|---|---|---|---|---|
| ROC-AUC | 0.6629 | 0.6638 | +0.000765 (-0.000564, +0.001994) | No |
| PR-AUC | 0.2726 | 0.2730 | +0.000164 (-0.001408, +0.001654) | No |
| Brier | 0.2366 | 0.2390 | +0.002410 (+0.002205, +0.002619) | No, worse |
| Log loss | 0.6671 | 0.6724 | +0.005314 (+0.004877, +0.005778) | No, worse |

These six features provide no clear benefit for this balanced LR experiment, so LR+lexical is not shortlisted. XGBoost plus lexical features was not tested. This does not establish whether NLP probabilities will help fusion; that remains a separate experiment.

## Task 5: raw predictions for calibration

Files are in `reports/tabular/raw_predictions/`.
Each candidate was fitted once on all `model_fit` rows. Every file covers each loan exactly once with a probability in [0, 1]: 26,940 calibration loans and 21,784 validation loans, 0 missing.

| Candidate | Calibration ROC-AUC / PR-AUC | Validation ROC-AUC / PR-AUC | Validation Brier (raw) |
|---|---|---|---|
| `logistic_regression` | 0.6619 / 0.2543 | 0.6660 / 0.2575 | 0.2363 |
| `xgboost_default` | 0.6626 / 0.2536 | 0.6660 / 0.2564 | 0.1225 |
| `xgboost_trial1` (selected) | 0.6613 / 0.2541 | 0.6661 / 0.2584 | 0.1224 |

These are informational only; nothing was selected using them. Platt/isotonic calibration, the final comparison and the champion recommendation belong to the calibration step. `manifest.json` lists the row counts, ranges, parameters and raw metrics.

Brandon should fit each calibrator using only the corresponding `*_calibration.csv` and its locked calibration labels. Apply it to `*_validation.csv` for evaluation. Do not fit on validation or test rows, and do not reuse a calibrator fitted on the earlier prediction files.

## Limitations
- The bootstrap resamples loans independently. Out-of-fold rows are clustered in time, so the CIs are probably somewhat narrow.
- `complexity` (trees × 2^depth) is a rough tie-break that ignores learning rate and regularisation.
- `raw_predictions` checks that the selection file matches `model_version`, but it does not hash the trial list. Rerun the search if the config changes.
- The reference results use the runtime recorded above. Different Python patches, operating systems or numerical packages require a fresh, consistent set of results rather than an assumption of bit-identical predictions.

## Reproduce (repository root, `data/raw/loan.csv` present)

Create a Python 3.13 environment and install `src/tabular/requirements.txt` first. Do not reuse the earlier unpinned environment. Run every experiment in the same environment, with search before prediction export.

```
python -m src.tabular.xgboost_baseline compare
python -m src.tabular.xgboost_baseline ablate
python -m src.tabular.xgboost_search
python -m src.tabular.lexical_ablation
python -m src.tabular.raw_predictions
```

For the full test suite in PowerShell:

```powershell
$env:LENDINGCLUB_RAW_CSV = "data/raw/loan.csv"
python -m pytest src/tabular -q
```

Without `LENDINGCLUB_RAW_CSV`, raw-data refit checks are skipped. The saved environment, prediction IDs and recorded metrics are still checked.

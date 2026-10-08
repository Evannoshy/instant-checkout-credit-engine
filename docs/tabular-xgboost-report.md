# Tabular XGBoost challenger — results and selected configuration

**Analyst:** Allan · **Config:** `configs/tabular_xgboost_v1.toml` (`tabular-xgboost-v1`) · **Feature set:** `tabular_features_v1` (9 features) · **Seed:** 42

## How every number here was produced
- **Rows:** all models were fitted on the `model_fit` role only (59,353 loans, 2007-06 to 2012-12). The calibration and validation rows were not used to choose anything. The test split was never loaded: `test_rows_available_to_model = 0` in every output.
- **Cross-validation:** expanding-window folds by `issue_month` inside `model_fit`. The first 40% of rows are always used for training. The remaining rows form 5 contiguous holdout blocks (2011-11 to 2012-12, about 6–8k loans each), and no month is split between blocks. Each fold trains only on earlier months.
- **Uncertainty:** model differences are paired bootstrap 95% CIs (1,000 resamples) on the pooled out-of-fold rows.

## Task 1 — default XGBoost vs logistic regression (same folds and rows)
| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|
| Logistic regression (approved settings) | 0.663 ± 0.011 | 0.273 ± 0.017 | 0.237 | 0.667 |
| XGBoost default (300 trees, depth 3) | 0.660 ± 0.009 | 0.269 ± 0.013 | 0.130 | 0.423 |
| XGBoost − LR (95% CI) | −0.003 (−0.006, +0.000) | −0.003 (−0.008, +0.001) | — | — |

Ranking is effectively tied. The large Brier and log-loss gap comes mostly from LR's `class_weight="balanced"`, which inflates its raw probabilities. Compare those two metrics only after calibration.

## Task 2 — feature-family ablation (default XGBoost, change in ROC-AUC / PR-AUC)
| Removed family | ΔROC-AUC (95% CI) | ΔPR-AUC (95% CI) | Reading |
|---|---|---|---|
| loan_information (loan_amnt, term) | −0.058 (−0.066, −0.053) | −0.057 (−0.064, −0.049) | Strongest family |
| income_affordability (annual_inc_log, dti, home_ownership) | −0.024 (−0.029, −0.019) | −0.016 (−0.021, −0.012) | Useful |
| credit_behaviour (revol_util, delinq_2yrs, inq_last_6mths) | −0.025 (−0.029, −0.019) | −0.015 (−0.019, −0.008) | Useful |
| credit_history (credit_history_months) | −0.001 (−0.002, +0.001) | +0.001 (−0.002, +0.002) | No detectable effect |

Three of the four families carry clear, independent signal. Credit history adds nothing detectable once the other eight features are present. As `tabular_ablation_v1.toml` notes, this is a lower bound, because other features overlap with it.

## Task 3 — bounded search and the selected configuration
All 20 predeclared trials ran (0 failed, 0 unstable). Full log: `reports/tabular/xgboost_trials.csv`. Decision: `reports/tabular/xgboost_selection.json`.

**Selected: trial 1, with `n_estimators=200, max_depth=2`.** Everything else is unchanged from the default: learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, min_child_weight 5, reg_lambda 1, reg_alpha 0.

Why it was selected:
- **It was the best stable mean**, CV ROC-AUC 0.6630 ± 0.012, and the simplest model in the eligible set (complexity 800 = 200 × 2²).
- **It doesn't win by a meaningful margin.** 14 of 20 trials are within the equivalence margin (0.0053, one standard error) of it, including the default (0.6603). The rule picks the simplest equivalent model, and here that happens to be the top scorer too.
- **Deeper and larger trials did worse.** Trials 18 and 19 (depth 5, min_child_weight 1) dropped to 0.648 and 0.637. With 9 features, shallow trees are enough.
- **It matches LR's ranking without beating it.** CV ROC-AUC is 0.663 vs LR's 0.663, so there is no ranking reason to prefer XGBoost yet. The decision rests on calibrated performance, which is the next step.

## Task 4 — lexical features (logistic regression, 9 vs 9 + 6 lexical, 59,353 aligned loan_ids)
| Metric | 9 features | 9 + 6 lexical | Δ (95% CI) | Improves? |
|---|---|---|---|---|
| ROC-AUC | 0.6629 | 0.6638 | +0.0008 (−0.0006, +0.0020) | No |
| PR-AUC | 0.2726 | 0.2730 | +0.0004 (−0.0014, +0.0017) | No |
| Brier | 0.2366 | 0.2390 | +0.0024 (+0.0022, +0.0026) | No (worse) |
| Log loss | 0.6671 | 0.6724 | +0.0053 (+0.0049, +0.0058) | No (worse) |

The lexical features do not improve any metric, so LR+lexical is not shortlisted. The optional XGBoost + lexical run was not done.

## Task 5 — raw predictions for calibration (`reports/tabular/raw_predictions/`)
Each candidate was fitted once on all `model_fit` rows. Every file covers each loan exactly once with a probability in [0, 1]: 26,940 calibration loans and 21,784 validation loans, 0 missing.

| Candidate | Calibration ROC-AUC / PR-AUC | Validation ROC-AUC / PR-AUC | Validation Brier (raw) |
|---|---|---|---|
| `logistic_regression` | 0.662 / 0.254 | 0.666 / 0.258 | 0.236 |
| `xgboost_default` | 0.664 / 0.255 | 0.665 / 0.257 | 0.123 |
| `xgboost_trial1` (selected) | 0.662 / 0.254 | 0.666 / 0.259 | 0.122 |

These are informational only; nothing was selected using them. Platt/isotonic calibration, the final comparison and the champion recommendation belong to the calibration step. `manifest.json` lists the row counts, ranges, parameters and raw metrics.

## Limitations
- The bootstrap resamples loans independently. Out-of-fold rows are clustered in time, so the CIs are probably somewhat narrow.
- `complexity` (trees × 2^depth) is a rough tie-break that ignores learning rate and regularisation.
- `raw_predictions` checks that the selection file matches `model_version`, but it does not hash the trial list. Rerun the search if the config changes.

## Reproduce (repository root, `data/raw/loan.csv` present)
```
python -m src.tabular.xgboost_baseline compare
python -m src.tabular.xgboost_baseline ablate
python -m src.tabular.xgboost_search
python -m src.tabular.lexical_ablation
python -m src.tabular.raw_predictions
LENDINGCLUB_RAW_CSV=data/raw/loan.csv python -m pytest src/tabular -v
```

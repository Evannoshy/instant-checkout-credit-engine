# Fusion Handoff v0 — Base-Model Scores for Fusion

**Handoff version:** `fusion-handoff-v0` · **Status:** exploratory
**Dataset:** `kaggle-adarshsng-local-2026-09-09` · **Split:** `split-v1` · **Cohort:** `real_text_matured_v1`
**Feature set:** `tabular_features_v1` · **Development roles:** `tabular_development_v1` · **Calibration:** `tabular_calibration_v1`
**Produced by:** `python -m src.tabular.fusion_handoff` · **Manifest:** `reports/tabular/fusion_handoff_v0_manifest.json`

## 1. Purpose

This is the first input for the fusion track. Under D-014, fusion v1 is logistic stacking over
base-model probabilities, using out-of-fold scores for the training rows. The handoff provides one
row per loan with three raw base-model scores, for training rows and for validation rows. It also
provides calibrated tabular validation probabilities for later policy work.

The handoff joins, checks and records existing scores. It computes no new score. The only new
values are the calibrated tabular validation probabilities, produced by the existing
`tabular_calibration_v1` Platt mapping. That mapping was refitted on calibration rows and
reproduces the merged evidence (§4).

**This handoff is exploratory.** It is not a model result, and nothing in it has been evaluated as
a fusion model.

## 2. What was joined

Each source was checked against pinned expectations before use: exact loan coverage, one row per
loan, correct split label, numeric probabilities in [0, 1] with no NaN or infinity, one pinned
model name, only pinned model versions, and an approved source checksum.

| Source | Score column | Rows | Pinned version(s) | SHA-256 (first 12) |
|---|---|---:|---|---|
| `data/tabular/tabular_oof_train_v1.parquet` | `p_default_tabular_raw` | 35,234 | `tabular-logistic-v1-time-oof` | `1905ed923969` |
| `reports/tabular/raw_predictions/logistic_regression_validation.csv` | `p_default_tabular` | 21,784 | `tabular-logistic-v1-model-fit` | `2c1a666997d1` |
| `data/nlp/tfidf_oof_train.parquet` | `prob_default_tfidf` | 86,293 | `nlp-tfidf-tuned-v1` | `bd29849563f6` |
| `data/nlp/tfidf_oof_validate.parquet` | `prob_default_tfidf` | 21,784 | `nlp-tfidf-tuned-v1` | `f83d88312a47` |
| `data/nlp/distilbert_scores_train.parquet` | `prob_default_distilbert` | 86,293 | `nlp-distilbert-10k-v1`, `nlp-distilbert-10k-holdout-v1` | `bd3aa33d0e8c` |
| `data/nlp/distilbert_scores_val.parquet` | `prob_default_distilbert` | 21,784 | `nlp-distilbert-10k-v1` | `b1a43c79d64a` |

Full checksums are in the manifest under `sources`.

All six source checksums are pinned in `EXPECTED_SOURCES` and checked before
calibration, table construction or output writes. Parquet files require exact
byte identity. The tabular validation CSV permits only LF/CRLF line-ending
conversion; changed scores, IDs or other content are still rejected. A missing
checksum also stops the build. Rebuilding does not approve new source bytes by
recording their hash. Source updates require a reviewed pin change.

- **The tabular scores are the balanced v1 model**, not the unweighted LR experiment. The
  unweighted file's version (`tabular-logistic-unweighted-v1-model-fit`) is not in the pinned set
  and would be refused.
- **The NLP files were checked against the full frozen sets** (86,293 train, 21,784 validation)
  before any subset was taken, so a broken NLP file cannot hide behind the selection.
- **The tabular OOF file** has no frozen ID list of its own. It was accepted because:
  - its SHA-256 equals the one in its own manifest (`data/tabular/tabular_oof_train_v1_manifest.json`);
  - its row count and per-fold counts equal that manifest's;
  - every loan in it is a `model_fit` loan.

  Its leakage rules (each loan held out from its fold; fit months earlier than scored months) are
  proven by `src/tabular/test_export_oof_scores.py`. That suite passed in full on 17 Oct 2026,
  including the raw-CSV refit, which reproduces every committed score to 1e-10. The folds are not
  re-derived in this handoff.

## 3. Row accounting

| Rows | Count | In this handoff |
|---|---:|---|
| `model_fit`, Jun 2007 – Oct 2011 | 24,119 | **Absent.** These are training-only rows for the first time fold, so no model scored them out of fold |
| `model_fit`, Nov 2011 – Dec 2012 (OOF-scored) | 35,234 | `fusion_train_v0` |
| `calibration`, Jan – Jul 2013 | 26,940 | Not in the fusion tables. The tabular Platt mapping was fitted here |
| `validation`, Aug – Dec 2013 | 21,784 | `fusion_val_v0` and `tabular_calibrated_val` |
| `test`, Jan – Mar 2014 | 14,922 | Locked; never read |

**Tabular folds in `fusion_train_v0`** (`tabular_fold_id`, 0-based):

| Fold | Fit months | Fit rows | Scored months | Rows |
|---:|---|---:|---|---:|
| 0 | 2007-06 – 2011-10 | 24,119 | 2011-11 – 2012-03 | 7,022 |
| 1 | 2007-06 – 2012-03 | 31,141 | 2012-04 – 2012-06 | 6,033 |
| 2 | 2007-06 – 2012-06 | 37,174 | 2012-07 – 2012-08 | 7,095 |
| 3 | 2007-06 – 2012-08 | 44,269 | 2012-09 – 2012-10 | 7,883 |
| 4 | 2007-06 – 2012-10 | 52,152 | 2012-11 – 2012-12 | 7,201 |

**DistilBERT versions:**
- `fusion_train_v0`: 31,060 `nlp-distilbert-10k-v1` and 4,174 `nlp-distilbert-10k-holdout-v1`.
- `fusion_val_v0`: all 21,784 `nlp-distilbert-10k-v1`.

## 4. Outputs

| File | Rows | Columns | SHA-256 (first 12) |
|---|---:|---:|---|
| `data/fusion/fusion_train_v0.parquet` | 35,234 | 12 | `33e426514c4b` |
| `data/fusion/fusion_val_v0.parquet` | 21,784 | 11 | `4d4a911ab17b` |
| `data/tabular/tabular_calibrated_val.parquet` | 21,784 | 7 | `ba9b60eee57c` |

All three files were written by pyarrow 23.0.1 (`parquet-cpp-arrow version 23.0.1`) and
pandas 3.0.5, on Python 3.13.9 / Windows AMD64.

**Fusion table schema:**

| Column | Train | Validation | Role |
|---|:-:|:-:|---|
| `p_default_tabular_raw`, `prob_default_tfidf`, `prob_default_distilbert` | ✓ | ✓ | **The three proposed fusion inputs** |
| `loan_id`, `split` | ✓ | ✓ | Metadata |
| `tabular_fold_id` | ✓ | — | Metadata. Validation has no folds, so there is no null column to fill |
| `{tabular,tfidf,distilbert}_model_name`, `_model_version` | ✓ | ✓ | Metadata |

- Only the three score columns are model inputs. `loan_id`, `split`, `tabular_fold_id` and the
  version fields are metadata and must not be fed to a stacker.
- The score columns keep their source names, so each can be traced back to the file it came from.
- Neither table contains a target or any outcome column.

**How the tables were built:**
- NLP scores were **left-joined** onto the tabular rows by `loan_id`, one-to-one.
- Every loan found a score in every track. A missing score stops the build. It is never filled
  with zero, a mean or an in-sample prediction.

### Calibrated tabular validation export

The `tabular_calibration_v1` Platt mapping was refitted with `calibrate.fit_platt` on the 26,940
calibration-role loans only, then applied to the 21,784 validation loans. Validation outcomes were
used only to check that the merged evidence reproduces. They are not in the file.

| | Value | Merged evidence |
|---|---|---|
| Platt slope | 0.991260 | `calibration_metrics.json` |
| Platt intercept | −1.773576 | `calibration_metrics.json` |
| Reproduction tolerance | 1e-6 (absolute) | `unweighted_comparison.REPRODUCTION_TOLERANCE` |

| Validation (21,784 loans, 3,298 defaults; observed rate 0.1514) | Raw | Platt |
|---|---:|---:|
| Mean probability | 0.4829 | 0.1510 |
| ROC-AUC | 0.665999 | 0.665999 |
| PR-AUC | 0.257538 | 0.257538 |
| Brier score | 0.236309 | 0.122502 |
| Log loss | 0.666574 | 0.403051 |

Both rows reproduce the `logistic_regression` validation rows of `calibration_results.csv`
within 1e-6.

How to read them:
- Platt scaling is a monotone transform, so ROC-AUC and PR-AUC are identical before and after.
- What changes is the level. The −1.77 intercept largely undoes the upward shift from
  `class_weight="balanced"`, bringing the mean from 0.4829 to 0.1510 against an observed 0.1514.
- This is calibration on validation **for reporting only**. Validation grades and never fits, and
  nothing here selects a model or a threshold.

## 5. Score semantics

**The fusion scores are raw scores, i.e. inputs to a stacker. They are not credit
probabilities.** None of them should be read as a probability of default.

| Mean score | Tabular | TF-IDF | DistilBERT |
|---|---:|---:|---:|
| `fusion_train_v0` (observed default rate 0.1629) | 0.4848 | 0.4836 | 0.3838 |
| `fusion_val_v0` (observed default rate 0.1514) | 0.4829 | 0.4933 | 0.4102 |

- Class weighting inflates the tabular and TF-IDF levels.
- DistilBERT is also uncalibrated: it was trained with class weights [1.0, 5.54].
- The fusion tables carry no calibrated score. The v1 Platt mapping belongs to the model fitted
  on all `model_fit` rows, not to the five fold models behind the train scores, so it must never
  be applied to `p_default_tabular_raw` in `fusion_train_v0`.
- **The three train scores are out-of-sample under different protocols.**
  - Tabular uses expanding time folds.
  - TF-IDF uses random 5-fold CV over all 86,293 train loans (`tfidf_tuning_metrics.json`:
    `shuffle=True`), so a loan's text score can come from a model trained on later loans.
  - DistilBERT routes loans between two models, so neither scores its own training loans.

## 6. Test-set non-access

- No code path in this handoff reads the locked test split.
- The expected train and validation ID sets were checked for overlap with `data/test_ids.csv`
  (an ID-only file); the overlap is 0.
- The manifest records `test_rows_available_to_model: 0` and `targets_included: false`.

## 7. Checks run

| Check | Result |
|---|---|
| `python -m pytest src/tabular/test_export_oof_scores.py -v -s -rs` with `LENDINGCLUB_RAW_CSV` set | 23 passed, 0 skipped (17 Oct 2026) |
| `python -m pytest src/tabular/test_fusion_handoff.py -v -s` | 32 passed |
| `LENDINGCLUB_RAW_CSV=data/raw/loan.csv python -m pytest src/tabular/ -v -s -rs` | 1 failed (mismatched environment, see note), 396 passed, 1 skipped |

**Note on the raw-CSV run.** `test_raw_predictions.py::test_real_candidate_refit_reproduces_committed_predictions`
fails by design in this environment. The test requires the exact environment recorded in
`reports/tabular/xgboost_selection.json`, which records Python 3.13.15; this run used 3.13.9.
Every other recorded package version matches. Neither that file nor `raw_predictions.py` is
changed by this handoff. The handoff only reads `logistic_regression_validation.csv`, and that
file's metrics reproduce `calibration_results.csv` (§4).

## 8. Checksum review follow-up (19 October 2026)

Ten regression checks were added for the source pins: changed scores with the
same IDs and model version are refused for all six sources; LF/CRLF copies of
the approved CSV still load; a missing checksum is refused; and a rejected
source leaves existing outputs and their manifest untouched.

The new changed-score checks reproduced the gap before the fix: five sources
were accepted despite changed scores, while the tabular OOF source was already
protected. All 42 handoff tests pass after the fix on Windows AMD64 with Python
3.13.15 and the pinned packages.

The broader `python -m pytest src/tabular -q` run passed with 399 tests, no
failures and 9 skips in 250.59 seconds. The skips are the eight raw-CSV checks
and the optional full calibration rerun; neither opt-in was enabled for this run.

The three committed Parquet files retain their original checksums. Only the
approved hashes were added to the manifest's source pins; its recorded output
hashes, writer details, model settings and calibration results are unchanged.

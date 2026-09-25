# NLP Tuning Report — TF-IDF Grid Search and Out-of-Fold Probabilities

**Document status:** Evidence record for NLP track review
**Version:** 0.1
**Date:** 25 September 2026
**Accountable owner:** NLP Track Lead
**Primary author:** Yati (NLP Track Analyst 2)
**Reviewers:** NLP Track Lead, Tabular Track Tech Lead (parquet consumer), Fusion/API Lead
**Dataset version under review:** `kaggle-adarshsng-local-2026-09-09`
**Split version under review:** `split-v1`
**Model version:** `nlp-tfidf-tuned-v1`
**Supersedes for scoring purposes:** `nlp-baseline-tfidf-v1` (`docs/nlp-baseline-report.md`)

## 1. Purpose

This report records the Week 2 hyperparameter search over the Week 1 TF-IDF baseline and the
out-of-fold probability export that decision D-014 requires of training predictions consumed by
fusion.

It answers three questions: which configuration wins, whether the gain over Week 1 is larger than
sampling noise, and what the tabular and fusion tracks need to know before merging
`prob_default_tfidf` into their models.

Every number is produced by the command in section 11 and written to `reports/nlp/` by the run
itself. None is transcribed by hand.

This report does not select a champion text model, calibrate the exported probability, propose an
operating threshold, or evaluate the locked test split.

## 2. Method

| Choice | Value | Source |
|---|---|---|
| Text input | Week 1 payload with field labels stripped, via `baseline_tfidf.load_baseline_split` | Imported, not re-implemented |
| Grid | `max_features` {3000, 5000, 10000} x `ngram_range` {(1,1), (1,2)} x `C` {0.01, 0.1, 1.0, 10.0} | Analyst 2 brief; 24 points |
| Fixed | `sublinear_tf=True`, `class_weight='balanced'`, `max_iter=1000`, `random_state=42` | `configs/nlp_tfidf_v2.toml` |
| Selection metric | Mean cross-validated PR-AUC (`average_precision`) | Brief lists PR-AUC first |
| Folds | `StratifiedKFold(5, shuffle=True, random_state=42)` | Brief; see NT-02 |
| Metrics | `src/tabular/evaluate.evaluate_predictions` | Shared implementation, not a second copy |

Three properties are load-bearing:

- **The vectoriser sits inside the scikit-learn `Pipeline`.** Fitting TF-IDF on all training rows
  before cross-validating would leak IDF statistics across folds and inflate every figure in this
  report. As a Pipeline step it is refit on each fold's training portion only. A regression test
  proves this by showing a model that ranks its own rows at ROC-AUC above 0.95 in sample collapses
  below 0.75 out of fold on a corpus built to be memorisable.
- **Selection never saw validation.** All 24 points were scored by cross-validation on training
  rows. Validation was scored once, after the winner was fixed.
- **Training probabilities are out of fold.** Each of the 86,293 training rows is scored by a model
  fitted without it. Validation probabilities come from a single fit on all training rows. The two
  files are therefore different quantities and are written separately.

`src/nlp/preprocess.py` is untouched, so the payload remains byte-identical across the TF-IDF and
transformer tracks.

## 3. Environment and artefacts

| Component | Version |
|---|---|
| Python | 3.13.3 |
| scikit-learn | 1.9.1 |
| pandas / numpy | 3.0.5 / 2.5.3 |
| pyarrow | 23.0.1 |
| Raw source CSV | `data/raw/loan.csv`, SHA-256 `23783ef3…9120c`, matching the `raw_csv_sha256` benchmark now published in `split_statistics.json` |

`test_rows_used` is written as `0` in the metrics JSON, as a machine-checkable record that the
locked split was never opened.

## 4. Grid search results

Best mean cross-validated PR-AUC **0.21504 ± 0.00297** (fold standard deviation) at:

| Parameter | Selected |
|---|---|
| `max_features` | 10000 |
| `ngram_range` | (1, 2) |
| `C` | 0.1 |
| `sublinear_tf` | True (fixed) |

Full results are in `reports/nlp/tfidf_grid_search_results.csv`. Marginal effect of each
hyperparameter, averaged over the others:

| `C` | Mean CV PR-AUC | | `max_features` | Mean CV PR-AUC | | `ngram_range` | Mean CV PR-AUC |
|---|---:|---|---|---:|---|---|---:|
| 0.01 | 0.20853 | | 3000 | **0.20440** | | (1,1) | 0.20167 |
| **0.10** | **0.21252** | | 5000 | 0.20306 | | **(1,2)** | **0.20471** |
| 1.00 | 0.20167 | | 10000 | 0.20212 | | | |
| 10.00 | 0.19005 | | | | | | |

**Regularisation strength is the only hyperparameter that matters.** `C` spans 0.0225 across the
grid with a clear peak at 0.1; the whole grid spans 0.18635 to 0.21504. Week 1's `C=1.0` was on the
wrong side of that peak, and `C=10.0` occupies all four worst positions.

`max_features` is inert, and *inverted*: more features is worse on average. See NT-03.

## 5. Results

### 5.1 Out-of-fold training and validation

| Metric | Train (out of fold) | Validation | Trivial-model baseline |
|---|---:|---:|---:|
| Rows | 86,293 | 21,784 | |
| Prevalence | 0.15293 | 0.15140 | |
| ROC-AUC | 0.6121 | **0.5867** | 0.5 |
| PR-AUC | 0.2144 | **0.1944** | prevalence |
| Brier score | 0.23438 | **0.24664** | 0.12848 (constant baseline) |
| Log loss | 0.66135 | 0.68666 | 0.42513 (constant baseline) |
| F1 at 0.50 | 0.2935 | 0.2788 | |
| Best swept F1 | 0.2962 at 0.4723 | 0.2860 at 0.4459 | |

### 5.2 Confusion matrices (validation)

Both thresholds are named because neither is privileged: `class_weight='balanced'` displaces the
natural operating point, so 0.5 carries no special meaning here.

| Threshold | TN | FP | FN | TP | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 0.5000 | 10,054 | 8,432 | 1,398 | 1,900 | 0.184 | 0.576 |
| 0.4459 (best F1) | 6,009 | 12,477 | 666 | 2,632 | 0.174 | 0.798 |

At either threshold the model flags between four and five non-defaulters for every defaulter it catches.
No threshold is recommended; that is a policy-layer decision under D-007.

### 5.3 Improvement over Week 1

Week 1's exact configuration was refitted inside this run as a like-for-like reference, reproducing
its published validation figures precisely (PR-AUC 0.18455, ROC-AUC 0.5693), which also confirms
Week 1 is reproducible.

| Comparison | PR-AUC | 95% interval |
|---|---:|---|
| Tuned (`nlp-tfidf-tuned-v1`) | 0.19437 | [0.18503, 0.20432] |
| Week 1 (`nlp-baseline-tfidf-v1`) | 0.18455 | — |
| **Paired difference** | **+0.00982** | **[0.00549, 0.01406]** |

The difference is computed by paired bootstrap over 1,000 resamples: both models are scored on the
same resampled rows, so the interval describes the difference rather than the sum of two
independent sampling errors. The tuned model wins in **1,000 of 1,000** resamples.

**The gain is real and small.** Tuning moved PR-AUC about one point, roughly a third of the
baseline's entire 3.3-point margin over its prevalence floor. It did not change what the text
modality is worth.

## 6. Calibration: what `prob_default_tfidf` is and is not

The exported column is an **uncalibrated PD** in the sense of D-006. Following repository precedent,
`class_weight='balanced'` is retained, and it inflates the probability scale: mean exported training
probability is **0.4786** against a training prevalence of **0.1529**, a factor of 3.1. Validation
Brier (0.24664) is nearly double the constant baseline's 0.12848, and log loss is worse too.

The winning configuration refitted with `class_weight=None` is recorded as a diagnostic:

| Variant | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| `class_weight='balanced'` (shipped) | 0.5867 | 0.1944 | 0.24664 | 0.68666 |
| `class_weight=None` (diagnostic) | **0.5876** | **0.1948** | **0.12708** | **0.41955** |
| Constant baseline | 0.5 | 0.1514 | 0.12848 | 0.42513 |

The unweighted variant is **better on every metric**, and its Brier of 0.12708 is the only figure in
this report that beats the constant baseline's 0.12848. Balanced weighting is costing calibration
and buying nothing measurable in ranking. See NT-01.

This does not invalidate the shipped file for its stated purpose: a logistic stacker is unaffected by
a monotone rescaling. It does mean nobody may read these values as probabilities of default.

## 7. Cross-track context

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|
| Tabular logistic (`tabular-logistic-v1`) | 0.6679 | 0.2594 |
| **TF-IDF tuned (`nlp-tfidf-tuned-v1`)** | **0.5867** | **0.1944** |
| TF-IDF Week 1 | 0.5693 | 0.1845 |
| DistilBERT prototype | 0.5046 | 0.1672 |
| Constant baseline | 0.5 | 0.1514 |

Text remains well behind the structured model. The DistilBERT prototype figures come from 1,000
training and 500 validation rows and are not comparable on sample size.

## 8. Findings and required follow-up

| ID | Finding | Severity | Required action | Owner |
|---|---|---|---|---|
| NT-01 | `class_weight='balanced'` degrades calibration for no ranking gain. Unweighted is better on ROC-AUC, PR-AUC, Brier and log loss, and is the only variant beating the constant-baseline Brier | High | Drop balanced weighting, or add an explicit calibrator, for the next text model version. **This applies to the tabular track too:** `tabular-logistic-v1` reports Brier 0.23214 and log loss 0.65770, both worse than the constant baseline, from the same cause | NLP + Tabular leads |
| NT-02 | Out-of-fold training figures are optimistic relative to chronological validation by 0.0200 PR-AUC and 0.0255 ROC-AUC. Random `StratifiedKFold` folds mix issue months, so an OOF row can be scored by a model fitted on later loans | High | Fusion must not read the OOF column's apparent strength as out-of-time generalisation. A time-aware fold scheme is the obvious v2 change; the brief specified 5-fold, so this is documented rather than silently altered | Fusion + NLP leads |
| NT-03 | `max_features=10000` is not meaningfully better than 3000: +0.00085 mean CV PR-AUC against a fold standard deviation of 0.00297. Averaged over the grid, more features is *worse* | Medium | Prefer 3000 in the next version: one third the vocabulary, no measurable loss, and it directly reduces the rare-term problem recorded as NB-02 in the Week 1 report | Analyst 2 |
| NT-04 | The tuning gain is +0.00982 PR-AUC, 95% interval [0.00549, 0.01406] | Medium | Real but small. Do not present tuning as having changed the text modality's value | Analyst 2 |
| NT-05 | The exported probability is uncalibrated, mean 0.4786 against prevalence 0.1529 | Medium | Safe for logistic stacking; unsafe to read as a probability. The `model_version` column exists so a later calibrated version is distinguishable | Fusion lead |
| NT-06 | Training payloads average 47.5 words, validation 26.2, because borrower descriptions shortened over the cohort period | Medium | The OOF train column and the validation column are drawn from measurably different text distributions, so a stacker fitted on the former may not transfer cleanly. Worth checking before fusion v1 is trusted | Fusion + NLP leads |

## 9. Limitations

- **One split, one seed.** Confidence intervals are reported for the validation PR-AUC and for the
  Week 1 comparison, but not for every figure in section 5.
- **No calibration assessment beyond Brier and log loss.** No reliability diagram is produced for
  the text model; the tabular track's `logistic_calibration.png` is the pattern to follow when a
  calibrated version exists.
- **`sublinear_tf` was fixed, not swept.** It reads as a stated setting in the brief. Week 1 used
  the scikit-learn default of `False`, so it is a confound between the two versions; the Week 1
  reference in section 5.3 holds it at `False` deliberately so the comparison is like for like.
- **Evidence lane, proxy population.** LendingClub instalment loans are not Singapore BNPL
  checkout. Nothing here supports a BNPL accuracy claim.
- **Test split untouched**, per D-015 and the locked-test-set guardrail.

## 10. Consuming the exported probabilities

Two Parquet files, snappy-compressed, written to tracked `data/nlp/` alongside
`text_lexical_features.parquet`. The brief named `data/processed/nlp/`, which is gitignored at
`.gitignore` line 28 and would not have reached the tabular track through version control.

| File | Rows | Contents |
|---|---:|---|
| `data/nlp/tfidf_oof_train.parquet` | 86,293 | Out-of-fold probabilities, one per training loan |
| `data/nlp/tfidf_oof_validate.parquet` | 21,784 | Single-fit probabilities for validation loans |

Columns: `loan_id`, `split`, `prob_default_tfidf`, `model_name`, `model_version`. The probability
name is as the brief specified; `model_name` and `model_version` are what D-014 means by versioned
probabilities keyed by locked `loan_id`.

```python
text = pd.read_parquet("data/nlp/tfidf_oof_train.parquet")
merged = tabular_train.merge(text[["loan_id", "prob_default_tfidf"]],
                             on="loan_id", how="inner", validate="one_to_one")
```

Both files were verified to cover their split's frozen ID set exactly, with no duplicates, no
nulls, and every value inside [0, 1]. A one-to-one merge against the manifest recovers all 86,293
training rows, and re-scoring the merged file reproduces ROC-AUC 0.6121.

## 11. Reproduction

From the repository root, with `data/raw/loan.csv` in place:

```
python -m pip install -r src/nlp/requirements.txt
python -m pytest src/nlp/test_evaluate_baseline.py -v -s
python -m src.nlp.evaluate_baseline
```

For a full repository run with nothing skipped, point the opt-in real-data checks at the verified
source CSV:

```
LENDINGCLUB_RAW_CSV="$PWD/data/raw/loan.csv" python -m pytest src/ scripts/ -q
```

Runtime is approximately 3 minutes 40 seconds for 120 grid fits plus the out-of-fold pass, the
final fit, two reference fits and 2,000 bootstrap resamples.

## 12. Verification

Thirty-three checks in `src/nlp/test_evaluate_baseline.py`. The suite loads no borrower data and
does not require `data/raw/loan.csv`, so it runs on a fresh checkout. Four are load-bearing for this
report's integrity:

- `test_out_of_fold_probabilities_are_not_in_sample` — the cross-fold leakage guard described in
  section 2.
- `test_pipeline_keeps_the_vectorizer_unfitted_so_folds_refit_it` — asserts the vectoriser carries
  no vocabulary before fitting, so cross-validation must rebuild it per fold.
- `test_validate_rejects_duplicate_loan_ids` and `test_validate_rejects_incomplete_coverage` — the
  conditions that would silently corrupt the tabular team's merge.

Repository-wide, **227 tests pass with none skipped**, across both tracks and the `scripts/`
integration checks. That figure includes the three opt-in real-data regression tests gated behind
`LENDINGCLUB_RAW_CSV`, which were run against the verified raw CSV rather than skipped.

## 13. Definition-of-done status

| Requirement | Status | Evidence |
|---|---|---|
| Grid search over the briefed TfidfVectorizer and LogisticRegression settings | Met | §4; 24 points, asserted in `test_shipped_config_loads_and_declares_the_briefed_grid` |
| Winning configuration trained on the full train split | Met | §4, §5.1 |
| Validation probabilities in (0,1) exported | Met | §10; range [0.17657, 0.84140], schema-validated |
| 5-fold out-of-fold probabilities for every training row | Met | §5.1; 86,293 rows, each scored by a model fitted without it |
| Parquet export for `pd.merge()` | Met | §10; one-to-one merge verified against the manifest |
| ROC-AUC, PR-AUC, confusion matrix and Brier score reported | Met | §5.1, §5.2 |
| Improvement over Week 1 recorded | Met | §5.3; +0.00982 PR-AUC, 95% interval [0.00549, 0.01406] |
| Locked test split untouched | Met | `test_rows_used: 0`; loader refuses `split="test"` |
| No raw LendingClub data committed | Met | `data/raw/` gitignored; exports carry `loan_id` and a probability only |

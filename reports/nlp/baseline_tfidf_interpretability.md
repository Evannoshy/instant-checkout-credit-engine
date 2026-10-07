# TF-IDF Baseline: Interpretability, Calibration and Threshold Trade-offs

**Document status:** Evidence record for NLP track review
**Version:** 0.1
**Date:** 7 October 2026
**Accountable owner:** NLP Track Lead
**Primary author:** Yati (NLP Track Analyst 2)
**Reviewers:** NLP Track Lead, Fusion/API Lead, Tabular Track Tech Lead (§4 applies to `tabular-logistic-v1`)
**Model under diagnosis:** `nlp-tfidf-tuned-v1` (`max_features=10000`, `ngram_range=(1,2)`, `C=0.1`, `sublinear_tf=True`)
**Diagnostic version:** `nlp-tfidf-interpretability-v1`

## 1. Purpose

This report diagnoses the tuned TF-IDF model `nlp-tfidf-tuned-v1`. It answers what the model has
learned, whether its probabilities can be made honest, and what it costs to use at any operating
threshold. **It changes no shipped artefact**: `data/nlp/tfidf_oof_*.parquet` and
`nlp-tfidf-tuned-v1` are unmodified.

Every figure is produced by the command in §7 and written to `reports/nlp/` by the run itself.

### 1.1 Rebuilding rather than loading

No fitted model is persisted anywhere in this repository — `*.joblib` and `*.pkl` are gitignored,
and `src/tabular/logistic_baseline.py` states the convention: *"No trained model is ever written to
disk: fitting is fast and deterministic."* The winning hyperparameters are therefore read from
`reports/nlp/tfidf_tuning_metrics.json` and refitted under the pinned `random_state=42`, which
reproduces the same pipeline in about 20 seconds without adding a binary artefact to version.

## 2. Coefficient extraction

Top 30 coefficients in each direction, with training document frequency, are in
`reports/nlp/tfidf_coefficients.csv`.

### 2.1 Artefact filtering

"Uninformative" is not left to taste. The three classes are the ones already recorded in
`docs/nlp-baseline-report.md` §6:

| Flag | Rule | Source finding |
|---|---|---|
| `low_document_frequency` | Appears in under 1% of training documents | NB-02 |
| `cross_field_bigram` | A bigram whose within-field occurrences are ≤20% of its payload occurrences | NB-03 |
| `period_specific` | Relative frequency swings more than 10x across issue-year buckets | NB-04 |

Of 60 ranked terms, **29 are flagged and 31 retained** (12 higher-risk, 19 lower-risk): 27 for low
document frequency, 4 as boundary bigrams, 4 as period-specific, with overlap.

Period-specific terms are measured as a share of all term occurrences in their bucket, not as raw
document frequency. Descriptions shortened steadily over the cohort period, so raw frequency falls
for almost every term for reasons of length alone. The thin 2007 and 2008 years (246 and 1,562 rows)
are merged into one 2007–2009 bucket, because a term with a few hundred documents is absent from
them by arithmetic rather than by era.

### 2.2 Retained higher-risk terms

| n-gram | Coefficient | Train doc % |
|---|---:|---:|
| `bills` | 1.616 | 12.09 |
| `business` | 1.606 | 5.14 |
| `need` | 1.243 | 9.81 |
| `help` | 1.068 | 12.06 |
| `00` | 0.799 | 2.24 |
| `loans` | 0.729 | 5.78 |
| `small business` | 0.690 | 3.02 |
| `plus` | 0.620 | 1.02 |
| `payment` | 0.616 | 21.82 |
| `lower monthly` | 0.612 | 1.26 |
| `thank` | 0.592 | 11.23 |
| `personal` | 0.591 | 6.14 |

### 2.3 Retained lower-risk terms (leading twelve of nineteen)

| n-gram | Coefficient | Train doc % |
|---|---:|---:|
| `rate` | −1.767 | 14.26 |
| `college` | −1.109 | 2.72 |
| `apr` | −0.987 | 2.51 |
| `rates` | −0.921 | 6.28 |
| `excellent credit` | −0.826 | 1.37 |
| `ve` | −0.824 | 3.86 |
| `late payment` | −0.744 | 1.54 |
| `lower rate` | −0.698 | 4.87 |
| `way` | −0.663 | 2.79 |
| `mortgage` | −0.654 | 2.47 |
| `balance` | −0.651 | 3.75 |
| `refinance` | −0.633 | 3.62 |

Coefficients are descriptive associations. Their magnitudes depend on IDF scaling and split across
correlated n-grams, so they identify neither risk drivers nor causes — the same boundary the project
applies to SHAP under D-009.

### 2.4 Rankings are materially cleaner than the untuned baseline

| | `nlp-baseline-tfidf-v1` (`C=1.0`, 5k, `sublinear_tf=False`) | `nlp-tfidf-tuned-v1` (`C=0.1`, 10k, `sublinear_tf=True`) |
|---|---:|---:|
| Ranked terms under 1% document frequency | 38 of 40 (95%) | 27 of 60 (45%) |
| Median document frequency of ranked terms | 0.181% | 1.246% |

Quadrupling the regularisation raised the median document frequency behind a ranked term **6.9x**.
`docs/nlp-baseline-report.md` recommended a `min_df` filter to fix this (NB-02); stronger regularisation
achieved it without one.

### 2.5 Two interpretability findings

**All four boundary bigrams are the same mechanism.** `purchase major`, `card credit`,
`improvement home` and `debts debt` are reversals of `major purchase`, `credit card`,
`home improvement` and `debt`/`debts` — produced where the `purpose` category repeats the title
across the payload join. `purchase major` occurs in 492 payload documents but inside a single field
in only 9. See NI-05: the baseline report's ablation already showed title and purpose contribute 0.0004 PR-AUC,
so dropping them from the TF-IDF input would remove this entire artefact class at no measurable cost.

**`late payment` ranks as protective** (−0.744, 1.54% of documents), which is not a sensible risk
association. The likely explanation is negation: borrowers writing "never had a late payment".
`src/nlp/preprocess.py` deliberately preserves negation, but a bag of unigrams and bigrams discards
it, so "no late payments" and "a late payment" are nearly indistinguishable to this model. See NI-06.

## 3. Calibration

Reliability diagrams: `reports/nlp/tfidf_calibration_curves.png`, 10 quantile bins, validation split.
Left panel contrasts the weighting choice; right panel shows the effect of calibrating the shipped
model.

| Variant | Brier | Log loss | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| `balanced` (as shipped, uncalibrated) | 0.24664 | 0.68666 | 0.5867 | 0.1944 |
| `unweighted` (`class_weight=None`) | **0.12708** | 0.41955 | 0.5876 | 0.1948 |
| `platt_oof` | 0.12742 | — | 0.5867 | 0.1944 |
| `isotonic_oof` | 0.12734 | — | 0.5868 | 0.1913 |
| `platt_role` | 0.12723 | — | 0.5822 | 0.1920 |
| `isotonic_role` | 0.12743 | — | 0.5815 | 0.1873 |
| `platt_classifier_cv` | 0.12726 | — | 0.5875 | 0.1948 |
| `isotonic_classifier_cv` | 0.12729 | — | 0.5872 | 0.1947 |
| Constant baseline (reference) | 0.12848 | 0.42513 | 0.5 | 0.1514 |

Reference sets: the out-of-fold variants calibrate on all 86,293 exported OOF probabilities. The
role variants mirror the chronological boundaries proposed in `configs/tabular_development_v1.toml`
on the tabular track — fit on `model_fit` months 2007-06 to 2012-12 (59,353 rows), calibrate on
2013-01 to 2013-07 (26,940 rows). Those months are read from the manifest here, so nothing in this
report depends on that unmerged branch. `CalibratedClassifierCV` is reported separately because it
cannot consume precomputed OOF probabilities: it wraps an estimator and runs its own internal
cross-validation, answering a slightly different question.

### 3.1 What the table says

**Every calibration route lands in the same place, and none beats simply not breaking the scale.**
The six calibrated variants span 0.12723 to 0.12743 — a range of 0.0002 — while `class_weight=None`
with no calibrator at all scores 0.12708. All eight beat the constant baseline's 0.12848; the shipped
`balanced` variant, at 0.24664, is the only one that does not.

Those differences are far too small to rank. No confidence intervals were computed for Brier, and a
0.0002 spread is not a measurable preference. The recommendation in NI-01 rests on simplicity — one
fewer fitted component to version, bundle and keep compatible under D-006 — not on a measured
advantage.

**Platt preserves the ranking exactly; isotonic does not.** `platt_oof` reproduces the shipped
ROC-AUC to six decimal places (0.586680 against 0.586680) because a logistic recalibration is
strictly monotone. Isotonic is monotone but ties scores together, costing 0.0030 PR-AUC
(0.1913 against 0.1944). Same Brier, worse ranking: there is no reason to prefer it here.

**The chronological role split costs ranking without buying calibration.** `role_base` scores
ROC-AUC 0.5822 against 0.5867, because it is fitted on 59,353 rows instead of 86,293. Its calibrated
Brier (0.12723) is indistinguishable from the OOF route's (0.12742). For the text model the
already-exported OOF probabilities are a sufficient calibration reference. This is not an argument
against the tabular track's role split, which exists for reasons beyond text calibration.

## 4. This applies to the tabular baseline too

`tabular-logistic-v1` uses `class_weight='balanced'` and reports Brier 0.23214 and log loss 0.65770,
both worse than a constant prediction (0.12848 / 0.42513). This report shows the mechanism is the
probability scale rather than the model, and that it is fully recoverable. `docs/nlp-tuning-report.md`
raised this as NT-01; §3 is the measured confirmation.

## 5. Threshold trade-offs

Swept on the shipped uncalibrated probabilities, so the figures are comparable with
`docs/nlp-tuning-report.md`. Full table: `reports/nlp/tfidf_threshold_sweep.csv`.

| Threshold | TP | FP | Precision | Recall | F1 | FP per TP | Share of applicants flagged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 3,298 | 18,486 | 0.1514 | 1.0000 | 0.2630 | 5.61 | 100.0% |
| 0.20 | 3,298 | 18,480 | 0.1514 | 1.0000 | 0.2630 | 5.60 | 100.0% |
| 0.30 | 3,270 | 18,192 | 0.1524 | 0.9915 | 0.2641 | 5.56 | 98.5% |
| 0.35 | 3,187 | 17,354 | 0.1552 | 0.9663 | 0.2674 | 5.45 | 94.3% |
| 0.40 | 2,965 | 15,412 | 0.1613 | 0.8990 | 0.2736 | 5.20 | 84.4% |
| 0.45 | 2,572 | 12,179 | 0.1744 | 0.7799 | **0.2850** | 4.74 | 67.7% |
| 0.50 | 1,900 | 8,432 | 0.1839 | 0.5761 | 0.2788 | 4.44 | 47.4% |
| 0.55 | 1,183 | 4,726 | 0.2002 | 0.3587 | 0.2570 | 3.99 | 27.1% |
| 0.60 | 547 | 1,923 | **0.2215** | 0.1659 | 0.1897 | 3.52 | 11.3% |

Because Platt calibration is strictly monotone, this trade-off is a property of the ranking and is
unchanged by whether a calibrated variant ships. Calibration re-indexes the thresholds; it does not
move the curve.

### 5.1 Recall targets

| Target | Threshold | TP | FP | Precision | FP per TP | Share of applicants flagged |
|---|---:|---:|---:|---:|---:|---:|
| ≥70% of defaults | 0.4714 | 2,309 | 10,615 | 0.1787 | 4.60 | **59.3%** |
| ≥80% of defaults | 0.4447 | 2,639 | 12,558 | 0.1737 | 4.76 | **69.8%** |

Thresholds are searched over observed score values rather than the 0.05 grid, so these are real
operating points.

**There is no usable standalone operating point.** Precision peaks at 0.2215 anywhere in the swept
range — 1.46x the 0.1514 base rate — and only while catching 16.6% of defaults. Capturing 70% of
defaults means flagging 59.3% of all applicants and wrongly flagging 10,615 non-defaulters to catch
2,309 defaulters. This model is a fusion input, not a decision gate. Threshold selection remains a
policy-layer decision under D-007; nothing here recommends one.

## 6. Findings and required follow-up

| ID | Finding | Severity | Required action | Owner |
|---|---|---|---|---|
| NI-01 | Calibration fully recovers the Brier score, but `class_weight=None` alone matches every calibrated route (0.12708 against a 0.12723–0.12743 spread) with no extra component. The spread is too small to rank | High | Drop balanced weighting in the next text model version rather than adding a calibrator. Recommendation rests on simplicity, not measured superiority | NLP lead |
| NI-02 | Platt preserves ranking exactly (ROC-AUC identical to six decimals); isotonic ties scores and costs 0.0030 PR-AUC for the same Brier | Medium | If a calibrator is wanted despite NI-01, use Platt. Do not use isotonic here | NLP lead |
| NI-03 | The chronological calibration-role variant costs 0.0045 ROC-AUC (27k fewer training rows) and does not improve calibrated Brier | Medium | For the text model, calibrate on the already-exported OOF probabilities. Not an argument against the tabular role split | NLP + Tabular leads |
| NI-04 | `C=0.1` raised the median document frequency behind a ranked term 6.9x versus the untuned baseline (0.181% to 1.246%), and cut sub-1% terms from 95% to 45% | Medium | The baseline report's NB-02 `min_df` recommendation can be closed: regularisation addressed it | Analyst 2 |
| NI-05 | Four of the top 30 higher-risk terms are payload-join artefacts, all caused by `purpose` repeating the title. The baseline report's ablation measured title and purpose as worth 0.0004 PR-AUC | Medium | Drop title and purpose from the TF-IDF input, or vectorise fields separately. Removes the whole artefact class at no measurable cost | Analyst 2 |
| NI-06 | `late payment` ranks as protective, almost certainly negation ("never had a late payment"). Bag-of-ngrams cannot represent negation, which preprocessing deliberately preserves | Medium | Treat as a known ceiling of the lexical approach and a concrete argument for the transformer track, which can model it | NLP lead |
| NI-07 | No usable standalone threshold. Precision peaks at 0.2215 against a 0.1514 base rate; 70% recall requires flagging 59.3% of applicants | High | Confirms the model's role as a fusion input only. No operating threshold should be quoted from this model alone | Fusion lead + co-leads |

## 7. Reproduction

```
python -m pip install -r src/nlp/requirements.txt
python -m pytest src/nlp/test_interpret_baseline.py -v -s
python -m src.nlp.interpret_baseline
```

Runtime is approximately 1 minute 55 seconds, including the two `CalibratedClassifierCV` runs
(`--skip-classifier-cv` omits them). For a full repository run with nothing skipped:

```
LENDINGCLUB_RAW_CSV="$PWD/data/raw/loan.csv" python -m pytest src/ scripts/ -q
```

## 8. Verification

Twenty-eight checks in `src/nlp/test_interpret_baseline.py`, loading no borrower data and not
requiring `data/raw/loan.csv`. Repository-wide, **255 tests pass with none skipped**.

Four are load-bearing for this report:

- `test_within_field_counts_miss_a_bigram_that_only_spans_a_boundary` — the mechanism behind §2.5.
- `test_cross_field_rule_tolerates_incidental_within_field_hits` — pins the ratio rule against the
  real `purchase major` case (9 of 492 documents), which an exact-zero rule misses entirely.
- `test_volatility_attributes_counts_to_the_right_term` — guards the index-alignment trap where
  `vocabulary_` keys are in insertion order while matrix columns follow index values.
- `test_platt_calibration_preserves_ranking_exactly` — the property §5 relies on.

## 9. Limitations

- **No confidence intervals on Brier.** The calibrated variants differ by 0.0002 and are not
  separable. §3.1 states this rather than ranking them.
- **One split, one seed.** Single chronological validation split; no repeated-seed variance.
- **Coefficients are associations, not drivers.** Magnitudes depend on IDF scaling and split across
  correlated n-grams (D-009 discipline).
- **The negation reading of `late payment` is inferred, not measured.** Confirming it would need a
  negation-scoped count, which this report does not perform.
- **Thresholds swept on uncalibrated probabilities.** Valid because Platt is monotone, but the
  threshold *values* would change under a calibrated model even though the trade-off would not.
- **Role boundaries mirror a proposed, unmerged config.** If `tabular_development_v1` changes before
  merge, §3's role variant would need rerunning.
- **Evidence lane, proxy population.** LendingClub instalment loans are not Singapore BNPL checkout.
- **Test split untouched** — `test_rows_used: 0`; the loader refuses `split="test"`.

## 10. Definition-of-done status

| Requirement | Status | Evidence |
|---|---|---|
| Ingest the fitted pipeline from `evaluate_baseline.py` | Met, by deterministic rebuild | §1.1; no persisted model exists by repository convention |
| Top 30 positive and top 30 negative coefficients | Met | §2.2, §2.3; full list in `tfidf_coefficients.csv` |
| Uninformative artefacts filtered, corpus document frequencies calculated | Met | §2.1; 29 of 60 flagged across three documented classes, document frequency on every term |
| 10-bin quantile reliability diagrams for `balanced` and `None` | Met | §3; left panel of `tfidf_calibration_curves.png` |
| Isotonic and Platt calibrators fitted, calibrated Brier on validation | Met | §3; four fitted calibrators plus the `CalibratedClassifierCV` cross-check |
| Plot saved to `reports/nlp/tfidf_calibration_curves.png` | Met | §3 |
| Precision, recall and FP:TP across thresholds 0.10–0.60 step 0.05 | Met | §5; 11 thresholds |
| Threshold for ≥70% and ≥80% default capture, with non-defaulters flagged | Met | §5.1; 0.4714 and 0.4447, flagging 59.3% and 69.8% of applicants |
| Compiled into `reports/nlp/baseline_tfidf_interpretability.md` | Met | This document |
| No raw LendingClub data committed | Met | `data/raw/` gitignored; outputs are aggregates and ranked terms only |

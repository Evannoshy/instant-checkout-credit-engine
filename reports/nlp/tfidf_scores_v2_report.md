# Unweighted TF-IDF Scores for Fusion (`nlp-tfidf-fusion-v2`)

**Document status:** Evidence record for the fusion experiment
**Version:** 0.1
**Date:** 6 November 2026
**Accountable owner:** NLP Track Lead
**Primary author:** Yati (NLP Track Analyst 2)
**Reviewers:** Allan (stacker consumer), Brandon (shared handoff), Fusion/API Lead
**Role contract:** `fusion_development_v1`, approved by Evan and Ariel on 28 October 2026
**Export:** `data/nlp/tfidf_scores_v2.parquet` — 48,724 loans

## 1. Purpose

Produces TF-IDF scores Allan can use directly in the late-fusion stacker, under the role boundaries
approved in `configs/fusion_development_v1.toml`. The existing balanced v1 exports are untouched and
keep their own version label.

Three properties define this export:

- **Everything is fitted on `base_fit` only.** Vocabulary, IDF weights, hyperparameter selection and
  the final coefficients all come from the 59,353 loans issued 2007-06 to 2012-12. No scoring-role
  loan influences anything it is later scored by.
- **Settings are chosen under time-ordered folds.** Expanding-window folds reused from
  `src.tabular.xgboost_baseline.time_aware_folds` with the tabular track's own `n_folds=5` and
  `min_train_share=0.4`, so both tracks select under comparable conditions. This replaces the random
  folds behind `nlp-tfidf-tuned-v1`, whose optimism was recorded as NT-02.
- **The probability is unweighted, not calibrated.** `class_weight=None`. Nothing post-hoc is applied
  and the export is not described as calibrated anywhere.

## 2. The role mapping is read as a contract

`src/tabular/fusion_roles.py` states that the CSV is the cross-track contract and other tracks read
the file rather than importing that module. `data/fusion/fusion_roles_v1.csv` is therefore loaded
directly, and its SHA-256 is pinned in `configs/nlp_tfidf_fusion_v2.toml` and verified before anything is
fitted. A changed mapping stops the run rather than silently rescoring a different population.

| Role | Months | Rows | In this export |
|---|---|---:|---|
| `base_fit` | 2007-06 → 2012-12 | 59,353 | No — fitting only |
| `fusion_fit` | 2013-01 → 2013-04 | 15,587 | Yes |
| `fusion_calibration` | 2013-05 → 2013-07 | 11,353 | Yes |
| `validation` | 2013-08 → 2013-12 | 21,784 | Yes |
| **Export total** | | **48,724** | |

Every loan ID, split label and role label was checked against the mapping by a validated one-to-one
merge. A missing or extra loan is an error; nothing is filled or dropped.

## 2.1 Naming and why this is a new module

Two review points worth stating plainly rather than leaving a reader to infer.

**The model version is the authoritative identifier**, and the config file is named for it:
`configs/nlp_tfidf_fusion_v2.toml` produces `nlp-tfidf-fusion-v2`, exported as
`tfidf_scores_v2.parquet`, matching the DistilBERT sibling's `nlp-distilbert-fusion-v2` /
`distilbert_scores_v2.parquet`. Note that the older `configs/nlp_tfidf_v2.toml` produces
`nlp-tfidf-tuned-v1` — that config's counter and its model version were never aligned. Nothing is
keyed by config filename; every export, manifest and test keys on `model_version`.

**This is a new module, not an extension of `evaluate_baseline.py`.** That runner produces the v1
out-of-fold export: it fits on all 86,293 training loans with random stratified folds, and its
schema and behaviour are pinned by 33 tests and consumed through the shipped v1 parquets. This
export fits on `base_fit` only, selects under expanding-window time-ordered folds, and writes a
different schema under a different role contract. Merging them would put two incompatible fold
schemes and two export schemas behind flags in one module, where a change meant for one could break
the other. The pipeline is genuinely reused rather than copied — `build_pipeline` and
`load_baseline_split` are imported from the v1 modules unchanged — and the v1 exports keep their own
version label, which a test asserts.

## 3. Selection

Twenty-four grid points scored on `base_fit` folds alone. Selected:
`max_features=10000`, `ngram_range=(1,2)`, `C=0.1`, `sublinear_tf=True`.
Mean cross-validated PR-AUC **0.22088 ± 0.01247**.

| Fold | Fit rows | Holdout rows | Holdout months |
|---|---:|---:|---|
| 0 | 24,119 | 7,022 | 2011-11 → 2012-03 |
| 1 | 31,141 | 6,033 | 2012-04 → 2012-06 |
| 2 | 37,174 | 7,095 | 2012-07 → 2012-08 |
| 3 | 44,269 | 7,883 | 2012-09 → 2012-10 |
| 4 | 52,152 | 7,201 | 2012-11 → 2012-12 |

The same settings won as under v1's random folds on all 86,293 rows, which is reassuring for
stability. But **the fold standard deviation is 0.01247, four times the 0.00297 seen under random
folds** — time-ordered folds across different periods disagree far more than random folds do. The
winner beats `max_features=3000` by 0.00035, which is 3% of one fold standard deviation. On this
evidence the three `max_features` values are indistinguishable, and NT-03's recommendation to prefer
3,000 stands on grounds of leanness rather than performance.

## 4. Matched balanced-versus-unweighted comparison

Same `base_fit` rows, same folds, same winning settings; only `class_weight` differs. The v1 balanced
figures are not a valid reference here — they used all 86,293 train rows and random folds.

| | Mean CV PR-AUC | Validation ROC-AUC | Validation PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| **Unweighted (shipped)** | 0.22088 | 0.5816 | **0.1928** | **0.12723** | **0.42026** |
| Balanced | 0.22105 | 0.5822 | 0.1920 | 0.24784 | 0.68919 |
| Constant baseline | — | 0.5 | 0.1514 | 0.12848 | 0.42513 |

Ranking is a tie: the two differ by 0.0008 PR-AUC and 0.0006 ROC-AUC, both far inside the 0.0125 fold
standard deviation, and balanced is marginally ahead in cross-validation while unweighted is
marginally ahead on validation. **Probability quality is not a tie.** Unweighted halves the Brier
score and is the only variant beating the constant baseline, confirming NI-01 under properly matched
conditions for the first time.

The practical consequence for the stacker: mean exported probability is **0.1581** against a 0.1514
validation prevalence. The v1 balanced export averaged 0.4786, a 3.1x inflation.

## 5. Vocabulary reach

`base_fit` ends in December 2012 and every scored loan is from 2013, so the vocabulary is fitted
entirely on months that predate everything it scores. Borrower descriptions also shortened sharply
over the cohort period. Coverage was therefore measured rather than assumed, at the level of analysed
n-gram occurrences:

| Role | Unigram | Bigram | Combined |
|---|---:|---:|---:|
| `fusion_fit` | 96.88% | 63.57% | 80.73% |
| `fusion_calibration` | 96.87% | 64.88% | 81.41% |
| `validation` | 97.25% | 67.24% | 82.78% |

This is the reassuring answer: the stricter fitting boundary did not degrade reach. Unigram coverage
is near-total and combined coverage above 80% in every role, and coverage is slightly *better* on
`validation` than on the earlier roles. Roughly a third of bigram occurrences still fall outside the
vocabulary, which is a property of a 10,000-feature cap shared between unigrams and bigrams rather
than of the role split.

An earlier exploratory measurement on the v1 train-to-validation boundary gave comparable figures,
but that analysis was not retained in the repository, so it is mentioned only as context and is not
reproducible from these artefacts. The numbers in the table above are.

## 6. Unusable text

The task sheet asked Evan and Ariel to confirm whether unusable text means an empty description or an
empty cleaned payload, and to record the definition in the handoff. **No definition has been recorded
in any config, document or module.** Both candidate definitions were measured on the 48,724 exported
loans:

| Definition | Rows affected |
|---|---:|
| Empty raw description | **0** |
| Empty cleaned title-purpose-description payload | **0** |
| No in-vocabulary token survives tokenisation | **3** |

Neither proposed definition selects any row, because the `real_text_matured_v1` cohort requires a
non-empty description and the shared loader rejects empty payloads. Three loans do produce an
all-zero feature vector, so their probability is the model intercept rather than a text signal —
arguably "inventing a text-model probability" in substance, which the sheet forbids.

No fallback flag is emitted: the agreed six-column schema has no field for one, and inventing a
column unilaterally would break the match with the DistilBERT export. The count is recorded in the
manifest. **Three rows out of 48,724 is small enough to decide calmly and too specific to leave
undefined** — this needs a ruling from Evan and Ariel.

## 7. Compatibility with the DistilBERT export

Checked directly against `distilbert_scores_v2.parquet` on `feature/distilbert-fusion-v2`:

| Property | Match |
|---|---|
| Column names and order | Yes |
| Column dtypes | Yes |
| Loan ID set | Yes, identical 48,724 |
| `(loan_id, split, role)` rows | Yes, identical after sorting |
| `raw_prob == sigmoid(raw_margin)` | Yes, both exports |

The two NLP exports are therefore joinable without reconciliation.

## 8. Findings and follow-up

| ID | Finding | Severity | Required action | Owner |
|---|---|---|---|---|
| FS-01 | No definition of unusable text has been recorded. Both proposed definitions select zero rows; 3 rows have an all-zero feature vector and receive an intercept-only probability | Medium | Rule on the definition and whether these 3 rows need a fallback. The schema has no flag column, so a change affects both NLP exports | Evan + Ariel |
| FS-02 | Unweighted and balanced are indistinguishable on ranking (0.0008 PR-AUC against a 0.0125 fold standard deviation) but unweighted halves the Brier score | Low | None for fusion — the unweighted export is shipped. Confirms NI-01 under matched conditions | NLP lead |
| FS-03 | Time-ordered fold variance is 4x that of random folds (0.01247 against 0.00297), so the grid cannot separate the three `max_features` values | Low | Treat the selected 10,000 as one of three equivalent choices. NT-03's preference for 3,000 is about leanness, not accuracy | Analyst 2 |
| FS-04 | Reusing `time_aware_folds` drags in xgboost, which imports at module level and needs an OpenMP runtime. The fold logic itself is pure pandas | Low | Consider moving the fold builder to a module without a model-specific dependency. Reuse was still the right call over reimplementing and risking divergence | Tabular lead |
| FS-05 | Roughly a third of bigram occurrences in each scoring role fall outside the base_fit vocabulary | Low | Informational for fusion: the text signal reaching the stacker is mostly unigram-driven | Fusion lead |

## 9. Limitations

- **Single seed, single configuration.** No repeated-seed variance and no confidence intervals on the
  validation figures in §4; the fold standard deviation is the only dispersion reported.
- **The unweighted probability is a raw model output.** Its level happens to sit near the base rate;
  that is a property of unweighted fitting, not a fitted calibrator, and it must not be called
  calibrated.
- **`fusion_fit` spans four months.** A stacker learns its weights from a short, recent window — noted
  in `fusion_roles.py` and inherited here, not introduced by this export.
- **The saved pipeline is gitignored.** `artifacts/nlp/tfidf_fusion_v2_pipeline.joblib` will not reach
  a reviewer through version control. The manifest is the contract: config, seed, fitting boundary and
  checksums are sufficient for a deterministic rebuild in about 20 seconds.
- **Evidence lane, proxy population.** LendingClub instalment loans are not Singapore BNPL checkout.
- **Test split untouched** — absent from the role mapping by construction, `test_rows_used: 0`.

## 10. Reproduction

```
python -m pip install -r src/nlp/requirements.txt
python -m pytest src/nlp/test_export_scores_v2.py -v -s
python -m src.nlp.export_scores_v2
```

Runtime about 6 minutes, dominated by 24 grid points x 5 folds. On macOS the xgboost dependency
(see FS-04) also needs the OpenMP runtime: `brew install libomp`.

Outputs: `data/nlp/tfidf_scores_v2.parquet`, `reports/nlp/tfidf_scores_v2_manifest.json`,
`reports/nlp/tfidf_scores_v2_grid.csv`, and the gitignored pipeline artefact.

## 11. Definition-of-done status

| Requirement | Status | Evidence |
|---|---|---|
| Reuse the existing TF-IDF pipeline | Met | Imports `build_pipeline` and `load_baseline_split`; v1 module unchanged |
| Vocabulary, training and selection restricted to 59,353 base_fit loans | Met | §1, §3; `fitted_on_scoring_roles: false` in the manifest |
| Time-ordered folds within the fitting pool | Met | §3; reuses the tabular track's `time_aware_folds` |
| Trained with `class_weight=None` | Met | §4; config refuses any other value |
| New model version, v1 not overwritten | Met | `nlp-tfidf-fusion-v2`; a test asserts v1 still reads `nlp-tfidf-tuned-v1` |
| Score 15,587 / 11,353 / 21,784 loans | Met | §2; exact counts asserted |
| Raw probability and margin from the same fitted model | Met | `raw_prob == sigmoid(raw_margin)` verified and tested |
| Balanced-versus-unweighted comparison, matched conditions | Met | §4 |
| Saved pipeline | Met, gitignored | §9; manifest supports deterministic rebuild |
| Manifest with boundaries, configuration, versions, checksums | Met | `tfidf_scores_v2_manifest.json` |
| Export coverage 48,724 with the six agreed columns | Met | §2, §7 |
| Tests for role coverage, unique IDs, finite margins, probabilities in [0,1] | Met | 31 checks in `src/nlp/test_export_scores_v2.py` |

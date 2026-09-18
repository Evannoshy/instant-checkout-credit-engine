# NLP Baseline Report — TF-IDF and Logistic Regression

**Document status:** Evidence record for NLP track review  
**Version:** 0.1  
**Date:** 18 September 2026  
**Accountable owner:** Ariel (NLP Track Lead)
**Primary author:** Yati (NLP Track Analyst 2)  
**Reviewers:** NLP Track Lead, Tabular Track Tech Lead, Fusion/API Lead  
**Dataset version under review:** `kaggle-adarshsng-local-2026-09-09`  
**Split version under review:** `split-v1`  
**Baseline version:** `nlp-baseline-tfidf-v1`  
**Preprocessing version:** `nlp-preprocess-v1`

## 1. Purpose

This report records the first measured text-only result for the NLP track: a TF-IDF and
logistic-regression baseline over the shared text payload, evaluated on the frozen validation
split. It exists to give the transformer work in `src/nlp/dataset.py` a reference point it must
beat, and to give the fusion track an honest estimate of what the text modality contributes.

Every number below is reproduced by the command in section 10 and written to `reports/nlp/` by the
run itself. None is copied by hand.

This report does not select a champion model, tune hyperparameters, or evaluate the test split.

## 2. Method

| Choice | Value | Source |
|---|---|---|
| Text input | `build_text_payload()` output, field labels stripped | Analyst 1 contract, unmodified |
| Vectoriser | `TfidfVectorizer`, unigrams + bigrams, 5,000 features, English stop words | Analyst 2 brief |
| Classifier | `LogisticRegression`, `class_weight='balanced'`, `max_iter=1000`, `random_state=42` | Analyst 2 brief |
| Fitted on | Training split only; validation is transformed, never fitted | `CONTRIBUTING.md` modelling standards |
| Evaluated on | Validation split | Test split is locked and was never loaded |

### 2.1 Field-label stripping

`build_text_payload()` prefixes each field, producing `Title: … / Purpose: … / Description: …`.
Those three label words occur in every document, are not English stop words, and would otherwise
consume vocabulary slots and appear in the ranked coefficients as artefacts of the payload format
rather than borrower language.

They are therefore removed **downstream**, in `strip_field_labels()` in `src/nlp/baseline_tfidf.py`.
`preprocess.py` is not modified: the payload remains byte-identical across the TF-IDF and
transformer tracks, and the labels are read from `preprocess.FIELD_LABELS` so the transform stays
tied to Analyst 1's field set. A parametrized test fails if that field set changes.

### 2.2 Ablation

A second configuration vectorises the cleaned `desc` narrative alone, dropping `title` and
`purpose`. `purpose` is a 14-level categorical the tabular track already owns
(`docs/feature-policy-draft.md` §6.3), so this separates genuine narrative signal from a
re-encoding of a structured field.

## 3. Environment

| Component | Version |
|---|---|
| Python | 3.13.3 |
| scikit-learn | 1.9.1 |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| Raw source CSV | `data/raw/loan.csv`, SHA-256 `23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c` |

The raw CSV hash matches the value cross-verified in `docs/data-quality-report.md` §3, so this
baseline is fitted on the same bytes as the tabular track's profiling. See finding NB-05 on the
Python version recorded in `src/nlp/requirements.txt`.

## 4. Results

Fitted on 86,293 training rows (13,197 positives). Evaluated on 21,784 validation rows
(3,298 positives, prevalence 0.1514). Solver converged in 42 of 1,000 permitted iterations.

| Metric | Value | Trivial-model baseline | Margin |
|---|---:|---:|---:|
| **PR-AUC** | **0.1845** | 0.1514 (prevalence) | +0.0332 |
| **ROC-AUC** | **0.5693** | 0.5 | +0.0693 |
| F1 at threshold 0.50 | 0.2698 | — | — |
| Best swept F1 | 0.2774 at threshold 0.4431 | — | — |

**Every metric is stated against the score a trivial model achieves.** PR-AUC's floor is the
positive-class prevalence, not zero; ROC-AUC's is 0.5. A PR-AUC of 0.1845 read without its 0.1514
floor overstates this model considerably.

### 4.1 Reading the F1 figures

`class_weight='balanced'` reweights the classes and moves the decision boundary. At a 15.1% base
rate, 0.5 is therefore not a meaningful operating point, and neither F1 figure should be quoted as
"the" F1 without its threshold. PR-AUC and ROC-AUC are rank-based and unaffected by the weighting.
No operating threshold is recommended here; that is a policy-layer decision, not a model output.

### 4.2 Ablation result

| Configuration | PR-AUC | ROC-AUC | F1 @ 0.5 |
|---|---:|---:|---:|
| Full payload (title + purpose + description) | 0.1845 | 0.5693 | 0.2698 |
| Description only | 0.1842 | 0.5688 | 0.2679 |
| **Difference** | **+0.0004** | **+0.0006** | +0.0019 |

Removing title and purpose changes PR-AUC by 0.0004. The two structured-ish fields contribute
essentially nothing, and the text signal comes from the borrower narrative rather than from a
re-encoding of the `purpose` category the tabular model already has. This is the most
decision-relevant result in this report for the fusion track.

## 5. Ranked n-grams

The full ranked lists are in `reports/nlp/baseline_tfidf_top_ngrams.csv` with document counts. The
strongest term in each direction is shown below; the caveat in section 6 applies to all of them.

| Direction | Leading terms | Train document % |
|---|---|---:|
| Higher risk | `purchase major`, `thats`, `time current`, `consolidation make`, `payday` | 0.57, 0.14, 0.07, 0.11, 0.08 |
| Lower risk | `rate`, `totaling`, `salary`, `fairly`, `thanks help` | 14.26, 0.19, 0.92, 0.15, 0.42 |

Coefficients are descriptive associations. Their magnitudes depend on IDF scaling and split across
correlated n-grams, so they do not identify risk drivers and are not evidence of causality. The
project treats SHAP the same way (`docs/decisions/decision-log.md`, D-009); the same discipline
applies here.

## 6. Findings and required follow-up

| ID | Finding | Severity | Required action | Owner |
|---|---|---|---|---|
| NB-01 | Text-only signal is weak: ROC-AUC 0.5693, PR-AUC 0.0332 above its prevalence floor | High | Set fusion expectations accordingly. Text should be assessed for *incremental* lift over the tabular score, not as a standalone modality | NLP + Fusion leads |
| NB-02 | Ranked keyword lists are dominated by rare terms: 38 of 40 appear in under 1% of training documents, median 156 documents (0.18%). Only `rate` (14.26%) and `excellent credit` (1.37%) exceed 1% | High | Do not quote the keyword lists without document frequency. Produce a frequency-filtered ranking before any of these terms reaches a slide or stakeholder summary | Analyst 2 |
| NB-03 | Bigrams span field boundaries in the concatenated payload. The top higher-risk term `purchase major` is `Title: Major purchase` running into `Purpose: major purchase`, not borrower language. Confirmed: it disappears entirely from the description-only ablation | Medium | Accepted and documented. Avoiding it requires vectorising fields separately, which is a different model. Reviewers must not read boundary bigrams as findings | Analyst 2 |
| NB-04 | `educational` ranks as higher-risk but is a retired `purpose` category, absent from the test split (`docs/feature-policy-draft.md` §4). It also disappears from the ablation | Medium | Treat as a vintage artefact of the chronological split, not a transferable signal. Re-check any purpose-derived term for era dependence | Analyst 2 |
| NB-05 | `src/nlp/requirements.txt` records verification under Python 3.14.6; this baseline ran on 3.13.3 and the data quality report on 3.13.9. No evidenced run on 3.14.6 exists in this repository | Low | Agree one Python version for the NLP track. Torch wheel availability, not scikit-learn, is the binding constraint — the transformer work needs it and it is absent from the current environment | NLP Track Lead |
| NB-06 | `docs/data-quality-report.md` §2 records the split manifest SHA-256 as `35e3545e…`; the tracked file hashes to `41686f71…` and has been unmodified since its only commit | Low | Correct the provenance table. No data defect: all 14 integrity checks pass and the manifest reconciles with `split_statistics.json` | Tabular Analyst 1 / data owner |

## 7. Limitations

- **One split, one seed, no interval.** These are point estimates on a single chronological
  validation split. No confidence interval or repeated-seed variance is reported, so small
  differences — including the ablation gap in section 4.2 — should not be read as significant.
- **No calibration assessment.** Balanced class weights distort predicted probabilities, and no
  Brier score, log loss or calibration curve is reported. The scores here are usable for ranking
  only, not as probabilities, and not yet as a fusion input under decision D-006.
- **The 5,000-feature cap is binding, not tuned.** The fitted vocabulary saturates the cap. The
  value comes from the analyst brief; whether more features help is untested.
- **Evidence lane, proxy population.** LendingClub instalment loans are not Singapore BNPL
  checkout. Per `README.md`, this result must not be presented as a BNPL accuracy claim.
- **Test split untouched**, per the locked-test-set guardrail.

## 8. What this does not decide

The champion text model, the operating threshold, the fusion weighting, and whether the text
modality earns a place in the served system. NB-01 is an input to those decisions, not a verdict.

## 9. Verification

Twenty-seven checks in `src/nlp/test_baseline_tfidf.py`, all passing, alongside Analyst 1's 35
preprocessing checks and the 14 data integrity checks — 76 in total across the repository.

The suite deliberately loads no borrower data and does not require `data/raw/loan.csv`, which is
gitignored and absent for reviewers and CI. It uses toy corpora and monkeypatching, per the
`CONTRIBUTING.md` rule on tiny synthetic fixtures in public tests. Two checks are load-bearing for
this report's integrity:

- `test_fit_baseline_learns_vocabulary_from_training_rows_only` enforces the no-validation-leakage
  rule in code rather than asserting it in prose.
- `test_strip_removes_every_label_the_shared_contract_defines` is parametrized over
  `preprocess.FIELD_LABELS`, so a new field in the shared payload fails the suite instead of
  silently entering the vocabulary.

## 10. Reproduction

From the repository root, with `data/raw/loan.csv` in place:

```
python -m pip install -r src/nlp/requirements.txt
python -m pytest src/nlp/ scripts/ -v -s
python -m src.nlp.baseline_tfidf
python -m src.nlp.baseline_tfidf --desc-only
```

Each run writes `baseline_tfidf_metrics*.json` and `baseline_tfidf_top_ngrams*.csv` to
`reports/nlp/`. Runtime is approximately 45 seconds per configuration.

## 11. Definition-of-done status

| Requirement | Status | Evidence |
|---|---|---|
| Concatenated text vectorised with TF-IDF, unigrams + bigrams, 5,000 features, English stop words | Met | §2; asserted in `test_fit_baseline_uses_the_specified_vectorizer_configuration` |
| `LogisticRegression` with `class_weight='balanced'` trained on the training split | Met | §2, §4; 86,293 rows, converged in 42 iterations |
| Predictions evaluated on the validation split | Met | §4; 21,784 rows |
| Validation PR-AUC recorded | Met | §4; 0.1845 against a 0.1514 floor |
| Validation ROC-AUC recorded | Met | §4; 0.5693 |
| Validation F1 recorded | Met | §4; 0.2698 at threshold 0.50, 0.2774 at 0.4431 |
| Ranked list of top 20 positive and negative n-grams | Met, with NB-02 | §5; `reports/nlp/baseline_tfidf_top_ngrams.csv`, with document frequency |
| No raw LendingClub data committed | Met | `data/raw/` is gitignored; only aggregate metrics and ranked terms are tracked |

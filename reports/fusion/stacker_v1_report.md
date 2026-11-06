# Late Fusion Stacker v1: First Comparison (tabular only)

**Version:** `fusion-stacker-v1` · **Status:** development, no recommendation
**Roles:** `fusion_development_v1` (approved by Evan and Ariel, 28 Oct 2026)
**Produced by:** `python -m src.fusion.stacker` · **Metrics:** `reports/fusion/stacker_v1_metrics.json`

## 1. What was run

| Step | Rows | Role |
|---|---:|---|
| Fit the stacker: unweighted logistic regression (`class_weight=None`, C = 1.0) on raw probabilities | 15,587 | `fusion_fit` |
| Fit the Platt calibrator on the stacker's outputs | 11,353 | `fusion_calibration` |
| Grade raw and calibrated outputs | 21,784 | `validation` |

The stacker reads score columns only. The one input is the raw balanced v1 tabular LR
(`tabular-logistic-v1-model-fit`), fitted on the 59,353 `base_fit` loans only. The test split was not read.

Only the **tabular-only** comparison is run. The tabular + TF-IDF and three-score variants are
defined in `configs/fusion_stacker_v1.toml` but cannot run until the v2 NLP exports are published
and pinned. Running them before then stops with an error rather than being skipped.

## 2. Validation results

| Version | ROC-AUC | PR-AUC | Brier | Log loss | Mean PD |
|---|---:|---:|---:|---:|---:|
| Calibrated tabular reference | 0.6660 | 0.2575 | 0.12250 | 0.40305 | 0.1510 |
| Tabular-only stacker, raw | 0.6660 | 0.2575 | 0.12247 | 0.40294 | 0.1476 |
| Tabular-only stacker + Platt | 0.6660 | 0.2575 | 0.12246 | 0.40292 | 0.1556 |

Observed default rate on validation: 0.1514 (3,298 of 21,784).

Paired bootstrap (1,000 resamples), stacker minus reference, 95% CI:

| Comparison | Brier | Log loss |
|---|---|---|
| Raw stacker − reference | −0.00003 [−0.00012, +0.00005] | −0.00011 [−0.00034, +0.00010] |
| Platt stacker − reference | −0.00004 [−0.00012, +0.00004] | −0.00013 [−0.00035, +0.00011] |

ROC-AUC and PR-AUC differences are exactly zero.

## 3. Reading the results

- **Ranking cannot change here.** A one-input logistic regression is a monotone transform of its
  input, so it orders loans exactly as the tabular model does. This comparison checks the pipeline
  and gives the baseline that text variants must beat. It does not test whether text helps.
- **Probability quality is the same.** Brier and log-loss differences are tiny, and every interval
  includes zero.
- **No winner is chosen.** The +0.005 margin is a proposed target only. Evan and Ariel have not yet
  confirmed the primary metric, probability-quality safeguards or decision rule.

## 4. Caveats

- The reference's Platt mapping (`tabular_calibration_v1`) was fitted on all 26,940 Jan–Jul 2013
  loans, including the 15,587 `fusion_fit` loans. It is a fair benchmark on validation, but it is not
  held out from the fusion rows.
- The bootstrap resamples loans independently. Validation loans are clustered in time, so the
  intervals are probably somewhat narrow.
- The stacker input is the balanced v1 LR, the version Brandon's handoff pins. The unweighted LR
  (`tabular-logistic-unweighted-v1-model-fit`) is also leak-safe. Which one feeds the stacker should
  be confirmed.

## 5. Prediction handoff

`src/fusion/predict.py` gives `predict(predictor, scores, text_usable=...)`. When text is marked
unusable, it returns the approved calibrated tabular-only score with `fallback=True` and refuses any
text score. A missing or invalid score raises an error and is never filled. Tests reproduce the
committed stacker predictions and the approved reference to 1e-12.

The caller decides `text_usable` until Evan and Ariel confirm the definition of unusable text.

## 6. Waiting on

| Item | Owner | Unblocks |
|---|---|---|
| `data/nlp/tfidf_scores_v2.parquet` + manifest | Yati | Tabular + TF-IDF comparison |
| `distilbert_scores_v2.parquet` merged (PR #25) | Vincent | Three-score comparison |
| Primary metric, safeguards, decision rule | Evan and Ariel | Any recommendation |
| Definition of unusable text | Evan and Ariel | Fallback rule in the demo |
| Saved tabular pipeline and reference calibrator | Brandon | End-to-end demo scoring from borrower fields |

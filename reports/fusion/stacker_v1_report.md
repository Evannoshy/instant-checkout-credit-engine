# Late Fusion Stacker v1: Tabular, TF-IDF and DistilBERT Comparison

**Version:** `fusion-stacker-v1` · **Status:** development, no recommendation
**Roles:** `fusion_development_v1` (approved by Evan and Ariel, 28 Oct 2026)
**Produced by:** `python -m src.fusion.stacker` · **Metrics:** `reports/fusion/stacker_v1_metrics.json`

## 1. What was run

| Step | Rows | Role |
|---|---:|---|
| Fit each stacker: unweighted logistic regression (`class_weight=None`, C = 1.0) on raw probabilities | 15,587 | `fusion_fit` |
| Fit each Platt calibrator on its stacker's outputs | 11,353 | `fusion_calibration` |
| Grade raw and calibrated outputs | 21,784 | `validation` |

The stackers read model scores only. All three inputs come from models fitted on `base_fit` months
(2007-06 to 2012-12) only, and the test split was not read.

| Track | Model version | Owner | Source |
|---|---|---|---|
| Tabular | `tabular-logistic-v1-model-fit` (balanced v1 LR) | Brandon | `reports/tabular/raw_predictions/` |
| TF-IDF | `nlp-tfidf-fusion-v2` (unweighted) | Yati, PR #29 | `data/nlp/tfidf_scores_v2.parquet` |
| DistilBERT | `nlp-distilbert-fusion-v2` (unweighted) | Vincent, PR #25 | `data/nlp/distilbert_scores_v2.parquet` |

Both NLP exports are raw, uncalibrated probabilities, and they are used only as stacker inputs.
Every file is pinned by SHA-256, and the NLP checksums match their owners' manifests.

| Variant | Inputs | Stacker weights (on probabilities) |
|---|---|---|
| `tabular_only` | tabular | 4.15 |
| `tabular_tfidf` | tabular, TF-IDF | 3.94, 4.90 |
| `tabular_tfidf_distilbert` | tabular, TF-IDF, DistilBERT | 3.85, 2.31, 2.20 |

## 2. Validation results

| Version | ROC-AUC ↑ | PR-AUC ↑ | Brier ↓ | Log loss ↓ | Mean PD |
|---|---:|---:|---:|---:|---:|
| Calibrated tabular reference | 0.6660 | 0.2575 | 0.12250 | 0.40305 | 0.1510 |
| Tabular-only stacker + Platt | 0.6660 | 0.2575 | 0.12246 | 0.40292 | 0.1556 |
| Tabular + TF-IDF + Platt | 0.6756 | 0.2628 | 0.12183 | 0.40040 | 0.1557 |
| Tabular + TF-IDF + DistilBERT + Platt | **0.6791** | **0.2665** | **0.12153** | **0.39924** | 0.1558 |

The raw (pre-Platt) rows are in `reports/fusion/stacker_v1_results.csv`. The observed default rate on
validation is 0.1514 (3,298 of 21,784).

Paired bootstrap (1,000 resamples) on the calibrated outputs, 95% CI. A positive ROC-AUC or PR-AUC
difference is better, and a negative Brier or log-loss difference is better.

| Comparison | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|
| Tabular-only − reference | 0 | 0 | −0.00004 [−0.00012, +0.00004] | −0.00013 [−0.00035, +0.00011] |
| + TF-IDF − tabular-only | +0.0096 [+0.0058, +0.0132] | +0.0053 [−0.0002, +0.0113] | −0.00063 [−0.00096, −0.00029] | −0.0025 [−0.0035, −0.0014] |
| + TF-IDF + DistilBERT − tabular-only | +0.0131 [+0.0084, +0.0180] | +0.0090 [+0.0023, +0.0163] | −0.00093 [−0.00137, −0.00048] | −0.0037 [−0.0050, −0.0022] |
| + DistilBERT on top of TF-IDF | +0.0036 [+0.0009, +0.0063] | +0.0037 [−0.0002, +0.0073] | −0.00030 [−0.00054, −0.00003] | −0.0012 [−0.0020, −0.0003] |

The differences against the reference itself are in the metrics JSON.

## 3. Reading the results

- **Text helps on validation.** Adding TF-IDF improves ROC-AUC, Brier and log loss, and the intervals
  exclude zero. Its PR-AUC gain (+0.0053) just passes the proposed +0.005 target, but the interval
  touches zero.
- **The three-score stacker is best on every metric.** Against tabular-only, every interval excludes
  zero, and the PR-AUC gain is +0.0090 [+0.0023, +0.0163].
- **DistilBERT adds a little on top of TF-IDF.** ROC-AUC, Brier and log loss improve with intervals
  that exclude zero, but the PR-AUC interval includes zero.
- **The gains are small in absolute terms.** ROC-AUC rises from 0.666 to 0.679. Text is a modest
  supplement to the borrower's financial fields, not a replacement for them.
- **No winner is chosen.** The +0.005 margin is a proposed target only. Evan and Ariel have not yet
  confirmed the primary metric, probability-quality safeguards or decision rule.

## 4. Caveats

- **Platt slightly over-predicts on validation.** Each calibrator is fitted on May–Jul 2013, when the
  default rate was 15.85%, against 15.14% in validation. All calibrated mean PDs are therefore about
  0.156. For the text variants, the raw stacker scores slightly better Brier and log loss than the
  calibrated ones. This is drift between time windows, not a bug, and it should be monitored.
- The reference's Platt mapping (`tabular_calibration_v1`) was fitted on all 26,940 Jan–Jul 2013
  loans, including the 15,587 `fusion_fit` loans. It is a fair benchmark on validation, but it is not
  held out from the fusion rows.
- The bootstrap resamples loans independently. Validation loans are clustered in time, so the
  intervals are probably somewhat narrow.
- The tabular input is the balanced v1 LR, the version Brandon's handoff pins. Which tabular model
  feeds the stacker should be confirmed.
- Neither NLP export flags unusable text. Yati's manifest counts 3 loans across the scoring roles
  whose TF-IDF vector is empty, and those loans still receive the model's intercept score.

## 5. Prediction handoff

`src/fusion/predict.py` gives `load_predictor(variant)` and `predict(predictor, scores, text_usable=...)`
for all three variants.

- **Text usable:** every track's raw score is required, and the result is the calibrated stacker output.
- **Text unusable:** the result is the approved calibrated tabular-only score, with `fallback=True`, and
  any text score passed in is refused.

A missing or invalid score raises an error and is never filled. Tests reproduce the committed
tabular-only and three-score predictions, and the approved reference, to 1e-12. The caller decides
`text_usable` until Evan and Ariel confirm the definition of unusable text.

## 6. Waiting on

| Item | Owner | Unblocks |
|---|---|---|
| Primary metric, safeguards, decision rule | Evan and Ariel | Choosing a variant |
| Definition of unusable text | Evan and Ariel | Fallback rule in the demo |
| Balanced or unweighted tabular LR as stacker input | Evan, Ariel, Brandon | Final stacker inputs |
| Saved tabular pipeline and reference calibrator | Brandon | End-to-end demo scoring from borrower fields |

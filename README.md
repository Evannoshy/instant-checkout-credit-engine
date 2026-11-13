# Instant Checkout Credit Engine

A research prototype comparing structured-data and borrower-text models for credit risk on a locked historical LendingClub cohort. The longer-term goal is to combine their scores behind a low-latency checkout API.

The project is led by NUS FinTech Society. Ariel and Evan are project co-leads; Evan is the Tabular Track Tech Lead.

## Project status

**Repository snapshot:** 13 November 2026<br>
**Current work:** local demo integration, model handoff and final evaluation preparation<br>
**Planning period:** 7 September to 30 November 2026

The repository contains executable preprocessing, constant/logistic/XGBoost baselines, calibration comparisons, TF-IDF and DistilBERT experiments, tests, and versioned score exports. The [tabular calibration report](reports/tabular/candidate_comparison.md) retains balanced logistic regression with Platt calibration as its reference under the predeclared selection rule. The newer [unweighted logistic comparison](reports/tabular/unweighted_lr_comparison.md) does not approve replacing that reference.

The [late-fusion comparison](reports/fusion/stacker_v1_report.md) is complete on validation. A [local API and demonstration form](docs/local-demo.md) connect raw borrower fields to the existing tabular model and fusion predictor. Tabular inference is available after rebuilding the local bundle. Live TF-IDF and DistilBERT routes require their verified original artifacts. Final model selection, deployment and the locked final evaluation are pending. Gate dates in the [handbook](docs/tabular-analyst-handbook.md) are the original plan; they do not establish that a gate passed.

## What the current models predict

The evidence lane uses **122,999 historical US originated loans** with a non-whitespace raw borrower description and issue months from June 2007 through March 2014. The frozen target is `Charged Off = 1` versus `Fully Paid = 0`; other statuses, including `Default`, are excluded. Text inputs are borrower `title`, `purpose` and `desc`, with separate description-only experiments. They are not bank transaction narratives.

This is a proxy experiment, not evidence of performance for Singapore BNPL applicants, declined applications or thin-file checkout customers. The proposed product target, 30+ days past due within 90 days of the first due date, remains distinct from the ultimate LendingClub outcome. See the [exact cohort contract](docs/cohort-definition.md), [problem definition](docs/problem-definition.md) and accepted decisions D-011 through D-016 in the [decision log](docs/decisions/decision-log.md).

| Frozen split | Issue months | Loans | Use |
|---|---|---:|---|
| Train | Jun 2007–Jul 2013 | 86,293 | Model development; tabular roles below |
| Validation | Aug–Dec 2013 | 21,784 | Reported model comparisons and selection |
| Test | Jan–Mar 2014 | 14,922 | Locked for final evaluation; unavailable to development loaders |

For current tabular comparisons, training is further divided into 59,353 `model_fit` loans (through December 2012) and 26,940 calibration loans (January–July 2013). The initial logistic baseline used all training rows; its results and model version must not be mixed with the later `model_fit` experiments.

## Evidence so far

The following are **validation** results on the same 21,784 loans, not final-test or checkout performance. The observed validation bad rate is 0.1514. PR-AUC means average precision.

| Model | ROC-AUC | PR-AUC | Brier score | Log loss | Evidence |
|---|---:|---:|---:|---:|---|
| Balanced tabular LR + Platt | 0.6660 | 0.2575 | 0.1225 | 0.4031 | [Calibration comparison](reports/tabular/candidate_comparison.md) |
| Selected XGBoost, raw | 0.6661 | 0.2584 | 0.1224 | 0.4030 | [Calibration comparison](reports/tabular/candidate_comparison.md) |
| Unweighted tabular LR, raw | 0.6662 | 0.2582 | 0.1225 | 0.4032 | [Unweighted comparison](reports/tabular/unweighted_lr_comparison.md) |
| Tabular + TF-IDF + Platt | 0.6756 | 0.2628 | 0.1218 | 0.4004 | [Fusion comparison](reports/fusion/stacker_v1_report.md) |
| Tabular + TF-IDF + DistilBERT + Platt | 0.6791 | 0.2665 | 0.1215 | 0.3992 | [Fusion comparison](reports/fusion/stacker_v1_report.md) |
| Tuned TF-IDF + balanced LR, raw | 0.5867 | 0.1944 | 0.2466 | 0.6867 | [NLP tuning report](docs/nlp-tuning-report.md) |

XGBoost and unweighted LR have not demonstrated a replacement-worthy gain over the calibrated tabular reference under the recorded margins and paired intervals. The fusion comparison shows a modest validation gain from combining the three scores; it does not approve a final champion. Base-model text probabilities remain raw stacker inputs. Earlier NLP v1 experiments use different fitting histories and are not the live v2 fusion inputs.

## Score handoffs and integration gaps

- **Tabular time-based OOF:** `data/tabular/tabular_oof_train_v1.parquet` contains 35,234 held-out `model_fit` loans from five expanding-window folds. The earliest 24,119 loans are fitting-only and have no OOF score. The [manifest](data/tabular/tabular_oof_train_v1_manifest.json) records folds, hashes and score semantics.
- **TF-IDF OOF:** `data/nlp/tfidf_oof_train.parquet` covers all 86,293 training loans using shuffled stratified folds. Validation scores are in `tfidf_oof_validate.parquet`, despite that filename's `oof` wording. See the [tuning report](docs/nlp-tuning-report.md).
- **DistilBERT scores:** train and validation Parquets are in `data/nlp/`. Two separately trained 10k-loan models avoid scoring their own fitting loans; the probabilities remain uncalibrated. See [the exporter](src/nlp/export_distilbert_scores.py) and `reports/nlp/distilbert_*metrics.json`.
- **Description-only distress density:** [the handoff](docs/nlp-desc-distress-export.md) covers 108,077 train/validation loans. D-017 remains proposed; exporting the feature does not add it to `tabular_features_v1`.

The bullets above describe historical v1 exports. The current fusion experiment uses `data/fusion/fusion_roles_v1.csv`: 59,353 base-fitting loans, 15,587 fusion-fitting loans, 11,353 fusion-calibration loans and 21,784 validation loans. The v2 NLP exports cover the same 48,724 scoring loans. See [the current fusion report](reports/fusion/stacker_v1_report.md). Join by verified `loan_id`, split and role. The old shuffled OOF exports are not substitutes for these inputs. Fusion consumes raw base-model probabilities and calibrates the stacked output.

## Local demo

```powershell
python -m src.serving.prepare_bundle --raw-csv "C:/path/to/loan.csv"
python -m src.serving.api --variant tabular_only --text-policy payload
```

Open `http://127.0.0.1:8000` and use fictional details. The explicit variant and text rule are preview settings, not final project decisions. See [setup, artifact requirements and API details](docs/local-demo.md).

## Setup and checks

Run from the repository root. Use separate environments for the two tracks; their recorded Python runtimes differ. The frozen XGBoost search/prediction reproduction requires Windows AMD64 and Python 3.13.15 as recorded in [the XGBoost report](docs/tabular-xgboost-report.md). See [Contributing](CONTRIBUTING.md) for test and raw-data details.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r src/tabular/requirements.txt
python -m pytest scripts src/tabular -q
```

The restricted raw CSV is not included. Tests needing it skip when unavailable; exact runtime checks still apply. Rebuilding models requires the approved `loan.csv` whose SHA-256 is recorded in the cohort contract. Point `LENDINGCLUB_RAW_CSV` at it for raw-data tests; model runners use their documented `--raw-csv` option or default `data/raw/loan.csv`.

With approved data and the recorded environment, the tabular experiment sequence is:

```powershell
python -m src.tabular.xgboost_baseline compare
python -m src.tabular.xgboost_baseline ablate
python -m src.tabular.xgboost_search
python -m src.tabular.lexical_ablation
python -m src.tabular.raw_predictions
python -m src.tabular.calibrate
python -m src.tabular.export_oof_scores
python -m src.tabular.logistic_unweighted
python -m src.tabular.unweighted_comparison
```

These commands regenerate evidence artifacts. Use a working branch and review the resulting changes. The [Windows test record](reports/tabular/oof_unweighted_lr_test_record.md) documents the existing reference run; it is not a test result for every future checkout.

## Repository layout

```text
.
|-- .github/pull_request_template.md
|-- configs/                 # Frozen TOML data, feature and experiment contracts
|-- data/                    # Shared IDs/manifests and reviewed derived score tables
|-- docs/                    # Contracts, design, analysis and limitations
|-- reports/                 # Aggregate metrics, prediction exports and figures
|-- scripts/                 # Cohort reproduction and data-integrity checks
|-- src/
|   |-- tabular/             # Preprocessing, models, calibration, OOF and tests
|   |-- nlp/                 # Text preprocessing, models, exports, notebooks and tests
|   |-- fusion/              # Checked score inputs, stacker and prediction handoff
|   `-- serving/             # Local borrower-field API, demo, bundle preparation and benchmark
|-- CONTRIBUTING.md
`-- updated-pitchdeck.pdf
```

The deck and original milestone plans are planning artifacts. Current claims should follow the evidence reports. CI workflows, a Dockerfile, deployment and a final release model card remain pending.

## Guardrails and next work

- Preserve frozen cohort IDs, labels, features and experiment configurations; changes need a new version and decision record.
- Keep the final test locked until an approved evaluation gate. Validation has already informed selection and is not an untouched final estimate.
- Keep the approved fusion roles and matched validation comparison intact; freeze the final variant and missing-text rule before release.
- Complete supported subgroup/robustness checks, loss scenarios and policy ownership before adopting decision thresholds.
- Package compatible preprocessing, model and calibration artifacts; implement contract/fallback tests and benchmark latency on agreed hardware.
- Keep raw/restricted data, secrets, local environments and model binaries out of Git. Synthetic systems data cannot establish real-world accuracy, fairness or inclusion.

See the [architecture](docs/architecture.md), [risks and assumptions](docs/risks-and-assumptions.md), [NLP limitations](docs/nlp-pipeline-limitations.md), [analyst handbook](docs/tabular-analyst-handbook.md) and [reading pack](docs/reading-list.md).

This is an educational research prototype. Production readiness, fairness, affordability, inclusion, causal explanations and regulatory compliance have not been established.

# Instant Checkout Credit Engine

A research prototype comparing structured-data and borrower-text models for credit risk on a locked historical LendingClub cohort. The longer-term goal is to combine their scores behind a low-latency checkout API.

The project is led by NUS FinTech Society. Ariel and Evan are project co-leads; Evan is the Tabular Track Tech Lead.

## Project status

**Repository snapshot:** 17 October 2026<br>
**Current work:** model comparison, calibration and cross-track score handoffs<br>
**Planning period:** 7 September to 30 November 2026

The repository contains executable preprocessing, constant/logistic/XGBoost baselines, calibration comparisons, TF-IDF and DistilBERT experiments, tests, and versioned score exports. The [tabular calibration report](reports/tabular/candidate_comparison.md) retains balanced logistic regression with Platt calibration as its reference under the predeclared selection rule. The newer [unweighted logistic comparison](reports/tabular/unweighted_lr_comparison.md) does not approve replacing that reference.

Fusion, a serving API, a packaged inference bundle, and end-to-end latency evidence are still pending. There is no measured sub-100 ms serving result in this repository. Gate dates in the [handbook](docs/tabular-analyst-handbook.md) are the original plan; they do not establish that a gate passed.

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
| Tuned TF-IDF + balanced LR, raw | 0.5867 | 0.1944 | 0.2466 | 0.6867 | [NLP tuning report](docs/nlp-tuning-report.md) |

XGBoost and unweighted LR have not demonstrated a replacement-worthy gain over the calibrated tabular reference under the recorded margins and paired intervals. The TF-IDF probabilities are uncalibrated. These text-only metrics do not establish whether text adds information beyond tabular scores; that needs a matched fusion comparison. DistilBERT reports use smaller training/validation samples and validation-based checkpoint selection, so they are not directly ranked in this table.

## Score handoffs and integration gaps

- **Tabular time-based OOF:** `data/tabular/tabular_oof_train_v1.parquet` contains 35,234 held-out `model_fit` loans from five expanding-window folds. The earliest 24,119 loans are fitting-only and have no OOF score. The [manifest](data/tabular/tabular_oof_train_v1_manifest.json) records folds, hashes and score semantics.
- **TF-IDF OOF:** `data/nlp/tfidf_oof_train.parquet` covers all 86,293 training loans using shuffled stratified folds. Validation scores are in `tfidf_oof_validate.parquet`, despite that filename's `oof` wording. See the [tuning report](docs/nlp-tuning-report.md).
- **DistilBERT scores:** train and validation Parquets are in `data/nlp/`. Two separately trained 10k-loan models avoid scoring their own fitting loans; the probabilities remain uncalibrated. See [the exporter](src/nlp/export_distilbert_scores.py) and `reports/nlp/distilbert_*metrics.json`.
- **Description-only distress density:** [the handoff](docs/nlp-desc-distress-export.md) covers 108,077 train/validation loans. D-017 remains proposed; exporting the feature does not add it to `tabular_features_v1`.

Join by verified `loan_id` and split, never row position. Matching IDs alone does not align fitting histories: shuffled NLP folds can use later months, and NLP training includes the tabular calibration months. Before treating combined scores as time-respecting fusion evidence, agree common fitting/holdout boundaries, calibration roles, eligible rows and a frozen evaluation protocol. Raw balanced-model scores must not be interpreted as calibrated default risk or silently passed through a calibrator fitted to another model.

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
|   `-- nlp/                 # Text preprocessing, models, exports, notebooks and tests
|-- CONTRIBUTING.md
`-- updated-pitchdeck.pdf
```

The deck and original milestone plans are planning artifacts. Current claims should follow the evidence reports. CI workflows, fusion/API source, a Dockerfile and a release model card have not yet been added.

## Guardrails and next work

- Preserve frozen cohort IDs, labels, features and experiment configurations; changes need a new version and decision record.
- Keep the final test locked until an approved evaluation gate. Validation has already informed selection and is not an untouched final estimate.
- Resolve fusion fold/role compatibility and compare tabular-only, text-only and fused models on identical rows.
- Complete supported subgroup/robustness checks, loss scenarios and policy ownership before adopting decision thresholds.
- Package compatible preprocessing, model and calibration artifacts; implement contract/fallback tests and benchmark latency on agreed hardware.
- Keep raw/restricted data, secrets, local environments and model binaries out of Git. Synthetic systems data cannot establish real-world accuracy, fairness or inclusion.

See the [architecture](docs/architecture.md), [risks and assumptions](docs/risks-and-assumptions.md), [NLP limitations](docs/nlp-pipeline-limitations.md), [analyst handbook](docs/tabular-analyst-handbook.md) and [reading pack](docs/reading-list.md).

This is an educational research prototype. Production readiness, fairness, affordability, inclusion, causal explanations and regulatory compliance have not been established.

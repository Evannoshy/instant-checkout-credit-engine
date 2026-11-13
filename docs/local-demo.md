# Running the local research demo

This demo connects the existing nine-feature tabular pipeline to the existing
fusion prediction code. It accepts borrower fields, builds the same features
used in training, and returns a historical default estimate. It applies no
approval or decline threshold.

## Build the local bundle

Use Python 3.13 and the pinned tabular requirements. From the repository root:

```powershell
python -m pip install -r src/tabular/requirements.txt
python -m src.serving.prepare_bundle --raw-csv "C:/path/to/loan.csv"
```

The command verifies the source CSV, rebuilds the approved tabular model,
reproduces all 48,724 existing development scores and writes a local bundle to
`artifacts/serving/demo_v1/`. Model binaries and raw records stay out of Git.
The bundle manifest is written last. Startup requires it and checks every
listed model hash before loading joblib.

To include TF-IDF, obtain Yati's original
`artifacts/nlp/tfidf_fusion_v2_pipeline.joblib` and pass it explicitly:

```powershell
python -m src.serving.prepare_bundle --raw-csv "C:/path/to/loan.csv" --tfidf-artifact "C:/path/to/tfidf_fusion_v2_pipeline.joblib"
```

Its hash must match `pipeline_artifact.sha256` in the published TF-IDF v2
manifest. The demo does not substitute a newly fitted text model for that file.
An integration rebuild attempt on 13 November did not reproduce the exported
TF-IDF scores, so the original artifact is required pending investigation.
The offline fusion results remain reproducible from the committed score files.

## Start the API and form

```powershell
python -m src.serving.api --variant tabular_only --text-policy payload
```

Open `http://127.0.0.1:8000`. This command is a research preview configuration,
not approval of a champion or missing-text policy. Both switches are required.

| Switch | Meaning |
|---|---|
| `--variant tabular_only` | Existing tabular-only stacker and calibrated fallback |
| `--variant tabular_tfidf` | Requires the verified original TF-IDF pipeline |
| `--variant tabular_tfidf_distilbert` | Also requires `--distilbert-dir` and the NLP dependencies |
| `--text-policy payload` | Text is usable when the cleaned combined payload is non-empty |
| `--text-policy description` | Text is usable when the cleaned description is non-empty |

For the three-score variant, obtain Vincent's private `distilbert_fusion_v2`
checkpoint. The loader checks the recorded weights, config and tokenizer hashes
before loading it. The checkpoint is not downloaded automatically. Its live
inference path has not been verified in this environment.

The current server binds only to loopback and handles one request at a time.
It is intended for a local university demonstration. It has no authentication,
deployment setup or load-balancing layer, and stores no requests.

## Request and response

`GET /api/health` returns the loaded variant and explicit text rule.
`POST /api/score` accepts JSON matching
`data/fixtures/demo_application.json`, which contains fictional details only.

- `decision_month` is `Mon-YYYY`. Replay a historical loan with its historical
  issue month. Do not calculate its credit tenure using today's date.
- `revol_util` uses percentage points without a percent sign: 35 means 35%.
- Unknown fields, including labels and repayment outcomes, are rejected.
- Loan amount must be positive. Other supplied numeric fields must be finite
  and nonnegative. Delinquencies and inquiries must be whole numbers.
- Optional missing feature values use the fitted imputation rules and appear
  in `quality_flags`. Unsupported categories are rejected by Stage A.
- Combined text is limited to 12,000 characters; JSON bodies to 32,768 bytes.

The response includes the probability, raw base scores, model versions, actual
tracks used, fallback status and processing times. Blank text follows the
selected rule and uses the approved calibrated tabular fallback. Failed models
return an error rather than an invented score. Model outages do not silently
switch to another variant.

## Checks and latency

```powershell
python -m pytest src/serving -q
$env:SERVING_BUNDLE = "$PWD/artifacts/serving/demo_v1"
$env:LENDINGCLUB_RAW_CSV = "C:/path/to/loan.csv"
python -m pytest src/serving -q -rs
python -m src.serving.benchmark --variant tabular_only --text-policy payload --output artifacts/serving/benchmark_tabular.json
```

The optional parity check scores 50 real validation loans from raw fields and
compares them with the existing offline predictions. It never evaluates the
locked final test. The benchmark measures sequential loopback HTTP requests,
including parsing, feature building, models, fusion and serialization. It uses
one fictional payload; it is not a deployed-service or concurrency benchmark.

## What still needs agreement

Evan and Ariel must choose the final model variant, missing-text rule and final
evaluation gate. The historical US cohort does not establish performance for
Singapore BNPL applicants. The demo makes no fairness, affordability, approval
or current-market performance claim.

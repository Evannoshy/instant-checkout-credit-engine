# Contributing

## Working agreement

Use issues and pull requests for all project work. Keep changes small enough to review, and link model or data claims to reproducible evidence.

## Before starting

1. Confirm the issue has an owner, outcome and acceptance criteria.
2. Identify data, label, split and cross-track dependencies.
3. Do not change a frozen problem, schema, label or split without a decision-log entry.
4. Create a short-lived branch from the latest `main`.

Suggested branch names:

- `docs/problem-definition`
- `data/source-assessment`
- `feature/cashflow-aggregates`
- `model/xgb-baseline`
- `fix/split-leakage`

## Development environments and checks

Run commands from the repository root. Keep track-specific environments: tabular
reference experiments use Python 3.13.15, while the NLP requirements document
separate TF-IDF and transformer runtimes. Install the requirements for the track
you are working on; changing frozen pins requires a new, consistently reproduced
experiment rather than an automatic dependency upgrade.

For tabular work on the reference Windows runtime:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r src/tabular/requirements.txt
python -m pytest scripts src/tabular -q
```

For NLP work, activate a separate environment with the appropriate Python runtime:

```powershell
python -m pip install -r src/nlp/requirements.txt
python -m pytest src/nlp -q
```

The restricted raw CSV and trained transformer checkpoints are not shipped.
Tests requiring the CSV skip when it is unavailable. To enable raw-data checks
with the approved source and the calibration rerun:

```powershell
$env:LENDINGCLUB_RAW_CSV = "C:\approved-data\loan.csv"
$env:CALIBRATION_RECOMPUTE = "1"
python -m pytest scripts src/tabular -q -rs
```

Replace the example path with the approved file location. The source must match
the SHA-256 in [the cohort contract](docs/cohort-definition.md). Model runners
default to `data/raw/loan.csv`; use their `--raw-csv` option when supported for an
external path. The test environment variable is not a universal runner setting.
Exact XGBoost prediction reproduction additionally requires the environment
recorded in [the challenger report](docs/tabular-xgboost-report.md); matching
package versions alone is insufficient. Report passes, failures, skips and the
runtime actually used. An existing test record is not evidence of a new rerun.

No CI workflow is currently committed. For documentation-only changes, check
links, cited counts/metrics and the diff; model refits are unnecessary when code,
configs and evidence artifacts are unchanged.

## Pull requests

Every pull request must:

- Link its issue.
- State what changed and why.
- Include tests or explain why they do not apply.
- Include an experiment ID for any model/result change.
- State data, schema, label, split and model versions affected.
- Record known limitations and rollback.
- Update documentation and the decision log when relevant.
- Avoid raw/restricted data, PII, secrets, model dumps and notebook-output bloat.

At least one reviewer is required. Problem-definition, label, split, schema, model-selection and release changes require the relevant track lead.

## Modeling standards

- Use only information available by the decision timestamp.
- Keep label generation separate from feature generation.
- Use the shared application/label/split manifest.
- Fit preprocessing, feature selection, calibration and resampling on allowed development data only.
- Protect the final test set until the approved evaluation gate.
- Reconcile fitting months, fold coverage and calibration roles before combining
  cross-track OOF exports. Shared IDs alone do not prove time-respecting fusion.
- Evaluate logistic regression and XGBoost on identical rows and metrics.
- Report uncertainty, calibration and policy trade-offs, not only AUROC.
- Treat SHAP as a diagnostic; do not describe it as proof of causality, fairness or compliance.

## Data and security

- Never commit live PII, raw bank transactions, credentials or restricted datasets.
- Record each data source, retrieval date, licence/terms, checksum and permitted use.
- Use tiny, clearly marked synthetic fixtures in public tests.
- Store sensitive or licensed data only in the approved external location.
- Immediately notify the project co-leads if sensitive material reaches Git history.

## Notebooks

Notebooks are for exploration and communication. Reusable label, feature, training, evaluation and inference logic belongs in `src/`. Clear large outputs and remove hidden state before review.

## Definition of done

A contribution is done only when:

- Acceptance criteria pass.
- Tests and required checks pass.
- Evidence is linked.
- Relevant documentation is updated.
- Downstream owners acknowledge contract changes.
- No sensitive or restricted artifact was introduced.

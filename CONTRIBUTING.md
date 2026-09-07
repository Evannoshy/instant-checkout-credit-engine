# Contributing

## Working agreement

Use issues and pull requests for all project work. Keep changes small enough to review, and link model or data claims to reproducible evidence.

Tabular-track contributors must follow the [Tabular Analyst Handbook](docs/tabular-analyst-handbook.md). Accepted decision records take precedence when project requirements change.

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

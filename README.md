# Instant Checkout Credit Engine

A research and engineering prototype for low-latency credit-risk inference at checkout. The proposed system combines a structured-data credit model with a transaction-text model, then fuses their outputs behind a versioned API.

The project is led by NUS FinTech Society. Ariel and Evan are project co-leads; Evan is the Tabular Track Tech Lead.

## Project status

**Current phase:** Week 1 - problem definition and architecture  
**Planning period:** 7 September to 30 November 2026  
**Next gate:** G0 Problem Freeze on 11 September 2026

No trained model or validated performance result exists yet. The immediate goal is to agree what the model predicts, what data is available at decision time, how outcomes mature, and how the tabular, NLP and fusion tracks share data and scores.

## Problem

Traditional credit information can be limited for thin-file applicants. The project investigates whether structured cash-flow and behavioural variables, together with transaction text, can add useful credit-risk information while supporting an instant-checkout user experience.

The proposed prediction unit is one request to finance one checkout basket. The current target proposal is the probability that an originated obligation reaches at least 30 days past due within 90 days of the first contractual due date, or is charged off earlier. This definition remains subject to the G0 review and data feasibility.

See [the problem definition](docs/problem-definition.md) for the full target, timing, population, exclusions and claim boundaries.

## Proposed system

The initial architecture has three model/service tracks:

1. **Tabular track:** a serious logistic-regression/scorecard baseline and an XGBoost challenger over structured application, affordability, cash-flow and prior-repayment features.
2. **NLP track:** a compact transaction-text model that produces a text-risk signal from information genuinely available at checkout.
3. **Fusion/API track:** late fusion over frozen base-model outputs, exposed through a typed service contract with explicit fallback and quality states.

The tabular model produces a risk score, not the final lending decision. The policy layer remains separate from the model and may return `APPROVE`, `DECLINE`, `REFER` or `INSUFFICIENT_DATA`.

See [the architecture](docs/architecture.md) for training, inference, data, contract and failure flows.

## Evidence strategy

The project uses two distinct data lanes unless representative partner data is approved and available:

- **Evidence lane:** permitted real credit data with observed, matured repayment outcomes and reliable timestamps. This supports honest baseline/model evaluation but may only be a proxy for BNPL.
- **Systems lane:** privacy-safe, BNPL-shaped synthetic data with aligned structured and text fields. This supports schema, API, fusion, latency, load and demo testing only.

Synthetic data must not be used to claim real-world default accuracy, fairness or financial inclusion. Public proxy data must not be presented as representative of Singapore BNPL users unless that is demonstrably true.

## Project guardrails

- Use only features available at or before the checkout decision timestamp.
- Keep label construction separate from feature construction.
- Never label a declined application as good simply because no repayment default is observed.
- Use the same application IDs, label version and split manifest across tabular and NLP tracks.
- Keep the final out-of-time test set locked until the agreed evaluation gate.
- Treat logistic regression as a real champion candidate, not a ceremonial baseline.
- Treat SHAP as a model diagnostic. It does not prove causality, fairness or regulatory compliance.
- Commit no secrets, live PII, raw bank transactions or restricted datasets.
- State performance and latency with the model/data version, sample size, split, hardware and measurement protocol.
- Describe the November result as a research prototype unless the evidence justifies a narrower or stronger statement.

## Provisional performance objectives

These are engineering objectives, not validated results:

- Compare models with AUROC, PR-AUC, Brier score, log loss and calibration plots.
- Report approval-rate/bad-rate and illustrative expected-loss/value curves.
- Use time-aware validation and confidence intervals.
- Evaluate missingness, time shift, feature-family ablations and supported subgroups.
- Target warmed tabular preprocessing + prediction + calibration p95 latency of at most 10 ms on agreed hardware.
- Target warmed full-system p95 latency of at most 100 ms under a frozen end-to-end benchmark.

Cold start, explanation time, concurrency, payload size and target hardware must be reported separately or explicitly included in the benchmark.

## Repository layout

```text
.
|-- README.md
|-- updated-pitchdeck.pdf
`-- docs/
    |-- architecture.md
    |-- problem-definition.md
    |-- reading-list.md
    |-- risks-and-assumptions.md
    |-- tabular-analyst-handbook.md
    `-- decisions/
        `-- decision-log.md
```

Code, configuration, tests and data documentation will be introduced after G0 as their contracts become stable. Do not create empty folders purely for appearance; add each area with its first owned artifact.

## Target repository layout

By the model-development and release phases, the repository should converge on:

```text
.
|-- .github/
|   |-- ISSUE_TEMPLATE/
|   |-- workflows/ci.yml
|   `-- pull_request_template.md
|-- configs/
|   |-- data.yaml
|   |-- features.yaml
|   |-- model_logistic.yaml
|   |-- model_xgb.yaml
|   `-- evaluation.yaml
|-- data/
|   |-- README.md
|   `-- fixtures/
|-- docs/
|   |-- architecture.md
|   |-- problem-definition.md
|   |-- data-source-assessment.md
|   |-- data-sheet.md
|   |-- feature-dictionary.md
|   |-- validation-plan.md
|   |-- validation-report.md
|   |-- fairness-report.md
|   |-- explanation-and-reason-policy.md
|   |-- model-card-tabular.md
|   |-- monitoring-and-rollback.md
|   |-- reading-list.md
|   |-- risks-and-assumptions.md
|   |-- tabular-analyst-handbook.md
|   `-- decisions/decision-log.md
|-- notebooks/
|   |-- 01_data_audit.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_baselines.ipynb
|   |-- 04_xgboost_experiments.ipynb
|   `-- 05_calibration_explainability.ipynb
|-- reports/
|   |-- figures/
|   `-- tables/
|-- scripts/
|   |-- reproduce_tabular.ps1
|   `-- generate_synthetic_fixture.py
|-- src/credit_engine/
|   |-- tabular/
|   |   |-- schema.py
|   |   |-- ingest.py
|   |   |-- labels.py
|   |   |-- split.py
|   |   |-- features.py
|   |   |-- train.py
|   |   |-- calibrate.py
|   |   |-- evaluate.py
|   |   |-- explain.py
|   |   `-- infer.py
|   |-- nlp/
|   |-- fusion/
|   `-- api/
|-- tests/
|   |-- unit/
|   |-- data/
|   |-- contract/
|   `-- integration/
|-- .env.example
|-- .gitignore
|-- CODEOWNERS
|-- CONTRIBUTING.md
|-- Dockerfile
|-- LICENSE
|-- README.md
|-- pyproject.toml
`-- dependency lock file
```

The repository should contain code, configuration, tiny synthetic fixtures, documentation and reproducible aggregate evidence. It should not contain raw consumer data, credentials, local environments, experiment caches or unrestricted collections of trained model binaries.

## Documentation

- [Problem definition](docs/problem-definition.md)
- [Architecture](docs/architecture.md)
- [Decision log](docs/decisions/decision-log.md)
- [Risks and assumptions](docs/risks-and-assumptions.md)
- [Tabular analyst handbook](docs/tabular-analyst-handbook.md)
- [Team reading pack](docs/reading-list.md)
- [Pitch deck](updated-pitchdeck.pdf)

## Week 1 definition of done

Week 1 is complete only when:

- The prediction unit, intended population, outcome and performance window are approved.
- The decision-time feature boundary and label maturity rules are approved.
- The treatment of declined and censored applications is explicit.
- The tabular, NLP and fusion tracks agree shared IDs, label version, splits and score semantics.
- The data-source decision is converted into a Week 2 plan.
- The architecture separates offline training, online inference, labels and audit-only data.
- Open risks and assumptions have owners and dates.
- The Tabular Track Tech Lead records `GO`, `GO WITH CONDITIONS` or `NO-GO` at G0.

## Development status

There is not yet an executable training or serving environment. Setup instructions will be added with the first implementation PR after the problem, data and interface contracts pass G0. Until then, contributors should work through reviewable documentation PRs and avoid committing generated or restricted artifacts.

## Governance and limitations

This repository demonstrates technical and model-risk practices for an educational project. It is not legal, regulatory, privacy, credit-risk or information-security advice. Any real lending use would require representative data, independent validation, data rights and consent review, security controls, production monitoring, human/accountability processes and qualified jurisdiction-specific review.

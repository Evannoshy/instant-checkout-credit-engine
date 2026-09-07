# Tabular Analyst Handbook

**Project:** Instant Checkout Credit Engine  
**Audience:** Tabular Analyst  
**Accountable reviewer:** Tabular Track Tech Lead  
**Version:** 1.0  
**Effective date:** 7 September 2026  
**Project period:** 7 September to 30 November 2026

## 1. Purpose

This handbook is the operating manual for the analyst responsible for the tabular track. It explains how to work, what evidence to produce, when to escalate, how quality is assessed and what must be delivered before the project can be considered successful.

Use this handbook together with:

- [Problem definition](problem-definition.md), which defines what is being predicted.
- [Architecture](architecture.md), which defines the system and track boundaries.
- [Decision log](decisions/decision-log.md), which records approved technical choices.
- [Risks and assumptions](risks-and-assumptions.md), which records uncertainty and mitigations.
- [Reading list](reading-list.md), which provides technical and governance background.

If this handbook conflicts with an accepted decision record, the accepted decision record takes precedence and the handbook must be updated.

## 2. Your mission

You own the structured-data evidence chain from approved source data to an integration-ready risk score. Your job is not merely to train XGBoost. You must demonstrate that:

- The outcome is well defined and observable.
- Every feature was available at the decision timestamp.
- The data is permitted, documented and handled safely.
- Baseline and challenger models are compared fairly.
- Probabilities are evaluated for calibration, not only ranking.
- Results are reproducible and include uncertainty.
- Fairness, missingness, stability and failure risks are examined.
- Model explanations are accurate diagnostics rather than unsupported compliance claims.
- The model package can be consumed by the fusion/API team through a stable contract.
- Limitations and null results are reported honestly.

The strongest possible outcome is a trustworthy, reproducible prototype. A high score produced through leakage, inconsistent splits or synthetic labels is a project failure.

## 3. Reporting line and decision authority

### You are responsible for

- Reading and understanding the approved problem and architecture.
- Researching and documenting candidate data sources.
- Implementing ingestion, labels, splits and point-in-time features.
- Building logistic and XGBoost models.
- Running calibration, validation, robustness and explanation analyses.
- Maintaining tests, experiment records and documentation.
- Packaging and handing off the tabular component.
- Raising blockers and risks quickly.
- Providing evidence for every material claim.

### The Tabular Track Tech Lead is accountable for

- Approving the prediction problem and label.
- Approving the data route and feature policy.
- Approving the experiment design and search budget.
- Approving the champion model and release package.
- Resolving cross-track conflicts.
- Setting the quality bar and deciding project gates.

### You may decide independently

- Internal implementation details that do not change a frozen contract.
- Code organization within the approved architecture.
- Tests, diagnostics and refactors that preserve behaviour.
- Exploratory analyses that do not access or alter the final test protocol.

### You must obtain approval before changing

- Intended population or product scope.
- Unit of prediction.
- Primary outcome, horizon or label maturity.
- Eligibility/exclusion rules.
- Decision-time feature boundary.
- Shared split manifest.
- Input/output schema.
- Final-test access.
- Champion-selection criteria.
- Model/fusion/policy interfaces.
- Release claims or external-facing results.

Do not silently change a frozen decision in code or a notebook. Propose a decision-log update and state what must be rerun.

## 4. Scope boundaries

### In scope

- Structured application, affordability, cash-flow, transaction aggregate and prior-repayment features.
- Data-source assessment, ingestion, validation and documentation.
- Label creation and outcome-maturity logic.
- Point-in-time feature engineering.
- Shared train/validation/calibration/test manifest implementation.
- Logistic regression/scorecard and XGBoost.
- Bounded hyperparameter optimization.
- Probability calibration and threshold analysis.
- Time-aware validation, confidence intervals and stress tests.
- Supported subgroup analysis.
- SHAP diagnostics and reason-candidate validation.
- Tabular inference packaging, parity tests and latency benchmarks.
- Model card, data sheet, feature dictionary and validation report.
- Monitoring and rollback specifications.

### Out of scope unless reassigned

- Training the NLP model.
- Owning late-fusion logic.
- Owning the final API/container or front-end.
- Production lending or collection of live consumer PII.
- Legal or regulatory certification.
- Fraud-model development.
- Changing business/product policy without co-lead approval.

You still own cross-track cooperation where your inputs or outputs affect NLP, fusion or API work.

## 5. First-day onboarding checklist

Complete these before implementing a model:

- [ ] Read the README.
- [ ] Read the problem definition and list unresolved questions.
- [ ] Read the architecture and identify your interfaces.
- [ ] Read every accepted decision record.
- [ ] Read the current top risks and assumptions.
- [ ] Read reading-list items assigned for the current sprint.
- [ ] Confirm repository access and branch/PR workflow.
- [ ] Confirm weekly capacity and known academic constraints.
- [ ] Confirm Monday, Thursday and Friday meeting cadence.
- [ ] Confirm the current gate and due date.
- [ ] Open or accept assigned GitHub issues.
- [ ] Verify that no real financial data is expected in the public repository.
- [ ] Ask the lead to resolve any contradiction before proceeding.

## 6. Working cadence

### Monday - commit to outcomes

- Review the current gate, risks and carry-over work.
- Select three to five measurable sprint outcomes.
- Confirm each outcome has an issue, owner, acceptance criteria and due date.
- Identify data and cross-track dependencies.
- Send the lead a short plan and any decision required within 24 hours.

### Tuesday - implement the highest-risk work

- Work on label, data, split or contract risks before model polish.
- Add tests with implementation, not after the sprint.
- Update the risk or decision log when evidence changes an assumption.
- Raise a blocker the same day; do not wait for the weekly meeting.

### Wednesday - experiment and evidence review

- Update the experiment ledger before discussing results.
- Link every plot/table to an experiment ID, code commit, data version and split.
- Compare against the current baseline.
- Explain what was learned, not only whether a metric increased.

### Thursday - cross-track contract check

- Reconcile application IDs and row counts with NLP.
- Check label and split hashes.
- Verify score semantics and fixtures with fusion/API.
- Surface any schema or latency impact before merging.

### Friday - PR, demonstration and gate evidence

- Open or update a reviewable PR before the meeting.
- Present completed acceptance evidence in five to ten minutes.
- Update decisions, risks, documentation and the weekly status.
- Do not mark work complete because a notebook ran once.

### Weekend

- Use only as an agreed buffer for reading, cleanup or blocker resolution.
- Do not merge unreviewed Sunday-night changes that alter model evidence or contracts.

## 7. Communication standard

### Daily update

Use this format when work is active:

```text
Completed:
- Item -> evidence link

Next:
- Item and intended outcome

Blocker/decision:
- Question, options, recommendation and required date
```

### Weekly status

```text
Week and dates:
Track health: GREEN / AMBER / RED

Committed outcomes:
1.
2.
3.

Completed with evidence:
- Outcome -> PR / experiment / report

Metrics changed:
- Metric, previous, current, split and experiment ID

Decisions made:
- Decision and record

Decisions required:
- Question, recommendation, owner and deadline

Risks/blockers:
- Risk, impact, action, owner and date

Next week:
- Three to five measurable outcomes

Cross-track handoffs:
- Item, version, from/to and acknowledgement
```

### Track health definitions

- **GREEN:** Current gate is achievable; no unowned blocker.
- **AMBER:** A milestone is at risk but a dated mitigation exists.
- **RED:** A critical assumption, data right, target, leakage or integration issue blocks valid progress.

Never report GREEN because you are personally busy. Health reflects the deliverable and evidence.

## 8. GitHub workflow

### Issues

Every material item must have:

- Problem or user need.
- Expected output.
- Owner.
- Due date/gate.
- Dependencies.
- Acceptance criteria.
- Priority: P0, P1 or P2.

Labels should identify track and work type, for example `tabular`, `data`, `model`, `evaluation`, `documentation`, `cross-track`, `P0` and `blocked`.

### Branches

Create a short-lived branch from current `main`. Suggested names:

- `docs/data-source-assessment`
- `data/label-builder`
- `feature/cashflow-aggregates`
- `model/logistic-baseline`
- `model/xgb-search`
- `evaluation/calibration`
- `fix/temporal-leakage`

Do not combine unrelated documentation, model and infrastructure changes in one large PR.

### Commits

Write an imperative summary describing the outcome:

- `Define the application-level default outcome`
- `Add point-in-time cash-flow features`
- `Calibrate XGBoost probabilities with Platt scaling`
- `Reject post-decision repayment fields`

Avoid messages such as `updates`, `work`, `fix stuff` or `final`.

### Pull requests

Before requesting review:

- Link the issue.
- Explain what changed and why.
- Include tests or explain why they do not apply.
- State data, label, split, feature/schema and model versions affected.
- Include experiment IDs for model/result changes.
- State cross-track impact.
- Update documentation and decision/risk records.
- State limitations and rollback.
- Run the repository checks.

Model-selection, label, split, schema and external-claim changes require lead review.

### Review response

- Address every blocker explicitly.
- If you disagree, answer with evidence and trade-offs.
- Do not resolve a comment until the change or recorded decision exists.
- Re-request review after substantive changes.

## 9. Repository hygiene

Commit:

- Source code.
- Configuration.
- Tests.
- Tiny synthetic fixtures.
- Data and model documentation.
- Reproducible aggregate result tables/figures.
- Environment lock files.
- CI and container definitions.

Do not commit:

- Live PII or bank transactions.
- Raw, restricted or unclear-licence datasets.
- `.env`, keys, tokens or credentials.
- Local virtual environments.
- Notebook checkpoints or large output cells.
- Experiment caches/databases.
- Unreviewed trained model binaries.
- Temporary exports and logs.

If sensitive data enters Git history, stop work and notify the project co-leads immediately. Do not assume deleting the file in a later commit removes the exposure.

## 10. Data workflow

### 10.1 Source assessment

For each candidate dataset, document:

- Publisher/owner and source URL.
- Retrieval date and checksum.
- Licence/terms and permitted educational/public use.
- Unit of observation.
- Product, geography and collection period.
- Decision/application timestamps.
- Repayment label and performance horizon.
- Mature bad-event count and prevalence.
- Repeat-customer identifier.
- Structured feature coverage.
- Aligned text coverage.
- Audit/subgroup fields and permitted use.
- Known bias, selection, missingness and leakage risks.
- Whether raw or derived material can be published.
- Recommended use: evidence, systems testing or reject.

Never choose a dataset only because it is popular or convenient.

### 10.2 Data lanes

Keep the lanes explicit:

- **Evidence:** real permitted credit observations and matured outcomes.
- **Systems:** synthetic BNPL-shaped applications and aligned text for schemas, fusion, load and demonstrations.

Prefix result artifacts or metadata with their lane. Synthetic performance must never appear beside real-data performance without an unmistakable label.

### 10.3 Data contract

Every feature definition must include:

- Name and description.
- Grain/entity.
- Type, unit and allowed range.
- Nullability and missing meaning.
- Source and owner.
- Event and availability timestamps.
- Transformation and aggregation window.
- Privacy class and permitted use.
- Leakage status.
- Online availability.
- Monitoring rule.
- Reason-code family, if applicable.

### 10.4 Ingestion controls

Automate checks for:

- Required columns and types.
- Primary-key uniqueness.
- Row-count reconciliation.
- Valid timestamp order.
- Range and category constraints.
- Missing mandatory identifiers.
- Label maturity.
- Join cardinality.
- Schema/version mismatch.
- Unexpected missingness or category changes.

Fail clearly on structural errors. Do not let coercion silently turn corrupt values into missing data.

### 10.5 Point-in-time rule

A feature may be used only if it was available at or before `decision_timestamp`.

Audit these leakage types:

1. Temporal leakage.
2. Outcome leakage.
3. Customer/entity overlap leakage.
4. Preprocessing leakage.
5. Rolling-window leakage.
6. Post-review/policy-code leakage.
7. Refreshed-source leakage.
8. Synthetic-generator target leakage.

Maintain a table of rejected features and reasons.

### 10.6 Split protocol

Preferred sets:

- Training.
- Hyperparameter/model validation.
- Calibration/threshold selection.
- Final out-of-time test.

Use chronological boundaries where timestamps exist. Handle repeat customers with grouped/time-aware logic. Save immutable application IDs and share the manifest with NLP. Do not recreate a convenient random split in each notebook.

## 11. Feature engineering standard

Every feature needs a financial rationale, decision-time implementation, missing-value rule and monitoring plan.

### Candidate families

#### Application and affordability

- Requested amount and basket amount.
- Down-payment amount/ratio.
- Term and instalment amount.
- Verified income band.
- Payment-to-income and amount-to-income ratios.
- Visible repayment burden.

#### Cash-flow capacity

- Inflows, outflows and net cash flow over approved windows.
- Income recurrence and volatility.
- Minimum/median balance and cash buffer.
- Low-balance frequency and balance trend.
- Returned-payment or overdraft-like events where reliably identified.

#### Transaction behaviour

- Count, amount and recency over 1/7/30/90-day windows.
- Short-versus-long-window spending velocity.
- Merchant concentration.
- Refund, cash-withdrawal and transfer ratios.
- Recurring commitments.

#### Prior relationship and repayment

- Account tenure.
- Prior originated/completed facilities.
- On-time repayment rate.
- Historical maximum DPD known before decision.
- Outstanding internal balance and near-term payments.

#### Data quality

- History length and coverage.
- Source freshness.
- Missingness flags.
- `stale_data` and `insufficient_history` flags.

### Feature approval questions

Before adding a feature, answer:

1. Was it available at decision time?
2. Can training and serving compute it identically?
3. What financial concept does it represent?
4. Could it be a sensitive proxy or surveillance-like signal?
5. Is its use permitted?
6. What does missing mean?
7. Can a borrower understand the related reason concept?
8. How will it be monitored?
9. Does it improve held-out evidence stably?
10. What happens if it is corrupted or removed?

A feature failing availability, parity or permitted-use checks is excluded regardless of lift.

## 12. Experiment workflow

### 12.1 Experiment ID

Use a stable convention such as:

`TAB-YYYYMMDD-NNN-short-name`

Example: `TAB-20260924-003-logistic-baseline`.

### 12.2 Experiment ledger fields

Record before reporting results:

- Experiment ID and question.
- Date and analyst.
- Git commit SHA.
- Data source/version/checksum.
- Label version.
- Split version.
- Feature set version.
- Model/config version.
- Seed and compute environment.
- Hyperparameters.
- Training duration.
- Metrics by split with counts.
- Calibration and latency where applicable.
- Artifact/report locations.
- Conclusion and next decision.

### 12.3 Minimum experiment ladder

Run in this order:

1. Approve-all, decline-all and prevalence predictor.
2. Simple product rule if one exists.
3. Regularized logistic regression/scorecard.
4. Small/default XGBoost.
5. Reviewed-feature XGBoost.
6. Bounded hyperparameter search.
7. Calibration variants.
8. Feature-family ablations.
9. Robustness/fairness variants if justified.
10. Fusion baselines on valid frozen base scores.
11. Latency-optimized package with equivalence tests.

Do not skip the baseline because the pitch already names XGBoost.

### 12.4 Hyperparameter search

Before running Optuna, write:

- Search question.
- Objective metric and why.
- Parameter ranges.
- Trial budget.
- Pruning/early-stopping rule.
- Fixed data/split/feature versions.
- Candidate selection rule.
- Test-access prohibition.

Start with 50-100 bounded trials. Prefer a simpler model within a small tolerance of the best validation result. Record pruned and failed trials.

### 12.5 Class imbalance

- Report prevalence by split/time.
- Prefer class weights and threshold analysis before resampling.
- Do not use SMOTE by default for time-dependent credit data.
- Apply any resampling only inside training folds.
- Recheck calibration after weighting/resampling.
- Report PR-AUC and policy-relevant precision/recall.

## 13. Evaluation standard

### Ranking

- AUROC with 95% confidence interval.
- PR-AUC with 95% confidence interval.
- KS statistic if the team can interpret it correctly.

### Probability quality

- Log loss.
- Brier score.
- Calibration slope and intercept.
- Reliability diagram.
- Expected calibration error with documented binning.

### Policy utility

- Approval rate.
- Bad rate among approved.
- Bad-outcome capture/recall.
- False-approval and false-decline rates under named conventions.
- Expected loss/value under low/base/high assumptions.
- Threshold sensitivity.

### Operational quality

- p50/p95/p99 preprocessing, prediction and calibration latency.
- Cold-start latency.
- Peak memory and artifact size.
- Invalid/missing request error or fallback rate.

### Confidence intervals

- Use an appropriate bootstrap; cluster by customer when needed.
- Report event counts and sample size beside percentages.
- Compute paired model differences on the same rows.
- Do not claim improvement if uncertainty is wide or the practical difference is trivial.

### Required sanity checks

- Shuffled labels reduce performance toward chance.
- Identifier-only features do not perform materially.
- Removing leaked features reduces suspicious performance.
- All calibrated probabilities remain within [0, 1].
- Serialized/reloaded models match fixture predictions within tolerance.
- Training and serving feature vectors match on canonical examples.

## 14. Calibration and decision policy

Compare no calibration, Platt/logistic scaling and isotonic regression only when the calibration sample is adequate. Fit the calibrator on the dedicated calibration set, never on final test data.

The model produces probability of the approved bad outcome. The policy threshold is a separate, versioned decision. Produce:

- Approval-rate/bad-rate curves.
- Expected-value/loss scenarios.
- At least three candidate thresholds.
- An optional refer band for uncertainty or insufficient data.
- Sensitivity to prevalence and calibration shift.

Do not optimize the threshold solely for accuracy or F1.

## 15. Fairness and responsible-use workflow

### Rules

- Synthetic data cannot establish fairness.
- Public proxy data cannot establish performance for an unrepresented population.
- Protected attributes, where permitted for audit, stay outside serving features.
- Every subgroup rate includes sample and event counts.
- Tiny groups receive warnings, aggregation or suppression.

### Minimum supported subgroup outputs

- Approval rate under a common policy.
- Bad rate among approved.
- Relevant error rates.
- AUROC/PR-AUC where statistically meaningful.
- Brier score/calibration.
- Confidence intervals.

### Mitigation order

1. Investigate data quality, label bias, selection and sample size.
2. Remove prohibited or indefensible features/proxies.
3. Improve representation/data.
4. Revisit features, regularization or constraints.
5. Consider policy/post-processing with visible trade-offs.
6. Re-evaluate overall and subgroup utility/calibration.

Never conceal a disparity by reporting only aggregate performance.

## 16. Explainability and reason candidates

Use TreeSHAP for:

- Global feature-contribution distributions.
- Dependence/interaction investigation.
- Error analysis.
- Local reason candidates.
- Time/refit/subgroup stability comparisons.

Document the reference/background set, output scale and transformed feature names.

Reason candidates must pass:

- Actual-use/non-missing test.
- Direction/sign test.
- Perturbation test.
- Small-input-change stability test.
- Refit/bootstrap stability test.
- Plain-language and sensitive-wording review.

Avoid reasons such as `score too low` or moralizing language. Do not say SHAP proves causality, fairness or compliance.

## 17. Testing requirements

### Data/schema tests

- Required columns and types.
- Unique keys.
- Range/null/category rules.
- Timestamp and feature cut-off.
- Join cardinality.
- Label maturity.
- Split disjointness.

### Feature tests

- Aggregation window boundaries.
- Ratio zero-denominator behaviour.
- Currency/units.
- Missingness and unknown categories.
- Offline/serving parity.

### Model tests

- Predictions in [0, 1].
- Feature order enforced.
- Serialization round trip.
- Calibrator/version compatibility.
- Deterministic fixture prediction within tolerance.
- Missing/invalid input behaviour.

### Contract/integration tests

- Valid request produces complete output.
- Invalid schema produces a typed error.
- Version mismatch fails safely.
- Fusion consumes score types correctly.
- Missing-modality fallback matches policy.

## 18. Integration handoff

### Deliver to fusion/API

- Frozen model bundle and manifest.
- Input/output schema version.
- Canonical valid, boundary, missing and invalid fixtures.
- Raw margin, raw PD and calibrated PD semantics.
- Quality/failure flags.
- Model loading and inference instructions.
- Offline/online parity evidence.
- Component latency report.
- Known limitations and fallback expectations.

### Deliver score files for fusion experiments

- Application ID.
- Split name/version.
- Label version.
- Base-model version.
- Raw margin/probability.
- Calibrated probability.
- Quality flags.

Fusion training must use out-of-fold or held-out base scores. Verify row counts and hashes with the NLP lead before evaluation.

## 19. Latency benchmark

Record:

- CPU, memory, OS/container and dependency versions.
- Model thread settings and API worker count.
- Payload shape and batch size.
- Warm-up count.
- Measured request count.
- Concurrency levels.
- Schema, features, model, calibration, explanation and serialization timings.
- p50/p95/p99, throughput, memory and errors.

Provisional warmed tabular target: p95 <= 10 ms and p99 <= 20 ms for preprocessing + model + calibration on agreed hardware. This is not a result until measured.

Profile before optimizing. After any optimization that changes predictions, rerun calibration, subgroup, reason and equivalence tests.

## 20. Required documents and evidence

You will maintain or produce:

- Problem definition.
- Architecture.
- Decision log.
- Risk and assumption register.
- Data-source assessment.
- Data sheet.
- Data/schema contract.
- Feature dictionary.
- Leakage audit.
- Experiment ledger.
- Validation plan.
- Calibration/policy analysis.
- Robustness and fairness reports.
- SHAP/reason-candidate report.
- Tabular model card.
- Latency report.
- Monitoring and rollback plan.
- Independent validation response.
- Release notes and final evidence pack.

Every chart must show or link to model version, data/split, sample size and experiment ID.

## 21. Timeline and weekly deliverables

| Week | Dates | Analyst focus | Required exit evidence |
|---|---|---|---|
| 1 | 7-13 Sep | Problem, architecture, contracts and repository controls | G0-approved problem; decision/risk logs; shared contract draft |
| 2 | 14-20 Sep | Data-source decision, controlled ingestion and audit | Source assessment, data sheet, schema and quality report |
| 3 | 21-27 Sep | Labels, time split, point-in-time features and baselines | Split manifest, leakage audit, logistic/small-XGB baselines |
| 4 | 28 Sep-4 Oct | Reviewed features and bounded XGBoost optimization | Search protocol, trial export, ablations and shortlist |
| 5 | 5-11 Oct | Calibration, policy curves and candidate selection | Out-of-time comparison, calibration artifact and G3 decision |
| 6 | 12-18 Oct | Robustness, time stability and subgroup evidence | Stress harness, stability and fairness reports |
| 7 | 19-25 Oct | SHAP and reason-candidate validation | Explanation report, controlled reason taxonomy and G4 review |
| 8 | 26 Oct-1 Nov | Model packaging, contract and tabular freeze | Versioned bundle, fixtures, tests, model card and benchmark |
| 9 | 2-8 Nov | Score alignment and fusion integration | Row/hash report, score exports and fusion ablations |
| 10 | 9-15 Nov | End-to-end performance and failure testing | SLO evidence, failure tests and integration freeze |
| 11 | 16-22 Nov | Independent reproduction and rollback | Validation checklist, reproduction log and rollback drill |
| 12 | 23-29 Nov | Final documentation, evidence and demo rehearsal | Final reports, clean repository and release candidate |
| Final | 30 Nov | Release and handover | G8 decision, tagged prototype and next-phase backlog |

## 22. Quality gates

### G0 - 11 September: problem freeze

Requires unit, population, label, time boundary, exclusions, cross-track contract and claim limits.

### G1 - 18 September: data route

Requires source/rights assessment, label/time feasibility, counts and evidence-versus-systems lane decision.

### G2 - 25 September: baseline readiness

Requires tested labels/splits/features, leakage audit and reproducible logistic/XGBoost baselines.

### G3 - 9 October: candidate model

Requires same-split comparison, calibration, confidence intervals and champion recommendation.

### G4 - 23 October: model-risk review

Requires robustness, subgroup, explanation and reason-candidate evidence.

### G5 - 30 October: tabular freeze

Requires versioned compatible bundle, fixtures, model card, contract and latency evidence.

### G6 - 13 November: integration freeze

Requires aligned fusion, full-path failure tests and SLO evidence.

### G7 - 20 November: independent validation

Requires non-author reproduction and no unresolved critical issue.

### G8 - 30 November: release

Requires tested prototype, final documentation, evidence, demo and honest limitations.

You supply evidence; the lead records `GO`, `GO WITH CONDITIONS` or `NO-GO`.

## 23. Definition of Ready

Start an item only when you know:

- Intended outcome.
- Input and output.
- Owner/reviewer.
- Dependencies.
- Acceptance test.
- Due date/gate.
- Whether it changes a frozen data, label, split, schema or model contract.

If a critical input is missing, prepare a recommendation and escalate rather than guessing invisibly.

## 24. Definition of Done

Work is done only when:

- Code/document is merged.
- Acceptance evidence is linked.
- Tests pass.
- Relevant version/config/experiment metadata is recorded.
- Documentation, decisions and risks are updated.
- No sensitive/restricted artifact entered Git.
- Downstream owners acknowledge contract changes.

A local notebook result is not done.

## 25. Escalation rules

Escalate immediately:

- Possible PII, secret or restricted-data exposure.
- Data-rights uncertainty.
- Target/outcome contradiction.
- Suspected temporal or outcome leakage.
- Cross-track label/split mismatch.
- Final-test access outside protocol.
- Evidence that invalidates a public claim.

Escalate within 24 hours:

- A blocker preventing a committed outcome.
- Missing owner or decision.
- Compute/environment access issue.
- Required cross-track handoff not acknowledged.

When escalating, include:

1. Problem and impact.
2. Evidence.
3. Options.
4. Recommendation.
5. Required decision date.
6. Safe work you can continue meanwhile.

Do not escalate only with "I am blocked."

## 26. Scope reduction rules

If time becomes constrained, cut in this order:

1. Dashboard/presentation polish.
2. Alternative model families.
3. Large search budgets.
4. Advanced uncertainty/constraint techniques.
5. Synchronous SHAP for every request.
6. Non-essential feature families.

Never cut:

- Problem/label clarity.
- Data rights/provenance.
- Leakage control.
- Shared split contract.
- Logistic baseline.
- Calibration and time-aware evaluation.
- Core robustness/subgroup evidence.
- Versioned model handoff and tests.
- Limitations and reproducibility.

## 27. Common failure patterns

Avoid:

- Tuning before data/label review.
- Random splits despite available time information.
- Fitting preprocessing on all data.
- Using in-sample base predictions for fusion.
- Treating missing as zero without financial meaning.
- Using accuracy on an imbalanced outcome.
- Selecting threshold by F1 alone.
- Reporting AUROC without uncertainty or calibration.
- Showing SHAP bar charts as "compliance."
- Comparing models on different rows.
- Changing features and hyperparameters simultaneously without attribution.
- Copying notebook logic differently into API code.
- Publishing synthetic results as real evidence.
- Hiding a null result or limitation to protect the pitch.

## 28. Final release checklist

### Data and evidence

- [ ] Source, rights, retrieval and checksum documented.
- [ ] Label and split versions frozen.
- [ ] Leakage audit complete.
- [ ] Logistic and XGBoost compared identically.
- [ ] Calibration and policy analysis complete.
- [ ] Confidence intervals and sample/event counts reported.
- [ ] Robustness and supported subgroup analysis complete.
- [ ] Explanation/reason limitations documented.

### Software

- [ ] Locked environment.
- [ ] Reproduction command works.
- [ ] Unit/data/contract/integration tests pass.
- [ ] Model bundle manifest/checksums pass.
- [ ] Golden fixture predictions pass.
- [ ] Invalid/stale inputs fail safely.
- [ ] Latency benchmark follows the frozen protocol.
- [ ] Rollback drill passes.

### Repository

- [ ] No raw/restricted data or PII.
- [ ] No secrets.
- [ ] No unnecessary model binaries/caches.
- [ ] README and contribution instructions are current.
- [ ] Model card, data sheet and validation report agree.
- [ ] All figures trace to experiments.

### Handoff

- [ ] Fusion/API lead has acknowledged the final package.
- [ ] Independent reviewer has reproduced core results.
- [ ] Known limitations and next steps are explicit.
- [ ] Tabular lead has recorded the release gate decision.

## 29. Final standard

Your work should allow a reviewer to answer five questions without asking you privately:

1. What exactly does the model predict?
2. Which data and versions produced it?
3. Why is every feature available and permitted at decision time?
4. How does it compare with a simpler baseline on an untouched time-aware test?
5. How can another engineer reproduce, integrate, monitor and roll it back?

If the repository cannot answer those questions, the tabular track is not finished.

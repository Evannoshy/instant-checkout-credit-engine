# Decision Log

**Project:** Instant Checkout Credit Engine  
**Document owner:** Tabular Track Tech Lead  
**Created:** 7 September 2026

## 1. Purpose

This log records decisions that affect the tabular track or its contracts with the NLP and fusion/API tracks. It prevents important choices from being buried in chats, notebooks or code.

## 2. Status definitions

- **Proposed:** Recommended direction awaiting the named approval gate.
- **Accepted:** Approved and active.
- **Accepted with conditions:** Active only under recorded conditions and dates.
- **Superseded:** Replaced by a newer decision; historical record remains.
- **Rejected:** Considered and not adopted.

## 3. Decision summary

| ID | Decision | Status | Owner | Decision date/gate | Revisit trigger |
|---|---|---|---|---|---|
| D-001 | Use one checkout financing application as the prediction unit | Proposed | Tabular lead | G0 - 11 Sep | Product scope changes materially |
| D-002 | Use `bad_30dpd_90d` as the proposed primary outcome | Proposed | Tabular lead + co-leads | G0/G1 | Data cannot support due dates, DPD or maturity |
| D-003 | Close all feature availability at the decision timestamp | Proposed | Tabular lead | G0 | Serving architecture proves a required source arrives later |
| D-004 | Use separate evidence and synthetic systems-data lanes | Proposed | Tabular lead + co-leads | G1 - 18 Sep | Representative permitted partner data becomes available |
| D-005 | Use one shared application/label/split manifest across model tracks | Proposed | Tabular + NLP leads | Cross-track review - 10 Sep | No application-level multimodal alignment is possible |
| D-006 | Export raw margin, raw PD, calibrated PD and quality metadata | Proposed | Tabular + fusion leads | Cross-track review - 10 Sep | Fusion design demonstrates a different justified contract |
| D-007 | Separate model score, calibration, fusion and decision policy | Proposed | Track leads | G0 - 11 Sep | Architecture review identifies a simpler auditable boundary |
| D-008 | Keep logistic regression eligible to be the final champion | Proposed | Tabular lead | G0 - 11 Sep | None; model selection evidence decides champion |
| D-009 | Treat SHAP as a diagnostic, not compliance evidence | Proposed | Project co-leads | G0 - 11 Sep | Qualified review approves narrower jurisdiction-specific usage |
| D-010 | Define sub-100 ms as a measured percentile SLO | Proposed | Fusion/API lead | Cross-track review - 10 Sep | Target hardware/product requirement changes |
| D-011 | Use the locked Kaggle LendingClub archive as the v1 evidence source | Accepted with conditions | Project co-leads | 10 Sep 2026 | Rights, checksum or source changes |
| D-012 | Lock the real-text cohort and chronological `split-v1` manifest | Accepted | Tabular + NLP leads | 10 Sep 2026 | Material cohort defect is demonstrated |
| D-013 | Generate 3,000 training-only synthetic descriptions with no synthetic labels | Accepted with conditions | Project co-leads | 10 Sep 2026 | Pilot fails quality, privacy or leakage checks |
| D-014 | Exchange out-of-fold default probabilities between tracks | Accepted | Tabular + NLP + fusion leads | 10 Sep 2026 | Fusion evaluation justifies a versioned replacement |
| D-015 | Freeze an executable, exact-match definition of the v1 LendingClub cohort | Accepted | Tabular + NLP leads | 19 Sep 2026 | Material cohort defect or source change is demonstrated |

## 4. Detailed decision records

## D-001 - Prediction unit

**Status:** Proposed  
**Owner:** Tabular Track Tech Lead  
**Decision:** One row represents one request to finance one checkout basket at one decision timestamp.

### Context

The pitch refers broadly to BNPL and microloans. Modeling cannot begin until the unit is stable.

### Options considered

1. Customer-level score.
2. Loan/facility-level score.
3. Application/checkout-level score.

### Recommendation

Use the application/checkout request because the system is called at checkout and the requested amount, basket and time are relevant to risk.

### Consequences

- Every modality joins on `application_id`.
- Repeat customers require grouped/time-aware validation.
- The output applies to the requested financing event, not the person's permanent creditworthiness.

### Does not decide

- Product eligibility.
- Final approval threshold.
- Whether one model covers multiple product terms.

## D-002 - Proposed primary outcome

**Status:** Proposed  
**Owner:** Tabular Track Tech Lead and project co-leads  
**Decision:** Use probability of 30+ days past due within 90 days of first contractual due date, or earlier charge-off.

### Rationale

The proposal is more meaningful than any late payment and fits a short-term credit prototype.

### Conditions

- Data must include sufficient due-date, payment and maturity information.
- If the data supports another outcome, create a new version rather than silently changing the target.
- Declined and censored applications are not marked good.

## D-003 - Point-in-time feature boundary

**Status:** Proposed  
**Owner:** Tabular Track Tech Lead  
**Decision:** A feature is eligible only when its availability timestamp is at or before the application decision timestamp.

### Consequences

- Feature records require event and/or availability timestamps.
- Later repayment, collection, manual-review and policy fields are prohibited.
- Rolling aggregates need boundary tests.
- Stale mandatory sources return quality/referral states.

## D-004 - Two-lane data strategy

**Status:** Proposed  
**Owner:** Tabular lead and project co-leads  
**Decision:** Use a real permitted credit dataset for model evidence and a separate BNPL-shaped synthetic dataset for systems/fusion/demo work, unless representative partner data is approved.

### Rationale

Synthetic data can validate engineering behaviour but cannot establish real credit performance or fairness. Public credit data may support model methodology but not match the intended BNPL population or text schema.

### Consequences

- Every result is labelled by lane and data version.
- No accuracy/fairness claim comes from the synthetic lane.
- Unrelated structured and text datasets are not joined and described as real multimodal evidence.

## D-005 - Shared cross-track manifest

**Status:** Proposed  
**Owner:** Tabular and NLP Track Leads  
**Decision:** Both tracks consume one versioned eligibility/label/split manifest.

### Minimum fields

- `application_id`
- `customer_id_hash`
- `decision_timestamp`
- `label_version`
- `split_name`
- `split_version`
- `eligibility_flag`
- `exclusion_reason`

### Consequences

- No independent random split per track.
- Fusion compares predictions for the same rows.
- Test access is governed centrally.

## D-006 - Tabular output semantics

**Status:** Proposed  
**Owner:** Tabular and Fusion/API Leads  
**Decision:** The tabular component exports raw margin, uncalibrated PD, calibrated PD, model/schema versions, quality flags, reason candidates and processing time.

### Rationale

The fusion team needs explicit score semantics. Calibration and raw decision functions serve different purposes and must not be confused.

### Consequences

- Output is typed and versioned.
- Calibrator is part of the compatible model bundle.
- Missing/invalid inputs are represented explicitly.

## D-007 - Separate score and policy

**Status:** Proposed  
**Owner:** Track leads  
**Decision:** Tabular/text models estimate risk; fusion combines evidence; the policy layer owns approve/decline/refer thresholds and fallback.

### Rationale

Separating these concerns makes calibration, thresholds, business assumptions and rollbacks auditable.

### Consequences

- Threshold changes do not require pretending the model changed.
- Model comparisons use score/probability quality as well as decisions.
- The policy version is recorded with every demo response.

## D-008 - Logistic regression is a real champion candidate

**Status:** Proposed  
**Owner:** Tabular Track Tech Lead  
**Decision:** Regularized logistic regression/scorecard remains eligible to win if XGBoost does not demonstrate material, stable and defensible improvement.

### Rationale

A simpler model may offer better calibration, explanation stability, operational simplicity and reproducibility.

### Consequences

- Logistic receives the same split, metrics, calibration and confidence intervals.
- XGBoost is not selected solely because it is named in the pitch.

## D-009 - SHAP claim boundary

**Status:** Proposed  
**Owner:** Project co-leads  
**Decision:** SHAP is used for global/local contribution diagnostics and reason-candidate testing. It is not described as proof of causality, fairness or regulatory compliance.

### Consequences

- Reason candidates require direction, perturbation and stability tests.
- Final materials use "explanation prototype" or "model diagnostics."
- Real deployment requires qualified jurisdiction-specific review.

## D-010 - Latency SLO definition

**Status:** Proposed  
**Owner:** Fusion/API Lead  
**Decision:** Treat the deck's under-100 ms goal as a warmed end-to-end p95 SLO on frozen hardware and concurrency; propose a warmed tabular p95 budget of 10 ms for feature transformation, prediction and calibration.

### Conditions to freeze

- Hardware/container.
- Payload/text length.
- Concurrency and worker count.
- Warm-up and sample size.
- Inclusion of validation, feature generation, explanation, fusion, policy and serialization.
- Cold-start reporting.

## D-011 - LendingClub evidence source

**Status:** Accepted with conditions
**Owner:** Project co-leads
**Decision date:** 10 September 2026

Use the Kaggle `adarshsng/lending-club-loan-data-csv` archive whose SHA-256 is
`c6255f6a8099b25303360976597fd8f86ae9087176598692a5a37dd8dc3339a1`.
The source is a historical US accepted-loan proxy and is not representative of
Singapore BNPL applicants. The raw file is not redistributed through Git.

The evidence target is ultimate `Charged Off`/`Default` versus `Fully Paid`; it is
not the intended 30+ DPD/90-day product outcome.

## D-012 - Real-text cohort and split v1

**Status:** Accepted
**Owner:** Tabular and NLP Track Leads
**Decision date:** 10 September 2026

Use rows whose raw `loan_status` is exactly `Fully Paid` or `Charged Off`, whose
raw `desc` field contains at least one non-whitespace character, and whose parsed
issue month is June 2007 through March 2014 inclusive. Read raw strings with
`keep_default_na=False`; this is required to preserve the original cohort rule.
Create stable IDs from locked one-based CSV data-row numbers because original IDs
are blank. Split by issue month: train through July 2013, validation
August-December 2013, and test January-March 2014.

The resulting 122,999 loans and derived ID lists are canonical. Neither track may
regenerate them independently or alter them after reviewing test performance.
The exact contract and read-only reproduction procedure are frozen by D-015.

## D-013 - Synthetic-description experiment

**Status:** Accepted with conditions
**Owner:** Project co-leads
**Decision date:** 10 September 2026

Generate 3,000 descriptions using `gpt-5.6-sol` at `xhigh` reasoning from coarse,
application-time fields for deterministically selected training loans. Do not send
the target, outcome, original description, employer, location, protected attributes
or post-origination variables. Do not generate labels.

Run and review 50 rows before scaling. Synthetic text stays in training, remains
grouped with its source loan in cross-validation, never duplicates tabular training
rows and is evaluated as an ablation against real-only training.

## D-014 - Cross-track output contract

**Status:** Accepted
**Owner:** Tabular, NLP and fusion leads
**Decision date:** 10 September 2026

The tabular and NLP tracks export versioned probabilities keyed by locked `loan_id`.
Training predictions consumed by fusion are out-of-fold. Fusion v1 begins with
logistic stacking over base-model probabilities and is evaluated on identical real
validation/test IDs.

## D-015 - Executable v1 cohort definition

**Status:** Accepted
**Owner:** Tabular and NLP Track Leads
**Decision date:** 19 September 2026

### Context

D-012 recorded the cohort at a high level, but the repository did not contain the
exact source parsing rule or executable generator. The source also existed in two
ZIP packages with different archive hashes.

### Decision

Freeze `configs/cohort_v1.toml` and `docs/cohort-definition.md` as the machine- and
human-readable definitions. Use `scripts/reproduce_locked_cohort.py` as a read-only
audit against `data/split_manifest.csv`; it must never rewrite the split files.

The raw CSV SHA-256
`23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c`
is the content-level source identity. Both observed archives contain this exact
`loan.csv`, despite their different ZIP hashes.

### Evidence

The approved implementation scanned 2,260,668 source rows and reproduced all
122,999 canonical rows: 86,293 train, 21,784 validation and 14,922 test. It found
zero missing IDs, zero extra IDs and zero mismatches in source row, split, issue
month, target, text-availability, cohort or dataset-version fields. The evidence is
stored in `reports/data/cohort_reproduction_v1.json`.

### Consequences

- Any rule change requires a new cohort and split version.
- Model-development loaders remain limited to train and validation.
- The audit may mechanically compare locked test membership and labels, but test
  features, outcomes and performance remain unavailable for model selection.
- Raw source data remains outside Git.

### Revisit trigger

A material cohort defect, source-file change or approved replacement dataset is
demonstrated.

## 5. How to add a decision

Use this structure:

```text
## D-XXX - Short title

Status:
Owner:
Decision date/gate:

Context:
Options considered:
Decision:
Evidence:
Consequences and trade-offs:
What this does not decide:
Revisit trigger:
```

Do not delete superseded records. Mark them superseded and link the replacement decision.


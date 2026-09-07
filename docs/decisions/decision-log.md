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

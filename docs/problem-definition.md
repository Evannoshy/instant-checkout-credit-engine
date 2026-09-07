# Tabular Credit Problem Definition

**Document status:** Proposed for G0 review  
**Version:** 0.1  
**Date:** 7 September 2026  
**Accountable owner:** Tabular Track Tech Lead  
**Primary author:** Tabular Analyst  
**Reviewers:** Project co-leads, NLP Track Lead and Fusion/API Lead  
**Target freeze date:** 11 September 2026

## 1. Purpose

This document defines the prediction problem for the structured-data component of the Instant Checkout Credit Engine. It exists to keep data collection, labels, features, model training, evaluation, fusion and product claims aligned.

The definitions below are proposals until the G0 review. Items marked as open questions must be resolved or converted into explicit, dated conditions before model training begins.

## 2. Business use case

The prototype assesses the repayment risk associated with a request to finance an e-commerce checkout basket through a short-term BNPL or microloan product. The model is called immediately before the credit decision, after any required identity and fraud checks.

The tabular component estimates repayment risk from structured information available at that moment. It does not determine legal eligibility, conduct fraud detection or make the final credit decision by itself.

### Intended consumer of the output

The immediate consumer is the late-fusion component. The fusion component combines the tabular signal with a text-model signal. A separate policy layer converts the fused result and quality flags into a prototype response.

### Intended project use

- Research comparison of a conventional logistic model and XGBoost.
- Demonstration of aligned tabular and text scoring.
- Demonstration of calibration, reason candidates, failure handling and low-latency serving.
- Documentation of the controls that would be required before any real use.

### Out of scope

- Real customer lending decisions.
- A claim of legal or regulatory compliance.
- A guarantee of affordability or financial inclusion.
- Fraud or identity-risk modeling unless separately defined.
- Causal conclusions about why an applicant defaults.

## 3. Unit of prediction

One row represents **one request to finance one checkout basket at one decision timestamp**.

Required entity fields:

| Field | Definition | Requirement |
|---|---|---|
| `application_id` | Immutable identifier for the financing request | Required and unique |
| `customer_id` | Pseudonymous identifier for repeat-customer grouping | Required where legally and technically available |
| `merchant_id` | Pseudonymous merchant identifier | Required if merchant features are used |
| `decision_timestamp` | Time at which the score is requested and the feature window closes | Required |
| `label_version` | Version of the outcome definition | Required in analytical data |
| `split_version` | Version of the train/validation/calibration/test assignment | Required in analytical data |

The same `application_id` and label definition must be used by the tabular, NLP and fusion tracks.

## 4. Intended population

The intended prototype population is applicants who request short-term unsecured financing during checkout and satisfy the product's non-model eligibility rules.

The following must be confirmed at G0 or G1:

- Whether the primary product is BNPL, a microloan, or a narrowly defined common prototype.
- Amount and repayment-term ranges.
- Geographic and age eligibility.
- Whether minimum account or transaction history is required.
- Whether repeat applications are permitted within a given period.

Any evidence dataset may represent a narrower or different population. Results must be described for the observed dataset population, not automatically generalized to Singapore, NUS, thin-file or BNPL users.

## 5. Decision point and data cut-off

The prediction is made at `decision_timestamp`, immediately before the credit approval/policy response.

Only information whose **availability timestamp** is at or before `decision_timestamp` may enter the model. Event time alone is insufficient if the data would not yet have reached the decision system.

### Permitted information categories, subject to data review

- Current application fields supplied before decision.
- Basket amount, requested financing amount, term and down payment.
- Merchant or channel attributes known before decision.
- Verified account/customer attributes already available.
- Historical structured transactions ending before the decision cut-off.
- Prior internal repayment outcomes that were already known before the new decision.
- Visible obligations known before decision.
- Data-quality, freshness and history-length indicators.

### Forbidden information

- The outcome of the current application.
- Payments, delinquency, collections or charge-off events occurring after decision.
- Manual-review results recorded after the automated score.
- Final approval/decline codes that embed later policy or human decisions.
- Future-refreshed balances, credit information or transaction aggregates.
- Any source without documented permission or a defensible relationship to credit risk.

### Staleness

The data contract must define acceptable freshness by source. If mandatory information is stale or missing, the system should return an explicit quality flag and, where warranted, `INSUFFICIENT_DATA` or `REFER` rather than silently substitute a neutral score.

## 6. Proposed primary outcome

The current proposal is:

> `bad_30dpd_90d = 1` when an originated obligation reaches at least 30 days past due within 90 days of the first contractual due date, or is charged off earlier. It is `0` only after the full 90-day performance window has matured without either event.

### Rationale

- The event is more meaningful than a single late payment.
- The horizon is short enough to suit a BNPL/microloan prototype.
- It allows a binary probability-of-bad outcome while preserving separate early-warning outcomes.

### Conditions

- The evidence dataset must support due dates, payment/delinquency status and adequate follow-up.
- If the dataset supports a materially different outcome, the team must approve a new label version and update every dependent artifact.
- Model results from different label versions must not be compared without clear reconciliation.

## 7. Secondary analytical outcomes

Keep these separate from the primary target:

| Outcome | Example definition | Intended use |
|---|---|---|
| `first_payment_default` | First required payment exceeds an agreed delinquency threshold | Early-risk sensitivity analysis |
| `ever_7dpd_30d` | At least 7 days past due in the first 30 days | Early-warning sensitivity analysis |
| `loss_amount` | Realized or proxy monetary loss | Expected-loss/value analysis |
| `exposure_at_default` | Outstanding financed amount at default | Loss normalization where available |

The primary model must not switch to a secondary label because it produces a more attractive score.

## 8. Observation and performance windows

### Observation windows

Candidate historical aggregate windows are 1, 7, 30, 90 and 180 days before `decision_timestamp`, subject to actual coverage. Every aggregation must end at or before the data-availability cut-off.

Examples:

- Seven-day spend amount.
- Thirty-day verified inflow amount.
- Ninety-day income regularity.
- Days since the last known overdue internal payment.

### Performance window

The proposed primary performance window ends 90 days after the first contractual due date.

### Label maturity and censoring

- An application is label-eligible only after its full performance window is observable, unless the bad event occurs earlier.
- Applications without sufficient follow-up are censored/excluded from primary supervised evaluation.
- The dataset cut-off date and all maturity calculations must be versioned.
- Censored rows must not be assumed good.

## 9. Eligibility and exclusions

### Candidate eligibility rules

- Valid, unique application and decision timestamp.
- Product falls inside the agreed financing definition.
- Required non-model product eligibility is satisfied.
- Data use is permitted for the project.
- Outcome is mature for supervised training/evaluation.

### Exclusions

- Test, employee or internal QA applications.
- Duplicate/corrupt application records.
- Applications outside the defined product/population.
- Rows with impossible or inconsistent timestamps.
- Known synthetic rows in the real-data evidence analysis.
- Cases whose outcome cannot be determined under the approved label.
- Fraudulent identity cases if fraud is managed by a separate process.

Every exclusion must have a coded reason and a row-count reconciliation.

## 10. Declined applications and selection bias

Repayment performance is normally observed only for originated/approved applications. A declined application must not be labelled good simply because it did not default.

Consequences:

- Primary supervised evaluation describes the historically originated population represented in the evidence data.
- Selection/reject-inference limitations must appear in the data sheet, validation report and model card.
- The initial prototype will not perform unvalidated reject inference.
- Approval-rate simulations outside the observed support must be labelled illustrative.

## 11. Tabular model output

The model output is a risk estimate, not a final decision.

Proposed `TabularRiskOutputV1`:

```json
{
  "application_id": "example-application",
  "model_version": "tabular-0.1.0",
  "schema_version": "tabular-input-1.0",
  "raw_margin": 0.0,
  "pd_raw": 0.0,
  "pd_calibrated": 0.0,
  "quality_flags": [],
  "reason_candidates": [],
  "processing_ms": 0.0
}
```

Definitions:

- `raw_margin`: model output before probability transformation/calibration.
- `pd_raw`: uncalibrated predicted probability of the approved bad outcome.
- `pd_calibrated`: post-hoc calibrated probability.
- `quality_flags`: missing, stale, out-of-range or out-of-distribution warnings.
- `reason_candidates`: reviewed model-contribution diagnostics, not a legal notice.

The model, preprocessing, calibrator, schema and reason map must be version-compatible.

## 12. Policy outputs

A separately versioned policy may return:

- `APPROVE`
- `DECLINE`
- `REFER`
- `INSUFFICIENT_DATA`

The threshold and any refer band must be justified using calibration, approval/bad-rate curves and transparent loss/value scenarios. Accuracy or F1 alone is insufficient.

## 13. Baselines and candidate model

The minimum comparison is:

1. Prevalence/constant predictor.
2. Simple product rule, if one exists.
3. Regularized logistic regression or scorecard.
4. Conservative XGBoost baseline.
5. Tuned XGBoost under a bounded, pre-declared search.

Logistic regression remains a valid champion if XGBoost does not produce material, stable and defensible improvement after calibration and complexity costs.

## 14. Evaluation design

### Preferred split

Use time-respecting train, model-validation, calibration and out-of-time test sets. Where customers repeat, group and order assignments to avoid identity/history leakage. Freeze the manifest and share it with the NLP track.

### Required metrics

- AUROC and PR-AUC with confidence intervals.
- Log loss and Brier score.
- Calibration slope/intercept and reliability plots.
- Approval rate and bad rate among approved.
- Error rates under explicit threshold definitions.
- Illustrative expected loss/value across scenarios.
- p50/p95/p99 latency and error rate under a frozen benchmark.

### Required robustness views

- Later time periods.
- Thin/short history versus established history.
- Missingness and stale-data scenarios.
- Unknown categories and boundary values.
- Feature-family ablations.
- Supported subgroup metrics with sample/event counts and uncertainty.

No arbitrary model-performance threshold will be frozen before the data profile is known. Selection criteria must be agreed before the final test is unlocked.

## 15. Error costs and consumer harms

### False approval

Potential consequences include borrower overextension, lender loss and repeated debt cycling.

### False decline

Potential consequences include exclusion of a creditworthy applicant, lost merchant conversion and uneven access across groups.

### Other harms

- Decisions based on stale or incomplete liabilities.
- Privacy-intrusive or indefensible alternative features.
- Unstable or misleading reasons.
- Overconfidence from poorly calibrated probabilities.
- Hidden subgroup errors in aggregate metrics.

The project must show trade-offs rather than optimizing a single error rate.

## 16. Data requirements

The ideal evidence dataset contains:

- Application-level identifiers and decision timestamps.
- Product terms and exposure.
- Structured application and historical cash-flow/behaviour fields with timestamps.
- Repayment schedule and matured outcomes.
- Repeat-customer identifier for grouped validation.
- Aligned transaction text if used for real multimodal evaluation.
- Permitted audit attributes or defensible segmentation fields for subgroup analysis.
- Documentation and lawful/permitted educational use.

If no one dataset supports both structured and text modalities, the team must not join unrelated sources and claim a real multimodal result.

## 17. Worked examples

### Example A - eligible and matured

- Decision: 1 January 2026 12:00.
- Latest allowed feature availability: at or before 1 January 2026 12:00.
- First due date: 15 January 2026.
- Performance window end: 15 April 2026.
- Dataset cut-off: after 15 April 2026.
- Result: eligible for supervised analysis and labelled from observed repayment history.

### Example B - not yet mature

- Decision: 1 August 2026.
- First due date: 15 August 2026.
- Performance window end: 13 November 2026.
- Dataset cut-off: 30 September 2026.
- Result: censored/excluded from the primary supervised set unless a bad event has already occurred under the approved rule. It must not be marked good.

### Example C - declined application

- Decision: 1 March 2026.
- Historical policy declined the request.
- No repayment schedule or observed performance exists.
- Result: no primary repayment label. The row may support volume/selection analysis but not a supervised good/bad target without a separately validated method.

### Example D - stale required data

- Decision: 1 June 2026.
- Latest cash-flow snapshot: 1 February 2026.
- Approved freshness rule: no more than 30 days old.
- Result: return a stale-data quality flag and follow the approved `REFER` or `INSUFFICIENT_DATA` policy.

### Example E - forbidden leaked feature

- Decision: 1 January 2026.
- Dataset contains `collections_status` updated 1 March 2026.
- Result: the field is excluded from features even if it is highly predictive, because it was unavailable at decision time and reflects the outcome process.

## 18. Open questions for G0/G1

| ID | Question | Decision owner | Required by |
|---|---|---|---|
| Q-001 | Is the primary product BNPL, microloan or one narrowly defined common product? | Project co-leads | G0, 11 Sep |
| Q-002 | Does the proposed 30+ DPD/90-day target match the data and product? | Tabular lead + co-leads | G0/G1 |
| Q-003 | What financing amount and term ranges define the intended population? | Project co-leads | G0 |
| Q-004 | What real labelled datasets or partner data are available and permitted? | Project co-leads | G1, 18 Sep |
| Q-005 | Are structured and text fields aligned to the same applications? | Tabular + NLP leads | G1 |
| Q-006 | What source freshness makes an application insufficient for automated scoring? | Tabular + API leads | Before schema freeze |
| Q-007 | Which subgroup attributes are available and permitted for audit-only evaluation? | Co-leads + data owner | Before fairness plan |
| Q-008 | Who owns the final policy threshold and refer band? | Project co-leads | Before calibration work |

## 19. Claim boundary

Until representative evidence is available, approved language is:

> The project is a research prototype investigating whether structured cash-flow and behavioural data, combined with transaction text, can improve credit-risk inference under an instant-checkout latency target.

Do not claim:

- Proven financial inclusion.
- Production readiness.
- Regulatory compliance.
- Real-world fairness based on synthetic data.
- Guaranteed sub-100 ms performance before the benchmark.
- Causal explanations from SHAP.

## 20. Change control

After G0, any change to the unit, population, target, horizon, feature cut-off, exclusions or label maturity requires:

1. A decision-log entry with reason and owner.
2. A new label/problem version where relevant.
3. Regenerated labels and split checks.
4. Re-run baselines, calibration and affected evaluations.
5. Notification to NLP and fusion/API leads.
6. Updated model/data documentation.

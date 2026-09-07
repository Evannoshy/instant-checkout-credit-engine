# Risks and Assumptions Register

**Project:** Instant Checkout Credit Engine  
**Document owner:** Tabular Track Tech Lead  
**Created:** 7 September 2026  
**Review cadence:** Weekly; critical changes reviewed immediately

## 1. Purpose

This register makes uncertainty explicit and actionable. A risk is a possible event that could harm the project. An assumption is an unverified statement currently used for planning. Each item has an owner, a validation trigger and a contingency.

## 2. Rating scale

### Probability

- **Low:** Unlikely during the project.
- **Medium:** Plausible and requires active monitoring.
- **High:** Expected unless action is taken.

### Impact

- **Low:** Minor rework; no milestone impact.
- **Medium:** A sprint or secondary deliverable may slip.
- **High:** A major model/integration milestone or claim is affected.
- **Critical:** The project becomes invalid, unsafe or unreleasable under its intended claim.

## 3. Top risks

| ID | Risk | Probability | Impact | Early trigger | Mitigation | Contingency | Owner | Review date |
|---|---|---:|---:|---|---|---|---|---|
| R-001 | No representative labelled BNPL/credit data is available | High | Critical | No viable source/permission by 14 Sep | Run data-source assessment before modeling; pursue partner and public routes | Use real proxy evidence + synthetic systems data; narrow all claims | Project co-leads | 18 Sep |
| R-002 | Primary outcome is unavailable, ambiguous or not mature | Medium | Critical | Candidate data lacks due dates/DPD/follow-up | Freeze label proposal and maturity rules; audit data before training | Version a feasible proxy outcome and restate scope; do not compare incompatible labels | Tabular lead | 18 Sep |
| R-003 | Temporal or outcome leakage inflates results | Medium | Critical | Suspiciously high metrics or post-decision fields | Availability-time registry, point-in-time tests, shuffled-label and suspicious-feature checks | Remove leaked fields, invalidate and rerun all affected experiments | Tabular analyst | Weekly |
| R-004 | Structured and text records do not align at application level | High | High | Different IDs, labels, populations or dates | Shared manifest and source matrix; verify joins before track development | Restrict real evaluation to separate tracks; use aligned synthetic data for system demo only | Tabular + NLP leads | 18 Sep |
| R-005 | Declined applications create selection/reject bias | High | High | Outcomes exist only for originated applications | Define population honestly and keep declines unlabeled | Limit claims to observed originated population; no unvalidated reject inference | Tabular lead | Data review |
| R-006 | Hidden BNPL liabilities make affordability/risk incomplete | High | High | External concurrent obligations are unavailable | Create visibility/freshness quality flags; document data boundary | Use refer/limitation; do not claim complete affordability assessment | Project co-leads | G0/G1 |
| R-007 | Too few bad events produce unstable metrics/calibration | Medium | High | Low event count by split/group | Count events before choosing complexity/splits; use CIs and simpler models | Reduce features/complexity, aggregate analyses and state insufficient evidence | Tabular analyst | 18 Sep |
| R-008 | XGBoost does not materially outperform logistic regression | Medium | Medium | Paired intervals overlap or lift is unstable | Treat logistic as a serious baseline; predefine practical selection criteria | Release logistic champion and present null result honestly | Tabular lead | 9 Oct |
| R-009 | Probability scores are poorly calibrated | Medium | High | Reliability plot/Brier/log loss are poor | Reserve calibration set; compare Platt and justified isotonic calibration | Use better-calibrated simpler model or narrow policy claims | Tabular analyst | 9 Oct |
| R-010 | Subgroup evidence is missing or statistically unstable | Medium | High | Small group/event counts or wide intervals | Plan audit-only attributes and sample thresholds early | Suppress/aggregate unstable results and state no fairness conclusion | Tabular lead | 16 Oct |
| R-011 | SHAP-based reason candidates are unstable or misleading | Medium | High | Direction/perturbation/refit tests fail | Controlled reason taxonomy and stability tests | Remove local reasons, simplify features/model or return diagnostic-only output | Tabular lead | 23 Oct |
| R-012 | Full system misses the under-100 ms target | Medium | High | Early p95 profile exceeds component budget | Freeze SLO, profile components, load models once and optimize measured bottlenecks | Relax/document SLO, use tabular-only fallback or separate explanation path | Fusion/API lead | 13 Nov |
| R-013 | Model, transformer, calibrator or schema versions mismatch | Low | High | Contract/golden fixture fails | One bundle manifest, checksums and strict version validation | Fail closed and roll back the compatible bundle | Tabular + API leads | 30 Oct onward |
| R-014 | Public repository receives PII, secrets or restricted data | Low | Critical | Sensitive file appears in a PR/history | Data policy, ignore rules, PR review, secret/PII scan | Stop release, restrict access, remove exposed material through controlled incident response | Project co-leads | Every PR |
| R-015 | Team capacity drops during academic deadlines | Medium | High | Two missed outcomes or unresolved blockers over 24 hours | Weekly capacity check and P0/P1/P2 scope | Cut stretch work, reassign support and preserve P0 evidence | Tabular lead | Weekly |
| R-016 | Late model/schema changes break fusion | Medium | High | Contract changes after 30 Oct freeze | Semantic versions, G5 freeze and contract tests | Keep last compatible package and defer non-critical changes | Tabular + fusion leads | 30 Oct onward |
| R-017 | Independent reproduction fails | Medium | High | Reviewer cannot set up or match metrics | Lock environment, one-command workflow, fixtures and manifests | Block release until P0 reproduction defects are fixed | Tabular lead | 20 Nov |
| R-018 | Final pitch overstates production readiness, inclusion or compliance | Medium | High | Unsupported wording returns to deck/demo | Evidence-to-claim review at every gate | Replace with prototype language and explicit limitations | Project co-leads | Weekly/final |

## 4. Planning assumptions

| ID | Assumption | Why currently needed | Validation/action | Owner | Validate by | If false |
|---|---|---|---|---|---|---|
| A-001 | One checkout financing application is the correct prediction unit | Aligns with instant-checkout invocation | Approve at G0 using worked examples | Tabular lead | 11 Sep | Rewrite problem, contracts and examples before modeling |
| A-002 | The proposed 30+ DPD within 90 days outcome is observable or can be mapped to a justified proxy | Needed to define labels and metrics | Inspect candidate data and product needs | Tabular lead + co-leads | 18 Sep | Version a feasible outcome and narrow the project claim |
| A-003 | The tabular analyst can commit 10-12 focused hours weekly | Used to plan November scope | Confirm at kickoff and weekly | Tabular lead | 7 Sep | Cut P2 and some P1 tasks; retain P0 |
| A-004 | The public repository remains the primary collaboration location | Drives data/security controls | Confirm with co-leads | Project co-leads | 7 Sep | Document private storage and public release workflow |
| A-005 | A legally/permissibly usable real credit dataset can be obtained | Needed for credible model evidence | Complete data-source matrix | Project co-leads | 18 Sep | Use proxy if possible; otherwise deliver engineering prototype without accuracy claims |
| A-006 | Aligned transaction text is available for the same application rows | Needed for real late-fusion evidence | Verify IDs, timestamps and labels | NLP lead | 18 Sep | Fusion evidence limited to synthetic/aligned demo data |
| A-007 | Audit groups may be available under permitted use | Needed for subgroup/fairness analysis | Review data dictionary and permissions | Project co-leads | Before Week 6 | Report only supported proxy segments; no broad fairness claim |
| A-008 | Target hardware supports a warmed p95 100 ms full-path goal | Drives architecture and component budgets | Freeze hardware and benchmark in November | Fusion/API lead | 10 Sep preliminary; 13 Nov final | Report measured SLO or redesign path honestly |
| A-009 | The fusion component can accept both raw and calibrated tabular outputs | Supports fair fusion experiments | Confirm score semantics Thursday | Fusion lead | 10 Sep | Revise typed contract and document conversion |
| A-010 | Explanations may be separated from the critical decision path if needed | Helps meet latency target | Product/API decision | Project co-leads + API lead | 10 Sep | Include explanation in the full SLO and simplify explanation work |
| A-011 | Logistic regression is acceptable as the final champion | Required for honest model selection | Approve at G0 | Tabular lead | 11 Sep | Document architecture-driven XGBoost constraint and associated trade-off |
| A-012 | 30 November is the final release/demo date | Drives all milestone dates | Confirm with project co-leads | Project co-leads | 7 Sep | Rebaseline the milestone plan immediately |

## 5. Week 1 issue log

| ID | Issue | Blocking? | Owner | Action | Due | Status |
|---|---|---:|---|---|---|---|
| I-001 | Tabular analyst has not been named in repository documents | Yes | Tabular lead | Confirm analyst and update owners | 7 Sep | Open |
| I-002 | Primary product definition is ambiguous between BNPL and microloans | Yes | Project co-leads | Choose one primary prototype or define shared scope | 11 Sep | Open |
| I-003 | No dataset is documented | Yes for modeling | Project co-leads + analyst | Create Week 2 source assessment | 18 Sep | Open |
| I-004 | Outcome definition is proposed but not approved | Yes | Tabular lead + co-leads | Review `bad_30dpd_90d` proposal | 11 Sep | Open |
| I-005 | Cross-track ID/label/split contract is not approved | Yes | Tabular + NLP leads | Hold contract review | 10 Sep | Open |
| I-006 | Full-system SLO lacks hardware/concurrency boundary | No for G0 if conditional | Fusion/API lead | Draft provisional benchmark contract | 10 Sep | Open |
| I-007 | Final policy owner is not confirmed | No for Week 1 modeling | Project co-leads | Assign policy owner | Before calibration work | Open |

## 6. Escalation rules

- Escalate any critical data-rights, privacy or leakage concern immediately.
- Escalate a blocker that remains unresolved for 24 hours to the Tabular Track Tech Lead.
- Escalate a cross-track schema/label/split conflict to both track leads and project co-leads.
- Do not bypass an unresolved blocking issue by changing the notebook or target privately.
- Record the resulting decision in the decision log.

## 7. Scope reduction order

If capacity falls, reduce work in this order:

1. Advanced dashboards and presentation polish.
2. Alternative model families and large searches.
3. Advanced uncertainty/constraint methods.
4. Online SHAP for every request.
5. Non-essential feature families.

Do not cut:

- Problem/label definition.
- Data rights and provenance.
- Leakage controls.
- Logistic baseline and same-split comparison.
- Time-aware evaluation and calibration.
- Core robustness/subgroup evidence.
- Versioned handoff and contract tests.
- Honest limitations and reproducibility.

## 8. Weekly review procedure

At the Friday review:

1. Re-rate every open high/critical risk.
2. Close only with linked evidence.
3. Convert invalidated assumptions into decisions, risks or changed scope.
4. Add new integration/model risks from the sprint.
5. Assign an owner/date to every mitigation.
6. Report the top three risks in the weekly team update.

The register is a control artifact, not a list created once for presentation.

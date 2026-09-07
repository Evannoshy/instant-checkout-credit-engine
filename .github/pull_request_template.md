## Summary

What changed and why?

## Linked issue

Closes #

## Change type

- [ ] Documentation/decision
- [ ] Data/schema
- [ ] Feature pipeline
- [ ] Model/training
- [ ] Evaluation/explainability
- [ ] API/fusion/integration
- [ ] Infrastructure/CI

## Versions affected

- Data source/version:
- Label version:
- Split version:
- Feature/schema version:
- Model/policy version:
- Experiment ID:

## Evidence

Provide test output, experiment/report links, before/after behaviour and sample/event counts where relevant.

## Risk checks

- [ ] No feature uses information unavailable at decision time.
- [ ] No label or outcome information enters the feature path.
- [ ] The shared application/label/split contract remains valid.
- [ ] Final-test data was not used for tuning or calibration.
- [ ] No PII, secrets, raw/restricted data or large generated artifacts are included.
- [ ] Claim language matches the evidence and prototype limitations.
- [ ] SHAP or local explanations are not presented as causal or compliance proof.

## Testing

- [ ] Tests added or updated.
- [ ] Existing relevant tests pass.
- [ ] Schema/contract compatibility checked.
- [ ] Reproducibility or golden-fixture check completed where relevant.

## Cross-track impact

Describe any impact on NLP, fusion, API, policy, latency, monitoring or release documentation. Name the downstream reviewer when a contract changes.

## Known limitations and rollback

What remains unresolved, and how can this change be reverted safely?

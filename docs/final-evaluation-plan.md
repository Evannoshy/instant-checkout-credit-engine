# Final evaluation preparation

Status: proposed checklist. The final test remains locked. This document does
not authorize reading its targets, scoring its loans or changing the loaders.

Before evaluation, Evan and Ariel should record:

1. The selected variant, base-model hashes, stacker and calibrator hashes, code
   commit, dependency versions and missing-text rule.
2. The primary metric and any probability-quality safeguards. The validation
   comparisons are already known; do not describe a new rule as predeclared
   before those comparisons.
3. Whether any demonstration threshold is needed. If so, choose it using
   development evidence and explain its assumptions before final evaluation.
4. The named operator, independent reviewer, date and approved evaluation
   procedure for the 14,922 January to March 2014 test loans.

The eventual runner should accept only the frozen release and approved test
manifest. It should verify IDs and source versions, create features without
outcome columns, score without refitting and join targets only for evaluation.
Keep the existing train/validation loaders' test refusal intact.

Report loan and default counts, ROC-AUC, average precision, Brier score, log
loss, calibration and the same frozen tabular reference. Include the confidence
interval method and its limitations. Report fallback counts and failures; do
not drop difficult loans silently.

Store the run command, environment, input and output hashes and reviewer signoff.
Final-test results must not feed another round of tuning while still being
presented as an untouched final assessment. If the result is disappointing,
report it and define a separate next experiment.

The current integration work provides a development-only local demo and parity
checks. A final-test runner and actual final evaluation are still pending the
approved gate.

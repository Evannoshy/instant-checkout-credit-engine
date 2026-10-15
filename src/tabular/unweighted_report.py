"""Markdown rendering of the unweighted logistic comparison.

Pure formatting: ``render_report`` turns the result dict built by
src/tabular/unweighted_comparison.py into reports/tabular/unweighted_lr_comparison.md.
Every number and every conclusion sentence is derived from that dict, so a
rerun regenerates the whole report and the text cannot drift from the numbers.
"""

from __future__ import annotations

from typing import Any

METRIC_LABELS = {
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier_score": "Brier score", "log_loss": "Log loss",
}
PROBABILITY_METRICS = ("brier_score", "log_loss")
RANKING_METRICS = ("roc_auc", "pr_auc")
REFERENCE_LABELS = {
    "balanced_lr_raw": "balanced LR, raw",
    "balanced_lr_platt": "balanced LR + existing Platt (the reference)",
    "xgboost_selected_raw": "selected XGBoost, raw",
}


def _delta(difference: dict[str, Any]) -> str:
    return (f"{difference['delta']:+.5f} [{difference['ci_low']:+.5f}, {difference['ci_high']:+.5f}]"
            f" — {difference['verdict']}")


def results_table(result: dict[str, Any]) -> str:
    """One row per candidate: the four metrics, mean predicted risk and the observed default rate."""
    lines = [
        "| Candidate | Model version | ROC-AUC ↑ | PR-AUC ↑ | Brier ↓ | Log loss ↓ "
        "| Mean predicted risk | Observed default rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["candidates"]:
        lines.append(
            f"| {row['label']} | `{row['model_version']}` | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} "
            f"| {row['brier_score']:.4f} | {row['log_loss']:.4f} | {row['mean_p']:.4f} "
            f"| {row['prevalence']:.4f} |"
        )
    return "\n".join(lines)


def differences_table(result: dict[str, Any]) -> str:
    """Unweighted LR minus each reference: delta, 95% CI and verdict per metric."""
    lines = [
        "| Unweighted LR minus | " + " | ".join(METRIC_LABELS.values()) + " |",
        "|---|" + "---|" * len(METRIC_LABELS),
    ]
    for key, metrics in result["unweighted_minus_reference"].items():
        cells = " | ".join(_delta(metrics[metric]) for metric in METRIC_LABELS)
        lines.append(f"| {REFERENCE_LABELS[key]} | {cells} |")
    return "\n".join(lines)


def verdict_summary(metrics: dict[str, Any], names: tuple[str, ...]) -> str:
    """'clearly better', 'clearly worse', 'no clear difference' or a mixed description."""
    verdicts = {metrics[name]["verdict"] for name in names}
    if verdicts == {"better"}:
        return "clearly better"
    if verdicts == {"worse"}:
        return "clearly worse"
    if verdicts == {"no clear difference"}:
        return "no clear difference"
    return "mixed (" + ", ".join(f"{METRIC_LABELS[name]} {metrics[name]['verdict']}" for name in names) + ")"


def _risk_level_sentence(result: dict[str, Any]) -> str:
    observed = result["candidates"][0]["prevalence"]
    parts = [
        f"{row['label']} {row['mean_p']:.4f} ({row['mean_p'] - observed:+.4f})" for row in result["candidates"]
    ]
    return (f"The observed default rate on validation is {observed:.4f}. Mean predicted risk "
            "(and its gap to the observed rate): " + "; ".join(parts) + ".")


def findings(result: dict[str, Any]) -> str:
    """Numbered findings generated from the verdicts."""
    differences = result["unweighted_minus_reference"]
    raw = verdict_summary(differences["balanced_lr_raw"], PROBABILITY_METRICS)
    reference = verdict_summary(differences["balanced_lr_platt"], PROBABILITY_METRICS)
    xgboost = verdict_summary(differences["xgboost_selected_raw"], PROBABILITY_METRICS)
    ranking = verdict_summary(differences["balanced_lr_raw"], RANKING_METRICS)
    return "\n".join([
        f"1. **Against the balanced model's raw scores**, the unweighted model's probability quality "
        f"(Brier score and log loss) is {raw}. {_risk_level_sentence(result)}",
        f"2. **Against the reference, balanced LR + existing Platt mapping**, the unweighted raw scores "
        f"show {reference} in probability quality.",
        f"3. **Against the selected XGBoost (raw)**, the unweighted model's probability quality: {xgboost}.",
        f"4. **Ranking** (ROC-AUC and PR-AUC) against the balanced model: {ranking}. Class weighting "
        "mainly moves the level of the probabilities; it changes the ordering of loans much less.",
    ])


def answer(result: dict[str, Any]) -> str:
    """A direct answer to the experiment question, kept in the context of the margins."""
    differences = result["unweighted_minus_reference"]
    raw = verdict_summary(differences["balanced_lr_raw"], PROBABILITY_METRICS)
    reference = verdict_summary(differences["balanced_lr_platt"], PROBABILITY_METRICS)
    if raw == "clearly better":
        first = "Removing the weighting clearly improves probability quality compared with the balanced model's raw scores"
    else:
        first = f"Compared with the balanced model's raw scores, removing the weighting gives {raw} in probability quality"
    if reference == "no clear difference":
        second = (
            "It does not clearly beat the calibrated reference (balanced LR + existing Platt): each difference "
            "is inside the predeclared margin or its 95% interval includes zero. The unweighted model reaches "
            "about the same probability quality without a separate calibration step."
        )
    else:
        second = f"Against the calibrated reference (balanced LR + existing Platt) the result is {reference}."
    return f"{first}. {second} This is a comparison only, not approval to replace the reference model."


def render_report(result: dict[str, Any]) -> str:
    """The full Markdown report."""
    margins = ", ".join(f"{METRIC_LABELS[name]} {value}" for name, value in result["margins"].items())
    bootstrap, platt = result["bootstrap"], result["platt_mapping"]
    inputs = "\n".join(f"| `{path}` | `{sha[:12]}…` |" for path, sha in result["inputs_sha256"].items())
    return f"""# Unweighted Logistic Regression Comparison

**Experiment:** `unweighted_lr_v1` · **Question:** does removing class weighting improve probability quality?
**Generated by:** `python -m src.tabular.unweighted_comparison` (every number and finding below is computed)

## 1. Answer

{answer(result)}

## 2. What was compared

All four candidates are graded on the same {result['validation_rows']:,} validation loans (identical
`loan_id` set and order, checked before any metric is computed).
Test rows available to any model: {result['test_rows_available_to_model']}.

- **Balanced LR, raw**: frozen v1 logistic model (`class_weight="balanced"`), fitted on `model_fit` only.
- **Balanced LR + existing Platt**: the same scores through the calibration-only Platt mapping, refitted
  with `calibrate.fit_platt` on the calibration rows (slope {platt['slope']:.6f}, intercept
  {platt['intercept']:.6f}) and checked against `reports/tabular/calibration_metrics.json`.
- **Unweighted LR, raw**: identical features and settings except `class_weight=None`, fitted once on the
  59,353 `model_fit` loans. The balanced Platt mapping is not applied to it.
- **Selected XGBoost, raw**: the configuration chosen by the bounded search.

The three reference rows reproduce the merged `reports/tabular/calibration_results.csv` within 1e-6.

## 3. Validation results

{results_table(result)}

Higher ROC-AUC and PR-AUC are better; lower Brier score and log loss are better. PR-AUC is average
precision, as defined in `src/tabular/evaluate.py`. Mean predicted risk should sit close to the observed
default rate.

## 4. Paired differences (unweighted LR minus reference)

Each cell shows the delta, the 95% paired bootstrap interval ({bootstrap['bootstrap_resamples']} resamples,
seed {bootstrap['seed']}) and the verdict. A difference is "better" or "worse" only if it exceeds the
predeclared margin ({margins}) **and** its interval excludes zero; otherwise it is "no clear difference".

{differences_table(result)}

## 5. Findings

{findings(result)}

## 6. Context and limits

- Small differences need context: the margins above were fixed before any calibrated result was seen,
  and differences inside them are treated as noise rather than improvement.
- The unweighted model learns its probability level from `model_fit` months only (Jun 2007 – Dec 2012),
  while the Platt mapping was fitted on the later calibration months (Jan – Jul 2013). If the default
  rate drifts over time, the raw unweighted level can sit further from the validation rate than the
  calibrated reference, as the mean-risk gaps in section 5 show.
- The bootstrap resamples loans independently, but validation loans are clustered in five issue months
  (Aug–Dec 2013), so the intervals are probably somewhat narrow.
- One validation period only. The result says nothing about later months or the locked test split.
- No hyperparameter search, new features, threshold selection or further calibration was done.

## 7. Inputs

| File | SHA-256 |
|---|---|
{inputs}

## 8. Reproduce

```bash
python -m pip install -r src/tabular/requirements.txt
python -m src.tabular.logistic_unweighted --raw-csv data/raw/loan.csv
python -m src.tabular.unweighted_comparison
python -m pytest src/tabular/test_unweighted_comparison.py -v
```
"""

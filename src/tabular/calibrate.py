"""Calibration of the shortlisted tabular candidates: raw, Platt and isotonic.

For each candidate in ``CANDIDATES``, read Allan's raw probabilities for the
``calibration`` role and the ``validation`` split, fit Platt scaling and
isotonic regression on the calibration rows, apply raw / Platt / isotonic to
both sets, and grade them on validation. Writes, to ``reports/tabular/``:
``calibration_results.csv`` (metrics per candidate x method x set),
``policy_curve.csv`` (approval and default rates per threshold, with counts),
``calibration_metrics.json`` (everything, plus calibrator parameters,
paired bootstrap CIs, the decision-rule outcome and provenance),
``calibration_<candidate>.png`` and ``policy_curve.png``.

Every choice is predeclared in ``configs/tabular_calibration_v1.toml``, written
before any calibrated validation result was seen: the methods, the probability
clip, the policy-curve grid, the bootstrap settings and the decision rule with
its margins. ``load_calibration_config`` checks that every setting has the
right type and range, and ``threshold_grid`` turns ``[policy] thresholds`` into
the curve grid.

Rule: calibrators are fit on the ``calibration`` role only (``fit_calibrators``
takes no validation data); validation grades and never fits; the locked test
split is never read. The raw CSV is never opened: outcomes come from the
manifest and are joined to predictions by ``loan_id``.

Inputs are checked by Allan's committed test
``test_raw_predictions.py::test_real_prediction_files_cover_every_calibration_and_validation_loan``
(coverage, split, model version, raw metrics against his manifest). This module
only re-runs ``evaluate.validate_prediction_frame`` on what it loads, so run
that test first.

Run from the repository root:
    python -m src.tabular.calibrate

Known limitations:
- Isotonic output is clipped to [probability_clip, 1 - probability_clip], so one
  wrong loan at an exact 0 or 1 cannot make log loss infinite.
- The paired bootstrap resamples loans independently; validation loans are
  clustered in time, so the CIs are probably somewhat narrow.
- Calibrators match models fit on ``model_fit`` rows only. A model refit on all
  train rows would need a new calibrator.
- Ties in a probability (isotonic steps) are broken by ``loan_id`` order in the
  threshold-free approval curve and the approval-rate table.
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.special import logit
from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.tabular import evaluate
from src.tabular.development import DEFAULT_DEVELOPMENT_CONFIG, assign_development_roles
from src.tabular.preprocess import DEFAULT_DATA_DIR, REPO_ROOT, load_feature_config
from src.tabular.xgboost_baseline import (
    HIGHER_IS_BETTER,
    count_test_rows_available,
    paired_bootstrap_delta,
    validate_runtime,
)

DEFAULT_CALIBRATION_CONFIG = REPO_ROOT / "configs" / "tabular_calibration_v1.toml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "tabular"

CANDIDATES = ("logistic_regression", "xgboost_default", "xgboost_trial1")
SETS = ("calibration", "validation")
METRICS = ("roc_auc", "pr_auc", "brier_score", "log_loss")
PROBABILITY_METRICS = ("brier_score", "log_loss")
PLOT_STRATEGIES = ("uniform", "quantile")
HISTOGRAM_BINS = 50
METHOD_COLOURS = {"raw": "#7f7f7f", "platt": "#1f77b4", "isotonic": "#ff7f0e"}
CANDIDATE_COLOURS = {
    "logistic_regression": "#1f77b4", "xgboost_default": "#2ca02c", "xgboost_trial1": "#d62728",
}

REQUIRED_KEYS = {
    "": {
        "calibration_version", "status", "development_version", "prediction_dir", "methods",
        "probability_clip", "calibration_plot_bins", "calibration_plot_strategy",
        "policy", "evaluation", "decision",
    },
    "policy": {"approve_rule", "thresholds", "illustrative_thresholds", "approval_rates"},
    "policy.thresholds": {"start", "stop", "step"},
    "evaluation": {"bootstrap_resamples", "seed"},
    "decision": {"method_metric", "incumbent", "challenger", "margins"},
}

# Every ValueError message raised by this module, as str.format templates.
ERROR_MESSAGES: dict[str, str] = {
    "missing_keys": "Calibration config [{section}] is missing keys: {missing}",
    "bad_value": "Calibration config {key} must be {requirement}; got {value!r}",
    "unknown_methods": "Unknown calibration methods {unknown}; known methods are {known}",
    "rows_not_aligned": "{first} and {second} validation rows are not the same loans in the same order",
}


def _is_number(value: Any) -> bool:
    """True for int or float, but not bool (TOML booleans are a separate type)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_increasing_probabilities(values: Any, *, include_one: bool) -> bool:
    """Non-empty, strictly increasing numbers in (0, 1), or (0, 1] if include_one."""
    return (
        isinstance(values, list)
        and len(values) > 0
        and all(_is_number(v) and (0 < v < 1 or (include_one and v == 1)) for v in values)
        and all(first < second for first, second in zip(values, values[1:]))
    )


def _require(condition: bool, key: str, requirement: str, value: Any) -> None:
    if not condition:
        raise ValueError(ERROR_MESSAGES["bad_value"].format(
            key=key, requirement=requirement, value=value,
        ))


def _require_keys(config: dict[str, Any]) -> None:
    """Every table in REQUIRED_KEYS exists and holds its keys ("" is the top level)."""
    for section, required in REQUIRED_KEYS.items():
        table = config
        for part in filter(None, section.split(".")):
            table = table.get(part, {})
        if missing := required - set(table):
            raise ValueError(ERROR_MESSAGES["missing_keys"].format(
                section=section or "top level", missing=sorted(missing),
            ))


def load_calibration_config(path: str | Path = DEFAULT_CALIBRATION_CONFIG) -> dict[str, Any]:
    """Load the calibration config, failing loudly on a missing key or a bad type or range."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    _require_keys(config)

    for key in ("calibration_version", "status", "development_version", "prediction_dir"):
        _require(isinstance(config[key], str) and config[key] != "", key, "a non-empty string",
                 config[key])
    methods = config["methods"]
    _require(isinstance(methods, list) and len(methods) > 0
             and all(isinstance(m, str) for m in methods),
             "methods", "a non-empty list of strings", methods)
    clip = config["probability_clip"]
    _require(_is_number(clip) and 0 < clip < 0.01, "probability_clip", "a number in (0, 0.01)", clip)
    bins = config["calibration_plot_bins"]
    _require(_is_integer(bins) and bins >= 2, "calibration_plot_bins", "an integer >= 2", bins)
    strategy = config["calibration_plot_strategy"]
    _require(strategy in PLOT_STRATEGIES, "calibration_plot_strategy",
             f"one of {list(PLOT_STRATEGIES)}", strategy)

    policy = config["policy"]
    _require(isinstance(policy["approve_rule"], str), "[policy] approve_rule", "a string",
             policy["approve_rule"])
    grid = policy["thresholds"]
    start, stop, step = grid["start"], grid["stop"], grid["step"]
    _require(all(_is_number(v) for v in (start, stop, step)) and 0 <= start < stop <= 1 and step > 0,
             "[policy] thresholds", "numbers with 0 <= start < stop <= 1 and step > 0", grid)
    steps = (stop - start) / step
    _require(math.isclose(steps, round(steps), abs_tol=1e-9), "[policy] thresholds",
             "a whole number of steps from start to stop", grid)
    _require(_is_increasing_probabilities(policy["illustrative_thresholds"], include_one=False),
             "[policy] illustrative_thresholds", "strictly increasing numbers in (0, 1)",
             policy["illustrative_thresholds"])
    _require(_is_increasing_probabilities(policy["approval_rates"], include_one=True),
             "[policy] approval_rates", "strictly increasing numbers in (0, 1]",
             policy["approval_rates"])

    evaluation = config["evaluation"]
    resamples = evaluation["bootstrap_resamples"]
    _require(_is_integer(resamples) and resamples >= 1, "[evaluation] bootstrap_resamples",
             "a positive integer", resamples)
    seed = evaluation["seed"]
    _require(_is_integer(seed) and seed >= 0, "[evaluation] seed", "a non-negative integer", seed)

    decision = config["decision"]
    _require(decision["method_metric"] in METRICS, "[decision] method_metric",
             f"one of {list(METRICS)}", decision["method_metric"])
    for key in ("incumbent", "challenger"):
        _require(decision[key] in CANDIDATES, f"[decision] {key}", f"one of {list(CANDIDATES)}",
                 decision[key])
    margins = decision["margins"]
    _require(set(margins) == set(METRICS), "[decision.margins]",
             f"a table with exactly the keys {sorted(METRICS)}", sorted(margins))
    for metric, value in margins.items():
        _require(_is_number(value) and value >= 0, f"[decision.margins] {metric}",
                 "a non-negative number", value)
    return config


def threshold_grid(config: dict[str, Any]) -> np.ndarray:
    """The policy-curve thresholds from [policy] thresholds, endpoints included.

    Built from a whole number of steps with ``np.linspace`` and rounded to
    12 decimals, so 0.07 is exactly 0.07 rather than an ``np.arange`` drift
    value such as 0.07000000000000001.
    """
    thresholds = config["policy"]["thresholds"]
    count = round((thresholds["stop"] - thresholds["start"]) / thresholds["step"]) + 1
    return np.round(np.linspace(thresholds["start"], thresholds["stop"], count), 12)


@dataclass(frozen=True)
class Scores:
    """One set's raw probabilities and outcomes, sorted by loan_id."""

    loan_ids: np.ndarray
    p: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """A shortlisted model's raw probabilities on the calibration and validation rows."""

    name: str
    model_name: str
    model_version: str
    sets: dict[str, Scores]


@dataclass(frozen=True)
class Calibrator:
    """A fitted score -> probability mapping and the parameters worth recording."""

    method: str
    transform: Callable[[np.ndarray], np.ndarray]
    details: dict[str, Any]


@dataclass(frozen=True)
class CalibrationRun:
    """Everything evaluate_candidates produces, before anything is written."""

    results: pd.DataFrame
    policy: pd.DataFrame
    summary: dict[str, Any]
    validation_versions: dict[str, dict[str, np.ndarray]]
    validation_y: np.ndarray


def load_candidates(
    prediction_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    development_config: str | Path = DEFAULT_DEVELOPMENT_CONFIG,
) -> dict[str, Candidate]:
    """Read each candidate's two prediction files and attach outcomes by loan_id.

    Expected IDs come from the development roles (calibration) and the manifest
    (validation), never from the files themselves. Each file is checked with
    ``evaluate.validate_prediction_frame``; the other input checks are in
    Allan's committed test. Rows are sorted by loan_id, so every candidate's
    arrays line up loan for loan.
    """
    manifest, _ = evaluate.load_manifest(data_dir)
    target = manifest.astype({"loan_id": str}).set_index("loan_id")["target"]
    roles = assign_development_roles(data_dir, development_config)
    expected_ids = {
        "calibration": set(roles.index[roles["role"].eq("calibration")]),
        "validation": set(evaluate.get_split_targets(manifest, "validation")["loan_id"].astype(str)),
    }

    candidates = {}
    for name in CANDIDATES:
        sets = {}
        for set_name in SETS:
            frame = pd.read_csv(
                Path(prediction_dir) / f"{name}_{set_name}.csv", dtype={"loan_id": str}
            )
            evaluate.validate_prediction_frame(frame, expected_ids[set_name])
            frame = frame.sort_values("loan_id", ignore_index=True)
            sets[set_name] = Scores(
                loan_ids=frame["loan_id"].to_numpy(dtype=str),
                p=frame["p_default_tabular"].to_numpy(dtype=np.float64),
                y=target.loc[frame["loan_id"]].to_numpy(dtype=np.int64),
            )
        candidates[name] = Candidate(
            name, str(frame["model_name"].iloc[0]), str(frame["model_version"].iloc[0]), sets
        )
    return candidates


def fit_raw(p: np.ndarray, y: np.ndarray, eps: float) -> Calibrator:
    """The uncalibrated baseline: probabilities pass through unchanged."""
    return Calibrator("raw", lambda values: np.asarray(values, dtype=np.float64), {})


def fit_platt(p: np.ndarray, y: np.ndarray, eps: float) -> Calibrator:
    """Platt scaling: sigmoid(slope * logit(p) + intercept), fit without a penalty.

    ``C=np.inf`` gives the unpenalised fit: the default ``C=1.0`` would shrink
    both parameters, and ``penalty=None`` is deprecated from scikit-learn 1.8.
    The logit input is clipped to [eps, 1 - eps] so logit(0) stays finite.
    """
    def logit_column(values: np.ndarray) -> np.ndarray:
        return logit(np.clip(values, eps, 1 - eps)).reshape(-1, 1)

    model = LogisticRegression(C=np.inf).fit(logit_column(p), y)
    return Calibrator(
        "platt",
        lambda values: model.predict_proba(logit_column(values))[:, 1],
        {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])},
    )


def fit_isotonic(p: np.ndarray, y: np.ndarray, eps: float) -> Calibrator:
    """Isotonic regression: a non-decreasing step function of p.

    Scores outside the calibration range take the nearest end value
    (``out_of_bounds="clip"``). The output is clipped to [eps, 1 - eps]: a step
    can be exactly 0 or 1, and one wrong loan there would make log loss infinite.
    """
    model = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(p, y)
    return Calibrator(
        "isotonic",
        lambda values: np.clip(model.predict(values), eps, 1 - eps),
        {"steps": int(np.unique(model.y_thresholds_).size)},
    )


FITTERS: dict[str, Callable[[np.ndarray, np.ndarray, float], Calibrator]] = {
    "raw": fit_raw, "platt": fit_platt, "isotonic": fit_isotonic,
}


def fit_calibrators(
    cal_p: np.ndarray, cal_y: np.ndarray, methods: list[str], eps: float
) -> dict[str, Calibrator]:
    """Fit every method on the calibration rows. Takes no validation data, by design."""
    if unknown := sorted(set(methods) - set(FITTERS)):
        raise ValueError(ERROR_MESSAGES["unknown_methods"].format(
            unknown=unknown, known=sorted(FITTERS),
        ))
    return {method: FITTERS[method](cal_p, cal_y, eps) for method in methods}


def metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Counts, mean probability against the observed rate, and the four shared metrics."""
    metrics = evaluate.evaluate_predictions(y, p)
    mean_p = float(p.mean())
    return {
        "rows": metrics["rows"],
        "positives": metrics["positives"],
        "prevalence": metrics["prevalence"],
        "mean_p": mean_p,
        "mean_p_minus_prevalence": mean_p - metrics["prevalence"],
        **{metric: metrics[metric] for metric in METRICS},
    }


def _bootstrap(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, metric: str,
    evaluation: dict[str, Any],
) -> dict[str, float]:
    return paired_bootstrap_delta(
        y, baseline, candidate, metric,
        resamples=evaluation["bootstrap_resamples"], seed=evaluation["seed"],
    )


def clearly_better(delta: dict[str, float], metric: str, margin: float) -> bool:
    """True if the candidate beats the baseline by more than margin and the 95% CI excludes zero.

    ``delta`` is metric(candidate) - metric(baseline) with its CI, as returned
    by ``paired_bootstrap_delta``.
    """
    if HIGHER_IS_BETTER[metric]:
        return delta["delta"] > margin and delta["ci_low"] > 0
    return -delta["delta"] > margin and delta["ci_high"] < 0


def compare_methods(
    y: np.ndarray, versions: dict[str, np.ndarray], metrics: list[str],
    evaluation: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """Paired deltas of every later method against every earlier one.

    Returns {"platt_minus_raw": {metric: {"delta", "ci_low", "ci_high"}}, ...}.
    """
    methods = list(versions)
    return {
        f"{later}_minus_{earlier}": {
            metric: _bootstrap(y, versions[earlier], versions[later], metric, evaluation)
            for metric in metrics
        }
        for position, earlier in enumerate(methods)
        for later in methods[position + 1:]
    }


def choose_method(
    comparisons: dict[str, dict[str, dict[str, float]]], methods: list[str], metric: str,
    margin: float,
) -> tuple[str, list[dict[str, Any]]]:
    """Decision rule 1: keep the earliest method unless a later one is clearly better on metric."""
    chosen = methods[0]
    steps = []
    for method in methods[1:]:
        delta = comparisons[f"{method}_minus_{chosen}"][metric]
        better = clearly_better(delta, metric, margin)
        steps.append({
            "baseline": chosen, "candidate": method, "metric": metric, **delta,
            "clearly_better": better,
        })
        if better:
            chosen = method
    return chosen, steps


def choose_champion(
    y: np.ndarray, incumbent_p: np.ndarray, challenger_p: np.ndarray,
    margins: dict[str, float], evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Decision rule 2: the challenger wins only if it is clearly better on log loss or
    Brier score and not worse on ROC-AUC by more than the margin."""
    deltas = {
        metric: _bootstrap(y, incumbent_p, challenger_p, metric, evaluation) for metric in METRICS
    }
    better = {metric: clearly_better(deltas[metric], metric, margins[metric]) for metric in METRICS}
    better_probabilities = any(better[metric] for metric in PROBABILITY_METRICS)
    ranking_not_worse = deltas["roc_auc"]["delta"] >= -margins["roc_auc"]
    return {
        "challenger_minus_incumbent": {
            metric: {**deltas[metric], "clearly_better": better[metric]} for metric in METRICS
        },
        "challenger_clearly_better_on_probability_quality": better_probabilities,
        "challenger_not_worse_on_roc_auc": ranking_not_worse,
        "challenger_wins": better_probabilities and ranking_not_worse,
    }


def policy_table(p: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Approve when p < threshold: approvals, defaults among them and both rates, per threshold.

    The default rate among approved is NaN where nothing is approved (a
    threshold at or below the lowest probability), since 0 / 0 has no value.
    """
    order = np.argsort(p, kind="stable")
    cumulative_defaults = np.concatenate([[0], np.cumsum(y[order])])
    approved = np.searchsorted(p[order], thresholds, side="left")
    defaults = cumulative_defaults[approved]
    default_rate = np.full(len(thresholds), np.nan)
    np.divide(defaults, approved, out=default_rate, where=approved > 0)
    return pd.DataFrame({
        "threshold": thresholds,
        "approved": approved,
        "approval_rate": approved / len(p),
        "defaults_among_approved": defaults,
        "default_rate_among_approved": default_rate,
    })


def approval_curve(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Threshold-free curve from approving the k lowest-risk loans, for k = 1..n.

    Returns (approval rate k / n, default rate among the k approved). It depends
    only on the order of p, so raw and Platt probabilities give the same curve.
    """
    order = np.argsort(p, kind="stable")
    approved = np.arange(1, len(p) + 1)
    return approved / len(p), np.cumsum(y[order]) / approved


def default_rate_at_approval(
    p: np.ndarray, y: np.ndarray, approval_rates: list[float]
) -> pd.DataFrame:
    """Default rate among the lowest-risk loans at each approval rate, with counts."""
    order = np.argsort(p, kind="stable")
    cumulative_defaults = np.cumsum(y[order])
    approved = np.maximum(np.round(np.asarray(approval_rates) * len(p)).astype(int), 1)
    defaults = cumulative_defaults[approved - 1]
    return pd.DataFrame({
        "approval_rate": approval_rates,
        "approved": approved,
        "defaults_among_approved": defaults,
        "default_rate_among_approved": defaults / approved,
    })


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-ready rows: native Python types, NaN as null."""
    return json.loads(frame.to_json(orient="records"))


def _require_aligned_validation(candidates: dict[str, Candidate]) -> None:
    """Paired comparisons need every candidate's validation rows to be the same loans in order."""
    names = list(candidates)
    first = candidates[names[0]].sets["validation"]
    for name in names[1:]:
        other = candidates[name].sets["validation"]
        if not (np.array_equal(first.loan_ids, other.loan_ids) and np.array_equal(first.y, other.y)):
            raise ValueError(ERROR_MESSAGES["rows_not_aligned"].format(first=names[0], second=name))


def evaluate_candidates(candidates: dict[str, Candidate], config: dict[str, Any]) -> CalibrationRun:
    """Fit, apply and grade every method for every candidate, then apply the decision rule."""
    methods, eps = config["methods"], config["probability_clip"]
    evaluation, decision, policy = config["evaluation"], config["decision"], config["policy"]
    method_metric = decision["method_metric"]
    compared_metrics = list(dict.fromkeys([method_metric, *PROBABILITY_METRICS]))
    thresholds = threshold_grid(config)
    _require_aligned_validation(candidates)

    result_rows, policy_frames, summaries, validation_versions = [], [], {}, {}
    for name, candidate in candidates.items():
        calibration, validation = candidate.sets["calibration"], candidate.sets["validation"]
        calibrators = fit_calibrators(calibration.p, calibration.y, methods, eps)
        versions = {
            set_name: {method: c.transform(scores.p) for method, c in calibrators.items()}
            for set_name, scores in candidate.sets.items()
        }
        for set_name, scores in candidate.sets.items():
            for method, p in versions[set_name].items():
                result_rows.append({
                    "candidate": name, "model_version": candidate.model_version,
                    "method": method, "set": set_name, **metric_row(scores.y, p),
                })

        comparisons = compare_methods(validation.y, versions["validation"], compared_metrics, evaluation)
        chosen, steps = choose_method(
            comparisons, methods, method_metric, decision["margins"][method_metric]
        )
        for method, p in versions["validation"].items():
            policy_frames.append(
                policy_table(p, validation.y, thresholds).assign(candidate=name, method=method)
            )
        chosen_p = versions["validation"][chosen]
        illustrative = np.asarray(policy["illustrative_thresholds"], dtype=np.float64)
        summaries[name] = {
            "model_name": candidate.model_name,
            "model_version": candidate.model_version,
            "rows": {
                set_name: {"rows": len(scores.y), "positives": int(scores.y.sum())}
                for set_name, scores in candidate.sets.items()
            },
            "calibrators": {method: c.details for method, c in calibrators.items() if c.details},
            "method_comparisons": comparisons,
            "method_selection": steps,
            "chosen_method": chosen,
            "policy": {
                "method": chosen,
                "at_illustrative_thresholds": _records(
                    policy_table(chosen_p, validation.y, illustrative)
                ),
                "at_approval_rates": _records(
                    default_rate_at_approval(chosen_p, validation.y, policy["approval_rates"])
                ),
            },
        }
        validation_versions[name] = versions["validation"]

    incumbent, challenger = decision["incumbent"], decision["challenger"]
    incumbent_method = summaries[incumbent]["chosen_method"]
    challenger_method = summaries[challenger]["chosen_method"]
    validation_y = candidates[incumbent].sets["validation"].y
    outcome = choose_champion(
        validation_y,
        validation_versions[incumbent][incumbent_method],
        validation_versions[challenger][challenger_method],
        decision["margins"], evaluation,
    )
    champion = {
        "incumbent": incumbent, "incumbent_method": incumbent_method,
        "challenger": challenger, "challenger_method": challenger_method,
        **outcome,
        "recommendation": challenger if outcome["challenger_wins"] else incumbent,
    }
    policy_columns = ["candidate", "method", "threshold", "approved", "approval_rate",
                      "defaults_among_approved", "default_rate_among_approved"]
    return CalibrationRun(
        results=pd.DataFrame(result_rows),
        policy=pd.concat(policy_frames, ignore_index=True)[policy_columns],
        summary={"candidates": summaries, "champion": champion},
        validation_versions=validation_versions,
        validation_y=validation_y,
    )


def plot_calibration(
    candidate: str, y: np.ndarray, versions: dict[str, np.ndarray], bins: int, strategy: str,
    path: Path,
) -> None:
    """Reliability curves for every method on one axis, with prediction histograms below."""
    fig, (curve_ax, hist_ax) = plt.subplots(
        2, 1, figsize=(7, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for position, (method, p) in enumerate(versions.items()):
        colour = METHOD_COLOURS.get(method)
        CalibrationDisplay.from_predictions(
            y, p, n_bins=bins, strategy=strategy, name=method, ref_line=position == 0,
            ax=curve_ax, color=colour, marker="o",
        )
        hist_ax.hist(p, bins=HISTOGRAM_BINS, range=(0, 1), histtype="step", color=colour,
                     label=method)
    curve_ax.set_xlabel("")
    curve_ax.set_title(
        f"{candidate}: calibration on validation\n"
        f"n={len(y):,}, defaults={int(y.sum()):,}, {bins} {strategy} bins"
    )
    hist_ax.set_xlabel("Predicted probability of default")
    hist_ax.set_ylabel("Loans")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy(curves: dict[str, dict[str, Any]], prevalence: float, path: Path) -> None:
    """Left: approval and default rates by threshold. Right: default rate by approval rate.

    ``curves`` maps candidate -> {"method", "table" (policy_table output),
    "approval" (approval_curve output)}. The right panel starts at 1% approval,
    where the default rate among a handful of loans is still noise.
    """
    fig, (threshold_ax, approval_ax) = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, curve in curves.items():
        colour = CANDIDATE_COLOURS.get(name)
        label = f"{name} ({curve['method']})"
        table = curve["table"]
        threshold_ax.plot(table["threshold"], table["approval_rate"], color=colour, label=label)
        threshold_ax.plot(table["threshold"], table["default_rate_among_approved"], color=colour,
                          linestyle="--")
        approval_rate, default_rate = curve["approval"]
        shown = approval_rate >= 0.01
        approval_ax.plot(approval_rate[shown], default_rate[shown], color=colour, label=label)

    handles, _ = threshold_ax.get_legend_handles_labels()
    threshold_ax.legend(handles=handles + [
        Line2D([], [], color="black", label="% approved"),
        Line2D([], [], color="black", linestyle="--", label="Default rate among approved"),
    ], fontsize=8)
    threshold_ax.set(
        xlabel="Threshold on predicted probability of default (approve if p < threshold)",
        ylabel="Share of loans", title="Approval and default rate by threshold",
    )
    approval_ax.axhline(prevalence, color="grey", linestyle=":",
                        label=f"Overall default rate ({prevalence:.1%})")
    approval_ax.legend(fontsize=8)
    approval_ax.set(
        xlabel="Approval rate (lowest-risk loans first)", ylabel="Default rate among approved",
        title="Default rate by approval rate (threshold-free)",
    )
    approval_ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in (threshold_ax, approval_ax):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    fig.suptitle("Validation policy curves (illustrative; not a threshold proposal)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _display_path(path: str | Path) -> str:
    """Repository-relative path where possible, for portable metadata."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def provenance(
    data_dir: str | Path, config: dict[str, Any], prediction_dir: str | Path,
    runtime: dict[str, str],
) -> dict[str, Any]:
    """Versions, row roles, the test-row count and the environment, written beside the results."""
    _, stats = evaluate.load_manifest(data_dir)
    return {
        "calibration_version": config["calibration_version"],
        "status": config["status"],
        "development_version": config["development_version"],
        "feature_set_version": load_feature_config()["feature_set_version"],
        "dataset_version": stats["dataset_version"],
        "split_version": stats.get("split_version"),
        "cohort": stats["cohort"],
        "prediction_dir": _display_path(prediction_dir),
        "calibrator_fit_rows": "calibration role only",
        "graded_rows": "validation split only",
        "test_rows_available_to_model": count_test_rows_available(data_dir),
        "environment": runtime,
    }


def write_outputs(
    run: CalibrationRun, config: dict[str, Any], output_dir: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write the results and policy CSVs, the plots and calibration_metrics.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run.results.to_csv(output_dir / "calibration_results.csv", index=False)
    run.policy.to_csv(output_dir / "policy_curve.csv", index=False)

    curves = {}
    for name, versions in run.validation_versions.items():
        plot_calibration(
            name, run.validation_y, versions, config["calibration_plot_bins"],
            config["calibration_plot_strategy"], output_dir / f"calibration_{name}.png",
        )
        chosen = run.summary["candidates"][name]["chosen_method"]
        selected = run.policy["candidate"].eq(name) & run.policy["method"].eq(chosen)
        curves[name] = {
            "method": chosen,
            "table": run.policy.loc[selected],
            "approval": approval_curve(versions[chosen], run.validation_y),
        }
    plot_policy(curves, float(run.validation_y.mean()), output_dir / "policy_curve.png")

    settings = ("methods", "probability_clip", "calibration_plot_bins",
                "calibration_plot_strategy", "policy", "evaluation", "decision")
    payload = {
        **metadata,
        "settings": {key: config[key] for key in settings},
        **run.summary,
        "results": _records(run.results),
    }
    (output_dir / "calibration_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_calibration(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    prediction_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    config_path: str | Path = DEFAULT_CALIBRATION_CONFIG,
    development_config: str | Path = DEFAULT_DEVELOPMENT_CONFIG,
) -> dict[str, Any]:
    """Check the runtime, load config and candidates, calibrate, grade and write everything."""
    runtime = validate_runtime()
    config = load_calibration_config(config_path)
    prediction_dir = Path(prediction_dir) if prediction_dir else REPO_ROOT / config["prediction_dir"]
    candidates = load_candidates(prediction_dir, data_dir, development_config)
    run = evaluate_candidates(candidates, config)
    metadata = provenance(data_dir, config, prediction_dir, runtime)
    return write_outputs(run, config, output_dir, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the shortlisted tabular candidates.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--prediction-dir", default=None, type=Path,
                        help="defaults to prediction_dir in the calibration config")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--config", default=DEFAULT_CALIBRATION_CONFIG, type=Path)
    args = parser.parse_args()
    payload = run_calibration(
        args.data_dir, args.prediction_dir, args.output_dir, config_path=args.config
    )
    for name, summary in payload["candidates"].items():
        row = next(
            r for r in payload["results"]
            if r["candidate"] == name and r["method"] == summary["chosen_method"]
            and r["set"] == "validation"
        )
        print(f"{name}: method={summary['chosen_method']} roc_auc={row['roc_auc']:.4f} "
              f"pr_auc={row['pr_auc']:.4f} brier={row['brier_score']:.4f} "
              f"log_loss={row['log_loss']:.4f}")
    print(f"recommendation: {payload['champion']['recommendation']}")


if __name__ == "__main__":
    main()


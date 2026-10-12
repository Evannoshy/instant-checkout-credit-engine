"""Calibration of the shortlisted tabular candidates: raw, Platt and isotonic.

Every choice is predeclared in ``configs/tabular_calibration_v1.toml``, written
before any calibrated validation result was seen: the methods, the probability
clip, the policy-curve grid, the bootstrap settings and the decision rule with
its margins. ``load_calibration_config`` checks that every setting has the
right type and range, and ``threshold_grid`` turns ``[policy] thresholds`` into
the curve grid.

Rule for every later step: calibrators are fit on the ``calibration`` role
only; validation grades and never fits; the locked test split is never read.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from src.tabular.preprocess import REPO_ROOT

DEFAULT_CALIBRATION_CONFIG = REPO_ROOT / "configs" / "tabular_calibration_v1.toml"

CANDIDATES = ("logistic_regression", "xgboost_default", "xgboost_trial1")
METRICS = ("roc_auc", "pr_auc", "brier_score", "log_loss")
PLOT_STRATEGIES = ("uniform", "quantile")

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

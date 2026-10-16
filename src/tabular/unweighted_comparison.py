"""Validation comparison: does removing class weighting improve probability quality?

Four candidates are graded on the same 21,784 validation loans:
- balanced LR, raw: the frozen v1 logistic predictions;
- balanced LR + Platt: the same scores through the existing calibration-only
  Platt mapping, refitted with ``calibrate.fit_platt`` on the calibration rows
  and checked against the parameters recorded in calibration_metrics.json;
- unweighted LR, raw: src/tabular/logistic_unweighted.py (class_weight=None);
- selected XGBoost, raw: the trial chosen in reports/tabular/xgboost_selection.json.

Every file is loaded through ``calibrate.load_candidate`` (coverage, split and
metadata checks) and every metric comes from ``calibrate.metric_row``. The
three reference rows must reproduce the merged calibration_results.csv, so the
report cannot drift from the approved evidence. The unweighted model is
compared with each reference by a paired bootstrap; a difference counts as
clear only if it exceeds the predeclared margin in
configs/tabular_calibration_v1.toml and its 95% CI excludes zero.

This is a comparison, not an approval to replace the reference model. No
hyperparameter search, threshold selection or further calibration is done,
and the unweighted model is never passed through the balanced Platt mapping.

Run from the repository root (after src.tabular.logistic_unweighted):
    python -m src.tabular.unweighted_comparison
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tabular import calibrate, logistic_unweighted, provenance, raw_predictions
from src.tabular.preprocess import DEFAULT_DATA_DIR
from src.tabular.unweighted_report import render_report
from src.tabular.xgboost_baseline import (
    DEFAULT_OUTPUT_DIR,
    count_test_rows_available,
    paired_bootstrap_delta,
    validate_runtime,
)

REPORT_PATH = DEFAULT_OUTPUT_DIR / "unweighted_lr_comparison.md"
RESULTS_PATH = DEFAULT_OUTPUT_DIR / "unweighted_lr_comparison.json"
REFERENCE_METRICS = DEFAULT_OUTPUT_DIR / "calibration_metrics.json"
REFERENCE_RESULTS = DEFAULT_OUTPUT_DIR / "calibration_results.csv"
SELECTION_PATH = DEFAULT_OUTPUT_DIR / "xgboost_selection.json"

BALANCED = "logistic_regression"
UNWEIGHTED = logistic_unweighted.EXPERIMENT_NAME
CHALLENGER_KEY = "unweighted_lr_raw"
REPRODUCTION_TOLERANCE = 1e-6  # the merged evidence was produced on another platform
REPRODUCED_COLUMNS = ("rows", "positives", "mean_p", *calibrate.METRICS)


@dataclass(frozen=True)
class Entry:
    """One compared row: which loaded candidate it uses and how its scores are mapped."""

    key: str
    label: str
    candidate: str
    method: str


def compared_entries(xgboost_candidate: str) -> tuple[Entry, ...]:
    """The four candidates, in report order."""
    return (
        Entry("balanced_lr_raw", "Balanced LR, raw", BALANCED, "raw"),
        Entry("balanced_lr_platt", "Balanced LR + existing Platt", BALANCED, "platt"),
        Entry(CHALLENGER_KEY, "Unweighted LR, raw", UNWEIGHTED, "raw"),
        Entry("xgboost_selected_raw", f"Selected XGBoost ({xgboost_candidate}), raw", xgboost_candidate, "raw"),
    )


def selected_xgboost_candidate(selection_path: str | Path = SELECTION_PATH) -> str:
    """The candidate name of the XGBoost trial chosen by the bounded search."""
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    return f"xgboost_trial{selection['selected_trial_id']}"


def prediction_paths(
    xgboost_candidate: str, prediction_dir: Path, unweighted_dir: Path
) -> dict[str, dict[str, Path]]:
    """Calibration and validation files of every loaded candidate."""
    paths = {
        name: {set_name: prediction_dir / f"{name}_{set_name}.csv" for set_name in calibrate.SETS}
        for name in (BALANCED, xgboost_candidate)
    }
    paths[UNWEIGHTED] = {set_name: unweighted_dir / f"{set_name}.csv" for set_name in calibrate.SETS}
    return paths


def load_compared_candidates(
    paths: dict[str, dict[str, Path]], data_dir: str | Path
) -> dict[str, calibrate.Candidate]:
    """Load each candidate through calibrate's checks and require identical validation loans."""
    expected_ids, target = calibrate.expected_prediction_ids(data_dir)
    candidates = {
        name: calibrate.load_candidate(name, files, expected_ids, target) for name, files in paths.items()
    }
    first, *others = candidates.values()
    for other in others:
        if not np.array_equal(first.sets["validation"].loan_ids, other.sets["validation"].loan_ids):
            raise ValueError(f"{first.name} and {other.name} are not graded on the same validation loans")
    return candidates


def validation_versions(
    entries: tuple[Entry, ...], candidates: dict[str, calibrate.Candidate], eps: float
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Each entry's validation probabilities, and the Platt parameters refitted on calibration rows.

    The Platt mapping is fitted on the balanced model's calibration scores and
    applied only to entries whose candidate is the balanced model.
    """
    balanced = candidates[BALANCED].sets["calibration"]
    mappings = {
        "raw": calibrate.fit_raw(balanced.p, balanced.y, eps),
        "platt": calibrate.fit_platt(balanced.p, balanced.y, eps),
    }
    versions = {}
    for entry in entries:
        if entry.method != "raw" and entry.candidate != BALANCED:
            raise ValueError(f"The balanced Platt mapping cannot be applied to {entry.candidate}")
        versions[entry.key] = mappings[entry.method].transform(candidates[entry.candidate].sets["validation"].p)
    return versions, mappings["platt"].details


def check_reference_reproduction(
    entries: tuple[Entry, ...],
    metric_rows: dict[str, dict[str, Any]],
    platt_details: dict[str, float],
    reference_metrics: dict[str, Any],
    reference_results: pd.DataFrame,
) -> None:
    """Require the Platt mapping and the three reference rows to match the merged evidence."""
    recorded = reference_metrics["candidates"][BALANCED]["calibrators"]["platt"]
    for parameter, value in platt_details.items():
        if abs(value - recorded[parameter]) > REPRODUCTION_TOLERANCE:
            raise ValueError(f"Platt {parameter} {value} differs from the recorded {recorded[parameter]}")
    for entry in entries:
        if entry.candidate == UNWEIGHTED:
            continue
        merged = reference_results[
            reference_results["candidate"].eq(entry.candidate)
            & reference_results["method"].eq(entry.method)
            & reference_results["set"].eq("validation")
        ]
        if len(merged) != 1:
            raise ValueError(f"No single merged validation row for {entry.candidate} {entry.method}")
        for column in REPRODUCED_COLUMNS:
            if abs(metric_rows[entry.key][column] - merged[column].iloc[0]) > REPRODUCTION_TOLERANCE:
                raise ValueError(f"{entry.label} {column} does not reproduce the merged calibration results")


def classify(delta: dict[str, float], metric: str, margin: float) -> str:
    """'better', 'worse' or 'no clear difference' for the challenger, by the predeclared rule."""
    if calibrate.clearly_better(delta, metric, margin):
        return "better"
    flipped = {"delta": -delta["delta"], "ci_low": -delta["ci_high"], "ci_high": -delta["ci_low"]}
    if calibrate.clearly_better(flipped, metric, margin):
        return "worse"
    return "no clear difference"


def paired_differences(
    entries: tuple[Entry, ...], versions: dict[str, np.ndarray], y: np.ndarray, config: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Unweighted LR minus each reference, per metric, with a 95% CI and a verdict."""
    evaluation, margins = config["evaluation"], config["decision"]["margins"]
    differences = {}
    for entry in entries:
        if entry.key == CHALLENGER_KEY:
            continue
        differences[entry.key] = {}
        for metric in calibrate.METRICS:
            delta = paired_bootstrap_delta(
                y, versions[entry.key], versions[CHALLENGER_KEY], metric,
                resamples=evaluation["bootstrap_resamples"], seed=evaluation["seed"],
            )
            differences[entry.key][metric] = {**delta, "verdict": classify(delta, metric, margins[metric])}
    return differences


def build_comparison(
    data_dir: str | Path,
    prediction_dir: Path,
    unweighted_dir: Path,
    config: dict[str, Any],
    xgboost_candidate: str,
) -> dict[str, Any]:
    """Load, grade and compare the four candidates; returns the JSON-ready result."""
    entries = compared_entries(xgboost_candidate)
    paths = prediction_paths(xgboost_candidate, prediction_dir, unweighted_dir)
    candidates = load_compared_candidates(paths, data_dir)
    versions, platt_details = validation_versions(entries, candidates, config["probability_clip"])
    y = candidates[BALANCED].sets["validation"].y
    metric_rows = {entry.key: calibrate.metric_row(y, versions[entry.key]) for entry in entries}
    check_reference_reproduction(
        entries, metric_rows, platt_details,
        json.loads(REFERENCE_METRICS.read_text(encoding="utf-8")), pd.read_csv(REFERENCE_RESULTS),
    )
    return {
        "candidates": [
            {"key": entry.key, "label": entry.label, "model_version": candidates[entry.candidate].model_version,
             "method": entry.method, **metric_rows[entry.key]}
            for entry in entries
        ],
        "platt_mapping": {"fit_on": "calibration role, balanced LR raw scores", **platt_details},
        "unweighted_minus_reference": paired_differences(entries, versions, y, config),
        "margins": config["decision"]["margins"],
        "bootstrap": config["evaluation"],
        "validation_rows": int(len(y)),
        "test_rows_available_to_model": count_test_rows_available(data_dir),
        "inputs_sha256": provenance.file_checksums(
            [path for files in paths.values() for path in files.values()]
            + [REFERENCE_METRICS, REFERENCE_RESULTS, SELECTION_PATH, calibrate.DEFAULT_CALIBRATION_CONFIG]
        ),
    }


def run_comparison(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    prediction_dir: str | Path = raw_predictions.DEFAULT_PREDICTION_DIR,
    unweighted_dir: str | Path = logistic_unweighted.DEFAULT_OUTPUT_DIR,
    report_path: str | Path = REPORT_PATH,
    results_path: str | Path = RESULTS_PATH,
) -> dict[str, Any]:
    """Check the runtime, build the comparison, and write the JSON results and the Markdown report."""
    runtime = validate_runtime()
    result = {
        **build_comparison(
            data_dir, Path(prediction_dir), Path(unweighted_dir),
            calibrate.load_calibration_config(), selected_xgboost_candidate(),
        ),
        "environment": runtime,
    }
    Path(results_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Path(report_path).write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare unweighted LR with the reference candidates.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--report", default=REPORT_PATH, type=Path)
    parser.add_argument("--results", default=RESULTS_PATH, type=Path)
    args = parser.parse_args()
    result = run_comparison(args.data_dir, report_path=args.report, results_path=args.results)
    for row in result["candidates"]:
        print(f"{row['label']}: roc_auc={row['roc_auc']:.4f} pr_auc={row['pr_auc']:.4f} "
              f"brier={row['brier_score']:.4f} log_loss={row['log_loss']:.4f} "
              f"mean_p={row['mean_p']:.4f} observed={row['prevalence']:.4f}")


if __name__ == "__main__":
    main()

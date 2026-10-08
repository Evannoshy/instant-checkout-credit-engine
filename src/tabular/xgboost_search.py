"""Bounded XGBoost search: run the predeclared trials and select one configuration.

Step 6 of docs/tabular-analyst-handbook.md Section 12.3. The search question,
objective, trial list, budget and selection rule are all fixed in
configs/tabular_xgboost_v1.toml before anything runs; this module only
executes them.

- Every trial is scored with the same expanding-window folds inside the
  ``model_fit`` role (src/tabular/xgboost_baseline.py). Calibration rows,
  validation rows and the locked test split are never loaded.
- Every attempted trial is recorded, including failed ones (status "failed"
  with the error message), in reports/tabular/xgboost_trials.csv.
- Selection does not simply take the top score. Unstable trials (fold std
  above max_std_ratio x the median) are excluded, every stable trial within
  max(tolerance, one standard error) of the best stable mean is treated as
  equivalent, and the least complex of those is chosen.

Run from the repository root:
    python -m src.tabular.xgboost_search
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.pipeline import Pipeline

from src.tabular import preprocess
from src.tabular.xgboost_baseline import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_XGBOOST_CONFIG,
    HIGHER_IS_BETTER,
    SUMMARY_METRICS,
    CrossValidationResult,
    Fold,
    build_folds,
    build_xgboost_pipeline,
    cross_validate,
    load_model_fit,
    load_xgboost_config,
    run_metadata,
)

REQUIRED_SEARCH_KEYS = {"max_trials", "primary_metric", "tolerance", "max_std_ratio", "trials"}
STATUS_OK = "ok"
STATUS_FAILED = "failed"

PipelineFactory = Callable[[dict[str, Any], dict[str, Any]], Pipeline]


def validate_trials(xgboost_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the predeclared trials, refusing an over-budget or malformed list before any run.

    Raises ValueError for missing [search] keys, an unknown primary metric, no
    trials, more trials than max_trials, duplicate trial ids, or a trial that
    overrides a parameter absent from [model] (which would usually be a typo).
    """
    search = xgboost_config["search"]
    if missing := REQUIRED_SEARCH_KEYS - set(search):
        raise ValueError(f"[search] is missing keys: {sorted(missing)}")
    if search["primary_metric"] not in HIGHER_IS_BETTER:
        raise ValueError(f"Unknown primary_metric {search['primary_metric']!r}")
    trials = search["trials"]
    if not trials:
        raise ValueError("[search] declares no trials")
    if len(trials) > search["max_trials"]:
        raise ValueError(
            f"[search] declares {len(trials)} trials but the budget is {search['max_trials']}"
        )
    ids = [trial["trial_id"] for trial in trials]
    if len(set(ids)) != len(ids):
        raise ValueError(f"[search] trial ids must be unique; got {ids}")
    for trial in trials:
        if unknown := set(trial.get("params", {})) - set(xgboost_config["model"]):
            raise ValueError(f"Trial {trial['trial_id']} sets unknown parameters: {sorted(unknown)}")
    return trials


def trial_params(xgboost_config: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    """The default [model] parameters with the trial's overrides applied (a new dict)."""
    return {**xgboost_config["model"], **trial.get("params", {})}


def complexity(params: dict[str, Any]) -> int:
    """Upper bound on leaves: trees x 2^depth. Used only to break near-ties toward simpler models."""
    return int(params["n_estimators"]) * 2 ** int(params["max_depth"])


def _trial_row(trial_id: int, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        **{f"param_{name}": value for name, value in params.items()},
        "complexity": complexity(params),
    }


def _result_columns(result: CrossValidationResult, primary: str) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        columns[f"{metric}_mean"] = result.summary[metric]["mean"]
        columns[f"{metric}_std"] = result.summary[metric]["std"]
    for fold in result.fold_metrics:
        columns[f"{primary}_fold{fold['fold']}"] = fold[primary]
    return columns


def run_trials(
    features: pd.DataFrame,
    target: pd.Series,
    folds: tuple[Fold, ...],
    feature_config: dict[str, Any],
    xgboost_config: dict[str, Any],
    *,
    make_pipeline: PipelineFactory = build_xgboost_pipeline,
) -> pd.DataFrame:
    """Cross-validate every predeclared trial; a failing trial is recorded, not skipped."""
    primary = xgboost_config["search"]["primary_metric"]
    rows = []
    for trial in validate_trials(xgboost_config):
        params = trial_params(xgboost_config, trial)
        row = _trial_row(trial["trial_id"], params)
        started = time.perf_counter()
        try:
            result = cross_validate(
                lambda: make_pipeline(feature_config, params), features, target, folds
            )
        except Exception as error:  # recorded in the trial log, never silently dropped
            row.update(status=STATUS_FAILED, error=f"{type(error).__name__}: {error}")
        else:
            row.update(status=STATUS_OK, error="", **_result_columns(result, primary))
        row["runtime_seconds"] = round(time.perf_counter() - started, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def select_configuration(
    trials: pd.DataFrame, search: dict[str, Any], n_folds: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the predeclared selection rule; return the annotated trials and the decision.

    Returns a new frame with boolean columns stable, eligible and selected.
    Raises ValueError if no trial succeeded.
    """
    primary = search["primary_metric"]
    sign = 1.0 if HIGHER_IS_BETTER[primary] else -1.0
    mean_col, std_col = f"{primary}_mean", f"{primary}_std"

    succeeded = trials["status"].eq(STATUS_OK)
    if not succeeded.any():
        raise ValueError("No search trial succeeded; nothing can be selected")
    median_std = float(trials.loc[succeeded, std_col].median())
    stable = succeeded & trials[std_col].le(search["max_std_ratio"] * median_std)

    stable_rows = trials.loc[stable]
    reference = stable_rows.loc[(sign * stable_rows[mean_col]).idxmax()]
    standard_error = float(reference[std_col]) / math.sqrt(n_folds)
    margin = max(float(search["tolerance"]), standard_error)
    gap = sign * (float(reference[mean_col]) - trials[mean_col])
    eligible = stable & gap.le(margin)

    chosen = trials.loc[eligible].sort_values(["complexity", "trial_id"]).iloc[0]
    annotated = trials.assign(
        stable=stable, eligible=eligible, selected=trials["trial_id"].eq(chosen["trial_id"])
    )
    param_columns = [name for name in trials.columns if name.startswith("param_")]
    decision = {
        "primary_metric": primary,
        "rule": {key: search[key] for key in ("tolerance", "max_std_ratio")},
        "trials_attempted": int(len(trials)),
        "trials_failed": int((~succeeded).sum()),
        "trials_unstable": int((succeeded & ~stable).sum()),
        "trials_eligible": int(eligible.sum()),
        "median_fold_std": median_std,
        "reference_trial_id": int(reference["trial_id"]),
        "reference_mean": float(reference[mean_col]),
        "equivalence_margin": margin,
        "selected_trial_id": int(chosen["trial_id"]),
        "selected_mean": float(chosen[mean_col]),
        "selected_std": float(chosen[std_col]),
        "selected_complexity": int(chosen["complexity"]),
        "selected_params": {
            name.removeprefix("param_"): _plain(chosen[name]) for name in param_columns
        },
    }
    return annotated, decision


def _plain(value: Any) -> Any:
    """Convert numpy scalars to plain Python values for JSON."""
    return value.item() if hasattr(value, "item") else value


def run_search(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    output_dir: str | Path,
    *,
    xgboost_config_path: str | Path = DEFAULT_XGBOOST_CONFIG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load model_fit, run every trial, select one, and write the trial log and decision."""
    xgboost_config = load_xgboost_config(xgboost_config_path)
    validate_trials(xgboost_config)
    feature_config = preprocess.load_feature_config()
    features, target, issue_month = load_model_fit(data_dir, raw_csv_path)
    folds = build_folds(issue_month, xgboost_config)

    trials = run_trials(features, target, folds, feature_config, xgboost_config)
    annotated, decision = select_configuration(trials, xgboost_config["search"], len(folds))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_dir / "xgboost_trials.csv", index=False)
    selection = {**run_metadata(data_dir, xgboost_config, folds), **decision}
    (output_dir / "xgboost_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    return annotated, selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded XGBoost search on model_fit rows.")
    parser.add_argument("--data-dir", default=preprocess.DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--raw-csv", default=preprocess.DEFAULT_RAW_CSV, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--xgboost-config", default=DEFAULT_XGBOOST_CONFIG, type=Path)
    args = parser.parse_args()
    trials, selection = run_search(
        args.data_dir, args.raw_csv, args.output_dir, xgboost_config_path=args.xgboost_config
    )
    primary = selection["primary_metric"]
    columns = ["trial_id", "status", f"{primary}_mean", f"{primary}_std", "complexity",
               "stable", "eligible", "selected"]
    print(trials[columns].to_string(index=False))
    print(json.dumps({key: selection[key] for key in (
        "reference_trial_id", "equivalence_margin", "selected_trial_id", "selected_params",
    )}, indent=2))


if __name__ == "__main__":
    main()

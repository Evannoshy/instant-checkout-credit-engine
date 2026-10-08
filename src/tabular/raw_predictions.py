"""Raw (uncalibrated) candidate probabilities for the calibration handoff.

For each shortlisted model: fit once on all ``model_fit`` rows, then predict
the ``calibration`` role and the ``validation`` split. Calibration and final
evaluation happen downstream; nothing here is fitted on calibration or
validation rows, and the locked test split is never requested.

Shortlist:
- ``logistic_regression``: the approved logistic pipeline, unchanged settings.
- ``xgboost_default``: the Task 1 default (configs/tabular_xgboost_v1.toml [model]).
- ``xgboost_trial<id>``: the configuration chosen by src/tabular/xgboost_search.py
  (read from reports/tabular/xgboost_selection.json), only if it differs from
  the default.

Every file uses the shared prediction schema (evaluate.PREDICTION_COLUMNS)
with ``split`` set to "calibration" or "validation", and is checked with
``evaluate.validate_prediction_frame`` against the full expected loan_id set
before it is written. A manifest.json records row counts, coverage,
probability ranges and raw metrics per file.

Run from the repository root (after the search):
    python -m src.tabular.raw_predictions
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.tabular import development, evaluate, logistic_baseline, preprocess
from src.tabular.xgboost_baseline import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_XGBOOST_CONFIG,
    MODEL_NAME,
    build_xgboost_pipeline,
    count_test_rows_available,
    environment,
    load_model_fit,
    load_xgboost_config,
    validate_runtime,
)

DEFAULT_SELECTION = DEFAULT_OUTPUT_DIR / "xgboost_selection.json"
DEFAULT_PREDICTION_DIR = DEFAULT_OUTPUT_DIR / "raw_predictions"


@dataclass(frozen=True)
class Candidate:
    """A shortlisted model: how to build it and how to label its predictions."""

    name: str
    model_name: str
    model_version: str
    params: dict[str, Any]
    make_pipeline: Callable[[], Pipeline]


def build_shortlist(
    feature_config: dict[str, Any],
    xgboost_config: dict[str, Any],
    logistic_config: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[Candidate, ...]:
    """LR, default XGBoost and (if different) the selected XGBoost configuration."""
    if selection.get("model_version") != xgboost_config["model_version"]:
        raise ValueError(
            f"Selection was made for {selection.get('model_version')!r}, "
            f"not {xgboost_config['model_version']!r}; rerun the search"
        )
    version = xgboost_config["model_version"]
    default_params = dict(xgboost_config["model"])
    selected_params = dict(selection["selected_params"])
    candidates = [
        Candidate(
            "logistic_regression", logistic_baseline.MODEL_NAME,
            f"{logistic_baseline.BASELINE_VERSION}-model-fit", dict(logistic_config["model"]),
            lambda: logistic_baseline.build_pipeline(feature_config, logistic_config),
        ),
        Candidate(
            "xgboost_default", MODEL_NAME, f"{version}-default", default_params,
            lambda: build_xgboost_pipeline(feature_config, default_params),
        ),
    ]
    if selected_params != default_params:
        trial = selection["selected_trial_id"]
        candidates.append(Candidate(
            f"xgboost_trial{trial}", MODEL_NAME, f"{version}-trial{trial}", selected_params,
            lambda: build_xgboost_pipeline(feature_config, selected_params),
        ))
    return tuple(candidates)


def _require_disjoint(
    fit_ids: pd.Index, scoring: dict[str, tuple[pd.DataFrame, pd.Series]]
) -> None:
    """Fit rows and every scoring set must be pairwise disjoint."""
    named = {"model_fit": set(fit_ids), **{name: set(x.index) for name, (x, _) in scoring.items()}}
    names = list(named)
    for position, first in enumerate(names):
        for second in names[position + 1:]:
            if overlap := named[first] & named[second]:
                raise ValueError(f"{first} and {second} share {len(overlap)} loan_id values")


def _file_summary(frame: pd.DataFrame, target: pd.Series, candidate: Candidate) -> dict[str, Any]:
    values = frame["p_default_tabular"].to_numpy(dtype=np.float64)
    labels = target.loc[frame["loan_id"]].to_numpy()
    return {
        "model_name": candidate.model_name,
        "model_version": candidate.model_version,
        "params": candidate.params,
        "rows": len(frame),
        "expected_rows": len(target),
        "missing_ids": 0,  # validate_prediction_frame has already enforced exact coverage
        "min_probability": float(values.min()),
        "max_probability": float(values.max()),
        "raw_metrics": evaluate.evaluate_predictions(labels, values),
    }


def predict_candidates(
    candidates: tuple[Candidate, ...],
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    scoring: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Fit each candidate on the fit rows and score every scoring set.

    Returns ({"<candidate>_<set>": prediction frame}, {"<candidate>_<set>": summary}).
    Raises ValueError if any two row sets overlap or a frame fails validation.
    """
    _require_disjoint(fit_features.index, scoring)
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for candidate in candidates:
        pipeline = candidate.make_pipeline().fit(fit_features, fit_target)
        for set_name, (features, target) in scoring.items():
            frame = evaluate.build_prediction_frame(
                loan_ids=features.index, split_name=set_name,
                probabilities=pipeline.predict_proba(features)[:, 1],
                model_name=candidate.model_name, model_version=candidate.model_version,
            )
            evaluate.validate_prediction_frame(frame, expected_loan_ids=set(features.index))
            key = f"{candidate.name}_{set_name}"
            frames[key] = frame
            summaries[key] = _file_summary(frame, target, candidate)
    return frames, summaries


def run_raw_predictions(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    output_dir: str | Path = DEFAULT_PREDICTION_DIR,
    *,
    selection_path: str | Path = DEFAULT_SELECTION,
    xgboost_config_path: str | Path = DEFAULT_XGBOOST_CONFIG,
) -> dict[str, Any]:
    """Load the three row sets, predict every candidate, write CSVs and manifest.json."""
    runtime = validate_runtime()
    xgboost_config = load_xgboost_config(xgboost_config_path)
    feature_config = preprocess.load_feature_config()
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection.get("environment") != runtime:
        raise ValueError(
            "Search selection used a different environment. Rerun the search in this "
            "environment before exporting calibration and validation predictions."
        )
    candidates = build_shortlist(
        feature_config, xgboost_config, logistic_baseline.load_logistic_config(), selection
    )

    fit_features, fit_target, _ = load_model_fit(data_dir, raw_csv_path)
    scoring = {
        "calibration": development.load_tabular_role(data_dir, raw_csv_path, "calibration"),
        "validation": preprocess.load_tabular_split(data_dir, raw_csv_path, "validation"),
    }
    frames, summaries = predict_candidates(candidates, fit_features, fit_target, scoring)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in frames.items():
        frame.to_csv(output_dir / f"{key}.csv", index=False)
    manifest = {
        "fit_rows": {"role": "model_fit", "rows": len(fit_features)},
        "scoring_rows": {name: len(target) for name, (_, target) in scoring.items()},
        "test_rows_available_to_model": count_test_rows_available(data_dir),
        "selected_trial_id": selection["selected_trial_id"],
        "note": "Raw, uncalibrated probabilities. Fit a calibrator on the calibration files only.",
        "files": {f"{key}.csv": summary for key, summary in summaries.items()},
        "environment": environment(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write raw candidate probabilities for calibration.")
    parser.add_argument("--data-dir", default=preprocess.DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--raw-csv", default=preprocess.DEFAULT_RAW_CSV, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_PREDICTION_DIR, type=Path)
    parser.add_argument("--selection", default=DEFAULT_SELECTION, type=Path)
    args = parser.parse_args()
    manifest = run_raw_predictions(
        args.data_dir, args.raw_csv, args.output_dir, selection_path=args.selection
    )
    for name, summary in manifest["files"].items():
        metrics = summary["raw_metrics"]
        print(f"{name}: rows={summary['rows']} roc_auc={metrics['roc_auc']:.4f} "
              f"pr_auc={metrics['pr_auc']:.4f} brier={metrics['brier_score']:.4f}")


if __name__ == "__main__":
    main()

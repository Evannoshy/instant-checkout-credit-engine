"""Unweighted logistic regression: the v1 logistic model with class_weight=None.

A separate, versioned experiment. It uses the same nine features and the same
settings as the frozen v1 model; the only change is class_weight, set to None
in a copy of the v1 config before ``logistic_baseline.build_pipeline`` is
called. configs/tabular_logistic_v1.toml is never edited.

The model is fitted once on the 59,353 model_fit loans and scores all 26,940
calibration loans and 21,784 validation loans. Nothing is fitted on
calibration or validation rows, and the locked test split is never requested.
The old logistic baseline runner is not used: it fits all 86,293 train rows.

Outputs go to their own directory, never over the frozen v1 files:
    reports/tabular/raw_predictions/unweighted_lr_v1/calibration.csv
    reports/tabular/raw_predictions/unweighted_lr_v1/validation.csv
    reports/tabular/raw_predictions/unweighted_lr_v1/manifest.json

The probabilities are raw. The v1 Platt mapping was fitted to the balanced
model's scores and must not be applied to this model.

Run from the repository root:
    python -m src.tabular.logistic_unweighted
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.tabular import development, logistic_baseline, preprocess, provenance, raw_predictions
from src.tabular.xgboost_baseline import count_test_rows_available, load_model_fit, validate_runtime

EXPERIMENT_NAME = "unweighted_lr_v1"
MODEL_VERSION = "tabular-logistic-unweighted-v1-model-fit"
CHANGED_SETTING = "class_weight"
BASE_CLASS_WEIGHT = "balanced"
FROZEN_PREDICTION_DIR = raw_predictions.DEFAULT_PREDICTION_DIR
DEFAULT_OUTPUT_DIR = FROZEN_PREDICTION_DIR / EXPERIMENT_NAME
EXPECTED_ROWS = {"model_fit": 59_353, "calibration": 26_940, "validation": 21_784}


def unweighted_config(base_config: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of the balanced v1 config with only class_weight set to None.

    Raises ValueError if the base config is not the balanced v1 config, so the
    experiment cannot silently start from a different model.
    """
    if base_config["model"][CHANGED_SETTING] != BASE_CLASS_WEIGHT:
        raise ValueError(
            f"Expected the balanced v1 config (class_weight={BASE_CLASS_WEIGHT!r}); "
            f"got {base_config['model'][CHANGED_SETTING]!r}"
        )
    copied = copy.deepcopy(base_config)
    return {**copied, "model": {**copied["model"], CHANGED_SETTING: None}}


def build_candidate(
    feature_config: dict[str, Any], base_config: dict[str, Any]
) -> raw_predictions.Candidate:
    """The unweighted logistic model, labelled with its own experiment version."""
    config = unweighted_config(base_config)
    return raw_predictions.Candidate(
        name=EXPERIMENT_NAME,
        model_name=logistic_baseline.MODEL_NAME,
        model_version=MODEL_VERSION,
        params=dict(config["model"]),
        make_pipeline=lambda: logistic_baseline.build_pipeline(feature_config, config),
    )


def predict_unweighted(
    candidate: raw_predictions.Candidate,
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    scoring: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Fit once on the fit rows and score each scoring set; results keyed by set name.

    Reuses ``raw_predictions.predict_candidates``, which refuses overlapping row
    sets and validates every frame against its full expected loan_id set.
    """
    frames, summaries = raw_predictions.predict_candidates(
        (candidate,), fit_features, fit_target, scoring
    )
    prefix = f"{candidate.name}_"
    return (
        {key.removeprefix(prefix): frame for key, frame in frames.items()},
        {key.removeprefix(prefix): summary for key, summary in summaries.items()},
    )


def require_expected_rows(fit_rows: int, scoring: dict[str, tuple[pd.DataFrame, pd.Series]]) -> None:
    """The frozen data must give exactly the agreed fit, calibration and validation counts."""
    observed = {"model_fit": fit_rows, **{name: len(target) for name, (_, target) in scoring.items()}}
    if observed != EXPECTED_ROWS:
        raise ValueError(f"Expected rows {EXPECTED_ROWS}; got {observed}")


def write_predictions(frames: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    """Write <set>.csv files, refusing the frozen v1 prediction directory."""
    output_dir = Path(output_dir)
    if output_dir.resolve() == FROZEN_PREDICTION_DIR.resolve():
        raise ValueError(
            f"{provenance.display_path(output_dir)} holds the frozen v1 predictions; "
            "write the unweighted experiment to its own directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for set_name, frame in frames.items():
        paths[set_name] = output_dir / f"{set_name}.csv"
        frame.to_csv(paths[set_name], index=False)
    return paths


def file_records(paths: dict[str, Path], summaries: dict[str, Any]) -> dict[str, Any]:
    """Per written file: its summary (rows, probability range, raw metrics) and SHA-256."""
    checksums = provenance.file_checksums(list(paths.values()))
    return {
        path.name: {**summaries[set_name], "sha256": checksums[provenance.display_path(path)]}
        for set_name, path in paths.items()
    }


def input_checksums(data_dir: str | Path, raw_csv_path: str | Path, raw_csv_sha256: str) -> dict[str, str]:
    """SHA-256 of the raw CSV, the locked split files and every config the run reads."""
    return {
        provenance.display_path(raw_csv_path): raw_csv_sha256,
        **provenance.file_checksums([
            Path(data_dir) / "split_manifest.csv",
            Path(data_dir) / provenance.SPLIT_STATISTICS,
            preprocess.DEFAULT_FEATURE_CONFIG,
            logistic_baseline.DEFAULT_LOGISTIC_CONFIG,
            development.DEFAULT_DEVELOPMENT_CONFIG,
        ]),
    }


def run_unweighted(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Verify inputs, fit on model_fit, score calibration and validation, write files and manifest."""
    runtime = validate_runtime()
    raw_csv_sha256 = provenance.require_locked_raw_csv(data_dir, raw_csv_path)
    base_config = logistic_baseline.load_logistic_config()
    candidate = build_candidate(preprocess.load_feature_config(), base_config)

    fit_features, fit_target, _ = load_model_fit(data_dir, raw_csv_path)
    scoring = {
        "calibration": development.load_tabular_role(data_dir, raw_csv_path, "calibration"),
        "validation": preprocess.load_tabular_split(data_dir, raw_csv_path, "validation"),
    }
    require_expected_rows(len(fit_features), scoring)
    frames, summaries = predict_unweighted(candidate, fit_features, fit_target, scoring)
    paths = write_predictions(frames, output_dir)

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "model_name": candidate.model_name,
        "model_version": MODEL_VERSION,
        "base_config": {
            "path": provenance.display_path(logistic_baseline.DEFAULT_LOGISTIC_CONFIG),
            "frozen_version": logistic_baseline.BASELINE_VERSION,
            "settings": base_config["model"],
        },
        "changed_setting": {CHANGED_SETTING: {"from": BASE_CLASS_WEIGHT, "to": None}},
        "params": candidate.params,
        "fit_rows": {"role": "model_fit", "rows": len(fit_features)},
        "scoring_rows": {name: len(target) for name, (_, target) in scoring.items()},
        "test_rows_available_to_model": count_test_rows_available(data_dir),
        "note": (
            "Raw, uncalibrated probabilities. Do not apply the balanced v1 Platt mapping: "
            "it was fitted to a different model's scores."
        ),
        "files": file_records(paths, summaries),
        "inputs_sha256": input_checksums(data_dir, raw_csv_path, raw_csv_sha256),
        "environment": runtime,
        "reproduce": [
            "python -m pip install -r src/tabular/requirements.txt",
            "python -m src.tabular.logistic_unweighted --raw-csv data/raw/loan.csv",
            "LENDINGCLUB_RAW_CSV=data/raw/loan.csv python -m pytest src/tabular/test_logistic_unweighted.py -v",
        ],
    }
    (Path(output_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the unweighted logistic experiment.")
    parser.add_argument("--data-dir", default=preprocess.DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--raw-csv", default=preprocess.DEFAULT_RAW_CSV, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    args = parser.parse_args()
    manifest = run_unweighted(args.data_dir, args.raw_csv, args.output_dir)
    for name, summary in manifest["files"].items():
        metrics = summary["raw_metrics"]
        print(f"{name}: rows={summary['rows']} roc_auc={metrics['roc_auc']:.4f} "
              f"pr_auc={metrics['pr_auc']:.4f} brier={metrics['brier_score']:.4f} "
              f"log_loss={metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()

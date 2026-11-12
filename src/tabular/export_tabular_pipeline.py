"""Saved tabular pipeline (tabular-pipeline-v1): the balanced v1 LR, refitted, checked and saved.

Fits the approved balanced logistic pipeline (``configs/tabular_logistic_v1.toml``,
``tabular_features_v1``) once on the 59,353 ``model_fit`` loans, then proves it
is the model behind every fusion result before anything is written:

1. Reproduction: it scores the 26,940 calibration-role loans (``fusion_fit`` +
   ``fusion_calibration``) and the 21,784 validation loans, and every
   probability must equal the committed
   ``reports/tabular/raw_predictions/logistic_regression_{calibration,validation}.csv``
   within ``rtol = atol = 1e-10``. Those files are checked against the pins in
   ``configs/fusion_handoff_v1.toml`` first: the pins the fusion tables use.
2. Reference calibrator: the ``tabular_calibration_v1`` Platt mapping is refitted
   on the committed calibration scores by
   ``fusion_handoff.build_calibrated_validation`` and must reproduce
   ``calibration_metrics.json`` and ``calibration_results.csv`` within 1e-6.

Only then is the fitted sklearn Pipeline saved with joblib, reloaded and
re-checked, and the manifest written. The Platt mapping is recorded as plain
numbers in the manifest, never pickled.

Contract:
- ``artifacts/tabular/tabular_logistic_v1_model_fit.joblib`` (gitignored): the
  fitted Pipeline (impute, scale, one-hot, LR). It takes the nine **built**
  ``tabular_features_v1`` columns, not raw borrower fields: Stage A
  (``preprocess._build_features``) must run first. ``predict_proba(X)[:, 1]`` is
  the raw probability, the stacker's tabular input.
- ``reports/tabular/tabular_pipeline_v1_manifest.json`` (committed): versions,
  the artifact's SHA-256, the ``input_contract``, the reproduction record, the
  Platt numbers, input checksums and the environment. It is the contract; the
  artifact is rebuilt from it by rerunning this module.

Labels are read only to fit on ``model_fit`` and inside the Platt reproduction
check. Nothing is fitted on calibration or validation rows, and the locked test
split is never requested.

Run from the repository root (needs the restricted raw CSV):
    python -m src.tabular.export_tabular_pipeline --raw-csv data/raw/loan.csv

Known limitations:
- The artifact loads reliably only under the scikit-learn version recorded in
  the manifest (1.9.1); rebuild after an upgrade.
- ``joblib.load`` can run code. Verify the artifact's SHA-256 against the
  manifest before loading it.
- The artifact does no feature building. Serving code must run Stage A exactly
  as training did; ``reports/tabular/tabular_pipeline_v1.md`` documents it.
- The Platt mapping was fitted on all 26,940 Jan-Jul 2013 loans, including the
  15,587 ``fusion_fit`` loans: fine for the tabular-only fallback, but not held
  out from the fusion rows.
- The committed CSVs were written under Python 3.13.15; this module accepts any
  Python 3.13 with the pinned packages (``validate_runtime``). The 1e-10 check
  is what shows the refit matches across that difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.nlp.preprocess import sha256_file
from src.tabular import calibrate, development, evaluate, logistic_baseline, preprocess, provenance
from src.tabular.fusion_handoff import build_calibrated_validation, load_score_file
from src.tabular.fusion_handoff_v1 import DEFAULT_CONFIG as DEFAULT_HANDOFF_CONFIG
from src.tabular.fusion_handoff_v1 import load_handoff_config
from src.tabular.logistic_unweighted import input_checksums
from src.tabular.preprocess import DEFAULT_DATA_DIR, DEFAULT_RAW_CSV, REPO_ROOT
from src.tabular.raw_predictions import DEFAULT_PREDICTION_DIR
from src.tabular.unweighted_comparison import REFERENCE_METRICS, REFERENCE_RESULTS
from src.tabular.xgboost_baseline import count_test_rows_available, load_model_fit, validate_runtime

PIPELINE_VERSION = "tabular-pipeline-v1"
MODEL_NAME = logistic_baseline.MODEL_NAME
MODEL_VERSION = f"{logistic_baseline.BASELINE_VERSION}-model-fit"  # the version the committed CSVs carry
REQUIRED_CLASS_WEIGHT = "balanced"
EXPECTED_ROWS = {"model_fit": 59_353, "calibration": 26_940, "validation": 21_784}
SCORING_SETS = ("calibration", "validation")
# Same values as the existing refit-vs-committed tests (test_raw_predictions.py, test_logistic_unweighted.py).
REPRODUCTION_RTOL = 1e-10
REPRODUCTION_ATOL = 1e-10

DEFAULT_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "tabular" / "tabular_logistic_v1_model_fit.joblib"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "reports" / "tabular" / "tabular_pipeline_v1_manifest.json"
INPUT_DOCUMENT = "reports/tabular/tabular_pipeline_v1.md"

# Every ValueError message raised by this module, as str.format templates.
ERROR_MESSAGES: dict[str, str] = {
    "not_balanced": "Expected the balanced v1 logistic config (class_weight={expected!r}); got {actual!r}",
    "wrong_version": "Committed tabular files are pinned as {actual!r}; this module saves {expected!r}",
    "row_counts": "Expected rows {expected}; got {actual}",
    "sets_overlap": "{first} and {second} share {count} loan_ids",
    "committed_file_missing": "No pinned tabular file with file_split {split!r} in the handoff config",
    "not_reproduced": (
        "{split}: {count} of {rows} refitted probabilities differ from the committed file "
        "beyond rtol={rtol}, atol={atol} (max abs difference {max_diff:.3e})"
    ),
    "reload_differs": "The reloaded artifact's {split} predictions differ from the fitted pipeline's",
}


# --------------------------------------------------------------------------- data and fit


def load_rows(
    data_dir: str | Path, raw_csv_path: str | Path
) -> tuple[tuple[pd.DataFrame, pd.Series], dict[str, tuple[pd.DataFrame, pd.Series]]]:
    """(model_fit features, target) and {set: (features, target)} for calibration and validation.

    The same loaders as ``logistic_unweighted.run_unweighted``. Raises ValueError
    unless the counts are exactly 59,353 / 26,940 / 21,784 and the three sets
    share no loan.
    """
    fit_features, fit_target, _ = load_model_fit(data_dir, raw_csv_path)
    scoring = {
        "calibration": development.load_tabular_role(data_dir, raw_csv_path, "calibration"),
        "validation": preprocess.load_tabular_split(data_dir, raw_csv_path, "validation"),
    }
    observed = {"model_fit": len(fit_features), **{name: len(target) for name, (_, target) in scoring.items()}}
    if observed != EXPECTED_ROWS:
        raise ValueError(ERROR_MESSAGES["row_counts"].format(expected=EXPECTED_ROWS, actual=observed))
    named = {"model_fit": set(fit_features.index), **{name: set(x.index) for name, (x, _) in scoring.items()}}
    names = list(named)
    for position, first in enumerate(names):
        for second in names[position + 1:]:
            if overlap := named[first] & named[second]:
                raise ValueError(ERROR_MESSAGES["sets_overlap"].format(first=first, second=second, count=len(overlap)))
    return (fit_features, fit_target), scoring


def fit_reference_pipeline(
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    feature_config: dict[str, Any],
    logistic_config: dict[str, Any],
) -> Pipeline:
    """The balanced v1 LR pipeline fitted on the model_fit rows. Refuses any other class_weight."""
    class_weight = logistic_config["model"]["class_weight"]
    if class_weight != REQUIRED_CLASS_WEIGHT:
        raise ValueError(ERROR_MESSAGES["not_balanced"].format(expected=REQUIRED_CLASS_WEIGHT, actual=class_weight))
    return logistic_baseline.build_pipeline(feature_config, logistic_config).fit(fit_features, fit_target)


def predict_frames(pipeline: Pipeline, scoring: dict[str, tuple[pd.DataFrame, pd.Series]]) -> dict[str, pd.DataFrame]:
    """The shared prediction frame per scoring set, each checked against its full loan_id set."""
    frames = {}
    for split, (features, _) in scoring.items():
        frame = evaluate.build_prediction_frame(
            loan_ids=features.index, split_name=split,
            probabilities=pipeline.predict_proba(features)[:, 1],
            model_name=MODEL_NAME, model_version=MODEL_VERSION,
        )
        evaluate.validate_prediction_frame(frame, expected_loan_ids=set(features.index))
        frames[split] = frame
    return frames


# --------------------------------------------------------------------------- checks


def committed_files(handoff_config: dict[str, Any], root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """{file_split: {path, sha256, ...}} for the pinned committed tabular files, from the handoff config."""
    source = handoff_config["sources"]["tabular"]
    if source["model_version"] != MODEL_VERSION:
        raise ValueError(ERROR_MESSAGES["wrong_version"].format(actual=source["model_version"], expected=MODEL_VERSION))
    files = {entry["file_split"]: {**entry, "resolved": root / entry["path"]} for entry in source["files"]}
    for split in SCORING_SETS:
        if split not in files:
            raise ValueError(ERROR_MESSAGES["committed_file_missing"].format(split=split))
    return files


def check_reproduction(
    frames: dict[str, pd.DataFrame], files: dict[str, dict[str, Any]], probability_column: str
) -> dict[str, Any]:
    """Require every refitted probability to equal the committed file's for that loan_id.

    Each committed file is checksum-verified (LF/CRLF accepted), then checked by
    ``fusion_handoff.load_score_file`` against the refit's exact loan_id set,
    split label, model name and version. Returns rows, max abs difference and
    the pinned SHA per set.
    """
    record = {}
    for split, frame in frames.items():
        entry = files[split]
        provenance.verify_file_checksum(entry["resolved"], entry["sha256"], allow_crlf=True)
        committed = load_score_file(
            entry["resolved"], probability_column=probability_column, split=split,
            expected_ids=set(frame["loan_id"]), model_name=MODEL_NAME, allowed_versions=frozenset({MODEL_VERSION}),
        )
        expected = committed.set_index("loan_id")[probability_column].loc[frame["loan_id"]].to_numpy(np.float64)
        actual = frame["p_default_tabular"].to_numpy(np.float64)
        difference = np.abs(actual - expected)
        if count := int((difference > REPRODUCTION_ATOL + REPRODUCTION_RTOL * np.abs(expected)).sum()):
            raise ValueError(ERROR_MESSAGES["not_reproduced"].format(
                split=split, count=count, rows=len(frame), rtol=REPRODUCTION_RTOL, atol=REPRODUCTION_ATOL,
                max_diff=float(difference.max()),
            ))
        record[split] = {
            "rows": len(frame),
            "max_abs_difference": float(difference.max()),
            "committed_file": entry["path"],
            "committed_sha256_lf": entry["sha256"],
        }
    return record


def check_reference_calibrator(
    data_dir: str | Path,
    prediction_dir: str | Path,
    reference_metrics_path: str | Path,
    reference_results_path: str | Path,
    calibration_config_path: str | Path,
) -> dict[str, Any]:
    """The tabular_calibration_v1 Platt numbers, after they reproduce the merged calibration evidence.

    Reuses ``fusion_handoff.build_calibrated_validation``: Platt is refitted on
    the committed calibration-role scores only and must reproduce
    calibration_metrics.json and the validation rows of calibration_results.csv
    within 1e-6, or it raises.
    """
    _, details = build_calibrated_validation(
        data_dir, prediction_dir, reference_metrics_path, reference_results_path, calibration_config_path,
    )
    platt = details["platt_mapping"]
    return {
        "method": "platt",
        "slope": platt["slope"],
        "intercept": platt["intercept"],
        "probability_clip": calibrate.load_calibration_config(calibration_config_path)["probability_clip"],
        "formula": "sigmoid(slope * logit(clip(p_raw, clip, 1 - clip)) + intercept)",
        "calibration_version": details["calibration_version"],
        "fit_on": "tabular_development_v1 calibration role (26,940 loans: fusion_fit + fusion_calibration)",
        "reproduced_against": [provenance.display_path(reference_metrics_path),
                               provenance.display_path(reference_results_path)],
        "reproduction_tolerance": details["reproduction_tolerance"],
        "calibration_metrics_sha256": sha256_file(reference_metrics_path),
    }


# --------------------------------------------------------------------------- save and record


def save_artifact(pipeline: Pipeline, artifact_path: Path, scoring: dict[str, tuple[pd.DataFrame, pd.Series]]) -> str:
    """Dump to a temporary file, reload it, require identical predictions, then move it into place.

    A failed reload check deletes the temporary file, so no artifact is left behind.
    Returns the artifact's SHA-256.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_name(artifact_path.name + ".tmp")
    joblib.dump(pipeline, temporary)
    try:
        reloaded = joblib.load(temporary)  # the file this function just wrote
        for split, (features, _) in scoring.items():
            if not np.array_equal(reloaded.predict_proba(features), pipeline.predict_proba(features)):
                raise ValueError(ERROR_MESSAGES["reload_differs"].format(split=split))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(artifact_path)
    return sha256_file(artifact_path)


def input_contract(feature_config: dict[str, Any]) -> dict[str, Any]:
    """What the artifact expects as input, machine-readable. INPUT_DOCUMENT explains it in prose."""
    numeric, categorical = feature_config["model"]["numeric"], feature_config["model"]["categorical"]
    categories = feature_config["categories"]
    return {
        "feature_set_version": feature_config["feature_set_version"],
        "columns": [
            {"name": column, "kind": "numeric" if column in numeric else "categorical"}
            for column in preprocess.feature_columns(feature_config)
        ],
        "categories": {column: categories[column] for column in categorical},
        "built_by": "src/tabular/preprocess.py _build_features (Stage A)",
        "input_form": "pandas DataFrame with exactly these columns, in this order; one row per loan",
        "missing_values": "allowed; imputed by the pipeline (numeric: median, categorical: 'missing')",
        "documentation": INPUT_DOCUMENT,
    }


def build_manifest(
    *,
    data_dir: Path,
    raw_csv_path: Path,
    raw_csv_sha256: str,
    artifact_path: Path,
    artifact_sha256: str,
    feature_config: dict[str, Any],
    logistic_config: dict[str, Any],
    calibration_config_path: Path,
    handoff_config_path: Path,
    reference_metrics_path: Path,
    reference_results_path: Path,
    reproduction: dict[str, Any],
    calibrator: dict[str, Any],
    runtime: dict[str, str],
) -> dict[str, Any]:
    """Versions, artifact, input contract, reproduction and calibrator records, inputs and environment."""
    _, stats = evaluate.load_manifest(data_dir)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "status": "development",
        "cohort": stats["cohort"],
        "split_version": stats.get("split_version"),
        "dataset_version": stats["dataset_version"],
        "feature_set_version": feature_config["feature_set_version"],
        "development_version": development.load_development_config()["development_version"],
        "calibration_version": calibrator["calibration_version"],
        "params": logistic_config["model"],
        "fit_rows": {"role": "model_fit", "rows": EXPECTED_ROWS["model_fit"]},
        "artifact": {
            "path": provenance.display_path(artifact_path),
            "sha256": artifact_sha256,
            "format": "joblib",
            "contents": "fitted sklearn Pipeline: preprocess (impute, scale, one-hot) + LogisticRegression",
            "gitignored": True,
            "load_with": {"scikit_learn": runtime["scikit_learn"], "joblib": runtime["joblib"]},
            "verify_before_loading": "joblib.load can run code: check this sha256 first",
            "rebuild": "python -m src.tabular.export_tabular_pipeline --raw-csv data/raw/loan.csv",
        },
        "input_contract": input_contract(feature_config),
        "output": {
            "raw": "predict_proba(X)[:, 1]: the raw probability; the stacker's tabular input",
            "calibrated": "reference_calibrator applied to the raw probability: the tabular-only fallback PD",
        },
        "reproduction": {
            "rtol": REPRODUCTION_RTOL,
            "atol": REPRODUCTION_ATOL,
            "sets": reproduction,
        },
        "reference_calibrator": calibrator,
        "inputs_sha256": {
            **input_checksums(data_dir, raw_csv_path, raw_csv_sha256),
            **provenance.file_checksums([
                calibration_config_path, handoff_config_path, reference_metrics_path, reference_results_path,
            ]),
        },
        "test_rows_available_to_model": count_test_rows_available(data_dir),
        "commands": [
            "python -m pip install -r src/tabular/requirements.txt",
            "python -m src.tabular.export_tabular_pipeline --raw-csv data/raw/loan.csv",
            "python -m pytest src/tabular/test_export_tabular_pipeline.py -v -s",
        ],
        "environment": runtime,
    }


# --------------------------------------------------------------------------- run


def run_export(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    raw_csv_path: str | Path = DEFAULT_RAW_CSV,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    prediction_dir: str | Path = DEFAULT_PREDICTION_DIR,
    handoff_config_path: str | Path = DEFAULT_HANDOFF_CONFIG,
    reference_metrics_path: str | Path = REFERENCE_METRICS,
    reference_results_path: str | Path = REFERENCE_RESULTS,
    calibration_config_path: str | Path = calibrate.DEFAULT_CALIBRATION_CONFIG,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Fit, prove reproduction and the reference calibrator, then save the artifact and write the manifest.

    Every check runs before the artifact is written, and the manifest is
    written last, so a failed check leaves neither behind. ``root`` resolves
    the handoff config's repository-relative paths.
    """
    data_dir, raw_csv_path = Path(data_dir), Path(raw_csv_path)
    artifact_path, manifest_path = Path(artifact_path), Path(manifest_path)
    runtime = validate_runtime()
    raw_csv_sha256 = provenance.require_locked_raw_csv(data_dir, raw_csv_path)
    feature_config = preprocess.load_feature_config()
    logistic_config = logistic_baseline.load_logistic_config()
    handoff_config = load_handoff_config(handoff_config_path)
    files = committed_files(handoff_config, root)

    (fit_features, fit_target), scoring = load_rows(data_dir, raw_csv_path)
    pipeline = fit_reference_pipeline(fit_features, fit_target, feature_config, logistic_config)
    reproduction = check_reproduction(
        predict_frames(pipeline, scoring), files, handoff_config["sources"]["tabular"]["probability_column"],
    )
    calibrator = check_reference_calibrator(
        data_dir, prediction_dir, reference_metrics_path, reference_results_path, calibration_config_path,
    )

    artifact_sha256 = save_artifact(pipeline, artifact_path, scoring)
    manifest = build_manifest(
        data_dir=data_dir, raw_csv_path=raw_csv_path, raw_csv_sha256=raw_csv_sha256,
        artifact_path=artifact_path, artifact_sha256=artifact_sha256,
        feature_config=feature_config, logistic_config=logistic_config,
        calibration_config_path=Path(calibration_config_path), handoff_config_path=Path(handoff_config_path),
        reference_metrics_path=Path(reference_metrics_path), reference_results_path=Path(reference_results_path),
        reproduction=reproduction, calibrator=calibrator, runtime=runtime,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit, check and save the balanced v1 tabular pipeline.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--raw-csv", default=DEFAULT_RAW_CSV, type=Path)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT_PATH, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, type=Path)
    args = parser.parse_args()
    manifest = run_export(args.data_dir, args.raw_csv, args.artifact, args.manifest)
    for split, record in manifest["reproduction"]["sets"].items():
        print(f"{split}: rows={record['rows']} max_abs_difference={record['max_abs_difference']:.3e}")
    calibrator = manifest["reference_calibrator"]
    print(f"platt slope={calibrator['slope']:.6f} intercept={calibrator['intercept']:.6f}")
    print(f"artifact -> {manifest['artifact']['path']} (sha256 {manifest['artifact']['sha256']})")


if __name__ == "__main__":
    main()

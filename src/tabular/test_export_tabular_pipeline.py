"""Saved tabular pipeline tests: the real-data reproduction, plus the guards that stop a wrong save.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_export_tabular_pipeline.py -v -s

The real export test needs the restricted raw CSV and is skipped without it:
    $env:LENDINGCLUB_RAW_CSV = "data/raw/loan.csv"

The real export test and the committed-manifest test also need
reports/tabular/tabular_pipeline_v1_manifest.json, written by
``python -m src.tabular.export_tabular_pipeline --raw-csv data/raw/loan.csv``.
The artifact itself is gitignored, so no test reads the committed .joblib.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nlp.preprocess import sha256_file
from src.tabular import calibrate, evaluate, logistic_baseline, preprocess, provenance
from src.tabular.export_tabular_pipeline import (
    DEFAULT_MANIFEST_PATH,
    ERROR_MESSAGES,
    MODEL_NAME,
    MODEL_VERSION,
    REPRODUCTION_ATOL,
    REPRODUCTION_RTOL,
    check_reproduction,
    fit_reference_pipeline,
    run_export,
)
from src.tabular.unweighted_comparison import REFERENCE_METRICS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPOSITORY_ROOT / "data"
# Hardcoded on purpose, so a silent change to the contract fails a test instead of being echoed back.
REAL_ROWS = {"calibration": 26_940, "validation": 21_784}


def error_pattern(key: str) -> str:
    """Regex for the ERROR_MESSAGES[key] text anywhere in a message, any value in each {placeholder}."""
    return ".*".join(re.escape(part) for part in re.split(r"\{[^}]*\}", ERROR_MESSAGES[key]))


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def committed_manifest() -> dict:
    return json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- real data


@pytest.mark.skipif(not os.environ.get("LENDINGCLUB_RAW_CSV"), reason="LENDINGCLUB_RAW_CSV is not set")
def test_real_export_reproduces_the_committed_scores_and_manifest(tmp_path: Path) -> None:
    """Real data: a fresh export reproduces all 48,724 committed tabular scores within 1e-10, the Platt
    numbers reproduce, and the committed manifest's records equal the fresh run's."""
    raw_csv = Path(os.environ["LENDINGCLUB_RAW_CSV"])
    fresh = run_export(
        REAL_DATA_DIR, raw_csv, tmp_path / "pipeline.joblib", tmp_path / "manifest.json",
    )
    assert (tmp_path / "pipeline.joblib").exists()
    assert sha256_file(tmp_path / "pipeline.joblib") == fresh["artifact"]["sha256"]
    for split, rows in REAL_ROWS.items():
        record = fresh["reproduction"]["sets"][split]
        assert record["rows"] == rows
        assert record["max_abs_difference"] <= REPRODUCTION_ATOL

    committed = committed_manifest()
    for key in ("pipeline_version", "model_name", "model_version", "params", "fit_rows",
                "input_contract", "test_rows_available_to_model"):
        assert committed[key] == fresh[key], key
    committed_calibrator = dict(committed["reference_calibrator"])
    fresh_calibrator = dict(fresh["reference_calibrator"])
    recorded_hash = committed_calibrator.pop("calibration_metrics_sha256")
    current_hash = fresh_calibrator.pop("calibration_metrics_sha256")
    assert committed_calibrator == fresh_calibrator
    provenance.verify_file_checksum(REFERENCE_METRICS, current_hash)
    provenance.verify_file_checksum(REFERENCE_METRICS, recorded_hash, allow_crlf=True)
    for split, rows in REAL_ROWS.items():
        assert committed["reproduction"]["sets"][split]["rows"] == rows
        assert committed["reproduction"]["sets"][split]["max_abs_difference"] <= REPRODUCTION_ATOL
    provenance.verify_reproduced_inputs(
        committed["inputs_sha256"], fresh["inputs_sha256"], raw_csv_path=raw_csv, data_dir=REAL_DATA_DIR,
    )


# --------------------------------------------------------------------------- guards


def _write_committed(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Write each frame as a committed CSV and pin it, in the shape committed_files returns."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, frame in frames.items():
        path = tmp_path / f"logistic_regression_{split}.csv"
        frame.to_csv(path, index=False)
        files[split] = {"path": path.name, "sha256": sha256_file(path), "resolved": path}
    return files


def test_a_score_that_differs_from_the_committed_file_is_refused(tmp_path: Path) -> None:
    """Identical committed scores pass with a zero difference; one committed probability off by 1e-8 is
    refused, so a refit that is not the committed model can never be saved."""
    frames = {
        split: evaluate.build_prediction_frame(
            loan_ids=pd.Index([f"lc_{split[0]}{number:08d}" for number in range(1, 5)]), split_name=split,
            probabilities=np.array([0.31, 0.47, 0.52, 0.66]), model_name=MODEL_NAME, model_version=MODEL_VERSION,
        )
        for split in REAL_ROWS
    }
    record = check_reproduction(frames, _write_committed(tmp_path / "same", frames), "p_default_tabular")
    assert {split: entry["max_abs_difference"] for split, entry in record.items()} == {
        "calibration": 0.0, "validation": 0.0,
    }

    tampered = copy.deepcopy(frames)
    tampered["validation"].loc[0, "p_default_tabular"] += 1e-8
    files = _write_committed(tmp_path / "tampered", tampered)
    with pytest.raises(ValueError, match=error_pattern("not_reproduced")):
        check_reproduction(frames, files, "p_default_tabular")


def test_only_the_balanced_logistic_config_is_accepted() -> None:
    """The pipeline saved for fusion must be the balanced v1 LR: any other class_weight is refused before
    anything is fitted."""
    config = copy.deepcopy(logistic_baseline.load_logistic_config())
    config["model"]["class_weight"] = None
    with pytest.raises(ValueError, match=error_pattern("not_balanced")):
        fit_reference_pipeline(pd.DataFrame(), pd.Series(dtype="int8"), preprocess.load_feature_config(), config)


def test_committed_manifest_matches_the_feature_contract_and_reference_calibrator() -> None:
    """The committed manifest documents the real pipeline input (the nine tabular_features_v1 columns in
    pipeline order) and the approved Platt numbers from calibration_metrics.json."""
    manifest = committed_manifest()
    feature_config = preprocess.load_feature_config()
    contract = manifest["input_contract"]
    assert contract["feature_set_version"] == feature_config["feature_set_version"]
    assert [column["name"] for column in contract["columns"]] == preprocess.feature_columns(feature_config)
    categorical = set(feature_config["model"]["categorical"])
    assert all((column["kind"] == "categorical") == (column["name"] in categorical) for column in contract["columns"])

    recorded = json.loads(REFERENCE_METRICS.read_text(encoding="utf-8"))
    platt = recorded["candidates"]["logistic_regression"]["calibrators"]["platt"]
    calibrator = manifest["reference_calibrator"]
    assert (calibrator["slope"], calibrator["intercept"]) == pytest.approx((platt["slope"], platt["intercept"]), abs=1e-6)
    assert calibrator["probability_clip"] == calibrate.load_calibration_config()["probability_clip"]
    assert manifest["params"]["class_weight"] == "balanced"
    assert manifest["model_version"] == MODEL_VERSION
    assert manifest["reproduction"]["rtol"] == REPRODUCTION_RTOL

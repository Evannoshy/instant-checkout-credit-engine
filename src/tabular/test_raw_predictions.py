"""Regression tests for the raw candidate predictions handed to calibration.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_raw_predictions.py -v -s

Uses synthetic frames (no borrower data) and the REAL frozen configs. The
committed-file check runs without the raw CSV. The refit check needs
LENDINGCLUB_RAW_CSV.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tabular import development, evaluate, logistic_baseline, preprocess, raw_predictions, xgboost_baseline
from src.tabular.test_xgboost_baseline import make_synthetic

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
XGBOOST_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_xgboost_v1.toml"
REAL_PREDICTION_DIR = REPOSITORY_ROOT / "reports" / "tabular" / "raw_predictions"


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def shortlist(selected_params: dict | None = None, model_version: str | None = None):
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    params = selected_params or {**config["model"], "max_depth": 2}
    selection = {
        "model_version": model_version or config["model_version"],
        "selected_trial_id": 1,
        "selected_params": params,
    }
    return raw_predictions.build_shortlist(
        preprocess.load_feature_config(), config, logistic_baseline.load_logistic_config(), selection
    )


def split_rows():
    """Synthetic loans cut into disjoint fit / calibration / validation sets."""
    features, target, _ = make_synthetic(rows=900)
    parts = {"fit": slice(0, 500), "calibration": slice(500, 700), "validation": slice(700, 900)}
    return {name: (features.iloc[rows], target.iloc[rows]) for name, rows in parts.items()}


def test_shortlist_has_lr_default_and_selected_xgboost() -> None:
    """The shortlist is LR, the default XGBoost and the selected trial when it differs."""
    names = [candidate.name for candidate in shortlist()]
    assert names == ["logistic_regression", "xgboost_default", "xgboost_trial1"]


def test_selected_trial_equal_to_default_is_not_duplicated() -> None:
    """If the search selects the default, it is shortlisted once."""
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    names = [candidate.name for candidate in shortlist(selected_params=dict(config["model"]))]
    assert names == ["logistic_regression", "xgboost_default"]


def test_selection_from_another_model_version_is_refused() -> None:
    """A selection file written for a different config version cannot be used."""
    with pytest.raises(ValueError, match="rerun the search"):
        shortlist(model_version="tabular-xgboost-v0")


def test_every_requested_loan_gets_a_valid_probability_in_the_shared_schema() -> None:
    """Each candidate covers every calibration and validation loan once, within [0, 1]."""
    rows = split_rows()
    fit_features, fit_target = rows.pop("fit")
    frames, summaries = raw_predictions.predict_candidates(shortlist(), fit_features, fit_target, rows)

    assert len(frames) == 6
    for key, frame in frames.items():
        set_name = key.rsplit("_", 1)[1]
        expected = set(rows[set_name][0].index)
        evaluate.validate_prediction_frame(frame, expected_loan_ids=expected)
        assert frame.columns.tolist() == list(evaluate.PREDICTION_COLUMNS)
        assert frame["split"].eq(set_name).all()
        assert summaries[key]["rows"] == len(expected)
        assert 0 <= summaries[key]["min_probability"] <= summaries[key]["max_probability"] <= 1


def test_overlapping_fit_and_scoring_rows_are_refused() -> None:
    """Calibration rows that also appear in the fit rows raise before any model is fitted."""
    rows = split_rows()
    fit_features, fit_target = rows.pop("fit")
    leaked = {**rows, "calibration": (fit_features.iloc[:50], fit_target.iloc[:50])}
    with pytest.raises(ValueError, match="model_fit and calibration share 50"):
        raw_predictions.predict_candidates(shortlist(), fit_features, fit_target, leaked)


def test_real_prediction_files_cover_every_calibration_and_validation_loan() -> None:
    """The committed files hold 26,940 calibration and 21,784 validation loans per candidate."""
    manifest = json.loads((REAL_PREDICTION_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_rows_available_to_model"] == 0
    assert manifest["scoring_rows"] == {"calibration": 26940, "validation": 21784}
    development_manifest, _ = evaluate.load_manifest(REPOSITORY_ROOT / "data")
    roles = development.assign_development_roles(REPOSITORY_ROOT / "data")
    expected = {
        "calibration": set(roles.index[roles["role"].eq("calibration")]),
        "validation": set(development_manifest.loc[
            development_manifest["split"].eq("validation"), "loan_id"
        ]),
    }
    target = development_manifest.set_index("loan_id")["target"]
    for name, summary in manifest["files"].items():
        frame = pd.read_csv(REAL_PREDICTION_DIR / name, dtype={"loan_id": str})
        split = name.removesuffix(".csv").rsplit("_", 1)[1]
        evaluate.validate_prediction_frame(frame, expected[split])
        assert frame["split"].eq(split).all()
        assert frame["model_version"].eq(summary["model_version"]).all()
        assert len(frame) == summary["expected_rows"] and frame["loan_id"].is_unique
        values = frame["p_default_tabular"].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        metrics = evaluate.evaluate_predictions(target.loc[frame["loan_id"]].to_numpy(), values)
        for metric, value in summary["raw_metrics"].items():
            assert metrics[metric] == pytest.approx(value, rel=1e-8, abs=1e-10)


@pytest.mark.skipif(
    not os.environ.get("LENDINGCLUB_RAW_CSV"), reason="LENDINGCLUB_RAW_CSV is not set"
)
def test_real_candidate_refit_reproduces_committed_predictions(tmp_path: Path) -> None:
    """A refit in the recorded environment reproduces every committed handoff probability."""
    rerun = raw_predictions.run_raw_predictions(
        REPOSITORY_ROOT / "data", Path(os.environ["LENDINGCLUB_RAW_CSV"]), tmp_path
    )
    saved = json.loads((REAL_PREDICTION_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert rerun["environment"] == saved["environment"]
    assert set(rerun["files"]) == set(saved["files"])
    for name in saved["files"]:
        expected = pd.read_csv(REAL_PREDICTION_DIR / name).set_index("loan_id").sort_index()
        actual = pd.read_csv(tmp_path / name).set_index("loan_id").sort_index()
        assert actual.index.equals(expected.index)
        assert actual["model_version"].equals(expected["model_version"])
        np.testing.assert_allclose(
            actual["p_default_tabular"], expected["p_default_tabular"], rtol=1e-10, atol=1e-10
        )

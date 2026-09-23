"""Regression tests for the tabular evaluation framework.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_evaluate.py -v -s

The -s flag shows print statements; -v shows each PASSED/FAILED result. Most
checks use a tiny, synthetic four-row manifest (no borrower data). One check
at the end reads the real, tracked data/split_manifest.csv and
data/split_statistics.json to confirm the loader and default-rate figure
against the frozen cohort; it never opens data/raw/loan.csv, which is
gitignored and may not exist on a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tabular.evaluate import (
    PREDICTION_COLUMNS,
    TEST_SPLIT_LOCKED_MESSAGE,
    build_prediction_frame,
    evaluate_predictions,
    get_split_targets,
    load_manifest,
    training_default_rate,
    validate_prediction_frame,
)

REAL_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def artificial_data_dir(tmp_path):
    """Four made-up rows spanning train/validation/test, no borrower data."""
    manifest = pd.DataFrame(
        {
            "loan_id": ["lc_000000001", "lc_000000002", "lc_000000003", "lc_000000004"],
            "source_row_number": [1, 2, 3, 4],
            "split": ["train", "train", "validation", "test"],
            "issue_month": ["2013-01", "2013-02", "2013-08", "2014-01"],
            "target": [0, 1, 0, "LOCKED_TEST_LABEL"],
            "text_available": [1, 1, 1, 1],
            "cohort": ["real_text_matured_v1"] * 4,
            "dataset_version": ["kaggle-adarshsng-local-2026-09-09"] * 4,
        }
    )
    manifest.to_csv(tmp_path / "split_manifest.csv", index=False)
    statistics = {
        "dataset_version": "kaggle-adarshsng-local-2026-09-09",
        "split_version": "split-v1",
        "cohort": "real_text_matured_v1",
        "eligible_rows": 4,
        "splits": {
            "train": {"rows": 2, "defaults": 1, "default_rate": 0.5},
            "validation": {"rows": 1, "defaults": 0, "default_rate": 0.0},
            "test": {"rows": 1, "defaults": 1, "default_rate": 1.0},
        },
    }
    (tmp_path / "split_statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    return tmp_path


# --- load_manifest ---


def test_load_manifest_accepts_a_well_formed_manifest(artificial_data_dir):
    """A well-formed manifest returns development rows and redacted stats."""
    manifest, stats = load_manifest(artificial_data_dir)
    assert len(manifest) == 3
    assert set(manifest["split"]) == {"train", "validation"}
    assert stats["splits"]["test"] == {"rows": 1}
    assert stats["split_version"] == "split-v1"


def test_load_manifest_does_not_expose_or_validate_test_targets(artificial_data_dir):
    """An intentionally non-binary test sentinel is never returned or validated."""
    manifest, stats = load_manifest(artificial_data_dir)
    assert "test" not in set(manifest["split"])
    assert "LOCKED_TEST_LABEL" not in set(manifest["target"].astype(str))
    assert set(stats["splits"]["test"]) == {"rows"}


@pytest.mark.parametrize("column", sorted({"loan_id", "split", "target", "cohort", "dataset_version"}))
def test_load_manifest_rejects_missing_required_columns(artificial_data_dir, column):
    """A manifest missing any required column is rejected."""
    path = artificial_data_dir / "split_manifest.csv"
    frame = pd.read_csv(path)
    frame.drop(columns=[column]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_manifest(artificial_data_dir)


def test_load_manifest_rejects_unknown_split_values(artificial_data_dir):
    """An unrecognized split label is rejected."""
    path = artificial_data_dir / "split_manifest.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "split"] = "holdout"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_manifest(artificial_data_dir)


def test_load_manifest_rejects_duplicate_loan_id(artificial_data_dir):
    """Duplicate loan_id values are rejected."""
    path = artificial_data_dir / "split_manifest.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "loan_id"] = frame.loc[0, "loan_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_manifest(artificial_data_dir)


def test_load_manifest_rejects_loan_id_source_row_mismatch(artificial_data_dir):
    """A loan_id that disagrees with its source_row_number is rejected."""
    path = artificial_data_dir / "split_manifest.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "loan_id"] = "lc_000000099"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_manifest(artificial_data_dir)


def test_load_manifest_rejects_non_binary_target(artificial_data_dir):
    """A target value outside {0, 1} is rejected."""
    path = artificial_data_dir / "split_manifest.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "target"] = '2'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_manifest(artificial_data_dir)


def test_load_manifest_rejects_row_count_mismatch_with_stats(artificial_data_dir):
    """A manifest row count disagreeing with eligible_rows is rejected."""
    stats_path = artificial_data_dir / "split_statistics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["eligible_rows"] = 999
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(artificial_data_dir)


def test_load_manifest_rejects_wrong_cohort(artificial_data_dir):
    """A cohort other than real_text_matured_v1 is rejected."""
    path = artificial_data_dir / "split_manifest.csv"
    frame = pd.read_csv(path)
    frame["cohort"] = "some_other_cohort"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_manifest(artificial_data_dir)


def test_load_manifest_rejects_dataset_version_mismatch(artificial_data_dir):
    """A manifest dataset_version disagreeing with split_statistics.json is rejected."""
    path = artificial_data_dir / "split_manifest.csv"
    frame = pd.read_csv(path)
    frame["dataset_version"] = "some-other-version"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_manifest(artificial_data_dir)


# --- get_split_targets ---


@pytest.mark.parametrize("split", ["train", "validation"])
def test_get_split_targets_returns_loan_id_and_target(artificial_data_dir, split):
    """train and validation each return their own loan_id/target rows."""
    manifest, _ = load_manifest(artificial_data_dir)
    targets = get_split_targets(manifest, split)
    assert set(targets.columns) == {"loan_id", "target"}
    assert (targets["loan_id"].str.len() > 0).all()


def test_get_split_targets_rejects_test_split_with_the_shared_lock_message(artificial_data_dir):
    """The test split is refused with the same message used across the repo."""
    manifest, _ = load_manifest(artificial_data_dir)
    with pytest.raises(ValueError, match=TEST_SPLIT_LOCKED_MESSAGE):
        get_split_targets(manifest, "test")


def test_get_split_targets_rejects_unknown_split_name(artificial_data_dir):
    """An unrecognized split name is rejected."""
    manifest, _ = load_manifest(artificial_data_dir)
    with pytest.raises(ValueError):
        get_split_targets(manifest, "holdout")


# --- training_default_rate ---


def test_training_default_rate_matches_manual_mean_on_synthetic_fixture(artificial_data_dir):
    """The training default rate equals the manual mean of train targets."""
    manifest, _ = load_manifest(artificial_data_dir)
    assert training_default_rate(manifest) == pytest.approx(0.5)


# --- build_prediction_frame / validate_prediction_frame ---


def test_build_prediction_frame_has_exact_column_set_and_order():
    """The prediction frame always has PREDICTION_COLUMNS in that exact order."""
    frame = build_prediction_frame(
        loan_ids=["lc_000000001", "lc_000000002"],
        split_name="validation",
        probabilities=np.array([0.2, 0.2]),
        model_name="constant_base_rate",
        model_version="tabular-baseline-constant-v1",
    )
    assert frame.columns.tolist() == list(PREDICTION_COLUMNS)
    assert frame["split"].tolist() == ["validation", "validation"]


def _sample_frame() -> pd.DataFrame:
    return build_prediction_frame(
        loan_ids=["lc_000000001", "lc_000000002"],
        split_name="validation",
        probabilities=np.array([0.2, 0.2]),
        model_name="constant_base_rate",
        model_version="tabular-baseline-constant-v1",
    )


def test_validate_prediction_frame_accepts_a_well_formed_frame():
    """A well-formed frame with matching coverage passes without error."""
    frame = _sample_frame()
    validate_prediction_frame(frame, expected_loan_ids={"lc_000000001", "lc_000000002"})


def test_validate_prediction_frame_rejects_wrong_column_order():
    """A frame with the wrong column order is rejected."""
    frame = _sample_frame()[list(reversed(PREDICTION_COLUMNS))]
    with pytest.raises(ValueError):
        validate_prediction_frame(frame, expected_loan_ids={"lc_000000001", "lc_000000002"})


def test_validate_prediction_frame_rejects_duplicate_loan_ids():
    """Duplicate loan_id rows are rejected."""
    frame = _sample_frame()
    frame.loc[1, "loan_id"] = frame.loc[0, "loan_id"]
    with pytest.raises(ValueError):
        validate_prediction_frame(frame, expected_loan_ids={"lc_000000001"})


def test_validate_prediction_frame_rejects_missing_id_coverage():
    """A prediction frame missing an expected loan_id is rejected."""
    frame = _sample_frame()
    with pytest.raises(ValueError):
        validate_prediction_frame(
            frame, expected_loan_ids={"lc_000000001", "lc_000000002", "lc_000000003"}
        )


def test_validate_prediction_frame_rejects_extra_id_coverage():
    """A prediction frame with an unexpected extra loan_id is rejected."""
    frame = _sample_frame()
    with pytest.raises(ValueError):
        validate_prediction_frame(frame, expected_loan_ids={"lc_000000001"})


def test_validate_prediction_frame_rejects_missing_probabilities():
    """A null p_default_tabular value is rejected."""
    frame = _sample_frame()
    frame.loc[0, "p_default_tabular"] = np.nan
    with pytest.raises(ValueError):
        validate_prediction_frame(frame, expected_loan_ids={"lc_000000001", "lc_000000002"})


def test_validate_prediction_frame_rejects_non_numeric_probabilities():
    """A non-numeric p_default_tabular column is rejected."""
    frame = _sample_frame()
    frame["p_default_tabular"] = frame["p_default_tabular"].astype(object)
    frame.loc[0, "p_default_tabular"] = "high"
    with pytest.raises(ValueError):
        validate_prediction_frame(frame, expected_loan_ids={"lc_000000001", "lc_000000002"})


@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_validate_prediction_frame_rejects_probabilities_outside_unit_interval(bad_value):
    """A probability below 0 or above 1 is rejected."""
    frame = _sample_frame()
    frame.loc[0, "p_default_tabular"] = bad_value
    with pytest.raises(ValueError):
        validate_prediction_frame(frame, expected_loan_ids={"lc_000000001", "lc_000000002"})


# --- evaluate_predictions ---


def test_evaluate_predictions_reports_perfect_ranking_as_one():
    """A perfectly separable score gives ROC-AUC and PR-AUC of 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_evaluate_predictions_constant_score_hits_exact_baselines():
    """A constant score gives chance-level ROC-AUC and prevalence-level PR-AUC exactly."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.full(10, 0.3)
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["roc_auc"] == 0.5
    assert metrics["roc_auc_baseline"] == 0.5
    assert metrics["pr_auc"] == pytest.approx(metrics["prevalence"])
    assert metrics["pr_auc_baseline"] == pytest.approx(metrics["prevalence"])


def test_evaluate_predictions_constant_score_f1_best_threshold_equals_the_constant_value():
    """For a constant score, the best F1 threshold degenerates to that same constant value."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    y_pred = np.full(10, 0.3)
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["f1_best_threshold"] == pytest.approx(0.3)


def test_evaluate_predictions_rejects_single_class_y_true():
    """y_true with only one class raises instead of silently returning NaN."""
    with pytest.raises(ValueError):
        evaluate_predictions(np.zeros(5), np.full(5, 0.2))


def test_evaluate_predictions_returns_exactly_the_specified_metric_keys():
    """The metrics dict exposes exactly the approved core-metric keys, no more, no fewer."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = evaluate_predictions(y_true, y_pred)
    assert set(metrics.keys()) == {
        "rows", "positives", "prevalence", "roc_auc", "roc_auc_baseline",
        "pr_auc", "pr_auc_baseline", "brier_score", "log_loss",
        "f1_at_0.5", "f1_best", "f1_best_threshold",
    }


# --- real-data regression check ---


def test_load_manifest_on_real_data_matches_split_statistics():
    """The real, tracked manifest loads and its training default rate matches the frozen cohort."""
    manifest, stats = load_manifest(REAL_DATA_DIR)
    assert len(manifest) == 86293 + 21784
    assert stats["eligible_rows"] == 122999
    assert set(manifest["split"]) == {"train", "validation"}
    assert set(stats["splits"]["test"]) == {"rows"}
    assert training_default_rate(manifest) == pytest.approx(13197 / 86293)

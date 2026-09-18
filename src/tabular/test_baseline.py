"""Regression tests for the constant base-rate baseline.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_baseline.py -v -s

Most checks use a tiny, synthetic manifest written under tmp_path (no
borrower data). The final check runs the real pipeline end to end against
the real, tracked data/ directory, writing output to tmp_path; it never
opens data/raw/loan.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.tabular import baseline, evaluate

REAL_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def artificial_data_dir(tmp_path):
    """Eight made-up rows spanning train/validation/test, no borrower data.

    Validation deliberately includes both classes (one default, one
    non-default) so evaluate_predictions' ranking metrics are well-defined.
    """
    manifest = pd.DataFrame(
        {
            "loan_id": [
                "lc_000000001", "lc_000000002", "lc_000000003", "lc_000000004",
                "lc_000000005", "lc_000000006", "lc_000000007", "lc_000000008",
            ],
            "source_row_number": [1, 2, 3, 4, 5, 6, 7, 8],
            "split": [
                "train", "train", "train", "train",
                "validation", "validation", "test", "test",
            ],
            "issue_month": [
                "2013-01", "2013-02", "2013-03", "2013-04",
                "2013-08", "2013-08", "2014-01", "2014-01",
            ],
            "target": [0, 1, 0, 0, 1, 0, 1, 0],
            "text_available": [1, 1, 1, 1, 1, 1, 1, 1],
            "cohort": ["real_text_matured_v1"] * 8,
            "dataset_version": ["kaggle-adarshsng-local-2026-09-09"] * 8,
        }
    )
    manifest.to_csv(tmp_path / "split_manifest.csv", index=False)
    statistics = {
        "dataset_version": "kaggle-adarshsng-local-2026-09-09",
        "split_version": "split-v1",
        "cohort": "real_text_matured_v1",
        "eligible_rows": 8,
        "splits": {
            "train": {"rows": 4, "defaults": 1, "default_rate": 0.25},
            "validation": {"rows": 2, "defaults": 1, "default_rate": 0.5},
            "test": {"rows": 2, "defaults": 1, "default_rate": 0.5},
        },
    }
    (tmp_path / "split_statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    return tmp_path


def test_run_writes_predictions_csv_with_exact_schema(artificial_data_dir, tmp_path):
    """run() writes a predictions CSV with exactly the agreed column set and order."""
    output_dir = tmp_path / "out"
    result = baseline.run(artificial_data_dir, output_dir)
    assert result["predictions"].columns.tolist() == list(evaluate.PREDICTION_COLUMNS)
    on_disk = pd.read_csv(output_dir / "constant_baseline_predictions.csv")
    assert on_disk.columns.tolist() == list(evaluate.PREDICTION_COLUMNS)


def test_run_predictions_are_all_equal_to_training_default_rate(artificial_data_dir, tmp_path):
    """Every validation prediction equals the training-set default rate exactly."""
    result = baseline.run(artificial_data_dir, tmp_path / "out")
    assert (result["predictions"]["p_default_tabular"] == 0.25).all()
    assert result["metrics"]["train"]["default_rate"] == pytest.approx(0.25)


def test_run_predictions_cover_every_validation_loan_id_exactly_once(artificial_data_dir, tmp_path):
    """The prediction frame has exactly one row per validation loan_id."""
    result = baseline.run(artificial_data_dir, tmp_path / "out")
    assert set(result["predictions"]["loan_id"]) == {"lc_000000005", "lc_000000006"}
    assert result["predictions"]["loan_id"].is_unique


def test_run_never_calls_get_split_targets_with_test_split(artificial_data_dir, tmp_path, monkeypatch):
    """run() never requests the locked test split from the manifest."""
    requested_splits: list[str] = []
    real_get_split_targets = evaluate.get_split_targets

    def spy(manifest, split):
        requested_splits.append(split)
        return real_get_split_targets(manifest, split)

    monkeypatch.setattr(baseline, "get_split_targets", spy)
    baseline.run(artificial_data_dir, tmp_path / "out")
    assert "test" not in requested_splits
    assert set(requested_splits) == {"train", "validation"}


def test_run_metrics_json_includes_test_set_not_accessed_confirmation(artificial_data_dir, tmp_path):
    """The written metrics JSON carries an explicit test_set_accessed: false field."""
    output_dir = tmp_path / "out"
    baseline.run(artificial_data_dir, output_dir)
    metrics = json.loads((output_dir / "constant_baseline_metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_set_accessed"] is False


def test_run_metrics_json_has_train_and_validation_sections_with_expected_keys(
    artificial_data_dir, tmp_path
):
    """The metrics JSON has train/validation sections with the expected keys."""
    result = baseline.run(artificial_data_dir, tmp_path / "out")
    metrics = result["metrics"]
    assert set(metrics["train"].keys()) == {"rows", "positives", "default_rate"}
    assert set(metrics["validation"].keys()) >= {"rows", "positives", "prevalence", "roc_auc", "pr_auc"}


def test_run_default_rate_can_differ_from_validation_prevalence(artificial_data_dir, tmp_path):
    """The predicted rate comes from train, not validation, and the two may differ."""
    result = baseline.run(artificial_data_dir, tmp_path / "out")
    metrics = result["metrics"]
    assert metrics["train"]["default_rate"] == pytest.approx(0.25)
    assert metrics["validation"]["prevalence"] == pytest.approx(0.5)


def test_run_is_idempotent_when_run_twice_on_the_same_synthetic_data(artificial_data_dir, tmp_path):
    """Running run() twice on unchanged input produces byte-identical output files."""
    output_dir = tmp_path / "out"
    baseline.run(artificial_data_dir, output_dir)
    first_predictions = (output_dir / "constant_baseline_predictions.csv").read_bytes()
    first_metrics = (output_dir / "constant_baseline_metrics.json").read_bytes()
    baseline.run(artificial_data_dir, output_dir)
    assert (output_dir / "constant_baseline_predictions.csv").read_bytes() == first_predictions
    assert (output_dir / "constant_baseline_metrics.json").read_bytes() == first_metrics


def test_run_on_real_data_produces_the_documented_training_default_rate(tmp_path):
    """Running the full pipeline on the real, tracked data/ dir matches the frozen figures."""
    result = baseline.run(REAL_DATA_DIR, tmp_path / "out")
    metrics = result["metrics"]
    assert metrics["train"]["default_rate"] == pytest.approx(13197 / 86293)
    assert metrics["validation"]["rows"] == 21784
    assert metrics["test_set_accessed"] is False
    assert len(result["predictions"]) == 21784

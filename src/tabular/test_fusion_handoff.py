"""Tests for the fusion handoff: the calibrated tabular validation export.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_fusion_handoff.py -v -s

The export reads only committed files (the split manifest and ID files, the raw
prediction CSVs and the merged calibration evidence), so every check here runs
on the real files and none needs LENDINGCLUB_RAW_CSV. The frame-validation check
uses a two-row fabricated frame (no borrower data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tabular import calibrate, fusion_handoff

COMMITTED_PATH = fusion_handoff.CALIBRATED_VAL_PATH


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture(scope="module")
def export() -> tuple[pd.DataFrame, dict]:
    """The calibrated frame and details, built once from the committed files."""
    return fusion_handoff.build_calibrated_validation()


def test_export_has_one_row_per_validation_loan_and_no_outcome_column(export) -> None:
    """21,784 rows, exactly the manifest's validation IDs, sorted, the agreed columns and no target."""
    frame, _ = export
    expected_ids, _ = calibrate.expected_prediction_ids()

    assert len(frame) == frame["loan_id"].nunique() == 21_784
    assert set(frame["loan_id"]) == expected_ids["validation"]
    assert frame["loan_id"].is_monotonic_increasing
    assert frame.columns.tolist() == list(fusion_handoff.CALIBRATED_COLUMNS)
    assert "target" not in frame.columns
    assert frame["split"].eq("validation").all()
    assert frame["model_version"].eq("tabular-logistic-v1-model-fit").all()
    assert frame["calibration_version"].eq("tabular_calibration_v1").all()


def test_platt_changes_the_level_but_not_the_ranking(export) -> None:
    """Raw and calibrated ROC-AUC and PR-AUC are equal (to 1e-12); only the mean probability moves."""
    _, details = export
    raw = details["validation_metrics"]["balanced_lr_raw"]
    platt = details["validation_metrics"]["balanced_lr_platt"]

    assert platt["roc_auc"] == pytest.approx(raw["roc_auc"], abs=1e-12)
    assert platt["pr_auc"] == pytest.approx(raw["pr_auc"], abs=1e-12)
    assert raw["mean_p"] > 0.4 > 0.2 > platt["mean_p"]


def _flip_labels(monkeypatch: pytest.MonkeyPatch, role: str) -> None:
    """Invert the outcomes of every loan in one set, and skip the reproduction check they would fail."""
    original = calibrate.expected_prediction_ids

    def flipped(*args, **kwargs):
        expected_ids, target = original(*args, **kwargs)
        ids = sorted(expected_ids[role])
        target = target.copy()
        target.loc[ids] = 1 - target.loc[ids]
        return expected_ids, target

    monkeypatch.setattr(calibrate, "expected_prediction_ids", flipped)
    monkeypatch.setattr(fusion_handoff, "check_reference_reproduction", lambda *args, **kwargs: None)


@pytest.mark.parametrize("role, changes", [("validation", False), ("calibration", True)])
def test_platt_mapping_is_fitted_on_calibration_labels_only(
    export, monkeypatch: pytest.MonkeyPatch, role: str, changes: bool
) -> None:
    """Flipping validation labels leaves the calibrated column unchanged; flipping calibration labels changes it."""
    committed, _ = export
    _flip_labels(monkeypatch, role)
    rebuilt, _ = fusion_handoff.build_calibrated_validation()

    np.testing.assert_array_equal(rebuilt["p_default_tabular_raw"], committed["p_default_tabular_raw"])
    same = np.array_equal(rebuilt["p_default_tabular_calibrated"], committed["p_default_tabular_calibrated"])
    assert same is not changes


def test_export_is_refused_when_the_merged_results_do_not_reproduce(tmp_path: Path) -> None:
    """A calibration_results.csv whose Platt validation Brier is off by 1e-5 (10x the tolerance) is refused."""
    results = pd.read_csv(fusion_handoff.REFERENCE_RESULTS)
    row = (
        results["candidate"].eq("logistic_regression")
        & results["method"].eq("platt")
        & results["set"].eq("validation")
    )
    results.loc[row, "brier_score"] += 1e-5
    results.to_csv(tmp_path / "calibration_results.csv", index=False)

    with pytest.raises(ValueError, match="brier_score does not reproduce"):
        fusion_handoff.build_calibrated_validation(reference_results_path=tmp_path / "calibration_results.csv")


def _small_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": ["lc_000000001", "lc_000000002"],
        "split": "validation",
        "p_default_tabular_raw": [0.4, 0.6],
        "p_default_tabular_calibrated": [0.1, 0.2],
        "model_name": "logistic_regression",
        "model_version": "tabular-logistic-v1-model-fit",
        "calibration_version": "tabular_calibration_v1",
    })


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("p_default_tabular_calibrated", np.nan, "missing p_default_tabular_calibrated"),
        ("p_default_tabular_raw", 1.5, "p_default_tabular_raw values must lie within"),
        ("split", "train", "only split 'validation'"),
    ],
)
def test_bad_calibrated_frames_are_rejected_naming_the_column(column: str, value, message: str) -> None:
    """A NaN, an out-of-range probability or a wrong split is rejected, and the error names the column."""
    frame = _small_frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        fusion_handoff.validate_calibrated_frame(frame, {"lc_000000001", "lc_000000002"})


def test_regenerated_export_reproduces_the_committed_file(tmp_path: Path) -> None:
    """Rerunning the export from the committed inputs reproduces the committed parquet
    (IDs and metadata exactly, probabilities to 1e-10)."""
    committed = pd.read_parquet(COMMITTED_PATH)
    result = fusion_handoff.run_calibrated_export(tmp_path / "tabular_calibrated_val.parquet")
    regenerated = pd.read_parquet(tmp_path / "tabular_calibrated_val.parquet")

    assert result["rows"] == len(committed) == 21_784
    assert regenerated.columns.tolist() == committed.columns.tolist()
    for column in ("loan_id", "split", "model_name", "model_version", "calibration_version"):
        assert regenerated[column].equals(committed[column]), column
    for column in fusion_handoff.CALIBRATED_PROBABILITY_COLUMNS:
        np.testing.assert_allclose(regenerated[column], committed[column], rtol=1e-10, atol=1e-10)

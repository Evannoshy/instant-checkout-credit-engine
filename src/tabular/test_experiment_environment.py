"""Checks for the runtime shared by the challenger experiments and handoff files."""

import copy
import json
from pathlib import Path

import pytest

from src.tabular import lexical_ablation, raw_predictions, xgboost_baseline, xgboost_search

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "tabular"
METADATA_FILES = (
    "xgboost_default_comparison.json", "xgboost_ablation_metadata.json",
    "xgboost_selection.json", "lexical_ablation_metadata.json", "raw_predictions/manifest.json",
)


def test_current_runtime_matches_repository_pins() -> None:
    assert xgboost_baseline.validate_runtime() == xgboost_baseline.environment()


@pytest.mark.parametrize("key", ["python", *xgboost_baseline.RUNTIME_PACKAGES.values()])
def test_mismatched_runtime_is_rejected(key: str) -> None:
    runtime = {**xgboost_baseline.environment(), key: "0.0.0"}
    with pytest.raises(ValueError, match="Experiment environment does not match"):
        xgboost_baseline.validate_environment(runtime)


def test_unreported_package_version_is_rejected() -> None:
    runtime = xgboost_baseline.environment()
    del runtime["scikit_learn"]
    with pytest.raises(ValueError, match="scikit-learn.*unreported"):
        xgboost_baseline.validate_environment(runtime)


@pytest.mark.parametrize("runner", [
    xgboost_baseline.run_default_comparison, xgboost_baseline.run_feature_ablation,
    xgboost_search.run_search, lexical_ablation.run_lexical_ablation,
    raw_predictions.run_raw_predictions,
])
def test_runners_reject_wrong_environment_before_reading_data(
    runner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = {**xgboost_baseline.environment(), "pandas": "0.0.0"}
    monkeypatch.setattr(xgboost_baseline, "environment", lambda: runtime)
    with pytest.raises(ValueError, match="pandas: expected"):
        runner(tmp_path / "missing-data", tmp_path / "missing.csv", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("filename", METADATA_FILES)
def test_saved_experiment_environments_match_repository_pins(filename: str) -> None:
    report = json.loads((REPORT_DIR / filename).read_text(encoding="utf-8"))
    xgboost_baseline.validate_environment(report["environment"])
    selection = json.loads((REPORT_DIR / "xgboost_selection.json").read_text(encoding="utf-8"))
    assert report["environment"] == selection["environment"]


def test_old_search_environment_cannot_be_used_for_prediction_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = {"environment": copy.deepcopy(xgboost_baseline.environment())}
    selection["environment"]["numpy"] = "0.0.0"
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(
        raw_predictions, "load_model_fit",
        lambda *args: pytest.fail("Data must not be read when the search environment differs"),
    )
    with pytest.raises(ValueError, match="Search selection used a different environment"):
        raw_predictions.run_raw_predictions(
            tmp_path, tmp_path / "missing.csv", tmp_path / "output", selection_path=path
        )
    assert not (tmp_path / "output").exists()

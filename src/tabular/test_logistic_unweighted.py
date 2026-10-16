"""Tests for the unweighted logistic experiment.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_logistic_unweighted.py -v -s

Most checks use synthetic loans (no borrower data) with the REAL logistic and
feature configs. The committed-file checks need only the committed prediction
files and the split manifest; the refit check also needs LENDINGCLUB_RAW_CSV.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tabular import calibrate, evaluate, logistic_baseline, logistic_unweighted, preprocess, provenance
from src.tabular.test_xgboost_baseline import make_synthetic

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPOSITORY_ROOT / "data"
COMMITTED_DIR = logistic_unweighted.DEFAULT_OUTPUT_DIR


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def split_rows() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Synthetic loans cut into disjoint model_fit / calibration / validation sets."""
    features, target, _ = make_synthetic(rows=900)
    parts = {"model_fit": slice(0, 500), "calibration": slice(500, 700), "validation": slice(700, 900)}
    return {name: (features.iloc[rows], target.iloc[rows]) for name, rows in parts.items()}


def frozen_v1_checksums() -> dict[str, str]:
    """SHA-256 of the v1 logistic config and every frozen v1 prediction file."""
    paths = [logistic_baseline.DEFAULT_LOGISTIC_CONFIG, *sorted(
        path for path in logistic_unweighted.FROZEN_PREDICTION_DIR.iterdir() if path.is_file()
    )]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.fixture
def fit_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record the rows and class_weight of every logistic pipeline fit."""
    fits: list[dict] = []
    real_build = logistic_baseline.build_pipeline

    def spying_build(feature_config, config):
        pipeline = real_build(feature_config, config)
        real_fit = pipeline.fit

        def fit(features, target):
            classifier = pipeline.named_steps["classify"]
            fits.append({"ids": set(features.index), "class_weight": classifier.get_params()["class_weight"]})
            return real_fit(features, target)

        pipeline.fit = fit
        return pipeline

    monkeypatch.setattr(logistic_baseline, "build_pipeline", spying_build)
    return fits


def test_copied_config_changes_only_class_weight_to_none() -> None:
    """The copy sets class_weight=None and keeps C, max_iter and random_state unchanged."""
    base = logistic_baseline.load_logistic_config()
    copied = logistic_unweighted.unweighted_config(base)
    assert copied["model"]["class_weight"] is None
    assert {k: v for k, v in copied["model"].items() if k != "class_weight"} == {
        k: v for k, v in base["model"].items() if k != "class_weight"
    }
    assert {k: v for k, v in copied.items() if k != "model"} == {k: v for k, v in base.items() if k != "model"}


def test_copying_the_config_leaves_the_v1_config_and_file_unchanged() -> None:
    """The loaded v1 dict keeps 'balanced' and the v1 TOML file bytes are untouched."""
    before = logistic_baseline.DEFAULT_LOGISTIC_CONFIG.read_bytes()
    base = logistic_baseline.load_logistic_config()
    logistic_unweighted.unweighted_config(base)
    assert base["model"]["class_weight"] == "balanced"
    assert logistic_baseline.DEFAULT_LOGISTIC_CONFIG.read_bytes() == before


def test_a_base_config_that_is_not_balanced_v1_is_refused() -> None:
    """Starting from anything but the balanced v1 config raises rather than silently changing two things."""
    base = logistic_baseline.load_logistic_config()
    with pytest.raises(ValueError, match="balanced v1 config"):
        logistic_unweighted.unweighted_config({**base, "model": {**base["model"], "class_weight": None}})


def test_candidate_builds_a_pipeline_with_class_weight_none() -> None:
    """build_pipeline receives class_weight=None; the candidate carries its own version."""
    candidate = logistic_unweighted.build_candidate(
        preprocess.load_feature_config(), logistic_baseline.load_logistic_config()
    )
    classifier = candidate.make_pipeline().named_steps["classify"]
    assert classifier.get_params()["class_weight"] is None
    assert (classifier.C, classifier.max_iter, classifier.random_state) == (1.0, 1000, 42)
    assert candidate.model_version == "tabular-logistic-unweighted-v1-model-fit"
    assert candidate.params["class_weight"] is None


def test_model_is_fitted_once_on_model_fit_rows_only(fit_spy: list[dict]) -> None:
    """Exactly one fit, on exactly the model_fit IDs, with class_weight=None."""
    rows = split_rows()
    fit_features, fit_target = rows.pop("model_fit")
    candidate = logistic_unweighted.build_candidate(
        preprocess.load_feature_config(), logistic_baseline.load_logistic_config()
    )
    logistic_unweighted.predict_unweighted(candidate, fit_features, fit_target, rows)
    assert fit_spy == [{"ids": set(fit_features.index), "class_weight": None}]


def test_every_calibration_and_validation_loan_gets_one_valid_probability() -> None:
    """One file per set, keyed by set name, covering every loan once with probabilities in [0, 1]."""
    rows = split_rows()
    fit_features, fit_target = rows.pop("model_fit")
    candidate = logistic_unweighted.build_candidate(
        preprocess.load_feature_config(), logistic_baseline.load_logistic_config()
    )
    frames, summaries = logistic_unweighted.predict_unweighted(candidate, fit_features, fit_target, rows)
    assert set(frames) == set(summaries) == {"calibration", "validation"}
    for set_name, frame in frames.items():
        evaluate.validate_prediction_frame(frame, expected_loan_ids=set(rows[set_name][0].index))
        assert frame["split"].eq(set_name).all()
        assert frame["model_version"].eq(logistic_unweighted.MODEL_VERSION).all()


def test_writing_into_the_frozen_v1_directory_is_refused(tmp_path: Path) -> None:
    """The experiment cannot overwrite the frozen v1 prediction files."""
    frames = {"validation": pd.DataFrame({"loan_id": ["a"]})}
    with pytest.raises(ValueError, match="frozen v1 predictions"):
        logistic_unweighted.write_predictions(frames, logistic_unweighted.FROZEN_PREDICTION_DIR)
    written = logistic_unweighted.write_predictions(frames, tmp_path / "unweighted")
    assert written == {"validation": tmp_path / "unweighted" / "validation.csv"}


def test_unexpected_row_counts_are_refused() -> None:
    """The run stops unless it fits 59,353 loans and scores 26,940 calibration and 21,784 validation loans."""
    def scoring(calibration: int, validation: int):
        return {"calibration": (None, range(calibration)), "validation": (None, range(validation))}

    logistic_unweighted.require_expected_rows(59_353, scoring(26_940, 21_784))
    for fit_rows, calibration, validation in (
        (86_293, 26_940, 21_784), (59_353, 26_939, 21_784), (59_353, 26_940, 21_785)
    ):
        with pytest.raises(ValueError, match="Expected rows"):
            logistic_unweighted.require_expected_rows(fit_rows, scoring(calibration, validation))


def test_runner_requests_only_model_fit_calibration_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fit_spy: list[dict]
) -> None:
    """The runner fits on the model_fit role, never asks for test rows, and leaves v1 files unchanged."""
    rows = split_rows()
    requested: list[str] = []

    def fake_model_fit(data_dir, raw_csv_path):
        requested.append("model_fit")
        features, target = rows["model_fit"]
        return features, target, None

    def fake_role(data_dir, raw_csv_path, role):
        requested.append(role)
        return rows[role]

    def fake_split(data_dir, raw_csv_path, split):
        requested.append(split)
        if split == "test":
            raise AssertionError("the locked test split was requested")
        return rows[split]

    monkeypatch.setattr(logistic_unweighted, "validate_runtime", lambda: {"python": "test"})
    monkeypatch.setattr(logistic_unweighted.provenance, "require_locked_raw_csv", lambda *_: "0" * 64)
    monkeypatch.setattr(logistic_unweighted, "load_model_fit", fake_model_fit)
    monkeypatch.setattr(logistic_unweighted.development, "load_tabular_role", fake_role)
    monkeypatch.setattr(logistic_unweighted.preprocess, "load_tabular_split", fake_split)
    monkeypatch.setattr(logistic_unweighted, "require_expected_rows", lambda *_: None)
    monkeypatch.setattr(logistic_unweighted, "input_checksums", lambda *_: {})

    before = frozen_v1_checksums()
    manifest = logistic_unweighted.run_unweighted(REAL_DATA_DIR, tmp_path / "loan.csv", tmp_path / "out")

    assert requested == ["model_fit", "calibration", "validation"]
    assert fit_spy == [{"ids": set(rows["model_fit"][0].index), "class_weight": None}]
    assert manifest["params"]["class_weight"] is None
    assert manifest["changed_setting"] == {"class_weight": {"from": "balanced", "to": None}}
    assert manifest["test_rows_available_to_model"] == 0
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "calibration.csv", "manifest.json", "validation.csv"
    ]
    assert frozen_v1_checksums() == before


def committed_outputs() -> dict:
    if not (COMMITTED_DIR / "manifest.json").exists():
        pytest.skip("The unweighted experiment has not been run yet")
    return json.loads((COMMITTED_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_committed_files_cover_every_calibration_and_validation_loan() -> None:
    """The committed files hold all 26,940 calibration and 21,784 validation loans, one row each."""
    manifest = committed_outputs()
    expected_ids, _ = calibrate.expected_prediction_ids(REAL_DATA_DIR)
    assert manifest["fit_rows"] == {"role": "model_fit", "rows": 59_353}
    assert manifest["scoring_rows"] == {"calibration": 26_940, "validation": 21_784}
    assert manifest["params"]["class_weight"] is None
    assert manifest["test_rows_available_to_model"] == 0
    for set_name, expected_rows in (("calibration", 26_940), ("validation", 21_784)):
        path = COMMITTED_DIR / f"{set_name}.csv"
        frame = pd.read_csv(path, dtype={"loan_id": str})
        assert len(expected_ids[set_name]) == expected_rows
        evaluate.validate_prediction_frame(frame, expected_ids[set_name])
        assert frame["split"].eq(set_name).all()
        assert frame["model_version"].eq(logistic_unweighted.MODEL_VERSION).all()
        provenance.verify_file_checksum(path, manifest["files"][path.name]["sha256"], allow_crlf=True)


@pytest.mark.skipif(
    not os.environ.get("LENDINGCLUB_RAW_CSV"), reason="LENDINGCLUB_RAW_CSV is not set"
)
def test_real_refit_reproduces_committed_unweighted_predictions(tmp_path: Path) -> None:
    """A refit on the checksum-verified CSV reproduces every committed probability and leaves v1 untouched."""
    saved = committed_outputs()
    before = frozen_v1_checksums()
    rerun = logistic_unweighted.run_unweighted(
        REAL_DATA_DIR, Path(os.environ["LENDINGCLUB_RAW_CSV"]), tmp_path
    )
    assert frozen_v1_checksums() == before
    provenance.verify_reproduced_inputs(
        saved["inputs_sha256"], rerun["inputs_sha256"],
        raw_csv_path=Path(os.environ["LENDINGCLUB_RAW_CSV"]), data_dir=REAL_DATA_DIR,
    )
    assert rerun["scoring_rows"] == saved["scoring_rows"]
    for name in ("calibration.csv", "validation.csv"):
        expected = pd.read_csv(COMMITTED_DIR / name, dtype={"loan_id": str})
        actual = pd.read_csv(tmp_path / name, dtype={"loan_id": str})
        assert actual["loan_id"].equals(expected["loan_id"])
        np.testing.assert_allclose(
            actual["p_default_tabular"], expected["p_default_tabular"], rtol=1e-10, atol=1e-10
        )

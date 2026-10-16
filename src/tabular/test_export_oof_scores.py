"""Tests for the out-of-fold logistic export.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_export_oof_scores.py -v -s

Most checks use synthetic loans shaped like the approved tabular_features_v1
output (no borrower data) with the REAL logistic and feature configs. The
committed-file checks need only the committed parquet and the split ID files;
the refit check also needs LENDINGCLUB_RAW_CSV.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tabular import development, export_oof_scores, logistic_baseline, preprocess, provenance, xgboost_baseline
from src.tabular.test_xgboost_baseline import make_synthetic
from src.tabular.xgboost_baseline import Fold

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPOSITORY_ROOT / "data"
N_FOLDS = 5
MIN_TRAIN_SHARE = 0.4


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture(scope="module")
def synthetic_run():
    """Synthetic model_fit loans, their folds, the OOF frame and every fitted pipeline."""
    features, target, issue_month = make_synthetic(rows=1200)
    folds = xgboost_baseline.time_aware_folds(issue_month, N_FOLDS, MIN_TRAIN_SHARE)
    fitted = []

    def make_pipeline():
        pipeline = logistic_baseline.build_pipeline(
            preprocess.load_feature_config(), logistic_baseline.load_logistic_config()
        )
        fitted.append(pipeline)
        return pipeline

    frame, fold_metrics = export_oof_scores.score_out_of_fold(
        make_pipeline, features, target, folds, issue_month
    )
    return {
        "features": features, "issue_month": issue_month, "folds": folds,
        "frame": frame, "fold_metrics": fold_metrics, "pipelines": fitted,
    }


def model_fit_ids(run) -> set[str]:
    return set(run["issue_month"].index)


def test_every_oof_loan_was_held_out_from_its_fold_and_fit_only_on_earlier_months(synthetic_run) -> None:
    """For every OOF row: its ID is not in that fold's fitting IDs and all fitting months come first."""
    issue_month, folds = synthetic_run["issue_month"], synthetic_run["folds"]
    for row in synthetic_run["frame"].itertuples(index=False):
        fold = folds[row.fold_id]
        assert row.loan_id in fold.holdout_ids
        assert row.loan_id not in fold.train_ids
        assert issue_month.loc[fold.train_ids].max() < issue_month.loc[row.loan_id]


def test_oof_covers_each_held_out_loan_once_and_skips_training_only_loans(synthetic_run) -> None:
    """Exactly one row per held-out loan; the earliest training-only loans are absent."""
    frame, folds, issue_month = synthetic_run["frame"], synthetic_run["folds"], synthetic_run["issue_month"]
    held_out = export_oof_scores.holdout_ids(folds)
    assert len(frame) == len(held_out) == frame["loan_id"].nunique()
    assert set(frame["loan_id"]) == set(held_out)
    training_only = issue_month.index.difference(held_out)
    assert not training_only.empty
    assert issue_month.loc[training_only].max() < issue_month.loc[held_out].min()
    assert not set(training_only) & set(frame["loan_id"])


def test_oof_frame_has_the_agreed_schema_fold_ids_and_finite_probabilities(synthetic_run) -> None:
    """Columns, split label, 0-4 fold IDs, one model version and finite probabilities in [0, 1]."""
    frame = synthetic_run["frame"]
    assert frame.columns.tolist() == list(export_oof_scores.OOF_COLUMNS)
    assert frame["split"].eq("train").all()
    assert sorted(frame["fold_id"].unique()) == list(range(N_FOLDS))
    assert frame["model_version"].eq("tabular-logistic-v1-time-oof").all()
    values = frame["p_default_tabular_raw"].to_numpy()
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
    assert frame["loan_id"].is_monotonic_increasing
    export_oof_scores.validate_oof_frame(frame, synthetic_run["folds"], model_fit_ids(synthetic_run))


def test_each_fold_fits_a_fresh_pipeline_that_learns_scaling_from_its_own_rows(synthetic_run) -> None:
    """One new pipeline per fold; its scaler mean equals that fold's training-row means only."""
    pipelines, folds, features = synthetic_run["pipelines"], synthetic_run["folds"], synthetic_run["features"]
    assert len(pipelines) == N_FOLDS
    assert len({id(pipeline) for pipeline in pipelines}) == N_FOLDS
    numeric = preprocess.load_feature_config()["model"]["numeric"]
    for pipeline, fold in zip(pipelines, folds, strict=True):
        fold_rows = features.loc[fold.train_ids, numeric]
        expected_means = fold_rows.fillna(fold_rows.median()).mean().to_numpy()
        scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-12)


def test_model_settings_are_the_unchanged_balanced_v1_settings(synthetic_run) -> None:
    """Every fold uses C=1, class_weight='balanced', max_iter=1000 and random_state=42."""
    for pipeline in synthetic_run["pipelines"]:
        params = pipeline.named_steps["classify"].get_params()
        assert (params["C"], params["class_weight"], params["max_iter"], params["random_state"]) == (
            1.0, "balanced", 1000, 42
        )


@pytest.mark.parametrize("foreign_id", ["calibration_loan", "validation_loan", "test_loan"])
def test_loans_outside_model_fit_are_rejected(synthetic_run, foreign_id: str) -> None:
    """A calibration, validation or test ID in the output is refused."""
    frame = synthetic_run["frame"].copy()
    frame.loc[0, "loan_id"] = foreign_id
    with pytest.raises(ValueError, match="outside model_fit"):
        export_oof_scores.validate_oof_frame(frame, synthetic_run["folds"], model_fit_ids(synthetic_run))


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate loan_id"),
        (lambda frame: frame.iloc[1:].reset_index(drop=True), "1 missing"),
        (lambda frame: frame.assign(fold_id=(frame["fold_id"] + 1) % N_FOLDS), "fold_id"),
        (lambda frame: frame.assign(split="validation"), "split must be 'train'"),
        (lambda frame: frame.assign(p_default_tabular_raw=np.nan), "finite"),
        (lambda frame: frame.assign(p_default_tabular_raw=np.inf), "finite"),
        (lambda frame: frame.assign(p_default_tabular_raw=1.5), "within \\[0, 1\\]"),
        (lambda frame: frame.drop(columns="fold_id"), "must have columns"),
    ],
    ids=["duplicate", "missing", "wrong_fold", "wrong_split", "nan", "inf", "above_one", "no_fold_id"],
)
def test_malformed_oof_frames_are_rejected(synthetic_run, corrupt, message: str) -> None:
    """Duplicates, gaps, wrong folds or splits, and invalid probabilities all fail loudly."""
    frame = corrupt(synthetic_run["frame"].copy())
    with pytest.raises(ValueError, match=message):
        export_oof_scores.validate_oof_frame(frame, synthetic_run["folds"], model_fit_ids(synthetic_run))


def test_folds_that_fit_on_their_holdout_or_later_months_are_refused() -> None:
    """A fold that fits a held-out loan, or fits a month not earlier than its block, is refused."""
    issue_month = pd.Series(
        ["2010-01", "2010-02", "2010-03", "2010-04"], index=pd.Index(["a", "b", "c", "d"], name="loan_id")
    )
    leaking = Fold(pd.Index(["a", "b", "c"]), pd.Index(["c", "d"]), "2010-03", "2010-04")
    with pytest.raises(ValueError, match="fits 1 of the loans it scores"):
        export_oof_scores.check_fold_boundaries((leaking,), issue_month)
    future = Fold(pd.Index(["a", "d"]), pd.Index(["c"]), "2010-03", "2010-03")
    with pytest.raises(ValueError, match="not earlier than"):
        export_oof_scores.check_fold_boundaries((future,), issue_month)
    first = Fold(pd.Index(["a"]), pd.Index(["b", "c"]), "2010-02", "2010-03")
    second = Fold(pd.Index(["a"]), pd.Index(["c", "d"]), "2010-03", "2010-04")
    with pytest.raises(ValueError, match="overlap"):
        export_oof_scores.check_fold_boundaries((first, second), issue_month)


def test_unexpected_row_counts_are_refused() -> None:
    """The run stops unless it fits 59,353 model_fit loans and scores exactly 35,234."""
    export_oof_scores.require_expected_counts(59_353, 35_234)
    assert export_oof_scores.EXPECTED_TRAINING_ONLY_ROWS == 24_119
    with pytest.raises(ValueError, match="35234 OOF rows"):
        export_oof_scores.require_expected_counts(59_353, 35_233)


def test_fold_table_reports_fit_and_holdout_months_and_counts(synthetic_run) -> None:
    """The manifest fold table records both month ranges, both counts and the holdout metrics."""
    table = export_oof_scores.fold_table(
        synthetic_run["folds"], synthetic_run["issue_month"], synthetic_run["fold_metrics"]
    )
    assert [row["fold_id"] for row in table] == list(range(N_FOLDS))
    for row, fold in zip(table, synthetic_run["folds"], strict=True):
        assert row["fit_rows"] == len(fold.train_ids)
        assert row["holdout_rows"] == len(fold.holdout_ids)
        assert row["fit_last_month"] < row["holdout_first_month"] <= row["holdout_last_month"]
        assert set(row["holdout_metrics"]) == set(export_oof_scores.FOLD_METRICS)


def committed_oof() -> tuple[pd.DataFrame, dict]:
    if not export_oof_scores.DEFAULT_OOF_PATH.exists():
        pytest.skip("The OOF parquet has not been generated yet")
    frame = pd.read_parquet(export_oof_scores.DEFAULT_OOF_PATH)
    manifest = json.loads(export_oof_scores.DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return frame, manifest


def test_committed_oof_file_covers_35234_model_fit_loans_and_no_other_split() -> None:
    """The committed parquet holds 35,234 unique model_fit loans and no calibration, validation or test ID."""
    frame, manifest = committed_oof()
    roles = development.assign_development_roles(REAL_DATA_DIR)
    validation_ids = set(pd.read_csv(REAL_DATA_DIR / "val_ids.csv", dtype=str)["loan_id"])
    test_ids = set(pd.read_csv(REAL_DATA_DIR / "test_ids.csv", dtype=str)["loan_id"])

    assert len(frame) == frame["loan_id"].nunique() == 35_234
    assert frame["loan_id"].isin(roles.index[roles["role"].eq("model_fit")]).all()
    assert not frame["loan_id"].isin(roles.index[roles["role"].eq("calibration")]).any()
    assert not set(frame["loan_id"]) & (validation_ids | test_ids)
    assert frame.columns.tolist() == list(export_oof_scores.OOF_COLUMNS)
    assert sorted(frame["fold_id"].unique()) == list(range(N_FOLDS))
    assert np.isfinite(frame["p_default_tabular_raw"]).all()
    assert frame["p_default_tabular_raw"].between(0, 1).all()

    assert manifest["rows"]["oof_scored"] == 35_234
    assert manifest["rows"]["training_only_unscored"] == 24_119
    assert manifest["test_rows_available_to_model"] == 0
    assert [fold["holdout_rows"] for fold in manifest["folds"]] == (
        frame["fold_id"].value_counts().sort_index().tolist()
    )


def test_every_committed_oof_row_was_held_out_and_fit_only_on_earlier_months() -> None:
    """On the real folds, each of the 35,234 committed rows is absent from its fold's fitting IDs,
    and every fitting month is earlier than that row's month (manifest only, no raw CSV)."""
    frame, _ = committed_oof()
    roles = development.assign_development_roles(REAL_DATA_DIR)
    issue_month = roles.loc[roles["role"].eq("model_fit"), "issue_month"]
    folds = xgboost_baseline.build_folds(issue_month, xgboost_baseline.load_xgboost_config())
    assert len(folds) == N_FOLDS

    for fold_id, fold in enumerate(folds):
        rows = frame.loc[frame["fold_id"].eq(fold_id), "loan_id"]
        assert set(rows) == set(fold.holdout_ids)
        assert not rows.isin(fold.train_ids).any()
        latest_fit_month = issue_month.loc[fold.train_ids].max()
        assert (issue_month.loc[rows] > latest_fit_month).all()


def test_committed_oof_manifest_checksum_matches_the_file() -> None:
    """The manifest's recorded SHA-256 is the committed parquet's actual SHA-256."""
    _, manifest = committed_oof()
    actual = provenance.file_checksums([export_oof_scores.DEFAULT_OOF_PATH])
    assert actual == {manifest["output"]["path"]: manifest["output"]["sha256"]}


@pytest.mark.skipif(
    not os.environ.get("LENDINGCLUB_RAW_CSV"), reason="LENDINGCLUB_RAW_CSV is not set"
)
def test_real_oof_refit_reproduces_the_committed_scores(tmp_path: Path) -> None:
    """A refit on the checksum-verified CSV reproduces every committed OOF score and fold ID."""
    committed, saved = committed_oof()
    rerun = export_oof_scores.run_export(
        REAL_DATA_DIR, Path(os.environ["LENDINGCLUB_RAW_CSV"]),
        tmp_path / "oof.parquet", tmp_path / "manifest.json",
    )
    actual = pd.read_parquet(tmp_path / "oof.parquet")
    assert rerun["rows"] == saved["rows"]
    provenance.verify_reproduced_inputs(
        saved["inputs_sha256"], rerun["inputs_sha256"],
        raw_csv_path=Path(os.environ["LENDINGCLUB_RAW_CSV"]), data_dir=REAL_DATA_DIR,
    )
    assert actual["loan_id"].equals(committed["loan_id"])
    assert actual["fold_id"].equals(committed["fold_id"])
    np.testing.assert_allclose(
        actual["p_default_tabular_raw"], committed["p_default_tabular_raw"], rtol=1e-10, atol=1e-10
    )

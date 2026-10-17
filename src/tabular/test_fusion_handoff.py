"""Tests for the fusion handoff: the calibrated tabular validation export and
the score-file loader.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_fusion_handoff.py -v -s

Everything reads only committed files (the split manifest and ID files, the
prediction CSVs, the NLP and OOF score files and the merged calibration
evidence), so no check needs LENDINGCLUB_RAW_CSV. Frame and loader checks use
tiny fabricated frames (no borrower data); OOF checks use edited copies of the
committed OOF file in a temporary directory.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nlp.preprocess import sha256_file
from src.tabular import calibrate, development, fusion_handoff

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


# ---------------------------------------------------------------------------
# Score-file loader

DISTILBERT_TRAIN = fusion_handoff.EXPECTED_SOURCES["distilbert_train"]
SCORE_IDS = ["lc_000000001", "lc_000000002", "lc_000000003"]


def _score_frame() -> pd.DataFrame:
    """Three DistilBERT-shaped train scores, with loan_id as pandas ``string`` like the NLP files."""
    return pd.DataFrame({
        "loan_id": pd.array(SCORE_IDS, dtype="string"),
        "split": "train",
        "prob_default_distilbert": [0.1, 0.5, 0.9],
        "model_name": "distilbert-base-uncased",
        "model_version": "nlp-distilbert-10k-v1",
    })


def _load(tmp_path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Write the frame as parquet and load it with the DistilBERT train pins."""
    path = tmp_path / "scores.parquet"
    frame.to_parquet(path, index=False)
    return fusion_handoff.load_score_file(
        path,
        probability_column=DISTILBERT_TRAIN["probability_column"],
        split=DISTILBERT_TRAIN["split"],
        expected_ids=set(SCORE_IDS),
        model_name=DISTILBERT_TRAIN["model_name"],
        allowed_versions=frozenset(DISTILBERT_TRAIN["allowed_versions"]),
    )


def _set(column: str, value):
    """Corruption that puts ``value`` in the first row of ``column``."""
    def corrupt(frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[column].tolist()
        values[0] = value
        return frame.assign(**{column: values})
    return corrupt


@pytest.mark.parametrize(
    "corrupt, message",
    [
        pytest.param(_set("loan_id", "lc_000000002"), "duplicate loan_id", id="duplicate"),
        pytest.param(lambda f: f.iloc[1:], "1 missing, 0 extra", id="missing-id"),
        pytest.param(_set("loan_id", "lc_000000099"), "1 missing, 1 extra", id="unexpected-id"),
        pytest.param(_set("split", "validation"), "split must be 'train'", id="wrong-split"),
        pytest.param(
            lambda f: f.assign(prob_default_distilbert=["high", "0.5", "0.9"]), "must be numeric", id="text-score"
        ),
        pytest.param(_set("prob_default_distilbert", np.nan), "missing prob_default_distilbert", id="nan"),
        pytest.param(_set("prob_default_distilbert", np.inf), r"within \[0, 1\]", id="inf"),
        pytest.param(_set("prob_default_distilbert", -0.1), r"within \[0, 1\]", id="below-0"),
        pytest.param(_set("prob_default_distilbert", 1.1), r"within \[0, 1\]", id="above-1"),
        pytest.param(_set("model_name", "bert-large"), "model_name must be", id="wrong-model"),
        pytest.param(_set("model_version", "nlp-distilbert-v2"), "not in the pinned set", id="unknown-version"),
    ],
)
def test_bad_score_files_are_refused(tmp_path: Path, corrupt, message: str) -> None:
    """Duplicates, missing or unexpected IDs, a wrong split, a bad score, or an unpinned model is refused."""
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, corrupt(_score_frame()))


def test_a_subset_with_one_of_two_pinned_versions_loads_sorted_with_str_ids(tmp_path: Path) -> None:
    """A file holding only one of DistilBERT's two pinned versions is accepted; loan_id comes back str and sorted."""
    frame = _score_frame().assign(model_version="nlp-distilbert-10k-holdout-v1").iloc[::-1]
    loaded = _load(tmp_path, frame)

    assert loaded["loan_id"].tolist() == SCORE_IDS
    assert loaded["loan_id"].map(type).eq(str).all()


def _load_with_edited_oof(tmp_path: Path, corrupt, *, re_sign: bool) -> None:
    """Load every source with an edited copy of the OOF file. ``re_sign`` writes a manifest
    copy whose SHA-256 matches the edit, so the checks after the checksum are reached."""
    sources = copy.deepcopy(fusion_handoff.EXPECTED_SOURCES)
    committed = pd.read_parquet(fusion_handoff.REPO_ROOT / sources["tabular_train"]["path"])
    path = tmp_path / "oof.parquet"
    corrupt(committed).to_parquet(path, index=False)
    sources["tabular_train"]["path"] = str(path)

    manifest_path = fusion_handoff.OOF_MANIFEST_PATH
    if re_sign:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output"]["sha256"] = sha256_file(path)
        manifest_path = tmp_path / "oof_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fusion_handoff.load_sources(sources=sources, oof_manifest_path=manifest_path)


def _first_calibration_id() -> str:
    roles = development.assign_development_roles(fusion_handoff.DEFAULT_DATA_DIR)
    return roles.index[roles["role"].eq("calibration")][0]


@pytest.mark.parametrize(
    "corrupt, re_sign, message",
    [
        pytest.param(
            lambda f: f.assign(p_default_tabular_raw=f["p_default_tabular_raw"] * 0.999), False,
            "Checksum mismatch", id="edited-bytes",
        ),
        pytest.param(lambda f: f.iloc[1:], True, "35233 rows", id="row-dropped"),
        pytest.param(
            lambda f: f.assign(loan_id=[_first_calibration_id(), *f["loan_id"].iloc[1:]]), True,
            "outside model_fit", id="calibration-loan",
        ),
        pytest.param(
            lambda f: f.assign(fold_id=[(f["fold_id"].iloc[0] + 1) % 5, *f["fold_id"].iloc[1:]]), True,
            "fold_id counts", id="fold-moved",
        ),
    ],
)
def test_oof_file_that_differs_from_its_manifest_or_role_is_refused(
    tmp_path: Path, corrupt, re_sign: bool, message: str
) -> None:
    """The OOF file is refused, naming the source, for changed bytes, a row count, a non-model_fit loan or fold counts
    that differ from its manifest."""
    with pytest.raises(ValueError, match=f"tabular_train .*{message}"):
        _load_with_edited_oof(tmp_path, corrupt, re_sign=re_sign)


def test_real_score_files_load_with_the_expected_counts() -> None:
    """All six committed sources load: 35,234 OOF rows in five folds, full NLP splits, and the
    DistilBERT train versions 31,060 / 4,174 inside the OOF loans."""
    frames = fusion_handoff.load_sources()

    assert {name: len(frame) for name, frame in frames.items()} == {
        "tabular_train": 35_234, "tabular_validation": 21_784,
        "tfidf_train": 86_293, "tfidf_validation": 21_784,
        "distilbert_train": 86_293, "distilbert_validation": 21_784,
    }
    assert frames["tabular_train"]["fold_id"].value_counts().sort_index().tolist() == [7022, 6033, 7095, 7883, 7201]
    distilbert = frames["distilbert_train"]
    inside_oof = distilbert.loc[distilbert["loan_id"].isin(frames["tabular_train"]["loan_id"]), "model_version"]
    assert inside_oof.value_counts().to_dict() == {"nlp-distilbert-10k-v1": 31_060, "nlp-distilbert-10k-holdout-v1": 4_174}

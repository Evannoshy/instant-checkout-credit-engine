"""Late-fusion input tests: synthetic tables for every reject rule, plus the committed real sources.

From the repository root:
    python -m pytest src/fusion/test_inputs.py -v

Nothing here reads data/raw/loan.csv. The real-data tests need the committed
role mapping, raw tabular prediction files and split manifest.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fusion import inputs
from src.fusion.inputs import ERROR_MESSAGES, NLP_SCORE_COLUMNS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "fusion_stacker_v1.toml"
REAL_ROLE_ROWS = {"fusion_fit": 15_587, "fusion_calibration": 11_353, "validation": 21_784}


def _message(key: str) -> str:
    """A regex for one whole ERROR_MESSAGES template: literal text escaped, each placeholder any text."""
    return "^" + ".*".join(re.escape(part) for part in re.split(r"\{[^}]*\}", ERROR_MESSAGES[key])) + "$"


def _write_config(tmp_path: Path, replace: tuple[str, str]) -> Path:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert replace[0] in text, f"fixture edit {replace[0]!r} not found in the config"
    path = tmp_path / "stacker.toml"
    path.write_text(text.replace(*replace), encoding="utf-8")
    return path


def _mapping() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": ["lc_1", "lc_2", "lc_3", "lc_4"],
        "split": ["train", "train", "train", "validation"],
        "role": ["base_fit", "fusion_fit", "fusion_calibration", "validation"],
        "issue_month": ["2012-12", "2013-01", "2013-05", "2013-08"],
    })


def _nlp_scores() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": ["lc_2", "lc_3", "lc_4"],
        "split": ["train", "train", "validation"],
        "role": ["fusion_fit", "fusion_calibration", "validation"],
        "raw_margin": [-1.0, 0.0, 1.0],
        "raw_prob": [0.27, 0.5, 0.73],
        "model_version": ["nlp-test-v2"] * 3,
    })


# --- config ---------------------------------------------------------------------------------

def test_committed_config_runs_only_the_tabular_variant_and_is_unweighted():
    config = inputs.load_stacker_config(CONFIG_PATH)
    assert config["run"]["variants"] == ["tabular_only"]
    assert config["stacker"]["class_weight"] == "none"
    assert config["evaluation"]["bootstrap_resamples"] == 1000
    assert config["evaluation"]["decision_rule_status"] == "pending lead confirmation"


@pytest.mark.parametrize("edit, key", [
    (('class_weight = "none"', 'class_weight = "balanced"'), "bad_setting"),
    (('input_space = "probability"', 'input_space = "logit"'), "bad_setting"),
    (('method = "platt"', 'method = "isotonic"'), "bad_setting"),
    (('variants = ["tabular_only"]', 'variants = ["missing"]'), "unknown_variant"),
])
def test_config_rejects_unapproved_settings(tmp_path, edit, key):
    with pytest.raises(ValueError, match=_message(key)):
        inputs.load_stacker_config(_write_config(tmp_path, edit))


def test_running_a_variant_whose_text_track_is_not_published_stops_instead_of_skipping(tmp_path):
    path = _write_config(tmp_path, ('variants = ["tabular_only"]', 'variants = ["tabular_only", "tabular_tfidf"]'))
    with pytest.raises(ValueError, match=_message("track_without_source")):
        inputs.load_stacker_config(path)


# --- NLP export contract ----------------------------------------------------------------------

def test_valid_nlp_export_passes():
    inputs._check_nlp_scores("tfidf", _nlp_scores(), _mapping(), "nlp-test-v2")


@pytest.mark.parametrize("change, key", [
    (lambda f: f.drop(columns="raw_margin"), "nlp_columns"),
    (lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True), "nlp_duplicates"),
    (lambda f: f.iloc[:2], "nlp_coverage"),
    (lambda f: f.assign(loan_id=["lc_1", "lc_3", "lc_4"]), "nlp_coverage"),
    (lambda f: f.assign(role=["fusion_fit", "validation", "validation"]), "nlp_mismatch"),
    (lambda f: f.assign(split=["validation", "train", "validation"]), "nlp_mismatch"),
    (lambda f: f.assign(raw_margin=[np.inf, 0.0, 1.0]), "not_finite"),
    (lambda f: f.assign(raw_prob=[np.nan, 0.5, 0.7]), "not_finite"),
    (lambda f: f.assign(raw_prob=[1.2, 0.5, 0.7]), "out_of_range"),
    (lambda f: f.assign(model_version=["nlp-test-v2", "old", "nlp-test-v2"]), "nlp_version"),
])
def test_nlp_export_rejects_contract_violations(change, key):
    with pytest.raises(ValueError, match=_message(key)):
        inputs._check_nlp_scores("tfidf", change(_nlp_scores()), _mapping(), "nlp-test-v2")


def test_nlp_export_with_base_fit_rows_is_rejected():
    extra = pd.DataFrame([{"loan_id": "lc_1", "split": "train", "role": "base_fit",
                           "raw_margin": 0.0, "raw_prob": 0.5, "model_version": "nlp-test-v2"}])
    with pytest.raises(ValueError, match=_message("nlp_coverage")):
        inputs._check_nlp_scores("tfidf", pd.concat([_nlp_scores(), extra]), _mapping(), "nlp-test-v2")


def test_score_columns_exclude_ids_roles_dates_and_versions():
    assert inputs.score_columns(["tabular", "tfidf"]) == ["p_default_tabular", "p_default_tfidf"]
    assert not set(inputs.score_columns(["tabular"])) & {*NLP_SCORE_COLUMNS, "issue_month", "target"}


# --- committed real sources --------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_tables():
    config = inputs.load_stacker_config(CONFIG_PATH)
    return config, inputs.build_role_tables(config, ["tabular"])


def test_real_tables_cover_each_role_exactly_once(real_tables):
    _, tables = real_tables
    assert {role: len(frame) for role, frame in tables.items()} == REAL_ROLE_ROWS
    all_ids = pd.concat([frame["loan_id"] for frame in tables.values()])
    assert not all_ids.duplicated().any()
    test_ids = set(pd.read_csv(REPOSITORY_ROOT / "data" / "test_ids.csv", dtype=str)["loan_id"])
    assert not set(all_ids) & test_ids


def test_real_tables_hold_only_ids_labels_and_finite_scores(real_tables):
    _, tables = real_tables
    for frame in tables.values():
        assert frame.columns.tolist() == ["loan_id", "target", "p_default_tabular"]
        assert frame["loan_id"].is_monotonic_increasing
        assert set(frame["target"].unique()) == {0, 1}
        scores = frame["p_default_tabular"].to_numpy()
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()


def test_real_reference_aligns_with_validation_rows(real_tables):
    config, tables = real_tables
    validation = tables["validation"]
    reference = inputs.load_reference(config, validation["loan_id"])
    assert reference["loan_id"].tolist() == validation["loan_id"].tolist()
    # The reference's raw score is the same balanced v1 LR score the stacker reads.
    np.testing.assert_allclose(reference["p_reference_raw"], validation["p_default_tabular"], atol=1e-12)


def test_tampered_role_mapping_is_refused(tmp_path):
    config = inputs.load_stacker_config(CONFIG_PATH)
    source = REPOSITORY_ROOT / config["role_mapping"]["path"]
    target = tmp_path / config["role_mapping"]["path"]
    target.parent.mkdir(parents=True)
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    target.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        inputs.load_role_mapping(config, root=tmp_path)

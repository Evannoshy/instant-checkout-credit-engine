"""Regression tests for the unweighted TF-IDF fusion export (fusion_development_v1).

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/nlp/test_export_scores_v2.py -v -s

No borrower data is loaded and ``data/raw/loan.csv`` is not required.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.nlp.export_scores_v2 import (
    DEFAULT_CONFIG,
    EXPORT_COLUMNS,
    attach_roles,
    build_export,
    grid_points,
    load_config,
    load_role_mapping,
    resolve_class_weight,
    sha256_file,
    validate_export,
    write_parquet,
)

ROLE_MAPPING = "data/fusion/fusion_roles_v1.csv"
CONFIG_FILENAME = "nlp_tfidf_fusion_v2.toml"


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def tiny_config() -> dict:
    """A config shaped like the real one but describing three toy loans."""
    return {
        "model_version": "test-v2",
        "model_name": "tfidf_logistic",
        "preprocess_version": "nlp-preprocess-v1",
        "roles": {
            "mapping_path": ROLE_MAPPING,
            "mapping_sha256": "unused-in-this-fixture",
            "fusion_development_version": "fusion_development_v1",
            "fit_role": "base_fit",
            "scoring_roles": ["fusion_fit", "fusion_calibration", "validation"],
        },
        "expected": {
            "role_rows": {
                "base_fit": 1,
                "fusion_fit": 1,
                "fusion_calibration": 1,
                "validation": 1,
                "scoring_total": 3,
            }
        },
        "model": {"class_weight": "none", "max_iter": 10, "random_state": 42, "sublinear_tf": True},
        "grid": {"max_features": [10], "ngram_range": [[1, 1]], "C": [1.0]},
        "cv": {"n_folds": 2, "min_train_share": 0.4, "scoring": "average_precision",
               "scheme": "expanding_window_time_ordered", "scheme_source": "x"},
        "comparison": {"class_weight": "balanced"},
        "output": {"scores_path": "x", "manifest_path": "y", "pipeline_path": "z"},
    }


@pytest.fixture
def tiny_mapping() -> pd.DataFrame:
    """Four loans, one per role, in the shared mapping's column order."""
    return pd.DataFrame(
        {
            "loan_id": pd.Series(["lc_000000001", "lc_000000002", "lc_000000003", "lc_000000004"], dtype="string"),
            "split": pd.Series(["train", "train", "train", "validation"], dtype="string"),
            "role": pd.Series(["base_fit", "fusion_fit", "fusion_calibration", "validation"], dtype="string"),
            "issue_month": ["2012-01", "2013-02", "2013-06", "2013-09"],
        }
    )


def scoring_export(mapping: pd.DataFrame, margins=None) -> pd.DataFrame:
    """Build a well-formed export covering exactly the three scoring roles."""
    scoring = mapping.loc[mapping["role"].ne("base_fit")].reset_index(drop=True)
    if margins is None:
        margins = np.array([-2.0, -1.0, 0.5])
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(margins, dtype=float)))
    return build_export(scoring, margins, probabilities, "test-v2")


# --- the shipped configuration -----------------------------------------------


def test_shipped_config_pins_the_published_role_mapping():
    """The config points at the approved mapping and its checksum still matches."""
    config = load_config(DEFAULT_CONFIG)
    assert config["roles"]["mapping_path"] == ROLE_MAPPING
    assert sha256_file(ROLE_MAPPING) == config["roles"]["mapping_sha256"]


def test_shipped_config_expects_the_approved_role_counts():
    """Role counts match fusion_development_v1 as approved on 2026-10-28."""
    expected = load_config(DEFAULT_CONFIG)["expected"]["role_rows"]
    assert expected["base_fit"] == 59353
    assert expected["fusion_fit"] == 15587
    assert expected["fusion_calibration"] == 11353
    assert expected["validation"] == 21784
    assert expected["scoring_total"] == 48724
    assert expected["fusion_fit"] + expected["fusion_calibration"] + expected["validation"] == 48724


def test_config_filename_matches_the_model_version_it_produces():
    """The config is named for its model version, so the two cannot drift apart.

    The older configs/nlp_tfidf_v2.toml produces nlp-tfidf-tuned-v1; that
    mismatch is the reason this rule is pinned rather than assumed.
    """
    config = load_config(DEFAULT_CONFIG)
    assert DEFAULT_CONFIG.name == CONFIG_FILENAME
    assert config["model_version"] == "nlp-tfidf-fusion-v2"
    assert DEFAULT_CONFIG.stem.replace("_", "-") == config["model_version"]


def test_shipped_export_is_not_the_v1_model_version():
    """The exported model_version is new; the v1 files keep theirs."""
    export = pd.read_parquet("data/nlp/tfidf_scores_v2.parquet")
    versions = export["model_version"].unique().tolist()
    assert versions == ["nlp-tfidf-fusion-v2"]
    assert "nlp-tfidf-tuned-v1" not in versions


def test_shipped_config_is_unweighted():
    """The fusion export must be unweighted; a weighted config is refused."""
    assert load_config(DEFAULT_CONFIG)["model"]["class_weight"] == "none"


def test_config_rejects_a_weighted_model(tmp_path):
    """class_weight other than 'none' fails loudly rather than shipping silently."""
    source = DEFAULT_CONFIG.read_text().replace('class_weight = "none"', 'class_weight = "balanced"', 1)
    path = tmp_path / "weighted.toml"
    path.write_text(source)
    with pytest.raises(ValueError, match="unweighted"):
        load_config(path)


@pytest.mark.parametrize("value,expected", [("none", None), (None, None), ("balanced", "balanced")])
def test_class_weight_sentinel_maps_to_scikit_learn(value, expected):
    """TOML has no null, so 'none' is translated rather than reaching the estimator."""
    assert resolve_class_weight(value) == expected


def test_grid_enumerates_every_declared_combination():
    """The grid is the product of the declared axes: 3 x 2 x 4 = 24."""
    points = grid_points(load_config(DEFAULT_CONFIG)["grid"])
    assert len(points) == 24
    assert len({(p["max_features"], p["ngram_range"], p["C"]) for p in points}) == 24


# --- the shared role mapping as a contract ------------------------------------


def test_role_mapping_checksum_mismatch_stops_the_run(tmp_path):
    """A changed shared contract halts scoring instead of rescoring quietly."""
    config = load_config(DEFAULT_CONFIG)
    config["roles"] = {**config["roles"], "mapping_sha256": "0" * 64}
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_role_mapping(tmp_path, config)


def test_role_mapping_loads_with_the_agreed_columns():
    """The published mapping has the four contract columns and no duplicate IDs."""
    mapping = load_role_mapping("data", load_config(DEFAULT_CONFIG))
    assert list(mapping.columns) == ["loan_id", "split", "role", "issue_month"]
    assert not mapping["loan_id"].duplicated().any()
    assert mapping["role"].value_counts().to_dict() == {
        "base_fit": 59353, "validation": 21784, "fusion_fit": 15587, "fusion_calibration": 11353
    }


def test_attach_roles_rejects_a_loan_absent_from_the_mapping(tiny_mapping):
    """An unmatched loan is an error, never filled or silently dropped."""
    rows = pd.DataFrame({"loan_id": ["lc_000000001", "lc_999999999"], "split": ["train", "train"]})
    with pytest.raises(ValueError, match="no row in the shared role mapping"):
        attach_roles(rows, tiny_mapping)


def test_attach_roles_rejects_a_disagreeing_split_label(tiny_mapping):
    """A split label differing from the mapping stops the run."""
    rows = pd.DataFrame({"loan_id": ["lc_000000002"], "split": ["validation"]})
    with pytest.raises(ValueError, match="disagrees with the shared role mapping"):
        attach_roles(rows, tiny_mapping)


# --- the export contract ------------------------------------------------------


def test_export_has_the_agreed_columns_in_order(tiny_mapping):
    """Six columns, in the order the fusion task sheet specifies."""
    assert list(scoring_export(tiny_mapping).columns) == list(EXPORT_COLUMNS)
    assert EXPORT_COLUMNS == ("loan_id", "split", "role", "raw_margin", "raw_prob", "model_version")


def test_export_covers_exactly_the_scoring_roles(tiny_mapping, tiny_config):
    """Exact role coverage: no base_fit row, nothing missing, nothing extra."""
    export = scoring_export(tiny_mapping)
    validate_export(export, tiny_mapping, tiny_config)
    assert "base_fit" not in set(export["role"])
    assert len(export) == tiny_config["expected"]["role_rows"]["scoring_total"]


def test_export_rejects_a_base_fit_row(tiny_mapping, tiny_config):
    """A base_fit row must never reach the fusion track."""
    export = build_export(tiny_mapping, np.zeros(4), np.full(4, 0.5), "test-v2")
    with pytest.raises(ValueError, match="does not cover the scoring roles exactly|base_fit"):
        validate_export(export, tiny_mapping, tiny_config)


def test_export_rejects_duplicate_loan_ids(tiny_mapping, tiny_config):
    """A duplicate would break the stacker's one-to-one join."""
    export = pd.concat([scoring_export(tiny_mapping)] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_export(export, tiny_mapping, tiny_config)


def test_export_rejects_incomplete_coverage(tiny_mapping, tiny_config):
    """A dropped loan is reported, not tolerated."""
    export = scoring_export(tiny_mapping).iloc[:-1]
    with pytest.raises(ValueError, match="does not cover the scoring roles exactly"):
        validate_export(export, tiny_mapping, tiny_config)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_export_rejects_a_non_finite_margin(tiny_mapping, tiny_config, bad):
    """Margins must be finite, as the task sheet requires."""
    export = scoring_export(tiny_mapping)
    export.loc[0, "raw_margin"] = bad
    with pytest.raises(ValueError, match="raw_margin|logistic transform"):
        validate_export(export, tiny_mapping, tiny_config)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_export_rejects_a_probability_outside_the_unit_interval(tiny_mapping, tiny_config, bad):
    """Probabilities must lie within [0, 1]."""
    export = scoring_export(tiny_mapping)
    export.loc[0, "raw_prob"] = bad
    with pytest.raises(ValueError, match="within"):
        validate_export(export, tiny_mapping, tiny_config)


def test_export_rejects_a_probability_that_is_not_its_own_margin(tiny_mapping, tiny_config):
    """raw_prob must be the logistic transform of raw_margin.

    The task sheet requires both columns to come from the same fitted model.
    For a logistic model that is exactly checkable, so it is checked rather
    than trusted.
    """
    export = scoring_export(tiny_mapping)
    export.loc[0, "raw_prob"] = 0.5
    with pytest.raises(ValueError, match="logistic transform"):
        validate_export(export, tiny_mapping, tiny_config)


def test_export_rejects_a_role_label_disagreeing_with_the_mapping(tiny_mapping, tiny_config):
    """Role labels are checked against the shared mapping, not just counted."""
    export = scoring_export(tiny_mapping)
    export.loc[export.index[0], "role"] = "validation"
    with pytest.raises(ValueError, match="role labels disagree|has .* rows, expected"):
        validate_export(export, tiny_mapping, tiny_config)


def test_export_round_trips_through_parquet(tmp_path, tiny_mapping):
    """What is written back is what was handed over."""
    export = scoring_export(tiny_mapping)
    written = write_parquet(export, tmp_path / "scores.parquet")
    assert list(written.columns) == list(EXPORT_COLUMNS)
    assert written["loan_id"].tolist() == export["loan_id"].tolist()
    np.testing.assert_allclose(written["raw_prob"], export["raw_prob"])


# --- the shipped artefacts -----------------------------------------------------


def test_shipped_export_matches_the_published_mapping_exactly():
    """The real export covers the 48,724 scoring loans and agrees on every label."""
    export = pd.read_parquet("data/nlp/tfidf_scores_v2.parquet")
    mapping = pd.read_csv(ROLE_MAPPING, dtype={"loan_id": "string", "split": "string", "role": "string"})
    expected = mapping.loc[mapping["role"].ne("base_fit")]
    assert len(export) == 48724
    assert set(export["loan_id"]) == set(expected["loan_id"])
    merged = export.merge(expected, on="loan_id", suffixes=("", "_expected"))
    assert merged["split"].equals(merged["split_expected"])
    assert merged["role"].equals(merged["role_expected"])


def test_shipped_export_is_internally_consistent():
    """Finite margins, probabilities in range, and each is the other's transform."""
    export = pd.read_parquet("data/nlp/tfidf_scores_v2.parquet")
    assert np.isfinite(export["raw_margin"]).all()
    assert ((export["raw_prob"] >= 0) & (export["raw_prob"] <= 1)).all()
    assert not export["loan_id"].duplicated().any()
    np.testing.assert_allclose(
        export["raw_prob"], 1 / (1 + np.exp(-export["raw_margin"])), atol=1e-9
    )


def test_shipped_export_probability_level_matches_the_base_rate():
    """Unweighted fitting leaves the probability on the right scale.

    The v1 balanced export averaged 0.4786 against a ~0.15 base rate. An
    unweighted model should land near prevalence without any calibrator.
    """
    export = pd.read_parquet("data/nlp/tfidf_scores_v2.parquet")
    assert abs(export["raw_prob"].mean() - 0.1514) < 0.03


def test_shipped_manifest_records_the_fitting_boundary_and_locked_test():
    """The manifest states where fitting stopped and that test was never read."""
    manifest = json.loads(open("reports/nlp/tfidf_scores_v2_manifest.json").read())
    assert manifest["test_rows_used"] == 0
    assert manifest["fitting_boundaries"]["fitted_on_scoring_roles"] is False
    assert manifest["fitting_boundaries"]["fit_rows"] == 59353
    assert manifest["fitting_boundaries"]["fit_first_month"] == "2007-06"
    assert manifest["fitting_boundaries"]["fit_last_month"] == "2012-12"
    assert manifest["output"]["sha256"] == sha256_file("data/nlp/tfidf_scores_v2.parquet")


def test_shipped_manifest_does_not_claim_calibration():
    """An unweighted probability must not be described as post-hoc calibrated."""
    manifest = json.loads(open("reports/nlp/tfidf_scores_v2_manifest.json").read())
    assert manifest["probability_semantics"] == "raw_unweighted_model_output_not_post_hoc_calibrated"


def test_shipped_export_does_not_overwrite_the_v1_files():
    """The balanced v1 exports keep their own version label and are untouched."""
    v1 = pd.read_parquet("data/nlp/tfidf_oof_train.parquet")
    assert v1["model_version"].unique().tolist() == ["nlp-tfidf-tuned-v1"]
    v2 = pd.read_parquet("data/nlp/tfidf_scores_v2.parquet")
    assert v2["model_version"].unique().tolist() == ["nlp-tfidf-fusion-v2"]

python -m pytest src/tabular/test_tabular_preprocess.py -v -s"""Regression tests using tiny, artificial examples (no borrower data).

From the repository root, show each check and its actual pytest result with:
    python -m pytest  -v -s

The -s flag shows print statements; -v shows each PASSED/FAILED result.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.tabular.preprocess import ERROR_MESSAGES, load_feature_config, load_tabular_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_features_v1.toml"
REAL_DATA_DIR = REPOSITORY_ROOT / "data"


def error_pattern(key: str) -> str:
    """Regex for the full ERROR_MESSAGES[key] text, allowing any value in each {placeholder}."""
    literal_parts = re.split(r"\{[^}]*\}", ERROR_MESSAGES[key])
    return "^" + ".*".join(re.escape(part) for part in literal_parts) + "$"


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def artificial_data_dir(tmp_path: Path) -> Path:
    """Six made-up records exercise split alignment, missing values and category normalization."""
    (tmp_path / "raw").mkdir()
    raw = pd.DataFrame(
        {
            "loan_amnt": ["11000", "10000", "13000", "9000", "20000", "7000"],
            "term": [
                " 60 months",
                " 36 months",
                " 36 months",
                " 36 months",
                " 60 months",
                " 36 months",
            ],
            "annual_inc": ["48000", "60000", "51000", "45000", "80000", "35000"],
            "dti": ["12.5", "", "18.0", "9.5", "22.0", "15.0"],
            "revol_util": ["", "34.5", "51.0", "19.0", "", "44.0"],
            "delinq_2yrs": ["0", "0", "1", "0", "2", "0"],
            "inq_last_6mths": ["1", "2", "0", "1", "3", "0"],
            "earliest_cr_line": [
                "Feb-2001",
                "Jan-2000",
                "May-2004",
                "Jun-2005",
                "",
                "Apr-2002",
            ],
            "home_ownership": ["NONE", "MORTGAGE", "RENT", "OWN", "OTHER", "RENT"],
            "issue_d": [
                "Aug-2013",
                "Mar-2013",
                "Jul-2012",
                "Jan-2014",
                "Dec-2013",
                "Apr-2013",
            ],
            "loan_status": [
                "Fully Paid",
                "Fully Paid",
                "Fully Paid",
                "Charged Off",
                "Charged Off",
                "Charged Off",
            ],
            "grade": ["SYNTHETIC_GRADE"] * 6,
            "int_rate": ["9.99%"] * 6,
            "desc": ["Synthetic description"] * 6,
            "total_pymnt": ["123.45"] * 6,
        }
    )
    raw.to_csv(tmp_path / "raw" / "loan.csv", index=False)

    version = "synthetic-tabular-fixture-v1"
    # Manifest order deliberately differs from raw-file order; source row 3 is absent.
    manifest = pd.DataFrame(
        {
            "loan_id": [
                "lc_000000005",
                "lc_000000001",
                "lc_000000004",
                "lc_000000002",
                "lc_000000006",
            ],
            "source_row_number": [5, 1, 4, 2, 6],
            "split": ["train", "validation", "test", "train", "validation"],
            "issue_month": ["2013-12", "2013-08", "2014-01", "2013-03", "2013-04"],
            "target": [1, 0, "LOCKED_TEST_LABEL", 0, 1],
            "text_available": [1, 1, 1, 1, 1],
            "cohort": ["real_text_matured_v1"] * 5,
            "dataset_version": [version] * 5,
        }
    )
    manifest.to_csv(tmp_path / "split_manifest.csv", index=False)
    manifest.loc[manifest["split"].eq("train"), ["loan_id"]].to_csv(
        tmp_path / "train_ids.csv", index=False
    )
    manifest.loc[manifest["split"].eq("validation"), ["loan_id"]].to_csv(
        tmp_path / "val_ids.csv", index=False
    )

    statistics = {
        "dataset_version": version,
        "split_version": "split-v1",
        "cohort": "real_text_matured_v1",
        "raw_rows": 6,
        "eligible_rows": 5,
        "splits": {
            "train": {"rows": 2},
            "validation": {"rows": 2},
            "test": {"rows": 1},
        },
    }
    (tmp_path / "split_statistics.json").write_text(
        json.dumps(statistics), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def feature_config() -> dict[str, Any]:
    """Load the repository's real tabular feature configuration."""
    return load_feature_config(FEATURE_CONFIG)


def _raw_path(data_dir: Path) -> Path:
    return data_dir / "raw" / "loan.csv"


def _write_modified_config(
    tmp_path: Path,
    filename: str,
    replacements: list[tuple[str, str]],
) -> Path:
    text = FEATURE_CONFIG.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in text, f"Expected feature-config text was not found: {old}"
        text = text.replace(old, new, 1)
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("split", "error_key"),
    [
        ("test", "test_split_locked"),
        ("TEST", "unknown_split"),
        ("holdout", "unknown_split"),
    ],
)
def test_loader_rejects_locked_and_unknown_splits(
    artificial_data_dir: Path, split: str, error_key: str
) -> None:
    """The locked test split and every unsupported split name are rejected before loading."""
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        load_tabular_split(
            artificial_data_dir,
            _raw_path(artificial_data_dir),
            split,
            FEATURE_CONFIG,
        )


def test_output_index_is_unique_sorted_named_and_target_aligned(
    artificial_data_dir: Path,
) -> None:
    """Features and targets share one unique, sorted loan_id index."""
    features, target = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    assert features.index.name == "loan_id"
    assert features.index.is_unique
    assert features.index.tolist() == sorted(features.index.tolist())
    assert features.index.equals(target.index)


def test_train_and_validation_ids_are_disjoint(artificial_data_dir: Path) -> None:
    """The development splits contain disjoint frozen loan_id sets."""
    train, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    validation, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "validation",
        FEATURE_CONFIG,
    )
    assert set(train.index).isdisjoint(validation.index)


def test_locked_test_rows_never_appear_in_development_splits(
    artificial_data_dir: Path,
) -> None:
    """The locked test row is absent from both the train and validation outputs."""
    train, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    validation, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "validation",
        FEATURE_CONFIG,
    )
    # lc_000000004 is the fixture's only test-split row.
    assert "lc_000000004" not in train.index
    assert "lc_000000004" not in validation.index


def test_missing_configured_source_column_is_rejected(
    artificial_data_dir: Path,
) -> None:
    """A configured source feature missing from the raw CSV header is rejected."""
    raw_path = _raw_path(artificial_data_dir)
    raw = pd.read_csv(raw_path)
    raw.drop(columns="annual_inc").to_csv(raw_path, index=False)
    with pytest.raises(ValueError, match=error_pattern("raw_missing_columns")):
        load_tabular_split(artificial_data_dir, raw_path, "train", FEATURE_CONFIG)


def test_feature_columns_follow_config_order(
    artificial_data_dir: Path,
    feature_config: dict[str, Any],
) -> None:
    """Model columns are exactly numeric then categorical features in configured order."""
    features, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    expected = feature_config["model"]["numeric"] + feature_config["model"]["categorical"]
    assert list(features.columns) == expected


def test_target_is_int8_binary_and_never_a_feature(
    artificial_data_dir: Path,
) -> None:
    """The aligned target is binary int8 and is excluded from model features."""
    features, target = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    assert "target" not in features.columns
    assert target.dtype == np.dtype("int8")
    assert set(target.tolist()) == {0, 1}


def test_prohibited_columns_are_excluded(
    artificial_data_dir: Path,
    feature_config: dict[str, Any],
) -> None:
    """No prohibited field enters features even when prohibited values exist in the raw CSV."""
    features, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    prohibited = {
        column
        for group in feature_config["prohibited"].values()
        for column in group
    }
    assert prohibited.isdisjoint(features.columns)


def test_config_rejects_prohibited_model_and_source_feature(
    tmp_path: Path,
) -> None:
    """A prohibited underwriting field cannot be added as a source or model feature."""
    path = _write_modified_config(
        tmp_path,
        "prohibited.toml",
        [
            (
                'features = ["loan_amnt", "term",',
                'features = ["int_rate", "loan_amnt", "term",',
            ),
            (
                'numeric = ["loan_amnt",',
                'numeric = ["int_rate", "loan_amnt",',
            ),
        ],
    )
    with pytest.raises(ValueError, match=error_pattern("prohibited_columns")):
        load_feature_config(path)


def test_config_cannot_remove_always_prohibited_columns(tmp_path: Path) -> None:
    """A post-outcome field stays prohibited even when the config's own lists omit it."""
    text = FEATURE_CONFIG.read_text(encoding="utf-8")
    text = re.sub(r"\[prohibited\].*", "[prohibited]\n", text, flags=re.DOTALL)
    for old, new in [
        ('features = ["loan_amnt", "term",', 'features = ["total_pymnt", "loan_amnt", "term",'),
        ('numeric = ["loan_amnt",', 'numeric = ["total_pymnt", "loan_amnt",'),
    ]:
        assert old in text, f"Expected feature-config text was not found: {old}"
        text = text.replace(old, new, 1)
    path = tmp_path / "empty_prohibited.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=error_pattern("prohibited_columns")):
        load_feature_config(path)


def test_blank_features_remain_nan_without_dropping_rows(
    artificial_data_dir: Path,
) -> None:
    """Blank raw features remain missing while every frozen split row is retained."""
    features, target = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    manifest = pd.read_csv(artificial_data_dir / "split_manifest.csv")
    expected_rows = int(manifest["split"].eq("train").sum())
    assert len(features) == len(target) == expected_rows
    assert pd.isna(features.loc["lc_000000002", "dti"])
    assert pd.isna(features.loc["lc_000000005", "revol_util"])
    assert pd.isna(features.loc["lc_000000005", "credit_history_months"])


def test_term_is_normalized_to_configured_values(
    artificial_data_dir: Path,
) -> None:
    """Raw term labels such as ' 36 months' map to their configured month counts."""
    train, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    assert train.loc["lc_000000002", "term"] == "36"
    assert train.loc["lc_000000005", "term"] == "60"


def test_home_ownership_is_normalized_to_configured_values(
    artificial_data_dir: Path,
) -> None:
    """Legacy home-ownership values map to the configured OTHER category."""
    train, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    validation, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "validation",
        FEATURE_CONFIG,
    )
    assert train.loc["lc_000000005", "home_ownership"] == "OTHER"
    assert validation.loc["lc_000000001", "home_ownership"] == "OTHER"


@pytest.mark.parametrize(
    ("column", "replacement", "error_key"),
    [
        ("term", "3 years", "unrecognised_term"),
        ("home_ownership", "CASTLE", "unrecognised_home_ownership"),
    ],
)
def test_unknown_categories_are_rejected(
    artificial_data_dir: Path,
    column: str,
    replacement: str,
    error_key: str,
) -> None:
    """Unexpected non-missing categorical values raise rather than becoming missing."""
    raw_path = _raw_path(artificial_data_dir)
    raw = pd.read_csv(raw_path)
    raw.loc[1, column] = replacement
    raw.to_csv(raw_path, index=False)
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        load_tabular_split(artificial_data_dir, raw_path, "train", FEATURE_CONFIG)


def test_log1p_transformation_matches_hand_computed_value(
    artificial_data_dir: Path,
) -> None:
    """Income log uses only the selected row's annual income."""
    features, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    assert features.loc["lc_000000002", "annual_inc_log"] == pytest.approx(np.log1p(60_000.0))


def test_months_between_matches_hand_computed_value(
    artificial_data_dir: Path,
) -> None:
    """Month-granularity credit history uses only the selected row's dates."""
    features, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    # Jan-2000 to Mar-2013: 13 years * 12 + 2 months.
    assert features.loc["lc_000000002", "credit_history_months"] == pytest.approx(158.0)


def test_feature_dtypes_match_model_contract(
    artificial_data_dir: Path,
    feature_config: dict[str, Any],
) -> None:
    """Numeric features are float64 and categorical features use object dtype."""
    features, _ = load_tabular_split(
        artificial_data_dir,
        _raw_path(artificial_data_dir),
        "train",
        FEATURE_CONFIG,
    )
    for column in feature_config["model"]["numeric"]:
        assert features[column].dtype == np.dtype("float64")
    for column in feature_config["model"]["categorical"]:
        assert features[column].dtype == np.dtype("object")


def test_loading_is_deterministic_across_calls_and_chunk_sizes(
    artificial_data_dir: Path,
) -> None:
    """Repeated loads and one-row chunks produce identical features and targets."""
    raw_path = _raw_path(artificial_data_dir)
    first_features, first_target = load_tabular_split(
        artificial_data_dir,
        raw_path,
        "train",
        FEATURE_CONFIG,
    )
    second_features, second_target = load_tabular_split(
        artificial_data_dir,
        raw_path,
        "train",
        FEATURE_CONFIG,
    )
    chunked_features, chunked_target = load_tabular_split(
        artificial_data_dir,
        raw_path,
        "train",
        FEATURE_CONFIG,
        chunksize=1,
    )
    assert_frame_equal(first_features, second_features)
    assert_series_equal(first_target, second_target)
    assert_frame_equal(first_features, chunked_features)
    assert_series_equal(first_target, chunked_target)


def test_id_file_disagreement_with_manifest_is_rejected(
    artificial_data_dir: Path,
) -> None:
    """A split ID file that disagrees with the frozen manifest stops loading."""
    pd.DataFrame(
        {"loan_id": ["lc_000000005", "lc_000000001"]}
    ).to_csv(artificial_data_dir / "train_ids.csv", index=False)
    with pytest.raises(ValueError, match=error_pattern("id_file_disagrees")):
        load_tabular_split(
            artificial_data_dir,
            _raw_path(artificial_data_dir),
            "train",
            FEATURE_CONFIG,
        )


@pytest.mark.parametrize(
    ("column", "replacement", "error_key"),
    [
        ("issue_d", "Apr-2013", "issue_month_alignment"),
        ("loan_status", "Charged Off", "status_target_alignment"),
    ],
)
def test_raw_reference_disagreement_with_manifest_is_rejected(
    artificial_data_dir: Path,
    column: str,
    replacement: str,
    error_key: str,
) -> None:
    """Raw issue month or outcome disagreement with the manifest stops loading."""
    raw_path = _raw_path(artificial_data_dir)
    raw = pd.read_csv(raw_path)
    raw.loc[1, column] = replacement
    raw.to_csv(raw_path, index=False)
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        load_tabular_split(artificial_data_dir, raw_path, "train", FEATURE_CONFIG)


@pytest.mark.parametrize(
    ("filename", "replacement", "error_key"),
    [
        (
            "wrong_from_count.toml",
            'annual_inc_log = { from = ["annual_inc", "loan_amnt"], transform = "log1p" }',
            "derived_wrong_from",
        ),
        (
            "missing_from.toml",
            'annual_inc_log = { transform = "log1p" }',
            "derived_wrong_from",
        ),
        (
            "unknown_transform.toml",
            'annual_inc_log = { from = ["annual_inc"], transform = "square_root" }',
            "unsupported_transform",
        ),
    ],
)
def test_config_rejects_malformed_derived_features(
    tmp_path: Path,
    filename: str,
    replacement: str,
    error_key: str,
) -> None:
    """Derived features require the implemented transform and its exact input arity."""
    original = 'annual_inc_log = { from = ["annual_inc"], transform = "log1p" }'
    path = _write_modified_config(tmp_path, filename, [(original, replacement)])
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        load_feature_config(path)


@pytest.mark.skipif(
    not os.getenv("LENDINGCLUB_RAW_CSV"),
    reason="Set LENDINGCLUB_RAW_CSV to run the locked real-data regression check.",
)
def test_real_development_split_sizes_and_disjointness() -> None:
    """The frozen real train and validation splits retain their expected disjoint rows."""
    raw_csv_path = Path(os.environ["LENDINGCLUB_RAW_CSV"])
    train, _ = load_tabular_split(
        REAL_DATA_DIR,
        raw_csv_path,
        "train",
        FEATURE_CONFIG,
    )
    validation, _ = load_tabular_split(
        REAL_DATA_DIR,
        raw_csv_path,
        "validation",
        FEATURE_CONFIG,
    )
    assert len(train) == 86_293
    assert len(validation) == 21_784
    assert set(train.index).isdisjoint(validation.index)

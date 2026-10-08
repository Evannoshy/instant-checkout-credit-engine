"""Development-role tests using tiny, artificial examples (no borrower data).

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_tabular_development.py -v -s

The -s flag shows print statements; -v shows each PASSED/FAILED result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.tabular.development import (
    ERROR_MESSAGES,
    get_ablation_feature_sets,
    assign_development_roles,
    join_lexical_features,
    load_ablation_config,
    load_development_config,
    load_tabular_role,
)
from src.tabular.preprocess import load_feature_config, load_tabular_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_features_v1.toml"
DEVELOPMENT_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_development_v1.toml"
ABLATION_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_ablation_v1.toml"
REAL_DATA_DIR = REPOSITORY_ROOT / "data"

# Hardcoded on purpose, so a silent edit to the config's boundaries fails a test
# instead of being echoed back by it.
EXPECTED_ROLE_MONTHS = {
    "model_fit": ("2007-06", "2012-12"),
    "calibration": ("2013-01", "2013-07"),
}
# Fixture train rows on each side of the role boundaries, with the role each must get.
FIXTURE_TRAIN_ROLES = {
    "lc_000000006": "model_fit",    # 2012-11
    "lc_000000005": "model_fit",    # 2012-12, last model_fit month
    "lc_000000003": "calibration",  # 2013-01, first calibration month
    "lc_000000001": "calibration",  # 2013-07, last train month
}
VALIDATION_ID = "lc_000000004"
TEST_ID = "lc_000000002"
MODEL_FIT_ID = "lc_000000005"
UNKNOWN_ID = "lc_000000099"  # in no split of the fixture manifest
# Every train and validation row of the fixture, with its manifest split.
FIXTURE_LEXICAL_SPLITS = {**dict.fromkeys(FIXTURE_TRAIN_ROLES, "train"), VALIDATION_ID: "validation"}
FIXTURE_COUNTS = [
    ("model_fit = 59353", "model_fit = 2"),
    ("calibration = 26940", "calibration = 2"),
]
LOCKED_FILES = (
    "split_manifest.csv", "train_ids.csv", "val_ids.csv", "test_ids.csv", "split_statistics.json",
)


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
    """Six made-up records: four train rows around the role boundary, one validation, one test."""
    (tmp_path / "raw").mkdir()
    raw = pd.DataFrame(
        {
            "loan_amnt": ["11000", "10000", "13000", "9000", "20000", "7000"],
            "term": [" 36 months", " 60 months", " 36 months", " 36 months", " 60 months",
                     " 36 months"],
            "annual_inc": ["48000", "60000", "51000", "45000", "80000", "35000"],
            "dti": ["12.5", "14.0", "18.0", "9.5", "22.0", "15.0"],
            "revol_util": ["30.0", "34.5", "51.0", "19.0", "62.0", "44.0"],
            "delinq_2yrs": ["0", "0", "1", "0", "2", "0"],
            "inq_last_6mths": ["1", "2", "0", "1", "3", "0"],
            "earliest_cr_line": ["Feb-2001", "Jan-2000", "May-2004", "Jun-2005", "Mar-1999",
                                 "Apr-2002"],
            "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT", "MORTGAGE", "RENT"],
            "issue_d": ["Jul-2013", "Jan-2014", "Jan-2013", "Aug-2013", "Dec-2012", "Nov-2012"],
            "loan_status": ["Fully Paid", "Charged Off", "Charged Off", "Fully Paid",
                            "Fully Paid", "Charged Off"],
        }
    )
    raw.to_csv(tmp_path / "raw" / "loan.csv", index=False)

    version = "synthetic-development-fixture-v1"
    manifest = pd.DataFrame(
        {
            "loan_id": [f"lc_{number:09d}" for number in range(1, 7)],
            "source_row_number": [1, 2, 3, 4, 5, 6],
            "split": ["train", "test", "train", "validation", "train", "train"],
            "issue_month": ["2013-07", "2014-01", "2013-01", "2013-08", "2012-12", "2012-11"],
            "target": [0, "LOCKED_TEST_LABEL", 1, 0, 0, 1],
            "text_available": [1] * 6,
            "cohort": ["real_text_matured_v1"] * 6,
            "dataset_version": [version] * 6,
        }
    )
    manifest.to_csv(tmp_path / "split_manifest.csv", index=False)
    for split, filename in [("train", "train_ids.csv"), ("validation", "val_ids.csv"),
                            ("test", "test_ids.csv")]:
        manifest.loc[manifest["split"].eq(split), ["loan_id"]].to_csv(
            tmp_path / filename, index=False
        )

    statistics = {
        "dataset_version": version,
        "split_version": "split-v1",
        "cohort": "real_text_matured_v1",
        "raw_rows": 6,
        "eligible_rows": 6,
        "splits": {"train": {"rows": 4}, "validation": {"rows": 1}, "test": {"rows": 1}},
    }
    (tmp_path / "split_statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    return tmp_path


def _write_development_config(
    tmp_path: Path,
    filename: str,
    replacements: list[tuple[str, str]],
) -> Path:
    text = DEVELOPMENT_CONFIG.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in text, f"Expected development-config text was not found: {old}"
        text = text.replace(old, new, 1)
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def development_config(tmp_path: Path) -> Path:
    """The real development config with expected role counts set to the fixture's 2 / 2."""
    return _write_development_config(tmp_path, "fixture_development.toml", FIXTURE_COUNTS)


def _load_role(
    data_dir: Path, role: str, development_config: Path
) -> tuple[pd.DataFrame, pd.Series]:
    return load_tabular_role(
        data_dir, data_dir / "raw" / "loan.csv", role, FEATURE_CONFIG, development_config,
    )


def test_roles_are_disjoint_and_cover_exactly_the_train_ids(
    artificial_data_dir: Path, development_config: Path
) -> None:
    """model_fit and calibration share no loan_id and together equal the train ID file."""
    model_fit, _ = _load_role(artificial_data_dir, "model_fit", development_config)
    calibration, _ = _load_role(artificial_data_dir, "calibration", development_config)
    train_ids = set(pd.read_csv(artificial_data_dir / "train_ids.csv")["loan_id"])
    assert set(model_fit.index).isdisjoint(calibration.index)
    assert set(model_fit.index) | set(calibration.index) == train_ids


def test_boundary_months_fall_in_the_hardcoded_roles(
    artificial_data_dir: Path, development_config: Path
) -> None:
    """Config boundaries equal the hardcoded months; 2012-12 is model_fit, 2013-01 calibration."""
    config = load_development_config(DEVELOPMENT_CONFIG)
    configured = {
        role: (bounds["first_month"], bounds["last_month"])
        for role, bounds in config["roles"].items()
    }
    assert configured == EXPECTED_ROLE_MONTHS

    roles = assign_development_roles(artificial_data_dir, development_config)
    assert roles["role"].to_dict() == FIXTURE_TRAIN_ROLES


def test_validation_and_test_ids_never_enter_a_role(
    artificial_data_dir: Path, development_config: Path
) -> None:
    """Neither role contains the fixture's validation row or its locked test row."""
    for role in ("model_fit", "calibration"):
        features, _ = _load_role(artificial_data_dir, role, development_config)
        assert VALIDATION_ID not in features.index
        assert TEST_ID not in features.index


@pytest.mark.parametrize(
    ("role", "error_key"),
    [
        ("test", "test_split_locked"),
        ("validation", "split_not_a_role"),
        ("train", "split_not_a_role"),
        ("other", "unknown_role"),
    ],
)
def test_loader_rejects_test_split_names_and_unknown_roles(
    tmp_path: Path, role: str, error_key: str
) -> None:
    """The locked test split, split names and unknown roles are rejected before any file is read."""
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        load_tabular_role(
            missing, missing / "loan.csv", role, missing / "features.toml",
            missing / "development.toml",
        )


def test_loan_ids_are_unique_within_each_role(
    artificial_data_dir: Path, development_config: Path
) -> None:
    """Each role's loan_id index has no duplicates."""
    for role in ("model_fit", "calibration"):
        features, _ = _load_role(artificial_data_dir, role, development_config)
        assert features.index.is_unique


def test_role_count_mismatch_is_rejected(artificial_data_dir: Path, tmp_path: Path) -> None:
    """Role counts that differ from [expected.role_rows] stop the assignment."""
    path = _write_development_config(
        tmp_path,
        "wrong_count.toml",
        [("model_fit = 59353", "model_fit = 3"), ("calibration = 26940", "calibration = 2")],
    )
    with pytest.raises(ValueError, match=error_pattern("role_count_mismatch")):
        assign_development_roles(artificial_data_dir, path)


@pytest.mark.parametrize(
    ("filename", "old", "new", "error_key"),
    [
        ("overlap.toml", 'first_month = "2013-01"', 'first_month = "2012-12"', "roles_overlap"),
        ("gap.toml", 'first_month = "2013-01"', 'first_month = "2013-02"', "roles_gap"),
        (
            "out_of_range.toml",
            'last_month = "2013-07"',
            'last_month = "2013-08"',
            "roles_do_not_cover_train",
        ),
    ],
)
def test_config_rejects_overlapping_gapped_and_out_of_range_months(
    tmp_path: Path, filename: str, old: str, new: str, error_key: str
) -> None:
    """Role months must be contiguous, non-overlapping and cover the train months exactly."""
    path = _write_development_config(tmp_path, filename, [(old, new)])
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        load_development_config(path)


def test_features_and_target_are_aligned_and_target_is_not_a_feature(
    artificial_data_dir: Path, development_config: Path
) -> None:
    """Each role's features and target share one index, and the target is not a column."""
    for role in ("model_fit", "calibration"):
        features, target = _load_role(artificial_data_dir, role, development_config)
        assert features.index.equals(target.index)
        assert target.name not in features.columns


def test_repeated_loads_are_identical(
    artificial_data_dir: Path, development_config: Path
) -> None:
    """Two loads of the same role produce identical features and targets."""
    first_features, first_target = _load_role(artificial_data_dir, "model_fit", development_config)
    second_features, second_target = _load_role(
        artificial_data_dir, "model_fit", development_config
    )
    assert_frame_equal(first_features, second_features)
    assert_series_equal(first_target, second_target)


def _file_hashes(data_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in LOCKED_FILES
    }


def test_locked_split_files_are_unchanged_after_loading(
    artificial_data_dir: Path, development_config: Path
) -> None:
    """Loading roles leaves the manifest, ID files and split statistics byte-identical."""
    before = _file_hashes(artificial_data_dir)
    assign_development_roles(artificial_data_dir, development_config)
    for role in ("model_fit", "calibration"):
        _load_role(artificial_data_dir, role, development_config)
    assert _file_hashes(artificial_data_dir) == before


@pytest.mark.skipif(
    not os.getenv("LENDINGCLUB_RAW_CSV"),
    reason="Set LENDINGCLUB_RAW_CSV to run the locked real-data regression check.",
)
def test_real_role_sizes_and_union() -> None:
    """The real roles hold 59,353 and 26,940 rows, together the 86,293 train rows."""
    raw_csv_path = Path(os.environ["LENDINGCLUB_RAW_CSV"])
    model_fit, _ = load_tabular_role(REAL_DATA_DIR, raw_csv_path, "model_fit")
    calibration, _ = load_tabular_role(REAL_DATA_DIR, raw_csv_path, "calibration")
    assert len(model_fit) == 59_353
    assert len(calibration) == 26_940
    assert len(set(model_fit.index) | set(calibration.index)) == 86_293


# --- Ablation families ---
# Families and columns are read from the configs, not hardcoded, so these tests
# keep working when features or families are added.


@pytest.fixture
def feature_config() -> dict:
    return load_feature_config(FEATURE_CONFIG)


def _families(feature_config: dict) -> dict[str, list[str]]:
    config = load_ablation_config(ABLATION_CONFIG, feature_config)
    return {name: spec["features"] for name, spec in config["families"].items()}


def _feature_sets(feature_config: dict) -> dict[str, dict]:
    return get_ablation_feature_sets(
        feature_config, load_ablation_config(ABLATION_CONFIG, feature_config)
    )


def _model_features(config: dict) -> list[str]:
    return [*config["model"]["numeric"], *config["model"]["categorical"]]


def test_every_model_feature_is_in_exactly_one_family(feature_config: dict) -> None:
    """The families list every model feature exactly once and nothing else."""
    listed = [name for features in _families(feature_config).values() for name in features]
    assert len(listed) == len(set(listed))
    assert set(listed) == set(_model_features(feature_config))


def test_all_set_equals_the_model_features(feature_config: dict) -> None:
    """The "all" variant keeps the feature config's lists; one without_ set follows per family."""
    sets = _feature_sets(feature_config)
    assert sets["all"]["model"] == feature_config["model"]
    assert list(sets) == ["all", *(f"without_{name}" for name in _families(feature_config))]


def test_each_without_set_removes_exactly_its_family_in_config_order(feature_config: dict) -> None:
    """Each without_<family> keeps every other model feature, in the original order per list."""
    sets = _feature_sets(feature_config)
    for family, removed in _families(feature_config).items():
        model = sets[f"without_{family}"]["model"]
        for kind in ("numeric", "categorical"):
            expected = [name for name in feature_config["model"][kind] if name not in removed]
            assert model[kind] == expected, family


def test_variants_are_independent_copies(feature_config: dict) -> None:
    """Building variants leaves the feature config intact; editing one variant changes no other."""
    before = copy.deepcopy(feature_config)
    sets = _feature_sets(feature_config)
    assert feature_config == before

    for name, variant in sets.items():
        if name != "all":
            variant["model"]["numeric"].clear()
            variant["model"]["categorical"].clear()
    assert sets["all"]["model"] == before["model"]
    assert feature_config == before


# --- Lexical join ---
# The fixture parquet covers the four train rows and the validation row. Each
# value encodes its loan and column (lc_000000005, column 2 -> 5.2), so a value
# that lands on the wrong row is detectable. Every file a test writes gets its
# own SHA-256 pin, so these tests reach the content checks, not the SHA check.


def _lexical_columns() -> list[str]:
    return load_development_config(DEVELOPMENT_CONFIG)["lexical"]["columns"]


def _lexical_value(loan_id: str, column: int) -> float:
    return int(loan_id.removeprefix("lc_")) + column / 10


@pytest.fixture
def lexical_table() -> pd.DataFrame:
    """A valid lexical file for the fixture's train and validation rows."""
    table = pd.DataFrame({
        "loan_id": list(FIXTURE_LEXICAL_SPLITS),
        "split": list(FIXTURE_LEXICAL_SPLITS.values()),
    })
    for column, name in enumerate(_lexical_columns()):
        table[name] = [_lexical_value(loan_id, column) for loan_id in table["loan_id"]]
    return table


def _write_lexical_config(tmp_path: Path, table: pd.DataFrame) -> Path:
    """Write the table as parquet and a fixture development config pinned to it."""
    lexical = load_development_config(DEVELOPMENT_CONFIG)["lexical"]
    path = tmp_path / "lexical.parquet"
    table.to_parquet(path, engine="pyarrow", index=False)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return _write_development_config(
        tmp_path,
        "lexical_development.toml",
        [
            *FIXTURE_COUNTS,
            (f'path = "{lexical["path"]}"', f'path = "{path.as_posix()}"'),
            (f'sha256 = "{lexical["sha256"]}"', f'sha256 = "{sha256}"'),
        ],
    )


def test_lexical_file_changed_after_pinning_is_rejected(
    artificial_data_dir: Path, tmp_path: Path, lexical_table: pd.DataFrame
) -> None:
    """A lexical file rewritten after its SHA-256 was pinned stops the join, even if otherwise valid."""
    config = _write_lexical_config(tmp_path, lexical_table)
    regenerated = lexical_table.copy()
    regenerated[_lexical_columns()[0]] += 1.0
    regenerated.to_parquet(
        load_development_config(config)["lexical"]["path"], engine="pyarrow", index=False,
    )
    features, target = _load_role(artificial_data_dir, "model_fit", config)
    with pytest.raises(ValueError, match=error_pattern("lexical_sha_mismatch")):
        join_lexical_features(features, target, artificial_data_dir, config)


def test_duplicate_loan_ids_are_rejected(
    artificial_data_dir: Path, tmp_path: Path, lexical_table: pd.DataFrame
) -> None:
    """A loan_id repeated in the lexical file or in the requested rows stops the join."""
    repeated_in_file = pd.concat([lexical_table, lexical_table.iloc[[0]]], ignore_index=True)
    config = _write_lexical_config(tmp_path, repeated_in_file)
    features, target = _load_role(artificial_data_dir, "model_fit", config)
    with pytest.raises(ValueError, match=error_pattern("lexical_bad_ids")):
        join_lexical_features(features, target, artificial_data_dir, config)

    config = _write_lexical_config(tmp_path, lexical_table)
    rows = [0, 0, 1]
    with pytest.raises(ValueError, match=error_pattern("lexical_input_not_unique")):
        join_lexical_features(
            features.iloc[rows], target.iloc[rows], artificial_data_dir, config,
        )


@pytest.mark.parametrize(
    ("split", "error_key"),
    [("validation", "lexical_split_mismatch"), ("holdout", "lexical_bad_splits")],
)
def test_file_split_must_match_the_manifest(
    artificial_data_dir: Path,
    tmp_path: Path,
    lexical_table: pd.DataFrame,
    split: str,
    error_key: str,
) -> None:
    """A train loan the file labels validation, or a split outside train/validation, stops the join."""
    table = lexical_table.copy()
    table.loc[table["loan_id"].eq(MODEL_FIT_ID), "split"] = split
    config = _write_lexical_config(tmp_path, table)
    features, target = _load_role(artificial_data_dir, "model_fit", config)
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        join_lexical_features(features, target, artificial_data_dir, config)


@pytest.mark.parametrize(
    ("loan_id", "error_key"),
    [(TEST_ID, "lexical_test_ids"), (UNKNOWN_ID, "lexical_ids_not_in_manifest")],
)
def test_unexpected_ids_in_the_file_are_rejected(
    artificial_data_dir: Path,
    tmp_path: Path,
    lexical_table: pd.DataFrame,
    loan_id: str,
    error_key: str,
) -> None:
    """A locked test ID or an ID absent from the manifest anywhere in the file stops the join."""
    extra = lexical_table.iloc[[0]].assign(loan_id=loan_id)
    config = _write_lexical_config(tmp_path, pd.concat([lexical_table, extra], ignore_index=True))
    features, target = _load_role(artificial_data_dir, "model_fit", config)
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        join_lexical_features(features, target, artificial_data_dir, config)


def test_join_adds_no_rows_for_other_loans_in_the_file(
    artificial_data_dir: Path, tmp_path: Path, lexical_table: pd.DataFrame
) -> None:
    """The file also covers other loans; the output holds exactly the requested rows."""
    config = _write_lexical_config(tmp_path, lexical_table)
    features, target = _load_role(artificial_data_dir, "model_fit", config)
    joined, _ = join_lexical_features(features, target, artificial_data_dir, config)
    assert len(lexical_table) > len(features)
    assert joined.index.equals(features.index)


def test_requested_loan_absent_from_the_file_is_rejected(
    artificial_data_dir: Path, tmp_path: Path, lexical_table: pd.DataFrame
) -> None:
    """A requested loan with no row in the lexical file stops the join."""
    table = lexical_table.loc[lexical_table["loan_id"].ne(MODEL_FIT_ID)]
    config = _write_lexical_config(tmp_path, table)
    features, target = _load_role(artificial_data_dir, "model_fit", config)
    assert MODEL_FIT_ID in features.index
    with pytest.raises(ValueError, match=error_pattern("lexical_ids_missing")):
        join_lexical_features(features, target, artificial_data_dir, config)


def test_missing_lexical_value_is_rejected(
    artificial_data_dir: Path, tmp_path: Path, lexical_table: pd.DataFrame
) -> None:
    """A requested loan whose lexical value is missing stops the join."""
    table = lexical_table.copy()
    table.loc[table["loan_id"].eq(MODEL_FIT_ID), _lexical_columns()[-1]] = float("nan")
    config = _write_lexical_config(tmp_path, table)
    features, target = _load_role(artificial_data_dir, "model_fit", config)
    with pytest.raises(ValueError, match=error_pattern("lexical_missing_values")):
        join_lexical_features(features, target, artificial_data_dir, config)


@pytest.mark.parametrize("rows", ["model_fit", "calibration", "validation"])
def test_rows_and_target_stay_aligned_when_the_file_is_shuffled(
    artificial_data_dir: Path, tmp_path: Path, lexical_table: pd.DataFrame, rows: str
) -> None:
    """With the file reversed, each row gets its own loan's values; index, features and target are unchanged."""
    config = _write_lexical_config(tmp_path, lexical_table.iloc[::-1])
    if rows == "validation":
        features, target = load_tabular_split(
            artificial_data_dir, artificial_data_dir / "raw" / "loan.csv", "validation",
            FEATURE_CONFIG,
        )
    else:
        features, target = _load_role(artificial_data_dir, rows, config)
    joined, joined_target = join_lexical_features(features, target, artificial_data_dir, config)

    columns = _lexical_columns()
    assert list(joined.columns) == [*features.columns, *columns]
    assert joined.index.equals(features.index)
    assert_frame_equal(joined[features.columns], features)
    assert_series_equal(joined_target, target)
    assert joined_target.index.equals(joined.index)
    expected = pd.DataFrame(
        {
            name: [_lexical_value(loan_id, column) for loan_id in features.index]
            for column, name in enumerate(columns)
        },
        index=features.index,
    )
    assert_frame_equal(joined[columns], expected)


@pytest.mark.skipif(
    not os.getenv("LENDINGCLUB_RAW_CSV"),
    reason="Set LENDINGCLUB_RAW_CSV to run the locked real-data regression check.",
)
def test_real_lexical_join_on_train_and_validation() -> None:
    """The pinned real file joins onto all 86,293 train and 21,784 validation rows with zero NaN."""
    raw_csv_path = Path(os.environ["LENDINGCLUB_RAW_CSV"])
    columns = _lexical_columns()
    for split, expected_rows in (("train", 86_293), ("validation", 21_784)):
        features, target = load_tabular_split(REAL_DATA_DIR, raw_csv_path, split)
        joined, joined_target = join_lexical_features(features, target, REAL_DATA_DIR)
        assert len(joined) == expected_rows, split
        assert joined.index.equals(features.index), split
        assert list(joined.columns) == [*features.columns, *columns], split
        assert int(joined[columns].isna().sum().sum()) == 0, split
        assert joined_target.index.equals(joined.index), split


def test_misaligned_features_and_target_are_rejected(
    artificial_data_dir: Path, tmp_path: Path, lexical_table: pd.DataFrame
) -> None:
    """Features and target whose rows are in different orders stop the join."""
    config = _write_lexical_config(tmp_path, lexical_table)
    features, target = _load_role(artificial_data_dir, "model_fit", config)
    with pytest.raises(ValueError, match=error_pattern("output_not_aligned")):
        join_lexical_features(features, target.iloc[::-1], artificial_data_dir, config)


def test_logistic_pipeline_uses_all_joined_lexical_features(
    artificial_data_dir: Path, tmp_path: Path, lexical_table: pd.DataFrame
) -> None:
    """All six joined columns reach the fitted logistic model without changing the v1 config."""
    from src.tabular.logistic_baseline import build_pipeline, load_logistic_config

    development_config = _write_lexical_config(tmp_path, lexical_table)
    features, target = _load_role(artificial_data_dir, "model_fit", development_config)
    joined, target = join_lexical_features(
        features, target, artificial_data_dir, development_config
    )

    feature_config = load_feature_config(FEATURE_CONFIG)
    original_config = copy.deepcopy(feature_config)
    lexical_config = copy.deepcopy(feature_config)
    lexical_columns = _lexical_columns()
    lexical_config["model"]["numeric"].extend(lexical_columns)

    pipeline = build_pipeline(lexical_config, load_logistic_config()).fit(joined, target)
    transformer = pipeline.named_steps["preprocess"]
    output_names = transformer.get_feature_names_out().tolist()
    transformed = transformer.transform(joined)

    assert feature_config == original_config
    assert list(pipeline.feature_names_in_) == list(joined.columns)
    assert len(pipeline.named_steps["classify"].coef_[0]) == len(output_names)

    for column in lexical_columns:
        name = f"numeric__{column}"
        assert output_names.count(name) == 1
        changed = joined.copy()
        changed.loc[joined.index[0], column] += 1.0
        changed_transformed = transformer.transform(changed)
        position = output_names.index(name)
        assert transformed[0, position] != changed_transformed[0, position], column

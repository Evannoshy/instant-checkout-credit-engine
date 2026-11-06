"""Fusion role-mapping tests: tiny artificial manifests plus the committed real manifest.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_fusion_roles.py -v -s

Nothing here reads data/raw/loan.csv, so no test is skipped for a missing
LENDINGCLUB_RAW_CSV. The two "committed" tests need data/fusion/fusion_roles_v1.csv
and its manifest, written by ``python -m src.tabular.fusion_roles``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.tabular import development, fusion_roles, preprocess, provenance
from src.tabular.fusion_roles import (
    ERROR_MESSAGES,
    MAPPING_COLUMNS,
    SCORING_ROLES,
    assign_fusion_roles,
    load_fusion_config,
    run_role_mapping,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FUSION_CONFIG = REPOSITORY_ROOT / "configs" / "fusion_development_v1.toml"
REAL_DATA_DIR = REPOSITORY_ROOT / "data"

# Hardcoded on purpose, so a silent edit to the config's boundaries fails a test
# instead of being echoed back by it.
EXPECTED_ROLE_MONTHS = {
    "base_fit": ("train", "2007-06", "2012-12"),
    "fusion_fit": ("train", "2013-01", "2013-04"),
    "fusion_calibration": ("train", "2013-05", "2013-07"),
    "validation": ("validation", "2013-08", "2013-12"),
}
REAL_ROLE_ROWS = {
    "base_fit": 59_353, "fusion_fit": 15_587, "fusion_calibration": 11_353, "validation": 21_784,
}
# Fixture rows on each side of every role boundary, with the role each must get.
FIXTURE_ROLES = {
    "lc_000000001": "base_fit",            # 2012-12, last base_fit month
    "lc_000000003": "fusion_fit",          # 2013-01, first fusion_fit month
    "lc_000000004": "validation",          # 2013-08, first validation month
    "lc_000000005": "fusion_fit",          # 2013-04, last fusion_fit month
    "lc_000000006": "fusion_calibration",  # 2013-05, first fusion_calibration month
    "lc_000000007": "validation",          # 2013-12, last validation month
    "lc_000000008": "fusion_calibration",  # 2013-07, last train month
    "lc_000000009": "base_fit",            # 2007-06, first train month
}
TEST_ID = "lc_000000002"
FIXTURE_COUNTS = [
    ("base_fit = 59353", "base_fit = 2"),
    ("fusion_fit = 15587", "fusion_fit = 2"),
    ("fusion_calibration = 11353", "fusion_calibration = 2"),
    ("validation = 21784", "validation = 2"),
    ("nlp_scoring_rows = 48724", "nlp_scoring_rows = 6"),
    ("locked_test_rows = 14922", "locked_test_rows = 1"),
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
    """Nine made-up manifest rows: six train and two validation around every boundary, one test."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    version = "synthetic-fusion-roles-fixture-v1"
    manifest = pd.DataFrame(
        {
            "loan_id": [f"lc_{number:09d}" for number in range(1, 10)],
            "source_row_number": list(range(1, 10)),
            "split": ["train", "test", "train", "validation", "train", "train", "validation",
                      "train", "train"],
            "issue_month": ["2012-12", "2014-01", "2013-01", "2013-08", "2013-04", "2013-05",
                            "2013-12", "2013-07", "2007-06"],
            "target": [0, "LOCKED_TEST_LABEL", 1, 0, 0, 1, 0, 1, 0],
            "text_available": [1] * 9,
            "cohort": ["real_text_matured_v1"] * 9,
            "dataset_version": [version] * 9,
        }
    )
    manifest.to_csv(data_dir / "split_manifest.csv", index=False)
    for split, filename in [("train", "train_ids.csv"), ("validation", "val_ids.csv"),
                            ("test", "test_ids.csv")]:
        manifest.loc[manifest["split"].eq(split), ["loan_id"]].to_csv(data_dir / filename, index=False)

    statistics = {
        "dataset_version": version,
        "split_version": "split-v1",
        "cohort": "real_text_matured_v1",
        "eligible_rows": 9,
        "splits": {"train": {"rows": 6}, "validation": {"rows": 2}, "test": {"rows": 1}},
    }
    (data_dir / "split_statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    return data_dir


def _write_fusion_config(tmp_path: Path, filename: str, replacements: list[tuple[str, str]]) -> Path:
    text = FUSION_CONFIG.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in text, f"Expected fusion-config text was not found: {old}"
        text = text.replace(old, new, 1)
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def fusion_config(tmp_path: Path) -> Path:
    """The real fusion config with expected counts set to the fixture's."""
    return _write_fusion_config(tmp_path, "fixture_fusion.toml", FIXTURE_COUNTS)


def _run(data_dir: Path, config: Path, output_dir: Path) -> dict:
    return run_role_mapping(
        data_dir, output_dir / "fusion_roles.csv", output_dir / "manifest.json", config,
    )


# --- Role assignment on the fixture ---


def test_boundary_months_fall_in_the_hardcoded_roles(
    artificial_data_dir: Path, fusion_config: Path
) -> None:
    """Config boundaries equal the hardcoded months, and each boundary row gets its role."""
    config = load_fusion_config(FUSION_CONFIG)
    configured = {
        role: (bounds["split"], bounds["first_month"], bounds["last_month"])
        for role, bounds in config["roles"].items()
    }
    assert configured == EXPECTED_ROLE_MONTHS

    mapping = assign_fusion_roles(artificial_data_dir, fusion_config)
    assert mapping.set_index("loan_id")["role"].to_dict() == FIXTURE_ROLES


def test_roles_are_disjoint_and_cover_exactly_the_split_ids(
    artificial_data_dir: Path, fusion_config: Path
) -> None:
    """Each loan appears once; train roles together equal train_ids.csv; validation equals val_ids.csv."""
    mapping = assign_fusion_roles(artificial_data_dir, fusion_config)
    assert mapping["loan_id"].is_unique
    train_ids = set(pd.read_csv(artificial_data_dir / "train_ids.csv")["loan_id"])
    val_ids = set(pd.read_csv(artificial_data_dir / "val_ids.csv")["loan_id"])
    train_roles = mapping.loc[mapping["role"].isin(["base_fit", "fusion_fit", "fusion_calibration"])]
    assert set(train_roles["loan_id"]) == train_ids
    assert set(mapping.loc[mapping["role"].eq("validation"), "loan_id"]) == val_ids


def test_test_ids_and_targets_never_enter_the_mapping(
    artificial_data_dir: Path, fusion_config: Path
) -> None:
    """The locked test row is absent, and the mapping has exactly the four label-free columns."""
    mapping = assign_fusion_roles(artificial_data_dir, fusion_config)
    test_ids = set(pd.read_csv(artificial_data_dir / "test_ids.csv")["loan_id"])
    assert TEST_ID in test_ids
    assert set(mapping["loan_id"]).isdisjoint(test_ids)
    assert list(mapping.columns) == list(MAPPING_COLUMNS)


def test_split_column_keeps_the_manifest_split(
    artificial_data_dir: Path, fusion_config: Path
) -> None:
    """Every loan's split equals the manifest's split; a role name never replaces it."""
    mapping = assign_fusion_roles(artificial_data_dir, fusion_config).set_index("loan_id")
    manifest = pd.read_csv(artificial_data_dir / "split_manifest.csv").set_index("loan_id")
    assert mapping["split"].eq(manifest.loc[mapping.index, "split"]).all()
    assert set(mapping["split"]) == {"train", "validation"}


@pytest.mark.parametrize(
    ("old", "new", "error_key"),
    [
        ('first_month = "2013-01"', 'first_month = "2012-12"', "roles_overlap"),
        ('first_month = "2013-01"', 'first_month = "2013-02"', "roles_gap"),
        ('last_month = "2013-07"', 'last_month = "2013-06"', "roles_do_not_cover_split"),
        ('first_month = "2013-08"', 'first_month = "2013-07"', "roles_do_not_cover_split"),
        ('split = "validation"', 'split = "test"', "test_split_locked"),
        ('split = "validation"', 'split = "holdout"', "bad_role_split"),
        ('first_month = "2013-08"', 'first_month = "2013-8"', "bad_month_from_development"),
        ("nlp_scoring_rows = 48724", "nlp_scoring_rows = 48725", "scoring_rows_mismatch"),
    ],
)
def test_config_rejects_bad_role_definitions(tmp_path: Path, old: str, new: str, error_key: str) -> None:
    """Overlapping, gapped, out-of-range, test, unknown-split, malformed or miscounted roles are rejected."""
    path = _write_fusion_config(tmp_path, "bad.toml", [(old, new)])
    if error_key == "bad_month_from_development":
        # Month format is checked by the reused development helper and its message.
        literal_parts = re.split(r"\{[^}]*\}", development.ERROR_MESSAGES["bad_month"])
        pattern = "^" + ".*".join(re.escape(part) for part in literal_parts) + "$"
    else:
        pattern = error_pattern(error_key)
    with pytest.raises(ValueError, match=pattern):
        load_fusion_config(path)


@pytest.mark.parametrize(
    ("replacements", "error_key"),
    [
        ([("base_fit = 2", "base_fit = 3")], "role_count_mismatch"),
        ([("locked_test_rows = 1", "locked_test_rows = 2")], "test_rows_mismatch"),
    ],
)
def test_assignment_rejects_counts_that_differ_from_the_manifest(
    artificial_data_dir: Path, tmp_path: Path, replacements: list, error_key: str
) -> None:
    """Role or test-row counts that differ from the manifest stop the assignment."""
    path = _write_fusion_config(tmp_path, "wrong_count.toml", [*FIXTURE_COUNTS, *replacements])
    with pytest.raises(ValueError, match=error_pattern(error_key)):
        assign_fusion_roles(artificial_data_dir, path)


def test_row_outside_every_role_is_rejected(artificial_data_dir: Path, fusion_config: Path) -> None:
    """A train row whose month no role covers stops the assignment instead of being dropped."""
    manifest_path = artificial_data_dir / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest.loc[manifest["loan_id"].eq("lc_000000009"), "issue_month"] = "2007-05"
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match=error_pattern("rows_not_in_one_role")):
        assign_fusion_roles(artificial_data_dir, fusion_config)


def test_mapping_id_listed_in_test_ids_is_rejected(
    artificial_data_dir: Path, fusion_config: Path
) -> None:
    """A mapping loan that test_ids.csv also lists stops the run, so the test-ID count is measured."""
    test_ids_path = artificial_data_dir / "test_ids.csv"
    test_ids = pd.read_csv(test_ids_path)
    pd.concat([test_ids, pd.DataFrame({"loan_id": ["lc_000000001"]})]).to_csv(test_ids_path, index=False)
    with pytest.raises(ValueError, match=error_pattern("test_ids_in_mapping")):
        assign_fusion_roles(artificial_data_dir, fusion_config)


def test_assignment_never_loads_the_raw_csv(
    artificial_data_dir: Path, fusion_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assigning roles works with every raw-CSV loader replaced by one that raises."""
    def refuse(*args, **kwargs):
        raise AssertionError("the raw CSV must not be read")

    monkeypatch.setattr(preprocess, "load_tabular_split", refuse)
    monkeypatch.setattr(development, "load_tabular_split", refuse)
    assert len(assign_fusion_roles(artificial_data_dir, fusion_config)) == len(FIXTURE_ROLES)


# --- Writing the mapping ---


def _file_hashes(data_dir: Path) -> dict[str, str]:
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in LOCKED_FILES}


def test_two_runs_write_identical_files_and_leave_locked_files_unchanged(
    artificial_data_dir: Path, fusion_config: Path, tmp_path: Path
) -> None:
    """Two runs give byte-identical CSVs and equal manifests; the split files are untouched."""
    before = _file_hashes(artificial_data_dir)
    first = _run(artificial_data_dir, fusion_config, tmp_path / "first")
    second = _run(artificial_data_dir, fusion_config, tmp_path / "second")
    assert _file_hashes(artificial_data_dir) == before

    first_csv = (tmp_path / "first" / "fusion_roles.csv").read_bytes()
    assert first_csv == (tmp_path / "second" / "fusion_roles.csv").read_bytes()
    assert b"\r\n" not in first_csv
    # Output paths differ by directory only; the CSV bytes were compared above.
    del first["output"], second["output"]
    assert first == second


def test_written_manifest_records_the_mapping(
    artificial_data_dir: Path, fusion_config: Path, tmp_path: Path
) -> None:
    """The manifest's checksum, counts and month ranges describe the CSV that was written."""
    manifest = _run(artificial_data_dir, fusion_config, tmp_path)
    csv_path = tmp_path / "fusion_roles.csv"
    written = pd.read_csv(csv_path, dtype=str)

    assert manifest["output"]["sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert manifest["output"]["rows"] == len(written) == len(FIXTURE_ROLES)
    assert manifest["output"]["columns"] == list(MAPPING_COLUMNS)
    assert manifest["roles"]["fusion_fit"] == {
        "split": "train", "rows": 2, "first_month": "2013-01", "last_month": "2013-04",
    }
    assert manifest["nlp_scoring_rows"] == 6
    assert manifest["test_rows_in_mapping"] == 0 and manifest["targets_included"] is False
    assert manifest["dataset_version"] == "synthetic-fusion-roles-fixture-v1"


# --- Real manifest (committed files only; no raw CSV) ---


def test_real_role_counts() -> None:
    """The real manifest gives 59,353 / 15,587 / 11,353 / 21,784 and 48,724 NLP scoring loans."""
    mapping = assign_fusion_roles(REAL_DATA_DIR)
    assert mapping["role"].value_counts().to_dict() == REAL_ROLE_ROWS
    assert int(mapping["role"].isin(SCORING_ROLES).sum()) == 48_724
    assert len(mapping) == 108_077


def test_real_roles_match_tabular_development_v1() -> None:
    """base_fit equals model_fit, and fusion_fit + fusion_calibration equal calibration."""
    config = load_fusion_config(FUSION_CONFIG)
    assert config["tabular_development_version"] == (
        development.load_development_config()["development_version"]
    )
    mapping = assign_fusion_roles(REAL_DATA_DIR)
    tabular = development.assign_development_roles(REAL_DATA_DIR)

    def ids(frame: pd.DataFrame, roles: list[str]) -> set[str]:
        return set(frame.loc[frame["role"].isin(roles), "loan_id"])

    tabular = tabular.reset_index()
    assert ids(mapping, ["base_fit"]) == ids(tabular, ["model_fit"])
    assert ids(mapping, ["fusion_fit", "fusion_calibration"]) == ids(tabular, ["calibration"])


def test_committed_mapping_equals_a_fresh_assignment() -> None:
    """The committed fusion_roles_v1.csv equals the mapping rebuilt from the manifest."""
    committed = pd.read_csv(fusion_roles.ROLE_MAPPING_PATH, dtype=str)
    assert_frame_equal(committed, assign_fusion_roles(REAL_DATA_DIR).astype(str))


def test_committed_manifest_checksums_match_the_files() -> None:
    """The committed manifest's output and input SHA-256 values match the files on disk."""
    manifest = json.loads(fusion_roles.ROLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    output = manifest["output"]
    assert REPOSITORY_ROOT / output["path"] == fusion_roles.ROLE_MAPPING_PATH
    provenance.verify_file_checksum(fusion_roles.ROLE_MAPPING_PATH, output["sha256"], allow_crlf=True)
    for path, sha256 in manifest["inputs_sha256"].items():
        provenance.verify_file_checksum(REPOSITORY_ROOT / path, sha256, allow_crlf=True)
    assert manifest["roles"] == {
        role: {"split": split, "rows": REAL_ROLE_ROWS[role], "first_month": first, "last_month": last}
        for role, (split, first, last) in EXPECTED_ROLE_MONTHS.items()
    }

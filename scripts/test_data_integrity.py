"""Data integrity checks for data/*_ids.csv and data/split_manifest.csv against data/raw/loan.csv.

From the repository root, show each check and its actual pytest result with:
    python -m pytest scripts/test_data_integrity.py -v -s

The -s flag shows print statements; -v shows each PASSED/FAILED result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

EXPECTED_COUNTS = {"train": 86293, "val": 21784, "test": 14922}
ID_FILES = {
    "train": DATA_DIR / "train_ids.csv",
    "val": DATA_DIR / "val_ids.csv",
    "test": DATA_DIR / "test_ids.csv",
}
MANIFEST_SPLIT_NAME = {"train": "train", "val": "validation", "test": "test"}


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture(scope="module")
def ids():
    return {name: pd.read_csv(path)["loan_id"] for name, path in ID_FILES.items()}


@pytest.fixture(scope="module")
def manifest():
    return pd.read_csv(DATA_DIR / "split_manifest.csv")


@pytest.fixture(scope="module")
def raw_row_count():
    with open(RAW_DIR / "loan.csv", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1  # exclude header


@pytest.fixture(scope="module")
def raw_fields_for_manifest_rows(manifest):
    """issue_d and loan_status from loan.csv, aligned to each manifest row's source_row_number."""
    wanted = np.fromiter(manifest["source_row_number"], dtype=np.int64)
    row_number = 0
    frames = []
    for chunk in pd.read_csv(RAW_DIR / "loan.csv", usecols=["issue_d", "loan_status"], chunksize=200_000):
        row_numbers = np.arange(row_number + 1, row_number + 1 + len(chunk))
        mask = np.isin(row_numbers, wanted)
        if mask.any():
            frames.append(
                pd.DataFrame(
                    {
                        "source_row_number": row_numbers[mask],
                        "raw_issue_d": chunk["issue_d"].to_numpy()[mask],
                        "raw_loan_status": chunk["loan_status"].to_numpy()[mask],
                    }
                )
            )
        row_number += len(chunk)
    return manifest.merge(pd.concat(frames, ignore_index=True), on="source_row_number", how="left")


@pytest.mark.parametrize("name", ["train", "val", "test"])
def test_ids_files_have_no_missing_or_malformed_values(ids, name):
    """Every loan_id in a *_ids.csv is present, non-blank and matches the lc_NNNNNNNNN convention."""
    series = ids[name].astype("string")
    assert series.isna().sum() == 0, f"{name}_ids.csv contains null loan_id values"
    malformed = sorted(series[~series.fillna("").str.fullmatch(r"lc_\d{9}")].tolist())
    assert not malformed, f"Malformed loan_id values in {name}_ids.csv: {malformed}"


def test_ids_files_have_no_internal_duplicates(ids):
    """Each *_ids.csv has the expected unique loan_id count and no duplicates."""
    for name, series in ids.items():
        duplicates = sorted(series[series.duplicated(keep=False)].unique())
        assert not duplicates, f"Duplicates found in {name}_ids.csv: {duplicates}"
        assert series.nunique() == EXPECTED_COUNTS[name]


def test_ids_files_do_not_overlap(ids):
    """No loan_id appears in more than one of the train/val/test ids files."""
    sets = {name: set(series) for name, series in ids.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            assert not overlap, f"Overlap between {names[i]}_ids.csv and {names[j]}_ids.csv: {sorted(overlap)}"


def test_split_manifest_row_counts_match_ids_files(manifest):
    """split_manifest row counts per split match the *_ids.csv counts."""
    manifest_counts = manifest["split"].value_counts()
    for name, expected in EXPECTED_COUNTS.items():
        manifest_split = MANIFEST_SPLIT_NAME[name]
        actual = int(manifest_counts.get(manifest_split, 0))
        assert actual == expected, f"split_manifest rows with split=='{manifest_split}' == {expected} (got {actual})"


@pytest.mark.parametrize("name", ["train", "val", "test"])
def test_ids_file_matches_split_manifest_classification(ids, manifest, name):
    """Every id in a *_ids.csv carries the matching split label in split_manifest, with none missing."""
    expected_split = MANIFEST_SPLIT_NAME[name]
    manifest_split_by_id = manifest.set_index("loan_id")["split"]
    id_list = ids[name].tolist()
    missing = [i for i in id_list if i not in manifest_split_by_id.index]
    mismatched = [
        (i, manifest_split_by_id.loc[i])
        for i in id_list
        if i in manifest_split_by_id.index and manifest_split_by_id.loc[i] != expected_split
    ]
    assert not missing, f"Missing from split_manifest: {missing}"
    assert not mismatched, f"Mismatched split label (id, actual_split): {mismatched}"


def test_loan_id_suffix_matches_source_row_number(manifest):
    """The numeric suffix of loan_id equals source_row_number for every manifest row."""
    row_number_from_id = manifest["loan_id"].str.replace("lc_", "", regex=False).astype(int)
    assert (row_number_from_id == manifest["source_row_number"]).all()


def test_each_loan_id_merges_to_exactly_one_raw_row(manifest, raw_row_count):
    """Each requested id matches exactly one raw row: a validated one-to-one merge with no unmatched ids."""
    raw_index = pd.DataFrame({"source_row_number": pd.RangeIndex(1, raw_row_count + 1)})
    merged = manifest.merge(raw_index, on="source_row_number", how="left", validate="one_to_one", indicator=True)
    unmatched = merged[merged["_merge"] == "left_only"]
    assert unmatched.empty, f"Unmatched loan_ids (source_row_number out of raw bounds): {unmatched['loan_id'].tolist()}"


def test_target_values_are_binary_with_no_nulls(manifest):
    """split_manifest target values are 0 or 1, with no nulls."""
    assert manifest["target"].isna().sum() == 0
    bad_targets = sorted(set(manifest["target"].dropna().unique()) - {0, 1})
    assert not bad_targets, f"Unexpected target values: {bad_targets}"


def test_issue_month_matches_raw_issue_d(raw_fields_for_manifest_rows):
    """split_manifest issue_month matches issue_d parsed from raw loan.csv."""
    df = raw_fields_for_manifest_rows
    raw_issue_month = pd.to_datetime(df["raw_issue_d"], format="%b-%Y", errors="coerce").dt.strftime("%Y-%m")
    mismatches = df[raw_issue_month != df["issue_month"]]
    assert mismatches.empty, (
        "issue_month mismatches:\n"
        f"{mismatches[['loan_id', 'source_row_number', 'issue_month', 'raw_issue_d']].to_string(index=False)}"
    )


def test_target_matches_raw_loan_status(raw_fields_for_manifest_rows):
    """split_manifest target matches the outcome implied by raw loan_status (Fully Paid -> 0, Charged Off -> 1)."""
    df = raw_fields_for_manifest_rows
    expected_target = df["raw_loan_status"].map({"Fully Paid": 0, "Charged Off": 1})
    unmapped = df[expected_target.isna()]
    assert unmapped.empty, (
        "Unexpected loan_status values outside the matured cohort:\n"
        f"{unmapped[['loan_id', 'source_row_number', 'raw_loan_status']].to_string(index=False)}"
    )
    mismatches = df[expected_target != df["target"]]
    assert mismatches.empty, (
        "target mismatches:\n"
        f"{mismatches[['loan_id', 'source_row_number', 'raw_loan_status', 'target']].to_string(index=False)}"
    )

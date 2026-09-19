"""Regression tests using tiny, artificial examples (no borrower data).

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/nlp/test_preprocess.py -v -s

The -s flag shows print statements; -v shows each PASSED/FAILED result.
"""

import json

import pandas as pd
import pytest

from src.nlp.preprocess import audit_source_files, build_text_payload, clean_text, load_original_split, sha256_file


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.mark.parametrize("archive_present", [False, True])
def test_source_audit_accepts_csv_with_missing_or_different_zip(artificial_data_dir, archive_present):
    """The approved CSV passes regardless of missing or differently packaged ZIP."""
    path = artificial_data_dir / "split_statistics.json"
    stats = json.loads(path.read_text())
    stats["raw_csv_sha256"] = sha256_file(artificial_data_dir / "raw" / "loan.csv")
    path.write_text(json.dumps(stats))
    if archive_present:
        (artificial_data_dir / "raw" / "lending-club-loan-data-csv.zip").write_bytes(b"Artificial archive stand-in")
    result = audit_source_files(artificial_data_dir)
    assert result["raw_csv_matches"] is True
    assert result["archive_matches"] is False
    assert result["raw_csv_sha256"] == stats["raw_csv_sha256"]


@pytest.mark.parametrize("benchmark", [None, "invalid", "0" * 64])
def test_source_audit_rejects_missing_invalid_or_mismatched_csv_hash(artificial_data_dir, benchmark):
    """Unapproved CSV identity stops analysis even when split structure is valid."""
    path = artificial_data_dir / "split_statistics.json"
    stats = json.loads(path.read_text())
    if benchmark is not None:
        stats["raw_csv_sha256"] = benchmark
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="benchmark"):
        audit_source_files(artificial_data_dir)


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), pd.NA, pd.NaT, "", "  ", "nan", "None", "NULL", "n/a", "NA", "<NA>"],
)
def test_clean_text_returns_empty_for_missing_values(value):
    """Missing values and exact missing markers become empty text."""
    assert clean_text(value) == ""


def test_clean_text_normalizes_markup_and_unicode_without_losing_meaning():
    """HTML, Unicode and spacing are cleaned while negation and punctuation remain."""
    source = " <p>Ｃａｆé&nbsp; &amp; tea</p>\n<p>I do NOT want another loan!</p> "
    assert clean_text(source) == "Café & tea I do NOT want another loan!"


def test_clean_text_removes_html_escaped_tags():
    """Escaped HTML tags are removed without joining separate sentences together."""
    assert clean_text("&lt;p&gt;Repairs.&lt;/p&gt;&lt;p&gt;No new debt.&lt;/p&gt;") == (
        "Repairs. No new debt."
    )


def test_clean_text_normalizes_characters_decoded_from_html_entities():
    """Characters decoded from HTML entities also receive Unicode normalization."""
    assert clean_text("&#65313;") == "A"


def test_clean_text_removes_borrower_update_wrapper():
    """Multiple borrower timestamp wrappers are removed while their prose remains."""
    source = "Borrower added on 01/02/13 > I need repairs.  Borrower added on 01/03/13 > Thanks!"
    assert clean_text(source) == "I need repairs. Thanks!"


def test_clean_text_preserves_missing_marker_words_inside_sentences():
    """A word such as 'None' inside a real sentence is not treated as missing text."""
    assert clean_text("None of this is optional.") == "None of this is optional."


@pytest.mark.parametrize("value", [42, 3.5, ["text"], {"desc": "text"}])
def test_clean_text_rejects_nontext_values(value):
    """Unexpected numbers and containers raise TypeError instead of becoming text."""
    with pytest.raises(TypeError):
        clean_text(value)


def test_build_payload_has_fixed_order_and_ignores_unapproved_fields():
    """Only approved fields enter the payload, in order; employment title is opt-in."""
    row = {
        "desc": "I do not want another credit card.",
        "target": 1,
        "loan_status": "Charged Off",
        "title": "A fresh start",
        "purpose": "debt_consolidation",
        "emp_title": "Sample occupation",
        "emp_length": "10+ years",
        "synthetic_desc": "DUMMY TEXT MUST NOT BE INCLUDED",
    }
    core = (
        "Title: A fresh start\n"
        "Purpose: debt consolidation\n"
        "Description: I do not want another credit card."
    )
    assert build_text_payload(row) == core
    assert build_text_payload(row, include_emp_title=True) == (
        core + "\nEmployment title: Sample occupation"
    )


def test_build_payload_skips_empty_fields_without_empty_labels():
    """Empty fields and their labels are omitted from the combined narrative."""
    assert build_text_payload({"title": pd.NA, "purpose": "n/a", "desc": "  Repairs. "}) == (
        "Description: Repairs."
    )
    assert build_text_payload({}) == ""


@pytest.fixture
def artificial_data_dir(tmp_path):
    """Four made-up records, deliberately stored in nonchronological order."""
    (tmp_path / "raw").mkdir()
    raw = pd.DataFrame(
        {
            "title": ["Validation example", "First train example", "Test example", "Second train example"],
            "purpose": ["other", "home_improvement", "other", "debt_consolidation"],
            "desc": ["Validation text.", "Repair a sample roof.", "Locked test text.", "Combine sample balances."],
            "emp_title": ["Example role A", "Example role B", "Example role C", "Example role D"],
            "issue_d": ["Aug-2013", "Jan-2013", "Jan-2014", "Feb-2013"],
            "loan_status": ["Fully Paid", "Fully Paid", "Charged Off", "Charged Off"],
            "synthetic_desc": ["Dummy batch text"] * 4,
            "emp_length": ["1 year"] * 4,
        }
    )
    raw.to_csv(tmp_path / "raw" / "loan.csv", index=False)
    version = "kaggle-adarshsng-local-2026-09-09"
    # Row positions are one-based data records, excluding the CSV header.
    manifest = pd.DataFrame(
        {
            "loan_id": ["lc_000000004", "lc_000000001", "lc_000000002", "lc_000000003"],
            "source_row_number": [4, 1, 2, 3],
            "split": ["train", "validation", "train", "test"],
            "issue_month": ["2013-02", "2013-08", "2013-01", "2014-01"],
            "target": [1, 0, 0, 1],
            "text_available": [1] * 4,
            "cohort": ["real_text_matured_v1"] * 4,
            "dataset_version": [version] * 4,
        }
    )
    manifest.to_csv(tmp_path / "split_manifest.csv", index=False)
    statistics = {
        "dataset": "Artificial Lending Club-shaped test fixture",
        "dataset_version": version,
        "archive_sha256": "0" * 64,
        "split_version": "split-v1",
        "split_method": "chronological_by_issue_month",
        "cohort": "real_text_matured_v1",
        "raw_rows": 4,
        "eligible_rows": 4,
        "invalid_eligible_dates": 0,
        "splits": {
            "train": {"rows": 2, "defaults": 1, "default_rate": 0.5},
            "validation": {"rows": 1, "defaults": 0, "default_rate": 0.0},
            "test": {"rows": 1, "defaults": 1, "default_rate": 1.0},
        },
    }
    (tmp_path / "split_statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    return tmp_path


def test_loader_uses_one_based_source_rows_across_chunk_boundaries(artificial_data_dir):
    """One-based CSV row IDs stay aligned with text and targets across read chunks."""
    frame = load_original_split(artificial_data_dir, "train", chunksize=1)
    assert set(frame["loan_id"]) == {"lc_000000002", "lc_000000004"}
    by_id = frame.set_index("loan_id")
    assert by_id.loc["lc_000000002", "desc"] == "Repair a sample roof."
    assert by_id.loc["lc_000000004", "desc"] == "Combine sample balances."
    assert by_id.loc["lc_000000002", "target"] == 0
    assert by_id.loc["lc_000000004", "target"] == 1
    assert set(frame["split"]) == {"train"}
    assert by_id.loc["lc_000000002", "text_payload"] == (
        "Title: First train example\nPurpose: home improvement\nDescription: Repair a sample roof."
    )
    assert {"loan_status", "issue_d", "emp_title", "emp_length", "synthetic_desc"}.isdisjoint(frame.columns)


def test_loader_supports_validation_and_explicit_employment_title(artificial_data_dir):
    """Validation loading works and includes employment title only when requested."""
    frame = load_original_split(artificial_data_dir, "validation", include_emp_title=True, chunksize=1)
    assert frame["loan_id"].tolist() == ["lc_000000001"]
    assert frame["emp_title"].tolist() == ["Example role A"]
    assert frame["text_payload"].iloc[0].endswith("Employment title: Example role A")


def test_loader_preserves_frozen_row_when_description_cleans_to_empty(artificial_data_dir):
    """A wrapper-only description does not remove its loan from the frozen split."""
    raw_path = artificial_data_dir / "raw" / "loan.csv"
    raw = pd.read_csv(raw_path)
    raw.loc[1, "desc"] = "Borrower added on 01/02/13 >"
    raw.to_csv(raw_path, index=False)
    frame = load_original_split(artificial_data_dir, "train", chunksize=1)
    assert set(frame["loan_id"]) == {"lc_000000002", "lc_000000004"}
    row = frame.set_index("loan_id").loc["lc_000000002"]
    assert row["text_available"] == 1
    assert clean_text(row["desc"]) == ""
    assert row["text_payload"] == "Title: First train example\nPurpose: home improvement"


@pytest.mark.parametrize("split", ["test", "val", "unknown"])
def test_loader_rejects_test_and_unknown_split_names(artificial_data_dir, split):
    """The loader refuses the held-out test split and unsupported split names."""
    with pytest.raises(ValueError):
        load_original_split(artificial_data_dir, split)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [("issue_d", "Mar-2013"), ("loan_status", "Charged Off"), ("desc", "")],
)
def test_loader_rejects_corrupted_source_mapping(artificial_data_dir, column, replacement):
    """Mismatched source dates, outcomes or description availability are rejected."""
    path = artificial_data_dir / "raw" / "loan.csv"
    raw = pd.read_csv(path)
    raw.loc[1, column] = replacement  # The selected training record at source row 2.
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_original_split(artificial_data_dir, "train", chunksize=1)


def test_loader_rejects_incorrect_text_availability(artificial_data_dir):
    """An incorrect text-availability flag is rejected even when prose is present."""
    path = artificial_data_dir / "split_manifest.csv"
    manifest = pd.read_csv(path)
    manifest.loc[manifest["source_row_number"].eq(2), "text_available"] = 0
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_original_split(artificial_data_dir, "train", chunksize=1)


def test_real_text_cohort_rejects_jointly_missing_description_and_availability(artificial_data_dir):
    """Missing prose and a missing-text flag cannot bypass the real-text cohort rule."""
    manifest_path = artificial_data_dir / "split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest.loc[manifest["source_row_number"].eq(2), "text_available"] = 0
    manifest.to_csv(manifest_path, index=False)
    raw_path = artificial_data_dir / "raw" / "loan.csv"
    raw = pd.read_csv(raw_path)
    raw.loc[1, "desc"] = ""
    raw.to_csv(raw_path, index=False)
    with pytest.raises(ValueError):
        load_original_split(artificial_data_dir, "train", chunksize=1)


def test_loader_rejects_raw_record_count_mismatch(artificial_data_dir):
    """A raw CSV with the wrong record count is rejected before using its rows."""
    raw_path = artificial_data_dir / "raw" / "loan.csv"
    raw = pd.read_csv(raw_path)
    raw.iloc[:-1].to_csv(raw_path, index=False)
    with pytest.raises(ValueError):
        load_original_split(artificial_data_dir, "train", chunksize=1)

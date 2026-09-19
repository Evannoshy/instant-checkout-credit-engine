"""Analyst 1: deterministic original-text preprocessing and development splits.

Use ``build_text_payload(row)`` in both the TF-IDF and transformer pipelines.
Case, punctuation, numbers and negation are preserved. Model-specific operations
such as stop-word removal, lowercasing and truncation belong downstream and are not part of the preprocessing pipeline.

The source manifest counts CSV *data records* from ONE (header excluded).
Physical line numbers are unsuitable because quoted descriptions can span lines.
``load_original_split`` validates this mapping and deliberately exposes only
train/validation. It never substitutes generated text or changes frozen labels.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_scalar

CORE_TEXT_COLUMNS = ("title", "purpose", "desc")
OPTIONAL_TEXT_COLUMNS = ("emp_title",)
PREPROCESS_VERSION = "nlp-preprocess-v1"
MISSING_MARKERS = frozenset({"nan", "none", "null", "n/a", "na", "<na>"})
FIELD_LABELS = {
    "title": "Title",
    "purpose": "Purpose",
    "desc": "Description",
    "emp_title": "Employment title",
}
_BORROWER_WRAPPER = re.compile(
    r"\bBorrower\s+added\s+on\s+\d{1,2}/\d{1,2}/\d{2,4}\s*>\s*", # borrower wrapper is a timestamp prefix found in some original loan descriptions, for example: "Borrower added on 07/12/13 > I want to consolidate my credit card debt."
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


class _PlainText(HTMLParser):
    """Separate HTML blocks while retaining text and ignoring script/style."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def clean_text(value: object) -> str:
    """Clean one original text field; missing values become an empty string.

    Normalises Unicode (NFKC), HTML, known loan.csv entry wrappers and
    whitespace. Dates in ordinary borrower prose are retained. Does not stem,
    remove stop words, anonymize, or infer missing content. Nonmissing numeric
    and container values raise TypeError so schema mistakes are visible.
    """
    if value is None or (is_scalar(value) and pd.isna(value)):
        return ""
    if not isinstance(value, str):
        raise TypeError(f"Expected a string or missing scalar; got {type(value).__name__}")
    text = unicodedata.normalize("NFKC", value).strip()
    if not text or text.casefold() in MISSING_MARKERS:
        return ""
    parser = _PlainText()
    parser.feed(html.unescape(text)) # unescape HTML entities before parsing to avoid losing text in tags
    parser.close()
    text = unicodedata.normalize("NFKC", "".join(parser.parts)) # normalise again after HTML unescape and parsing
    text = _BORROWER_WRAPPER.sub("", text) # remove the borrower wrapper prefix if present

    # remove control characters and format characters, replacing control characters with a space and removing format characters entirely.
    text = "".join(
        " " if unicodedata.category(char) == "Cc" else char # remove control characters, replace with space
        for char in text
        if unicodedata.category(char) != "Cf" # remove format characters, e.g., zero-width space
    )
    # collapse whitespace and strip leading/trailing spaces. If the cleaned text is a known missing marker, return an empty string.
    text = _WHITESPACE.sub(" ", text).strip()
    return "" if text.casefold() in MISSING_MARKERS else text


def build_text_payload(
    row: Mapping[str, object], *, include_emp_title: bool = False
) -> str:
    """Concatenate approved original fields in a stable, labelled order.

    Missing keys/values are omitted. ``purpose`` underscores become spaces.
    ``emp_title`` is opt-in; synthetic_desc, emp_length, IDs, labels, outcomes
    and all other fields are ignored, even if present in the row.
    """
    fields = CORE_TEXT_COLUMNS + (OPTIONAL_TEXT_COLUMNS if include_emp_title else ())
    parts = []
    for field in fields:
        value = clean_text(row.get(field))
        if field == "purpose":
            value = _WHITESPACE.sub(" ", value.replace("_", " ")).strip()
        if value:
            parts.append(f"{FIELD_LABELS[field]}: {value}")
    return "\n".join(parts)

""" List of helper functions for source file auditing and split loading. These are not part of the public API."""
def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def audit_source_files(data_dir: str | Path) -> dict[str, Any]:
    """Verify the extracted CSV against the team-approved SHA-256 benchmark.

    ZIP hashes are informational only: packaging can differ independently of
    the extracted bytes, and the archive may be deleted after extraction.
    Missing or mismatched CSV benchmarks fail closed before notebook EDA.
    Structural split checks remain separate from file identity verification.
    """
    data_dir = Path(data_dir)
    with (data_dir / "split_statistics.json").open(encoding="utf-8") as stream:
        stats = json.load(stream)
    archive = data_dir / "raw" / "lending-club-loan-data-csv.zip"
    actual = sha256_file(archive) if archive.is_file() else None
    expected = stats.get("archive_sha256")
    expected_csv = stats.get("raw_csv_sha256")
    if not isinstance(expected_csv, str) or re.fullmatch(r"[0-9a-f]{64}", expected_csv) is None:
        raise ValueError("Missing or invalid raw_csv_sha256 benchmark in split_statistics.json")
    actual_csv = sha256_file(data_dir / "raw" / "loan.csv")
    if actual_csv != expected_csv:
        raise ValueError("Raw CSV SHA-256 does not match the approved benchmark; stop source analysis")
    return {
        "dataset_version": stats.get("dataset_version"),
        "split_version": stats.get("split_version"),
        "preprocess_version": PREPROCESS_VERSION,
        "expected_archive_sha256": expected,
        "actual_archive_sha256": actual,
        "archive_matches": bool(expected and actual and actual == expected),
        "expected_raw_csv_sha256": expected_csv,
        "raw_csv_sha256": actual_csv,
        "raw_csv_matches": True,
        "manifest_sha256": sha256_file(data_dir / "split_manifest.csv"),
    }

def load_original_split(
    data_dir: str | Path,
    split: str = "train",
    *,
    include_emp_title: bool = False,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load one frozen development split using one-based source record numbers.

    Scans the CSV in chunks, retaining only requested records. Metadata for all
    splits is read to check uniqueness; held-out text is never returned or used
    for EDA. Only ``train`` and ``validation`` are allowed. Structural checks
    establish consistency, not exact archive identity; use audit_source_files
    to record provenance as well.

    The Fully Paid/0 and Charged Off/1 checks verify the mapping observed in
    real_text_matured_v1. They do not generate or overwrite any labels.
    """
    if split not in {"train", "validation"}:
        raise ValueError("Only train and validation are available; the final test set is locked.")
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    data_dir = Path(data_dir)
    manifest = pd.read_csv(data_dir / "split_manifest.csv", dtype={"loan_id": "string"})
    required = {
        "loan_id", "source_row_number", "split", "issue_month", "target",
        "text_available", "cohort", "dataset_version",
    }
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing required metadata")
    if not manifest["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Manifest has an unknown split")
    if manifest["loan_id"].duplicated().any() or manifest["source_row_number"].duplicated().any():
        raise ValueError("Manifest IDs and source row numbers must be unique across splits")
    numbers = manifest["source_row_number"]
    if (numbers < 1).any() or (numbers % 1 != 0).any():
        raise ValueError("source_row_number must contain positive one-based integers")
    expected_ids = numbers.map(lambda n: f"lc_{int(n):09d}")
    if not manifest["loan_id"].eq(expected_ids).all():
        raise ValueError("loan_id does not match the manifest source-row convention")
    if not manifest["target"].isin([0, 1]).all():
        raise ValueError("Manifest targets must be binary")
    if not manifest["text_available"].isin([0, 1]).all():
        raise ValueError("Manifest text_available must be binary")

    with (data_dir / "split_statistics.json").open(encoding="utf-8") as stream:
        stats = json.load(stream)
    if len(manifest) != stats["eligible_rows"]:
        raise ValueError("Manifest count differs from split_statistics.json")
    for split_name, split_info in stats["splits"].items():
        if int(manifest["split"].eq(split_name).sum()) != split_info["rows"]:
            raise ValueError(f"Unexpected manifest count for {split_name}")
    if not manifest["cohort"].eq("real_text_matured_v1").all():
        raise ValueError("Loader expects the real_text_matured_v1 cohort")
    if not manifest["dataset_version"].eq(stats["dataset_version"]).all():
        raise ValueError("Manifest dataset version disagrees with split statistics")

    selected = manifest.loc[manifest["split"].eq(split)].copy()
    if selected.empty:
        raise ValueError(f"No rows found for {split}")
    wanted = set(selected["source_row_number"])
    text_fields = list(CORE_TEXT_COLUMNS + (OPTIONAL_TEXT_COLUMNS if include_emp_title else ()))
    raw_fields = [*text_fields, "issue_d", "loan_status"]
    pieces = []
    records_seen = 0
    reader = pd.read_csv(
        data_dir / "raw" / "loan.csv", usecols=raw_fields, dtype="string",
        keep_default_na=False, chunksize=chunksize,
    )
    with reader:
        for chunk in reader:
            # Start at 1 and increment BEFORE any filtering or index reset.
            chunk["source_row_number"] = range(records_seen + 1, records_seen + len(chunk) + 1)
            records_seen += len(chunk)
            retained = chunk.loc[chunk["source_row_number"].isin(wanted)]
            if not retained.empty:
                pieces.append(retained.copy())
    if records_seen != stats["raw_rows"]:
        raise ValueError("Raw CSV record count disagrees with split statistics")
    if not pieces:
        raise ValueError("No requested source records were found")
    rows = selected.merge(
        pd.concat(pieces, ignore_index=True), on="source_row_number", how="left",
        validate="one_to_one", indicator=True, sort=False,
    )
    if not rows["_merge"].eq("both").all():
        raise ValueError("Some manifest rows have no matching source record")

    months = pd.to_datetime(rows["issue_d"], format="%b-%Y", errors="coerce").dt.strftime("%Y-%m")
    if not months.eq(rows["issue_month"]).all():
        raise ValueError("Issue-month alignment failed; check the source version and one-based row numbering")
    observed_target = rows["loan_status"].map({"Fully Paid": 0, "Charged Off": 1})
    if not observed_target.eq(rows["target"]).all():
        raise ValueError("Status/target alignment failed; do not use this source mapping")
    # text_available describes the raw source, before our cleaning policy.
    # Wrapper-only or marker-only descriptions may clean to empty; keep their
    # frozen rows and use the remaining original fields in the payload.
    has_desc = rows["desc"].fillna("").str.strip().ne("")
    if not has_desc.all() or not rows["text_available"].eq(1).all():
        raise ValueError("Original description availability disagrees with the manifest")
    rows = rows.drop(columns=["_merge", "issue_d", "loan_status"])
    rows["text_payload"] = [
        build_text_payload(row, include_emp_title=include_emp_title)
        for row in rows[text_fields].to_dict(orient="records")
    ]
    if rows["text_payload"].eq("").any():
        raise ValueError("Empty payload in the real-text cohort")
    return rows.reset_index(drop=True)

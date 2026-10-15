"""Tests for the checksum and path helpers written beside tabular outputs.

From the repository root:
    python -m pytest src/tabular/test_provenance.py -v -s

Uses small synthetic files in a temporary directory (no borrower data).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.tabular import provenance

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def write_locked_inputs(tmp_path: Path, csv_bytes: bytes, benchmark: str | None) -> Path:
    """A data dir with split_statistics.json and a raw CSV; returns the CSV path."""
    raw_csv = tmp_path / "raw" / "loan.csv"
    raw_csv.parent.mkdir()
    raw_csv.write_bytes(csv_bytes)
    stats = {} if benchmark is None else {"raw_csv_sha256": benchmark}
    (tmp_path / "split_statistics.json").write_text(json.dumps(stats), encoding="utf-8")
    return raw_csv


def test_matching_raw_csv_returns_its_checksum(tmp_path: Path) -> None:
    """A raw CSV identical to the locked benchmark passes and returns its SHA-256."""
    content = b"loan_id,loan_amnt\nlc_1,1000\n"
    expected = hashlib.sha256(content).hexdigest()
    raw_csv = write_locked_inputs(tmp_path, content, expected)
    assert provenance.require_locked_raw_csv(tmp_path, raw_csv) == expected


def test_changed_raw_csv_is_refused(tmp_path: Path) -> None:
    """A single changed byte in the raw CSV is refused before any model work."""
    benchmark = hashlib.sha256(b"loan_id,loan_amnt\nlc_1,1000\n").hexdigest()
    raw_csv = write_locked_inputs(tmp_path, b"loan_id,loan_amnt\nlc_1,1001\n", benchmark)
    with pytest.raises(ValueError, match="Use the verified loan.csv"):
        provenance.require_locked_raw_csv(tmp_path, raw_csv)


@pytest.mark.parametrize("benchmark", [None, "not-a-hash"])
def test_missing_or_malformed_benchmark_is_refused(tmp_path: Path, benchmark: str | None) -> None:
    """Without a valid 64-character benchmark the raw CSV cannot be verified."""
    raw_csv = write_locked_inputs(tmp_path, b"loan_id\n", benchmark)
    with pytest.raises(ValueError, match="no valid raw_csv_sha256"):
        provenance.require_locked_raw_csv(tmp_path, raw_csv)


def test_checksums_are_keyed_by_repository_relative_path() -> None:
    """Files inside the repository are recorded by relative path with their SHA-256."""
    config = REPOSITORY_ROOT / "configs" / "tabular_logistic_v1.toml"
    checksums = provenance.file_checksums([config])
    assert checksums == {
        "configs/tabular_logistic_v1.toml": hashlib.sha256(config.read_bytes()).hexdigest()
    }


def test_paths_outside_the_repository_stay_absolute(tmp_path: Path) -> None:
    """A path outside the repository is kept absolute rather than made misleadingly relative."""
    assert provenance.display_path(tmp_path) == tmp_path.resolve().as_posix()

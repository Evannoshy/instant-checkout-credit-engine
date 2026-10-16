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


@pytest.mark.parametrize("suffix", [".csv", ".json", ".toml"])
@pytest.mark.parametrize("recorded_eol,current_eol", [(b"\n", b"\r\n"), (b"\r\n", b"\n")])
def test_text_checksum_accepts_only_line_ending_conversion(tmp_path, suffix, recorded_eol, current_eol):
    path = tmp_path / f"input{suffix}"
    lines = [b"first", b"second", b""]
    recorded = hashlib.sha256(recorded_eol.join(lines)).hexdigest()
    path.write_bytes(current_eol.join(lines))
    provenance.verify_file_checksum(path, recorded, allow_crlf=True)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        provenance.verify_file_checksum(path, recorded)


@pytest.mark.parametrize("content", [b"first\nchanged\n", b"first \nsecond\n", b"second\nfirst\n"])
def test_text_checksum_rejects_changed_content(tmp_path, content):
    path = tmp_path / "input.csv"
    path.write_bytes(content)
    recorded = hashlib.sha256(b"first\nsecond\n").hexdigest()
    with pytest.raises(ValueError, match="Checksum mismatch"):
        provenance.verify_file_checksum(path, recorded, allow_crlf=True)


@pytest.mark.parametrize("suffix,content", [
    (".parquet", b"PAR1\r\n"), (".csv", b"\xff\r\n"), (".json", b"\x00\r\n"),
])
def test_binary_or_invalid_text_is_never_normalized(tmp_path, suffix, content):
    path = tmp_path / f"input{suffix}"
    path.write_bytes(content)
    recorded = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    with pytest.raises(ValueError, match="Checksum mismatch"):
        provenance.verify_file_checksum(path, recorded, allow_crlf=True)


def test_locked_raw_csv_rejects_line_ending_changes(tmp_path):
    original = b"loan_id,loan_amnt\nlc_1,1000\n"
    raw = write_locked_inputs(tmp_path, original.replace(b"\n", b"\r\n"), hashlib.sha256(original).hexdigest())
    with pytest.raises(ValueError, match="Use the verified loan.csv"):
        provenance.require_locked_raw_csv(tmp_path, raw)


def test_reproduced_inputs_accept_relocated_identical_raw_and_crlf_config(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path / "checkout")
    config = provenance.REPO_ROOT / "settings.toml"
    config.parent.mkdir()
    config.write_bytes(b"value = 1\r\n")
    content = b"loan_id\nlc_1\n"
    checksum = hashlib.sha256(content).hexdigest()
    raw = write_locked_inputs(tmp_path, content, checksum)
    assert raw.is_relative_to(tmp_path) and not raw.is_relative_to(provenance.REPO_ROOT)
    recorded = {"data/raw/loan.csv": checksum, "settings.toml": hashlib.sha256(b"value = 1\n").hexdigest()}
    current = {provenance.display_path(raw): checksum, **provenance.file_checksums([config])}
    provenance.verify_reproduced_inputs(recorded, current, raw_csv_path=raw, data_dir=tmp_path)


@pytest.mark.parametrize("defect", ["missing", "extra", "changed", "false_current_hash"])
def test_reproduced_inputs_reject_wrong_paths_content_or_hashes(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    path = tmp_path / "input.json"
    path.write_bytes(b'{"value":1}\n')
    recorded = provenance.file_checksums([path])
    current = dict(recorded)
    if defect == "missing":
        current.clear()
    elif defect == "extra":
        current["extra.json"] = "0" * 64
    elif defect == "changed":
        path.write_bytes(b'{"value":2}\n')
        current = provenance.file_checksums([path])
    else:
        current["input.json"] = "0" * 64
    with pytest.raises(ValueError):
        provenance.verify_reproduced_inputs(recorded, current)


@pytest.mark.parametrize("defect", ["recorded_hash", "current_hash", "changed_raw", "no_raw_path", "no_data_dir"])
def test_reproduced_raw_inputs_cannot_bypass_locked_identity(tmp_path, defect):
    content = b"loan_id\nlc_1\n"
    checksum = hashlib.sha256(content).hexdigest()
    raw = write_locked_inputs(tmp_path, content, checksum)
    recorded = {"data/raw/loan.csv": checksum}
    current = {provenance.display_path(raw): checksum}
    if defect == "recorded_hash":
        recorded["data/raw/loan.csv"] = "0" * 64
    elif defect == "current_hash":
        current[provenance.display_path(raw)] = "0" * 64
    elif defect == "changed_raw":
        raw.write_bytes(content.replace(b"\n", b"\r\n"))
        current = provenance.file_checksums([raw])
    with pytest.raises(ValueError):
        provenance.verify_reproduced_inputs(
            recorded, current,
            raw_csv_path=None if defect == "no_raw_path" else raw,
            data_dir=None if defect == "no_data_dir" else tmp_path,
        )

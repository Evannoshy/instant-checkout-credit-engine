"""Checksums and paths recorded beside tabular outputs, so a reader can tell
exactly which inputs produced a file.

``require_locked_raw_csv`` refuses a raw CSV whose SHA-256 differs from the
``raw_csv_sha256`` benchmark in ``data/split_statistics.json``: the frozen
split was built from that exact file, so any other copy could silently change
features. The hash itself reuses ``src.nlp.preprocess.sha256_file``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.nlp.preprocess import sha256_file
from src.tabular.preprocess import REPO_ROOT

SPLIT_STATISTICS = "split_statistics.json"
SHA256_HEX_LENGTH = 64
TEXT_SUFFIXES = {".csv", ".json", ".toml"}


def display_path(path: str | Path) -> str:
    """Repository-relative POSIX path where possible, for portable metadata."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def file_checksums(paths: list[str | Path]) -> dict[str, str]:
    """SHA-256 of each file, keyed by its display path."""
    return {display_path(path): sha256_file(path) for path in paths}


def verify_file_checksum(path: str | Path, expected: str, *, allow_crlf: bool = False) -> None:
    """Verify bytes, optionally allowing only LF/CRLF conversion in UTF-8 text.

    Raw source files must use the strict default. Parquet and other binary files
    are always strict. This does not change the byte hashes recorded by runners.
    """
    path = Path(path)
    if sha256_file(path) == expected:
        return
    if allow_crlf and path.suffix.lower() in TEXT_SUFFIXES:
        content = path.read_bytes()
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            if b"\x00" not in content:
                lf = content.replace(b"\r\n", b"\n")
                variants = (lf, lf.replace(b"\n", b"\r\n"))
                if expected in {hashlib.sha256(value).hexdigest() for value in variants}:
                    return
    raise ValueError(f"Checksum mismatch for {path}")


def verify_reproduced_inputs(
    recorded: dict[str, str],
    current: dict[str, str],
    *,
    raw_csv_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> None:
    """Check the same inputs after Git line-ending conversion or raw-file relocation.

    The recorded runs use data/raw/loan.csv. Only that source may change location,
    and it must still match the locked byte checksum. All other paths must match,
    their current byte hashes are verified, and only LF/CRLF text changes pass.
    """
    recorded, current = dict(recorded), dict(current)
    if "data/raw/loan.csv" in recorded and raw_csv_path is None:
        raise ValueError("raw_csv_path is required to verify raw source bytes")
    if raw_csv_path is not None:
        if data_dir is None:
            raise ValueError("data_dir is required to verify the locked raw CSV")
        locked = require_locked_raw_csv(data_dir, raw_csv_path)
        old_raw = recorded.pop("data/raw/loan.csv", None)
        new_raw = current.pop(display_path(raw_csv_path), None)
        if old_raw != locked or new_raw != locked:
            raise ValueError("Recorded and current raw CSV hashes must match the locked benchmark")
    if recorded.keys() != current.keys():
        raise ValueError("Recorded and current input paths differ")
    for name, expected in recorded.items():
        path = REPO_ROOT / name
        verify_file_checksum(path, current[name])
        verify_file_checksum(path, expected, allow_crlf=True)


def require_locked_raw_csv(data_dir: str | Path, raw_csv_path: str | Path) -> str:
    """Return the raw CSV's SHA-256 after checking it against the locked benchmark.

    Raises ValueError if split_statistics.json has no benchmark or the file differs.
    """
    stats = json.loads((Path(data_dir) / SPLIT_STATISTICS).read_text(encoding="utf-8"))
    expected = stats.get("raw_csv_sha256")
    if not isinstance(expected, str) or len(expected) != SHA256_HEX_LENGTH:
        raise ValueError(f"{SPLIT_STATISTICS} has no valid raw_csv_sha256 benchmark")
    actual = sha256_file(raw_csv_path)
    if actual != expected:
        raise ValueError(
            f"Raw CSV {raw_csv_path} has SHA-256 {actual}, but the locked split expects "
            f"{expected}. Use the verified loan.csv."
        )
    return actual

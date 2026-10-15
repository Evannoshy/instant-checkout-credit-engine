"""Checksums and paths recorded beside tabular outputs, so a reader can tell
exactly which inputs produced a file.

``require_locked_raw_csv`` refuses a raw CSV whose SHA-256 differs from the
``raw_csv_sha256`` benchmark in ``data/split_statistics.json``: the frozen
split was built from that exact file, so any other copy could silently change
features. The hash itself reuses ``src.nlp.preprocess.sha256_file``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.nlp.preprocess import sha256_file
from src.tabular.preprocess import REPO_ROOT

SPLIT_STATISTICS = "split_statistics.json"
SHA256_HEX_LENGTH = 64


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

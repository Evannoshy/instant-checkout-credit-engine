"""Reproduce and audit the locked LendingClub cohort without rewriting it.

This is a governance/audit utility, not a model-development loader. It reads the
raw source, applies ``configs/cohort_v1.toml``, and compares the reconstructed
rows with ``data/split_manifest.csv``. It never writes or replaces split files.

Example from the repository root::

    python scripts/reproduce_locked_cohort.py --raw-csv data/raw/loan.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cohort_v1.toml"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "split_manifest.csv"
DEFAULT_STATISTICS = REPO_ROOT / "data" / "split_statistics.json"
DEFAULT_RAW_CSV = REPO_ROOT / "data" / "raw" / "loan.csv"
MANIFEST_COLUMNS = (
    "loan_id",
    "source_row_number",
    "split",
    "issue_month",
    "target",
    "text_available",
    "cohort",
    "dataset_version",
)
RAW_COLUMNS = ("issue_d", "loan_status", "desc")
LITERAL_MISSING_MARKERS = frozenset({"nan", "none", "null", "n/a", "na", "<na>"})


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the frozen cohort configuration."""

    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config["eligibility"]["description_rule"] != "raw_non_whitespace":
        raise ValueError("Unsupported description rule")
    if not config["eligibility"]["preserve_default_na_tokens"]:
        raise ValueError("cohort_v1 requires raw NA-like strings to remain literal")
    if config["identifier"]["source_row_origin"] != 1:
        raise ValueError("cohort_v1 uses one-based CSV data-row numbers")
    return config


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01")


def _assign_splits(issue_dates: pd.Series, config: dict[str, Any]) -> pd.Series:
    splits = config["splits"]
    conditions = [
        issue_dates.between(
            _month(splits["train_first_month"]),
            _month(splits["train_last_month"]),
            inclusive="both",
        ),
        issue_dates.between(
            _month(splits["validation_first_month"]),
            _month(splits["validation_last_month"]),
            inclusive="both",
        ),
        issue_dates.between(
            _month(splits["test_first_month"]),
            _month(splits["test_last_month"]),
            inclusive="both",
        ),
    ]
    assigned = pd.Series(
        np.select(conditions, ["train", "validation", "test"], default=""),
        index=issue_dates.index,
        dtype="string",
    )
    if assigned.eq("").any():
        raise ValueError("An eligible issue month did not map to a frozen split")
    return assigned


def reconstruct_cohort(
    raw_csv: Path,
    config: dict[str, Any],
    *,
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the frozen rules and return the reconstructed manifest rows.

    ``keep_default_na=False`` is part of the contract. The source contains two
    literal ``None`` descriptions, one lowercase ``none`` description and one
    literal ``n/a`` description that are non-empty raw strings and were therefore
    admitted by the original rule.
    Downstream text cleaning may treat those descriptions as missing; their
    title/purpose fields still provide text. Changing CSV NA parsing here would
    silently remove canonical training IDs or make eligibility depend on pandas'
    evolving default missing-value vocabulary.
    """

    eligibility = config["eligibility"]
    allowed_statuses = set(eligibility["allowed_statuses"])
    target_map = {
        "Fully Paid": int(config["target"]["fully_paid"]),
        "Charged Off": int(config["target"]["charged_off"]),
    }
    first_month = _month(eligibility["first_issue_month"])
    last_month = _month(eligibility["last_issue_month"])

    parts: list[pd.DataFrame] = []
    raw_rows = 0
    marker_descriptions = 0

    for chunk in pd.read_csv(
        raw_csv,
        usecols=list(RAW_COLUMNS),
        chunksize=chunksize,
        keep_default_na=False,
        low_memory=False,
    ):
        source_rows = np.arange(raw_rows + 1, raw_rows + len(chunk) + 1)
        raw_rows += len(chunk)

        issue_dates = pd.to_datetime(
            chunk[eligibility["issue_date_column"]],
            format=eligibility["issue_date_format"],
            errors="coerce",
        )
        descriptions = chunk[eligibility["description_column"]].astype("string")
        description_available = descriptions.str.strip().ne("")
        eligible = (
            chunk["loan_status"].isin(allowed_statuses)
            & description_available
            & issue_dates.between(first_month, last_month, inclusive="both")
        )
        if not eligible.any():
            continue

        eligible_rows = source_rows[eligible.to_numpy()]
        eligible_dates = issue_dates.loc[eligible]
        eligible_statuses = chunk.loc[eligible, "loan_status"]
        eligible_descriptions = descriptions.loc[eligible].str.strip().str.casefold()
        marker_descriptions += int(eligible_descriptions.isin(LITERAL_MISSING_MARKERS).sum())

        frame = pd.DataFrame(
            {
                "loan_id": [f"lc_{row:09d}" for row in eligible_rows],
                "source_row_number": eligible_rows,
                "split": _assign_splits(eligible_dates, config).to_numpy(),
                "issue_month": eligible_dates.dt.strftime("%Y-%m").to_numpy(),
                "target": eligible_statuses.map(target_map).astype("int8").to_numpy(),
                "text_available": np.ones(len(eligible_rows), dtype="int8"),
                "cohort": config["cohort_version"],
                "dataset_version": config["dataset_version"],
            }
        )
        parts.append(frame)

    reconstructed = pd.concat(parts, ignore_index=True)
    diagnostics = {
        "raw_rows": raw_rows,
        "eligible_rows": len(reconstructed),
        "literal_missing_marker_descriptions": marker_descriptions,
    }
    return reconstructed, diagnostics


def compare_with_manifest(
    reconstructed: pd.DataFrame,
    manifest: pd.DataFrame,
    diagnostics: dict[str, int],
    raw_sha256: str,
    config: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    """Return machine-readable exact-match evidence."""

    missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    expected = config["expected"]
    expected_split_rows = expected["split_rows"]
    observed_split_rows = {
        name: int(count)
        for name, count in reconstructed["split"].value_counts().sort_index().items()
    }
    manifest_split_rows = {
        name: int(count)
        for name, count in manifest["split"].value_counts().sort_index().items()
    }

    left = reconstructed.loc[:, MANIFEST_COLUMNS].sort_values("loan_id").reset_index(drop=True)
    right = manifest.loc[:, MANIFEST_COLUMNS].sort_values("loan_id").reset_index(drop=True)
    reconstructed_ids = set(left["loan_id"])
    manifest_ids = set(right["loan_id"])
    missing_ids = sorted(manifest_ids - reconstructed_ids)
    extra_ids = sorted(reconstructed_ids - manifest_ids)

    shared = left.merge(right, on="loan_id", how="inner", suffixes=("_actual", "_locked"))
    mismatch_counts: dict[str, int] = {}
    for column in MANIFEST_COLUMNS[1:]:
        mismatch_counts[column] = int(
            shared[f"{column}_actual"].astype("string").ne(
                shared[f"{column}_locked"].astype("string")
            ).sum()
        )

    checks = {
        "raw_csv_sha256": raw_sha256 == config["source"]["raw_csv_sha256"],
        "raw_row_count": diagnostics["raw_rows"] == int(config["source"]["expected_raw_rows"]),
        "eligible_row_count": diagnostics["eligible_rows"] == int(expected["eligible_rows"]),
        "literal_marker_count": diagnostics["literal_missing_marker_descriptions"]
        == int(expected["known_literal_missing_marker_descriptions"]),
        "reconstructed_ids_unique": not reconstructed["loan_id"].duplicated().any(),
        "manifest_ids_unique": not manifest["loan_id"].duplicated().any(),
        "exact_id_set": not missing_ids and not extra_ids,
        "exact_row_contract": not any(mismatch_counts.values()),
        "expected_split_counts": observed_split_rows == {
            name: int(value) for name, value in expected_split_rows.items()
        },
        "manifest_split_counts": observed_split_rows == manifest_split_rows,
        "statistics_eligible_rows": diagnostics["eligible_rows"]
        == int(statistics["eligible_rows"]),
        "statistics_dataset_version": config["dataset_version"]
        == statistics["dataset_version"],
        "statistics_split_counts": all(
            observed_split_rows.get(name) == int(info["rows"])
            for name, info in statistics["splits"].items()
        ),
    }

    return {
        "ok": all(checks.values()),
        "cohort_version": config["cohort_version"],
        "split_version": config["split_version"],
        "raw_csv_sha256": raw_sha256,
        "diagnostics": diagnostics,
        "reconstructed_split_rows": observed_split_rows,
        "manifest_split_rows": manifest_split_rows,
        "missing_id_count": len(missing_ids),
        "extra_id_count": len(extra_ids),
        "first_missing_ids": missing_ids[:10],
        "first_extra_ids": extra_ids[:10],
        "field_mismatch_counts": mismatch_counts,
        "checks": checks,
    }


def reproduce(
    *,
    raw_csv: Path,
    manifest_path: Path,
    statistics_path: Path,
    config_path: Path,
    chunksize: int = 200_000,
) -> dict[str, Any]:
    """Run the complete, read-only cohort reproduction audit."""

    config = load_config(config_path)
    raw_sha256 = sha256_file(raw_csv)
    reconstructed, diagnostics = reconstruct_cohort(raw_csv, config, chunksize=chunksize)
    manifest = pd.read_csv(manifest_path, dtype={"loan_id": "string"})
    with statistics_path.open(encoding="utf-8") as stream:
        statistics = json.load(stream)
    return compare_with_manifest(
        reconstructed,
        manifest,
        diagnostics,
        raw_sha256,
        config,
        statistics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional evidence report path; split files are never modified.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.raw_csv, args.manifest, args.statistics, args.config):
        if not path.is_file():
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            return 2

    report = reproduce(
        raw_csv=args.raw_csv,
        manifest_path=args.manifest,
        statistics_path=args.statistics,
        config_path=args.config,
        chunksize=args.chunksize,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

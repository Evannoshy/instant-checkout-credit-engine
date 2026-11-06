"""Fusion development roles (fusion_development_v1): the shared loan ID -> role mapping.

``assign_fusion_roles(data_dir)`` labels every train and validation loan of
split-v1 with one of four roles from ``configs/fusion_development_v1.toml``,
using only the manifest's ``issue_month``:

- ``base_fit`` (train, 2007-06 .. 2012-12): fit base models and choose their
  settings (vocabulary, checkpoints, time-ordered folds).
- ``fusion_fit`` (train, 2013-01 .. 2013-04): train the stacker.
- ``fusion_calibration`` (train, 2013-05 .. 2013-07): fit the final fusion
  calibrator only.
- ``validation`` (the whole validation split, 2013-08 .. 2013-12): development
  comparisons; nothing is fitted on it.

The locked test split is not a role and is not in the mapping: rows come from
``evaluate.load_manifest``, which removes test rows before returning, so this
module cannot see a test ID. As an independent check, the finished mapping's
IDs are compared with ``test_ids.csv`` (an ID-set check; no test data is read)
and any overlap stops the run; the manifest records the measured count. The
config's ``locked_test_rows`` is checked against ``split_statistics.json`` as
an audit count only.

Cross-track contract: ``data/fusion/fusion_roles_v1.csv`` with columns
``loan_id, split, role, issue_month``, one row per loan, sorted by loan_id.
``split`` is the original split-v1 label and never a role name; ``role`` is
new. ``issue_month`` is audit metadata, never a model input. No target column.
NLP score exports must cover exactly the 48,724 loans whose role is not
``base_fit``, with ``split`` and ``role`` equal to this file's. The CSV is the
contract, so other tracks read the file and do not import this module.

Relation to ``tabular_development_v1``: ``base_fit`` covers the same months as
its ``model_fit``, and ``fusion_fit`` + ``fusion_calibration`` together cover
its ``calibration``. That is why the existing tabular scores fitted on
``model_fit`` are valid inputs here; test_fusion_roles.py checks the ID sets.

Run from the repository root (no raw CSV needed) to write the mapping and
reports/tabular/fusion_roles_v1_manifest.json:
    python -m src.tabular.fusion_roles

Known limitations:
- Reuses ``_month_index`` and ``_validate_role_months`` from
  src/tabular/development.py. The leading underscore marks them internal, so
  editing either there can change this module too.
- ``fusion_fit`` spans four months; a stacker learns its weights from a short
  window. ``fusion_calibration`` is 11,353 loans: enough for Platt, thin for
  isotonic.
- The roles exist only as this file and its config. Regenerating from a
  changed config changes the file's SHA-256, which consumers should pin.
- Two development configs now describe overlapping months:
  tabular_development_v1 (two roles, committed tabular experiments) and this
  one (four roles, fusion). Only the consistency test keeps them aligned.
"""

from __future__ import annotations

import argparse
import json
import platform
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

from src.nlp.preprocess import sha256_file
from src.tabular import provenance
from src.tabular.development import DEFAULT_COHORT_CONFIG, _month_index, _validate_role_months
from src.tabular.evaluate import (
    EXPECTED_COHORT,
    LOADABLE_SPLITS,
    TEST_SPLIT_LOCKED_MESSAGE,
    load_manifest,
)
from src.tabular.preprocess import DEFAULT_DATA_DIR, REPO_ROOT

DEFAULT_FUSION_CONFIG = REPO_ROOT / "configs" / "fusion_development_v1.toml"
ROLE_MAPPING_PATH = REPO_ROOT / "data" / "fusion" / "fusion_roles_v1.csv"
ROLE_MANIFEST_PATH = REPO_ROOT / "reports" / "tabular" / "fusion_roles_v1_manifest.json"

ROLES = ("base_fit", "fusion_fit", "fusion_calibration", "validation")
# Roles the NLP tracks score; base_fit is where they fit.
SCORING_ROLES = ("fusion_fit", "fusion_calibration", "validation")
MAPPING_COLUMNS = ("loan_id", "split", "role", "issue_month")

REQUIRED_CONFIG_KEYS = {
    "fusion_development_version", "status", "cohort_version", "split_version",
    "tabular_development_version", "roles", "expected",
}
REQUIRED_EXPECTED_KEYS = {"nlp_scoring_rows", "locked_test_rows", "role_rows"}

# Every ValueError message raised by this module, as str.format templates.
# The test-split message is owned by src/tabular/evaluate.py and only referenced.
ERROR_MESSAGES: dict[str, str] = {
    # Fusion config (load_fusion_config)
    "config_missing_keys": "Fusion development config is missing keys: {missing}",
    "config_wrong_cohort": "Fusion development config targets {actual!r}, not {expected!r}",
    "cohort_config_disagrees": (
        "Fusion development config {key} {actual!r} disagrees with the cohort config {expected!r}"
    ),
    "wrong_role_names": "Fusion development config must define exactly roles {expected}; got {actual}",
    "expected_missing_keys": "[expected] is missing keys: {missing}",
    "wrong_expected_roles": "[expected.role_rows] must list exactly roles {expected}; got {actual}",
    "bad_expected_count": "[expected] {key} must be a positive integer; got {count!r}",
    "test_split_locked": TEST_SPLIT_LOCKED_MESSAGE,
    "bad_role_split": "Role {role} split must be one of {allowed}; got {split!r}",
    "split_without_roles": "No role is carved from the {split} split",
    "roles_overlap": "Roles {earlier} and {later} overlap in {split}: {later} starts at {first}",
    "roles_gap": (
        "Gap between roles {earlier} and {later} in {split}: {earlier} ends {last}, "
        "{later} starts {first}"
    ),
    "roles_do_not_cover_split": (
        "{split} roles cover {first} .. {last} but the cohort {split} split is "
        "{split_first} .. {split_last}"
    ),
    "scoring_rows_mismatch": (
        "[expected] nlp_scoring_rows {expected} differs from the sum of {roles} role_rows ({actual})"
    ),
    # Manifest and role assignment (assign_fusion_roles)
    "split_version_mismatch": "Fusion development config split version disagrees with split statistics",
    "test_rows_mismatch": (
        "split_statistics.json has {actual} test rows but [expected] locked_test_rows is {expected}"
    ),
    "rows_not_in_one_role": (
        "{unassigned} train/validation rows match no role and {multiple} match more than one, "
        "e.g. {sample}"
    ),
    "role_count_mismatch": "Role row counts {observed} differ from [expected.role_rows] {expected}",
    "test_ids_in_mapping": "The mapping contains {count} locked test IDs, e.g. {sample}",
}


def _require_positive_int(key: str, count: Any) -> None:
    """Reject a count that is not a positive int; bool is an int subclass, so exclude it explicitly."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(ERROR_MESSAGES["bad_expected_count"].format(key=key, count=count))


def _validate_role_splits(config: dict[str, Any]) -> None:
    """Require every role to name train or validation; the locked test split gets the shared message."""
    for role, bounds in config["roles"].items():
        split = bounds.get("split")
        if split == "test":
            raise ValueError(ERROR_MESSAGES["test_split_locked"])
        if split not in LOADABLE_SPLITS:
            raise ValueError(ERROR_MESSAGES["bad_role_split"].format(
                role=role, allowed=list(LOADABLE_SPLITS), split=split,
            ))


def _validate_split_coverage(config: dict[str, Any], cohort: dict[str, Any]) -> None:
    """Per split, require roles to be contiguous, non-overlapping and to cover the cohort's months exactly.

    development._validate_role_coverage does this for the train split only;
    these roles span train and validation, so each split is checked on its own.
    """
    for split in LOADABLE_SPLITS:
        ordered = sorted(
            ((role, bounds) for role, bounds in config["roles"].items() if bounds["split"] == split),
            key=lambda item: item[1]["first_month"],
        )
        if not ordered:
            raise ValueError(ERROR_MESSAGES["split_without_roles"].format(split=split))
        for (earlier, before), (later, after) in zip(ordered, ordered[1:]):
            step = _month_index(after["first_month"]) - _month_index(before["last_month"])
            if step < 1:
                raise ValueError(ERROR_MESSAGES["roles_overlap"].format(
                    earlier=earlier, later=later, split=split, first=after["first_month"],
                ))
            if step > 1:
                raise ValueError(ERROR_MESSAGES["roles_gap"].format(
                    earlier=earlier, later=later, split=split,
                    last=before["last_month"], first=after["first_month"],
                ))

        first, last = ordered[0][1]["first_month"], ordered[-1][1]["last_month"]
        split_first = cohort["splits"][f"{split}_first_month"]
        split_last = cohort["splits"][f"{split}_last_month"]
        if (first, last) != (split_first, split_last):
            raise ValueError(ERROR_MESSAGES["roles_do_not_cover_split"].format(
                split=split, first=first, last=last, split_first=split_first, split_last=split_last,
            ))


def load_fusion_config(
    path: str | Path = DEFAULT_FUSION_CONFIG,
    *,
    cohort_config: str | Path = DEFAULT_COHORT_CONFIG,
) -> dict[str, Any]:
    """Load and validate a fusion development config.

    Raises ValueError for missing keys, a cohort or split version that
    disagrees with the cohort config, role names other than the four fusion
    roles, malformed expected counts, a role on the test split or an unknown
    split, malformed or reversed months, roles that overlap, leave a gap or
    miss part of a split, or an nlp_scoring_rows that is not the sum of the
    scoring roles' counts.
    """
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    with Path(cohort_config).open("rb") as stream:
        cohort = tomllib.load(stream)

    if missing := REQUIRED_CONFIG_KEYS - set(config):
        raise ValueError(ERROR_MESSAGES["config_missing_keys"].format(missing=sorted(missing)))
    if config["cohort_version"] != EXPECTED_COHORT:
        raise ValueError(ERROR_MESSAGES["config_wrong_cohort"].format(
            actual=config["cohort_version"], expected=EXPECTED_COHORT,
        ))
    for key in ("cohort_version", "split_version"):
        if config[key] != cohort.get(key):
            raise ValueError(ERROR_MESSAGES["cohort_config_disagrees"].format(
                key=key, actual=config[key], expected=cohort.get(key),
            ))

    if set(config["roles"]) != set(ROLES):
        raise ValueError(ERROR_MESSAGES["wrong_role_names"].format(
            expected=sorted(ROLES), actual=sorted(config["roles"]),
        ))
    expected = config["expected"]
    if missing := REQUIRED_EXPECTED_KEYS - set(expected):
        raise ValueError(ERROR_MESSAGES["expected_missing_keys"].format(missing=sorted(missing)))
    if set(expected["role_rows"]) != set(ROLES):
        raise ValueError(ERROR_MESSAGES["wrong_expected_roles"].format(
            expected=sorted(ROLES), actual=sorted(expected["role_rows"]),
        ))
    for role, count in expected["role_rows"].items():
        _require_positive_int(f"role_rows.{role}", count)
    for key in ("nlp_scoring_rows", "locked_test_rows"):
        _require_positive_int(key, expected[key])

    _validate_role_splits(config)
    _validate_role_months(config)
    _validate_split_coverage(config, cohort)

    scoring_total = sum(expected["role_rows"][role] for role in SCORING_ROLES)
    if expected["nlp_scoring_rows"] != scoring_total:
        raise ValueError(ERROR_MESSAGES["scoring_rows_mismatch"].format(
            expected=expected["nlp_scoring_rows"], roles=list(SCORING_ROLES), actual=scoring_total,
        ))
    return config


def _test_ids_in(mapping: pd.DataFrame, data_dir: str | Path) -> list[str]:
    """Mapping IDs that also appear in test_ids.csv, sorted (an ID-set check; no test data is read)."""
    test_ids = pd.read_csv(Path(data_dir) / "test_ids.csv", usecols=["loan_id"], dtype={"loan_id": str})
    return sorted(set(mapping["loan_id"]) & set(test_ids["loan_id"]))


def _assign(
    data_dir: str | Path, fusion_config: str | Path
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Role assignment, the redacted split statistics it was checked against, and the config."""
    config = load_fusion_config(fusion_config)
    manifest, stats = load_manifest(data_dir)  # train and validation rows only
    if config["split_version"] != stats.get("split_version"):
        raise ValueError(ERROR_MESSAGES["split_version_mismatch"])
    test_rows = stats.get("splits", {}).get("test", {}).get("rows")
    if test_rows != config["expected"]["locked_test_rows"]:
        raise ValueError(ERROR_MESSAGES["test_rows_mismatch"].format(
            actual=test_rows, expected=config["expected"]["locked_test_rows"],
        ))

    rows = manifest[["loan_id", "split", "issue_month"]].astype({"loan_id": str})
    rows["role"] = pd.Series(pd.NA, index=rows.index, dtype="object")
    matches = pd.Series(0, index=rows.index)
    for role, bounds in config["roles"].items():
        in_role = rows["split"].eq(bounds["split"]) & rows["issue_month"].between(
            bounds["first_month"], bounds["last_month"]
        )
        matches += in_role
        rows.loc[in_role, "role"] = role

    if not matches.eq(1).all():
        raise ValueError(ERROR_MESSAGES["rows_not_in_one_role"].format(
            unassigned=int(matches.eq(0).sum()), multiple=int(matches.gt(1).sum()),
            sample=rows.loc[matches.ne(1), "loan_id"].head(5).tolist(),
        ))
    # The config already ties nlp_scoring_rows to these counts, so matching
    # role_rows also fixes the scoring total.
    observed = {role: int(rows["role"].eq(role).sum()) for role in config["roles"]}
    if observed != config["expected"]["role_rows"]:
        raise ValueError(ERROR_MESSAGES["role_count_mismatch"].format(
            observed=observed, expected=config["expected"]["role_rows"],
        ))

    mapping = rows[list(MAPPING_COLUMNS)].sort_values("loan_id").reset_index(drop=True)
    # load_manifest already removed test rows; this re-checks against the locked ID file itself.
    if leaked := _test_ids_in(mapping, data_dir):
        raise ValueError(ERROR_MESSAGES["test_ids_in_mapping"].format(
            count=len(leaked), sample=leaked[:5],
        ))
    return mapping, stats, config


def assign_fusion_roles(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    fusion_config: str | Path = DEFAULT_FUSION_CONFIG,
) -> pd.DataFrame:
    """Return the mapping: columns loan_id, split, role, issue_month; one row per train/validation loan.

    Pure manifest logic: never opens the raw CSV or reads a target. Sorted by
    loan_id with a plain RangeIndex. Raises ValueError for an invalid config,
    a split version or test-row count that disagrees with
    split_statistics.json, a row in no role or in more than one, role
    counts that differ from [expected.role_rows], or a mapping ID listed in
    test_ids.csv.
    """
    mapping, _, _ = _assign(data_dir, fusion_config)
    return mapping


def build_role_manifest(
    mapping: pd.DataFrame,
    stats: dict[str, Any],
    config: dict[str, Any],
    data_dir: str | Path,
    fusion_config: str | Path,
    output_path: Path,
) -> dict[str, Any]:
    """Versions, input and output checksums, and per-role counts for a written mapping.

    Month ranges are read from the mapping itself, not echoed from the config,
    so the manifest records what was written.
    """
    roles = {}
    for role in ROLES:
        group = mapping.loc[mapping["role"].eq(role)]
        roles[role] = {
            "split": config["roles"][role]["split"],
            "rows": len(group),
            "first_month": group["issue_month"].min(),
            "last_month": group["issue_month"].max(),
        }
    data_dir = Path(data_dir)
    return {
        "fusion_development_version": config["fusion_development_version"],
        "status": config["status"],
        "approved_by": config.get("approved_by"),
        "approved_on": config.get("approved_on"),
        "approval_ref": config.get("approval_ref"),
        "cohort_version": config["cohort_version"],
        "split_version": config["split_version"],
        "dataset_version": stats["dataset_version"],
        "tabular_development_version": config["tabular_development_version"],
        "output": {
            "path": provenance.display_path(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(mapping),
            "columns": list(mapping.columns),
        },
        "roles": roles,
        "nlp_scoring_roles": list(SCORING_ROLES),
        "nlp_scoring_rows": int(mapping["role"].isin(SCORING_ROLES).sum()),
        "locked_test_rows": config["expected"]["locked_test_rows"],
        "test_rows_in_mapping": len(_test_ids_in(mapping, data_dir)),
        "targets_included": False,
        "inputs_sha256": provenance.file_checksums([
            data_dir / "split_manifest.csv",
            data_dir / "split_statistics.json",
            DEFAULT_COHORT_CONFIG,
            fusion_config,
        ]),
        "commands": [
            "python -m src.tabular.fusion_roles",
            "python -m pytest src/tabular/test_fusion_roles.py -v -s",
        ],
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "platform": platform.system(),
            "machine": platform.machine(),
        },
    }


def run_role_mapping(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = ROLE_MAPPING_PATH,
    manifest_path: str | Path = ROLE_MANIFEST_PATH,
    fusion_config: str | Path = DEFAULT_FUSION_CONFIG,
) -> dict[str, Any]:
    """Assign and check every role, then write the mapping CSV and its manifest JSON.

    Nothing is written unless the whole assignment passes. Both files use LF
    line endings, so the recorded SHA-256 does not depend on the OS.
    """
    output_path, manifest_path = Path(output_path), Path(manifest_path)
    mapping, stats, config = _assign(data_dir, fusion_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_path, index=False, lineterminator="\n")
    manifest = build_role_manifest(mapping, stats, config, data_dir, fusion_config, output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the fusion loan ID -> role mapping.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--fusion-config", default=DEFAULT_FUSION_CONFIG, type=Path)
    parser.add_argument("--output", default=ROLE_MAPPING_PATH, type=Path)
    parser.add_argument("--manifest", default=ROLE_MANIFEST_PATH, type=Path)
    args = parser.parse_args()
    manifest = run_role_mapping(args.data_dir, args.output, args.manifest, args.fusion_config)
    print(json.dumps({key: manifest[key] for key in ("output", "roles", "nlp_scoring_rows")}, indent=2))


if __name__ == "__main__":
    main()

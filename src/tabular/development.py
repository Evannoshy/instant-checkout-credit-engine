"""Tabular development roles: model_fit and calibration rows carved from split-v1 train.

``load_tabular_role(data_dir, raw_csv_path, role)`` returns ``(features, target)``
for one development role defined in ``configs/tabular_development_v1.toml``.
Roles split the ``train`` rows of split-v1 by manifest ``issue_month``:
``model_fit`` (2007-06 .. 2012-12) fits models and chooses their settings;
``calibration`` (2013-01 .. 2013-07) fits only the score -> probability mapping.
Validation still comes from ``preprocess.load_tabular_split(..., "validation")``
and is never used to fit anything; ``test`` is rejected before any file is opened.

Contract for the model pipelines (XGBoost search, logistic refit, ablations):
- ``features`` and ``target`` follow the ``load_tabular_split`` contract exactly
  (indexed and sorted by ``loan_id``, configured columns and dtypes), restricted
  to the role's rows. Only rows change; no feature is added or transformed here.
- Fit on ``model_fit`` rows only. Never fit on ``load_tabular_split(..., "train")``:
  it returns all train rows, calibration included, and nothing in code blocks it.
- ``assign_development_roles(data_dir)`` gives ``loan_id -> issue_month, role``
  for time-aware CV folds inside ``model_fit``. ``issue_month`` is not a model
  feature, so it is returned here by ``loan_id`` rather than in ``features``.

Training-time only: preprocess.py holds the serving-relevant transformations
(D-016) and is not changed; this module only selects rows from its output.
Manifest validation and the split-lock message are reused from
src/tabular/evaluate.py.

Run from the repository root to print each role's counts from the manifest
(no raw CSV needed):
    python -m src.tabular.development

Known limitations:
- Every ``load_tabular_role`` call scans the full raw CSV through
  ``load_tabular_split``, so loading both roles scans it twice. There is no cache.
- Roles derive only from manifest ``issue_month``; no new split or ID files are
  written, so the role of a loan exists only by rerunning this code with the
  same config.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

from src.tabular.evaluate import (
    EXPECTED_COHORT,
    LOADABLE_SPLITS,
    TEST_SPLIT_LOCKED_MESSAGE,
    get_split_targets,
    load_manifest,
)
from src.tabular.preprocess import (
    DEFAULT_DATA_DIR,
    DEFAULT_FEATURE_CONFIG,
    REPO_ROOT,
    load_feature_config,
    load_tabular_split,
)

DEFAULT_DEVELOPMENT_CONFIG = REPO_ROOT / "configs" / "tabular_development_v1.toml"
DEFAULT_COHORT_CONFIG = REPO_ROOT / "configs" / "cohort_v1.toml"

ROLES = ("model_fit", "calibration")
_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

REQUIRED_DEVELOPMENT_CONFIG_KEYS = {
    "development_version", "status", "cohort_version", "split_version", "source_split",
    "roles", "expected",
}

# Every ValueError message raised by this module, as str.format templates.
# Placeholders are filled at the raise site; messages with none are used as-is.
# The test-split message is owned by src/tabular/evaluate.py and only referenced.
ERROR_MESSAGES: dict[str, str] = {
    # Development config (load_development_config)
    "config_missing_keys": "Development config is missing keys: {missing}",
    "config_wrong_cohort": "Development config targets {actual!r}, not {expected!r}",
    "cohort_config_disagrees": (
        "Development config {key} {actual!r} disagrees with the cohort config {expected!r}"
    ),
    "source_split_not_train": "Roles must be carved from the train split; got {actual!r}",
    "wrong_role_names": "Development config must define exactly roles {expected}; got {actual}",
    "wrong_expected_roles": "[expected.role_rows] must list exactly roles {expected}; got {actual}",
    "bad_expected_count": "[expected.role_rows] {role} must be a positive integer; got {count!r}",
    "bad_month": "Role {role} {key} must be a 'YYYY-MM' string; got {value!r}",
    "role_range_reversed": "Role {role} starts after it ends: {first} > {last}",
    "roles_overlap": "Roles {earlier} and {later} overlap: {later} starts at {first}",
    "roles_gap": "Gap between roles {earlier} and {later}: {earlier} ends {last}, {later} starts {first}",
    "roles_do_not_cover_train": (
        "Roles cover {first} .. {last} but the cohort train split is "
        "{train_first} .. {train_last}"
    ),
    # Load parameters
    "test_split_locked": TEST_SPLIT_LOCKED_MESSAGE,
    "split_not_a_role": (
        "{role!r} is a split, not a development role; roles {roles} are carved from "
        "the train split only"
    ),
    "unknown_role": "Unknown role: {role!r}; expected one of {roles}",
    "feature_config_disagrees": (
        "Development config {key} {actual!r} disagrees with the feature config {expected!r}"
    ),
    # Manifest and role assignment
    "split_version_mismatch": "Development config split version disagrees with split statistics",
    "empty_source_split": "No manifest rows found for {split}",
    "rows_not_in_one_role": (
        "{unassigned} {split} rows match no role and {multiple} match more than one, "
        "e.g. {sample}"
    ),
    "role_count_mismatch": "Role row counts {observed} differ from [expected.role_rows] {expected}",
    # Output
    "role_rows_not_loaded": "Loaded {loaded} {role} rows but the manifest assigns {expected}",
    "output_not_aligned": "Feature and target rows are not aligned on loan_id",
}


def _month_index(month: str) -> int:
    """Map 'YYYY-MM' to a running month number so adjacent months differ by one."""
    year, number = month.split("-")
    return int(year) * 12 + int(number) - 1


def _validate_role_months(config: dict[str, Any]) -> None:
    """Require every role boundary to be a well-formed 'YYYY-MM' with first <= last."""
    for role, bounds in config["roles"].items():
        for key in ("first_month", "last_month"):
            value = bounds.get(key)
            if not isinstance(value, str) or not _MONTH_PATTERN.match(value):
                raise ValueError(ERROR_MESSAGES["bad_month"].format(role=role, key=key, value=value))
        if bounds["first_month"] > bounds["last_month"]:
            raise ValueError(ERROR_MESSAGES["role_range_reversed"].format(
                role=role, first=bounds["first_month"], last=bounds["last_month"],
            ))


def _validate_role_coverage(config: dict[str, Any], cohort: dict[str, Any]) -> None:
    """Require roles to be contiguous, non-overlapping and to cover the cohort train range exactly."""
    ordered = sorted(config["roles"].items(), key=lambda item: item[1]["first_month"])
    for (earlier, before), (later, after) in zip(ordered, ordered[1:]):
        step = _month_index(after["first_month"]) - _month_index(before["last_month"])
        if step < 1:
            raise ValueError(ERROR_MESSAGES["roles_overlap"].format(
                earlier=earlier, later=later, first=after["first_month"],
            ))
        if step > 1:
            raise ValueError(ERROR_MESSAGES["roles_gap"].format(
                earlier=earlier, later=later,
                last=before["last_month"], first=after["first_month"],
            ))

    first, last = ordered[0][1]["first_month"], ordered[-1][1]["last_month"]
    train_first = cohort["splits"]["train_first_month"]
    train_last = cohort["splits"]["train_last_month"]
    if (first, last) != (train_first, train_last):
        raise ValueError(ERROR_MESSAGES["roles_do_not_cover_train"].format(
            first=first, last=last, train_first=train_first, train_last=train_last,
        ))


def load_development_config(
    path: str | Path = DEFAULT_DEVELOPMENT_CONFIG,
    *,
    cohort_config: str | Path = DEFAULT_COHORT_CONFIG,
) -> dict[str, Any]:
    """Load and validate a tabular development config."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    with Path(cohort_config).open("rb") as stream:
        cohort = tomllib.load(stream)

    if missing := REQUIRED_DEVELOPMENT_CONFIG_KEYS - set(config):
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
    if config["source_split"] != "train":
        raise ValueError(ERROR_MESSAGES["source_split_not_train"].format(
            actual=config["source_split"],
        ))

    if set(config["roles"]) != set(ROLES):
        raise ValueError(ERROR_MESSAGES["wrong_role_names"].format(
            expected=sorted(ROLES), actual=sorted(config["roles"]),
        ))
    expected_rows = config["expected"].get("role_rows", {})
    if set(expected_rows) != set(ROLES):
        raise ValueError(ERROR_MESSAGES["wrong_expected_roles"].format(
            expected=sorted(ROLES), actual=sorted(expected_rows),
        ))
    for role, count in expected_rows.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(ERROR_MESSAGES["bad_expected_count"].format(role=role, count=count))

    _validate_role_months(config)
    _validate_role_coverage(config, cohort)
    return config


def _validate_role(role: str) -> None:
    """Reject the locked test split, the split names, and unknown roles before any file is read."""
    if role == "test":
        raise ValueError(ERROR_MESSAGES["test_split_locked"])
    if role in LOADABLE_SPLITS:
        raise ValueError(ERROR_MESSAGES["split_not_a_role"].format(role=role, roles=list(ROLES)))
    if role not in ROLES:
        raise ValueError(ERROR_MESSAGES["unknown_role"].format(role=role, roles=list(ROLES)))


def _validate_feature_config_versions(
    config: dict[str, Any], feature_config: dict[str, Any]
) -> None:
    """Require the development and feature configs to name the same cohort and split."""
    for key in ("cohort_version", "split_version"):
        if config[key] != feature_config[key]:
            raise ValueError(ERROR_MESSAGES["feature_config_disagrees"].format(
                key=key, actual=config[key], expected=feature_config[key],
            ))


def assign_development_roles(
    data_dir: str | Path,
    development_config: str | Path = DEFAULT_DEVELOPMENT_CONFIG,
) -> pd.DataFrame:
    """Return the role of every train row: columns issue_month, role, indexed by loan_id.

    Pure manifest logic: never opens the raw CSV. Rows are sorted by loan_id,
    so the index lines up by label with ``load_tabular_role`` features.
    Raises ValueError for an invalid development config, a split version that
    disagrees with split_statistics.json, a train row in no role or in more
    than one, or role counts that differ from [expected.role_rows].
    """
    config = load_development_config(development_config)
    manifest, stats = load_manifest(data_dir)
    if config["split_version"] != stats.get("split_version"):
        raise ValueError(ERROR_MESSAGES["split_version_mismatch"])

    split = config["source_split"]
    rows = (
        manifest.loc[manifest["split"].eq(split), ["loan_id", "issue_month"]]
        .astype({"loan_id": str})  # same index dtype as load_tabular_split
        .set_index("loan_id")
    )
    if rows.empty:
        raise ValueError(ERROR_MESSAGES["empty_source_split"].format(split=split))

    rows["role"] = pd.Series(pd.NA, index=rows.index, dtype="object")
    matches = pd.Series(0, index=rows.index)
    for role, bounds in config["roles"].items():
        in_role = rows["issue_month"].between(bounds["first_month"], bounds["last_month"])
        matches += in_role
        rows.loc[in_role, "role"] = role

    if not matches.eq(1).all():
        raise ValueError(ERROR_MESSAGES["rows_not_in_one_role"].format(
            unassigned=int(matches.eq(0).sum()), multiple=int(matches.gt(1).sum()), split=split,
            sample=rows.index[matches.ne(1)][:5].tolist(),
        ))
    observed = {role: int(rows["role"].eq(role).sum()) for role in config["roles"]}
    expected = config["expected"]["role_rows"]
    if observed != expected:
        raise ValueError(ERROR_MESSAGES["role_count_mismatch"].format(
            observed=observed, expected=expected,
        ))
    
    return rows.sort_index()


def _validate_output(
    features: pd.DataFrame, target: pd.Series, role: str, expected: int
) -> None:
    """Require the role's full row count, with features and target aligned."""
    if len(features) != expected:
        raise ValueError(ERROR_MESSAGES["role_rows_not_loaded"].format(
            loaded=len(features), role=role, expected=expected,
        ))
    if not features.index.equals(target.index):
        raise ValueError(ERROR_MESSAGES["output_not_aligned"])


def load_tabular_role(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    role: str,
    feature_config: str | Path = DEFAULT_FEATURE_CONFIG,
    development_config: str | Path = DEFAULT_DEVELOPMENT_CONFIG,
    *,
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load one development role as (features, target), both indexed by loan_id.

    Rejects the role before any file is opened, validates both configs and
    assigns roles from the manifest, then loads the train split once through
    ``load_tabular_split`` and keeps the role's rows in its loan_id order.
    Raises ValueError for the locked test split, a split name or unknown role,
    an invalid or disagreeing config, a role assignment or count failure, or
    any error ``load_tabular_split`` raises.
    """
    _validate_role(role)
    config = load_development_config(development_config)
    _validate_feature_config_versions(config, load_feature_config(feature_config))
    roles = assign_development_roles(data_dir, development_config)

    features, target = load_tabular_split(
        data_dir, raw_csv_path, config["source_split"], feature_config, chunksize=chunksize,
    )
    keep = features.index.isin(roles.index[roles["role"].eq(role)])
    features, target = features.loc[keep], target.loc[keep]
    _validate_output(features, target, role, config["expected"]["role_rows"][role])
    return features, target


def profile_roles(roles: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    """Summarise each role from the manifest: rows, defaults, default rate and month range."""
    targets = get_split_targets(manifest, "train")
    rows = roles.join(targets.set_index("loan_id"), how="left", validate="one_to_one")
    profile: dict[str, Any] = {}
    for role in ROLES:
        group = rows.loc[rows["role"].eq(role)]
        profile[role] = {
            "rows": len(group),
            "defaults": int(group["target"].sum()),
            "default_rate": float(group["target"].mean()),
            "first_month": group["issue_month"].min(),
            "last_month": group["issue_month"].max(),
        }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Print each development role's manifest counts.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--development-config", default=DEFAULT_DEVELOPMENT_CONFIG, type=Path)
    args = parser.parse_args()
    config = load_development_config(args.development_config)
    roles = assign_development_roles(args.data_dir, args.development_config)
    manifest, _ = load_manifest(args.data_dir)
    summary = {
        "development_version": config["development_version"],
        "status": config["status"],
        "source_split": config["source_split"],
        "roles": profile_roles(roles, manifest),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

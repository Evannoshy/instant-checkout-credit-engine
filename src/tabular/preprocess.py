"""Tabular preprocessing: deterministic feature rows for the logistic baseline.

``load_tabular_split(data_dir, raw_csv_path, split, feature_config)`` returns
``(features, target)`` for one development split of cohort
``real_text_matured_v1`` / ``split-v1``, using the feature set defined in
``configs/tabular_features_v1.toml``. Only ``train`` and ``validation`` are
available; ``test`` is rejected before any file is opened.

Contract for the model pipeline (src/tabular/logistic_baseline.py):
- ``features`` is indexed by ``loan_id``, sorted by ``loan_id``, and its
  columns are exactly ``[model].numeric + [model].categorical`` in config
  order. Numeric columns are float64; categorical columns are object dtype.
- ``target`` is the int8 manifest target (Fully Paid 0, Charged Off 1) on the
  same index. It is never a feature column, and raw ``loan_status`` is used
  only to cross-check alignment before being dropped.
- Missing values are preserved as NaN and no row is ever dropped for a
  missing feature. Every transformation here is a pure function of one row:
  no median, scale, category list or other statistic is learned. Imputation,
  scaling and one-hot encoding belong to the model pipeline and must be fitted
  on training rows only, so nothing learned from one split can reach another.
- Unexpected non-missing values (non-numeric text, an unknown ``term`` or
  ``home_ownership`` level, an unparseable date, negative income) raise
  ValueError rather than being coerced to missing.

The chunked raw read follows src/nlp/preprocess.load_original_split (one-based
data-record numbering, header excluded) but is deliberately copied rather than
imported so the two tracks' contracts stay independent. Manifest validation and
the split-lock message are reused from src/tabular/evaluate.py.

Run from the repository root to print a profile of one split:
    python -m src.tabular.preprocess --split train

Known limitations:
- The raw CSV's SHA-256 is not verified on every load (hashing 1.6 GB per
  call is slow); only the record count is checked against
  split_statistics.json, as in the NLP loader. Provenance is audited by
  scripts/reproduce_locked_cohort.py.
- ``credit_history_months`` has month granularity because both
  ``earliest_cr_line`` and ``issue_d`` are month-level. Negative values are
  kept, not clipped; ``main()`` reports how many occur.
- ``dti`` stops at LendingClub's underwriting cut-off (policy §4); the loader
  does not correct this.
- Every call scans all ~2.26M raw records (a minute or two).
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The implementations for the following imports may be shifted here
from src.tabular.evaluate import (
    EXPECTED_COHORT,
    LOADABLE_SPLITS,
    TEST_SPLIT_LOCKED_MESSAGE,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RAW_CSV = DEFAULT_DATA_DIR / "raw" / "loan.csv"
DEFAULT_FEATURE_CONFIG = REPO_ROOT / "configs" / "tabular_features_v1.toml"

ID_FILES = {"train": "train_ids.csv", "validation": "val_ids.csv"}
# Columns read only to cross-check the positional join; never model features.
ALIGNMENT_COLUMNS = ("issue_d", "loan_status")
STATUS_TO_TARGET = {"Fully Paid": 0, "Charged Off": 1}
DATE_FORMAT = "%b-%Y"
_TERM_PATTERN = re.compile(r"^(\d+) months$")

REQUIRED_FEATURE_CONFIG_KEYS = {
    "feature_set_version", "cohort_version", "split_version", "identifier_column",
    "target_column", "source", "derived", "model", "categories", "prohibited",
}

# Every ValueError message raised by this module, as str.format templates.
# Placeholders are filled at the raise site; messages with none are used as-is.
# The test-split message is owned by src/tabular/evaluate.py and only referenced.
ERROR_MESSAGES: dict[str, str] = {
    # Feature config (load_feature_config, _resolve_transformation)
    "config_missing_keys": "Feature config is missing keys: {missing}",
    "config_wrong_cohort": "Feature config targets {actual!r}, not {expected!r}",
    "derived_undeclared_columns": "Derived feature {feature} uses undeclared columns: {columns}",
    "duplicate_model_features": "Model feature names must be unique",
    "prohibited_columns": "Prohibited columns used by feature configuration: {columns}",
    "identifier_or_target_as_feature": "Identifier/target configured as features: {columns}",
    "unsourced_features": "Model features with no source column or derivation: {columns}",
    "missing_categorical_rule": "No normalisation rule or category list for {column!r}",
    "derived_not_table": "Derived feature {feature} must be a table with 'transform' and 'from'",
    "unsupported_transform": (
        "Unsupported transform for {feature}: {transform!r}; supported: {supported}"
    ),
    "derived_wrong_from": (
        "Derived feature {feature} ({transform}) needs 'from' to list exactly "
        "{count} column name(s); got {sources!r}"
    ),
    # Raw values (feature transformations and categorical normalisers)
    "non_numeric_values": "Non-numeric values in {column}: {examples}",
    "negative_values": "Negative values in {column}: {examples}",
    "unparseable_dates": "Unparseable dates in {column}: {examples}",
    "unrecognised_term": "Unrecognised term values: {examples}",
    "unrecognised_home_ownership": "Unrecognised home_ownership values: {examples}",
    "unsupported_categorical": "Unsupported categorical column: {column!r}; supported: {supported}",
    # Load parameters
    "test_split_locked": TEST_SPLIT_LOCKED_MESSAGE,
    "unknown_split": "Unknown split: {split!r}",
    "non_positive_chunksize": "chunksize must be positive",
    # Manifest and locked ID files
    "split_version_mismatch": "Feature config split version disagrees with split statistics",
    "missing_raw_rows": "split_statistics.json has no raw_rows count",
    "empty_split": "No rows found for {split}",
    "id_file_no_loan_id": "{filename} has no loan_id column",
    "id_file_missing_or_duplicate": "{filename} contains missing or duplicate loan_id values",
    "id_file_disagrees": (
        "{filename} disagrees with the manifest {split} split: "
        "{only_in_file} IDs only in the file, "
        "{only_in_manifest} only in the manifest"
    ),
    # Raw CSV read and join
    "raw_missing_columns": "Raw CSV is missing configured columns: {columns}",
    "raw_record_count": "Raw CSV record count disagrees with split statistics",
    "no_source_records": "No requested source records were found",
    "unmatched_manifest_rows": "{count} manifest rows have no source record, e.g. {sample}",
    "issue_month_alignment": (
        "Issue-month alignment failed; check the source version and one-based row numbering"
    ),
    "status_target_alignment": "Status/target alignment failed; do not use this source mapping",
    # Output
    "output_not_one_to_one": "Output rows do not match the manifest split one-to-one",
    "output_not_aligned": "Feature and target rows are not aligned on loan_id",
}


def load_feature_config(path: str | Path = DEFAULT_FEATURE_CONFIG) -> dict[str, Any]:
    """Load and validate a tabular feature config.

    Raises ValueError if a required section is missing, a model feature is
    prohibited, the identifier or target is listed as a feature, a feature has
    no source or derivation, a derived feature's ``from`` list is malformed or
    has the wrong number of columns, or a transform/categorical has no
    implementation. The prohibited check covers every column the loader
    reads for features: model features, source features, reference columns
    and derived-feature inputs.
    """
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)

    if missing := REQUIRED_FEATURE_CONFIG_KEYS - set(config):
        raise ValueError(ERROR_MESSAGES["config_missing_keys"].format(missing=sorted(missing)))
    if config["cohort_version"] != EXPECTED_COHORT:
        raise ValueError(ERROR_MESSAGES["config_wrong_cohort"].format(
            actual=config["cohort_version"], expected=EXPECTED_COHORT,
        ))

    source = list(config["source"]["features"])
    reference = list(config["source"].get("reference_columns", []))
    derived = config["derived"]
    features = feature_columns(config)
    prohibited = {column for group in config["prohibited"].values() for column in group}

    derived_inputs: set[str] = set()
    for name, spec in derived.items():
        _, inputs = _resolve_transformation(name, spec)
        if unknown := set(inputs) - set(source) - set(reference):
            raise ValueError(ERROR_MESSAGES["derived_undeclared_columns"].format(
                feature=name, columns=sorted(unknown),
            ))
        derived_inputs.update(inputs)

    if len(features) != len(set(features)):
        raise ValueError(ERROR_MESSAGES["duplicate_model_features"])
    if banned := (set(features) | set(source) | set(reference) | derived_inputs) & prohibited:
        raise ValueError(ERROR_MESSAGES["prohibited_columns"].format(columns=sorted(banned)))
    if leaked := {config["identifier_column"], config["target_column"]} & set(features):
        raise ValueError(
            ERROR_MESSAGES["identifier_or_target_as_feature"].format(columns=sorted(leaked))
        )
    if unsourced := set(features) - set(source) - set(derived):
        raise ValueError(ERROR_MESSAGES["unsourced_features"].format(columns=sorted(unsourced)))
    for column in config["model"]["categorical"]:
        if column not in _CATEGORICAL_NORMALISERS or column not in config["categories"]:
            raise ValueError(ERROR_MESSAGES["missing_categorical_rule"].format(column=column))
    return config


def feature_columns(config: dict[str, Any]) -> list[str]:
    """Return the model feature columns in their fixed order: numeric, then categorical."""
    return [*config["model"]["numeric"], *config["model"]["categorical"]]


def _blank_to_missing(values: pd.Series) -> pd.Series:
    """Strip whitespace and turn '' into <NA>; keep_default_na=False reads blanks as ''."""
    stripped = values.str.strip()
    return stripped.mask(stripped.eq(""))


def _examples(frame: pd.DataFrame, mask: pd.Series, column: str) -> str:
    """Format up to three offending loan_id=value pairs for an error message."""
    sample = frame.loc[mask, ["loan_id", column]].head(3)
    return ", ".join(f"{row.loan_id}={row[column]!r}" for _, row in sample.iterrows())


def _to_float(frame: pd.DataFrame, column: str) -> pd.Series:
    """Convert a string column to float64; missing stays NaN, non-numeric text raises."""
    values = frame[column]
    converted = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & converted.isna()
    if invalid.any():
        raise ValueError(ERROR_MESSAGES["non_numeric_values"].format(
            column=column, examples=_examples(frame, invalid, column),
        ))
    return converted.astype("float64")


def _as_object(values: pd.Series) -> pd.Series:
    """Return object dtype with np.nan for missing, which scikit-learn imputers accept."""
    return values.astype(object).where(values.notna(), np.nan)


def _normalise_term(frame: pd.DataFrame, allowed: list[str]) -> pd.Series:
    """Map ' 36 months' -> '36'; missing stays missing, unknown values raise."""
    values = frame["term"]
    parsed = values.str.extract(_TERM_PATTERN, expand=False)
    invalid = values.notna() & ~parsed.isin(allowed)
    if invalid.any():
        raise ValueError(ERROR_MESSAGES["unrecognised_term"].format(
            examples=_examples(frame, invalid, "term"),
        ))
    return _as_object(parsed)


def _normalise_home_ownership(
    frame: pd.DataFrame, allowed: list[str], to_other: list[str]
) -> pd.Series:
    """Collapse rare levels to OTHER; missing stays missing, unknown values raise."""
    values = frame["home_ownership"].str.upper()
    collapsed = values.mask(values.isin(to_other), "OTHER")
    invalid = values.notna() & ~collapsed.isin(allowed)
    if invalid.any():
        raise ValueError(ERROR_MESSAGES["unrecognised_home_ownership"].format(
            examples=_examples(frame, invalid, "home_ownership"),
        ))
    return _as_object(collapsed)


# Categorical column -> normaliser, called as fn(frame, config["categories"]).
# To support a new categorical feature, write its normaliser and register it here.
_CATEGORICAL_NORMALISERS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], pd.Series]] = {
    "term": lambda frame, categories: _normalise_term(frame, categories["term"]),
    "home_ownership": lambda frame, categories: _normalise_home_ownership(
        frame, categories["home_ownership"], categories.get("home_ownership_to_other", []),
    ),
}


def _normalise_categorical(
    frame: pd.DataFrame,
    column: str,
    categories: dict[str, Any],
) -> pd.Series:
    """Normalise one configured categorical column with its registered rule."""
    normaliser = _CATEGORICAL_NORMALISERS.get(column)
    if normaliser is None:
        raise ValueError(ERROR_MESSAGES["unsupported_categorical"].format(
            column=column, supported=sorted(_CATEGORICAL_NORMALISERS),
        ))
    return normaliser(frame, categories)


def _log1p(frame: pd.DataFrame, column: str) -> pd.Series:
    """log(1 + x); negative values raise because they have no financial meaning here."""
    values = _to_float(frame, column)
    negative = values < 0
    if negative.any():
        raise ValueError(ERROR_MESSAGES["negative_values"].format(
            column=column, examples=_examples(frame, negative, column),
        ))
    return np.log1p(values)


def _months_between(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Whole months from `start` to `end`, both 'Mon-YYYY'; missing start stays NaN."""
    start_dates = pd.to_datetime(frame[start], format=DATE_FORMAT, errors="coerce")
    invalid = frame[start].notna() & start_dates.isna()
    if invalid.any():
        raise ValueError(ERROR_MESSAGES["unparseable_dates"].format(
            column=start, examples=_examples(frame, invalid, start),
        ))
    end_dates = pd.to_datetime(frame[end], format=DATE_FORMAT)
    months = (
        (end_dates.dt.year - start_dates.dt.year) * 12
        + (end_dates.dt.month - start_dates.dt.month)
    )
    return months.astype("float64")


# Transform name -> (number of source columns, implementation). The
# implementation is called as fn(frame, *spec["from"]), so `from` order is the
# argument order. To add a transformation, write it above and register it here.
_TRANSFORMATIONS: dict[str, tuple[int, Callable[..., pd.Series]]] = {
    "log1p": (1, _log1p),                    # from = [column]
    "months_between": (2, _months_between),  # from = [start, end]
}


def _resolve_transformation(
    feature: str, spec: Any
) -> tuple[Callable[..., pd.Series], list[str]]:
    """Validate one [derived] entry and return its (implementation, source columns)."""
    if not isinstance(spec, dict):
        raise ValueError(ERROR_MESSAGES["derived_not_table"].format(feature=feature))
    transform = spec.get("transform")
    if transform not in _TRANSFORMATIONS:
        raise ValueError(ERROR_MESSAGES["unsupported_transform"].format(
            feature=feature, transform=transform, supported=sorted(_TRANSFORMATIONS),
        ))
    source_count, implementation = _TRANSFORMATIONS[transform]
    sources = spec.get("from")
    if (
        not isinstance(sources, list)
        or not all(isinstance(column, str) and column for column in sources)
        or len(sources) != source_count
    ):
        raise ValueError(ERROR_MESSAGES["derived_wrong_from"].format(
            feature=feature, transform=transform, count=source_count, sources=sources,
        ))
    return implementation, sources


def _apply_transformation(
    frame: pd.DataFrame,
    feature: str,
    spec: dict[str, Any],
) -> pd.Series:
    """Compute one derived feature from its configured transform and source columns."""
    implementation, sources = _resolve_transformation(feature, spec)
    return implementation(frame, *sources)


def _validate_split_ids(data_dir: Path, split: str, manifest_ids: set[str]) -> None:
    """Require the split's locked ID file to be duplicate-free and equal to the manifest split."""
    filename = ID_FILES[split]
    ids = pd.read_csv(data_dir / filename, dtype={"loan_id": "string"})
    if "loan_id" not in ids.columns:
        raise ValueError(ERROR_MESSAGES["id_file_no_loan_id"].format(filename=filename))
    if ids["loan_id"].isna().any() or ids["loan_id"].duplicated().any():
        raise ValueError(ERROR_MESSAGES["id_file_missing_or_duplicate"].format(filename=filename))
    file_ids = set(ids["loan_id"])
    if file_ids != manifest_ids:
        raise ValueError(ERROR_MESSAGES["id_file_disagrees"].format(
            filename=filename, split=split,
            only_in_file=len(file_ids - manifest_ids),
            only_in_manifest=len(manifest_ids - file_ids),
        ))


def _validate_load_params(split: str, chunksize: int) -> None:
    """Reject the locked test split, unknown splits and a non-positive chunksize."""
    if split == "test":
        raise ValueError(ERROR_MESSAGES["test_split_locked"])
    if split not in LOADABLE_SPLITS:
        raise ValueError(ERROR_MESSAGES["unknown_split"].format(split=split))
    if chunksize < 1:
        raise ValueError(ERROR_MESSAGES["non_positive_chunksize"])


def _load_split_manifest(
    data_dir: str | Path, split: str, config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load the validated manifest and return (split statistics, the split's rows)."""
    manifest, stats = load_manifest(data_dir)
    if config["split_version"] != stats.get("split_version"):
        raise ValueError(ERROR_MESSAGES["split_version_mismatch"])
    if "raw_rows" not in stats:
        raise ValueError(ERROR_MESSAGES["missing_raw_rows"])
    split_manifest = manifest.loc[manifest["split"].eq(split)].copy()
    if split_manifest.empty:
        raise ValueError(ERROR_MESSAGES["empty_split"].format(split=split))
    return stats, split_manifest


def _validate_header(raw_fields: list[str], header: set[str]) -> None:
    """Require every column the loader reads to be present in the raw CSV header."""
    if missing := [column for column in raw_fields if column not in header]:
        raise ValueError(ERROR_MESSAGES["raw_missing_columns"].format(columns=missing))


def _validate_read(
    stats: dict[str, Any], pieces: list[pd.DataFrame], records_seen: int
) -> None:
    """Require the full raw record count and at least one retained split record."""
    if records_seen != stats["raw_rows"]:
        raise ValueError(ERROR_MESSAGES["raw_record_count"])
    if not pieces:
        raise ValueError(ERROR_MESSAGES["no_source_records"])


def _merge_and_validate_source_rows(
    split_manifest: pd.DataFrame, pieces: list[pd.DataFrame]
) -> pd.DataFrame:
    """Join raw records to the split manifest and verify issue-month and status alignment."""
    rows = split_manifest.merge(
        pd.concat(pieces, ignore_index=True), on="source_row_number", how="left",
        validate="one_to_one", indicator=True, sort=False,
    )
    unmatched = ~rows["_merge"].eq("both")
    if unmatched.any():
        sample = rows.loc[unmatched, "loan_id"].head(5).tolist()
        raise ValueError(ERROR_MESSAGES["unmatched_manifest_rows"].format(
            count=int(unmatched.sum()), sample=sample,
        ))

    issue_dates = pd.to_datetime(rows["issue_d"], format=DATE_FORMAT, errors="coerce")
    months = issue_dates.dt.strftime("%Y-%m")
    if not months.eq(rows["issue_month"]).all():
        raise ValueError(ERROR_MESSAGES["issue_month_alignment"])
    observed_target = rows["loan_status"].map(STATUS_TO_TARGET)
    if not observed_target.eq(rows["target"]).all():
        raise ValueError(ERROR_MESSAGES["status_target_alignment"])
    # loan_status is the label's source field; drop it before any feature code runs.
    return rows.drop(columns=["_merge", "loan_status"])


def _build_features(
    rows: pd.DataFrame, config: dict[str, Any], index: pd.Index
) -> pd.DataFrame:
    """Apply the row-wise feature transformations; return them on `index`, sorted."""
    rows = rows.copy()
    for column in config["source"]["features"]:
        rows[column] = _blank_to_missing(rows[column])

    features = pd.DataFrame(index=rows.index)
    for column in config["model"]["numeric"]:
        spec = config["derived"].get(column)
        if spec is None:
            features[column] = _to_float(rows, column)
        else:
            features[column] = _apply_transformation(rows, column, spec)
    for column in config["model"]["categorical"]:
        features[column] = _normalise_categorical(rows, column, config["categories"])

    return features[feature_columns(config)].set_axis(index).sort_index()


def _build_target(rows: pd.DataFrame, config: dict[str, Any], index: pd.Index) -> pd.Series:
    """Return the int8 manifest target on `index`, sorted and named by the config."""
    target = (
        rows["target"].astype("int8").set_axis(index).sort_index()
        .rename(config["target_column"])
    )
    return target


def _validate_output(
    split_manifest: pd.DataFrame, features: pd.DataFrame, target: pd.Series
) -> None:
    """Require one unique row per manifest row, with features and target aligned."""
    if not features.index.is_unique or len(features) != len(split_manifest):
        raise ValueError(ERROR_MESSAGES["output_not_one_to_one"])
    if not features.index.equals(target.index):
        raise ValueError(ERROR_MESSAGES["output_not_aligned"])


def load_tabular_split(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    split: str,
    feature_config: str | Path = DEFAULT_FEATURE_CONFIG,
    *,
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load one development split as (features, target), both indexed by loan_id.

    Scans the raw CSV in chunks, numbering data records from one before any
    filtering, and keeps only the split's source rows. Cross-checks issue month
    and loan status against the manifest to catch a misaligned join, applies
    the configured row-wise transformations, and returns rows sorted by
    loan_id. Raises ValueError for the locked test split, an unknown split, a
    missing configured column, an ID mismatch or any unexpected raw value.
    """
    _validate_load_params(split, chunksize)
    data_dir = Path(data_dir)
    raw_csv_path = Path(raw_csv_path)
    config = load_feature_config(feature_config)

    stats, split_manifest = _load_split_manifest(data_dir, split, config)
    _validate_split_ids(data_dir, split, set(split_manifest["loan_id"]))

    source_columns = list(config["source"]["features"])
    raw_fields = list(dict.fromkeys([
        *source_columns, *config["source"].get("reference_columns", []), *ALIGNMENT_COLUMNS,
    ]))
    header = set(pd.read_csv(raw_csv_path, nrows=0).columns)
    _validate_header(raw_fields, header)

    wanted = set(split_manifest["source_row_number"])
    pieces: list[pd.DataFrame] = []
    records_seen = 0
    reader = pd.read_csv(
        raw_csv_path, usecols=raw_fields, dtype="string",
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
    _validate_read(stats, pieces, records_seen)
    rows = _merge_and_validate_source_rows(split_manifest, pieces)

    index = pd.Index(rows["loan_id"].astype(str), name=config["identifier_column"])
    features = _build_features(rows, config, index)
    target = _build_target(rows, config, index)
    _validate_output(split_manifest, features, target)
    return features, target


def profile_split(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Summarise a loaded split: row count, missingness, ranges and category counts."""
    profile: dict[str, Any] = {
        "rows": len(features),
        "positives": int(target.sum()),
        "missing": {column: int(features[column].isna().sum()) for column in features.columns},
        "numeric_range": {},
        "categories": {},
    }
    for column in features.columns:
        if pd.api.types.is_float_dtype(features[column]):
            profile["numeric_range"][column] = {
                "min": float(features[column].min()), "max": float(features[column].max()),
            }
        else:
            counts = features[column].value_counts(dropna=False)
            profile["categories"][column] = {
                str(level): int(count) for level, count in counts.items()
            }
    if "credit_history_months" in features.columns:
        negative = features["credit_history_months"] < 0
        profile["negative_credit_history_months"] = int(negative.sum())
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and profile one tabular development split.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--raw-csv", default=DEFAULT_RAW_CSV, type=Path)
    parser.add_argument("--split", default="train", choices=LOADABLE_SPLITS)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, type=Path)
    args = parser.parse_args()
    features, target = load_tabular_split(
        args.data_dir, args.raw_csv, args.split, args.feature_config
    )
    config = load_feature_config(args.feature_config)
    summary = {
        "feature_set_version": config["feature_set_version"],
        "split": args.split,
        "features": feature_columns(config),
        **profile_split(features, target),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Checked fusion tables (fusion-handoff-v1): join, check and record base-model scores per role.

Builds six tables, two versions x three scoring roles of fusion_development_v1:

- ``fusion-handoff-v1-two-score``: tabular + TF-IDF.
- ``fusion-handoff-v1-three-score``: tabular + TF-IDF + DistilBERT.

for ``fusion_fit`` (15,587 loans), ``fusion_calibration`` (11,353) and
``validation`` (21,784). ``base_fit`` gets no table: the base models were
fitted on it, so its scores would be in-sample. Within a role, both versions
hold exactly the same loans with identical tabular and TF-IDF values; the
three-score table only adds a column.

Contract of every table: one row per loan, sorted by loan_id, columns
``loan_id, split, role, p_default_<track>..., <track>_model_version...`` in the
track order of ``configs/fusion_handoff_v1.toml`` (tabular first). ``split`` is
the original split-v1 label and ``role`` the fusion role, both taken from the
pinned role mapping and never from a score file (the tabular calibration CSV
says ``split = "calibration"``; those loans are ``train``). Scores are the raw
base-model probabilities from committed files: this module computes no score.
The ``p_default_<track>`` columns are the stacker inputs; the version columns
are metadata, not model inputs. No target, outcome field or date is in any
table: labels come through ``evaluate.load_manifest`` on the reader's side.

Every source is pinned by SHA-256 in the config (NLP pins must also equal the
hash in their owner's manifest) and checked in full before use. Scores are
left-joined onto the mapping's rows by loan_id: a missing or duplicate score
stops the run and is never filled or dropped. All six frames are built and
checked before anything is written. This module reads the role mapping, the
score files, the owner manifests, ``split_statistics.json`` and
``test_ids.csv`` (an ID-set check); never the raw CSV, ``split_manifest.csv``
or a target.

Writes data/fusion/fusion_v1_{two,three}_score_{role}.parquet and
reports/tabular/fusion_handoff_v1_{two,three}_score_manifest.json.

Run from the repository root (no raw CSV needed):
    python -m src.tabular.fusion_handoff_v1

The v0 handoff (``src/tabular/fusion_handoff.py``, ``fusion_*_v0.parquet``) is
unchanged; its helpers ``read_score_file``, ``load_score_file``,
``join_scores`` and ``parquet_writer_metadata`` are imported, not copied.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from src.nlp.preprocess import sha256_file
from src.tabular import evaluate, provenance
from src.tabular.fusion_handoff import (
    OUTCOME_COLUMNS,
    join_scores,
    load_score_file,
    parquet_writer_metadata,
    read_score_file,
)
from src.tabular.fusion_roles import SCORING_ROLES, _test_ids_in
from src.tabular.preprocess import DEFAULT_DATA_DIR, REPO_ROOT
from src.tabular.xgboost_baseline import validate_runtime

HANDOFF_VERSION = "fusion-handoff-v1"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fusion_handoff_v1.toml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "fusion"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "reports" / "tabular"

TABULAR = "tabular"
NLP_COLUMNS = ("loan_id", "split", "role", "raw_margin", "raw_prob", "model_version")
NLP_PROBABILITY_COLUMN = "raw_prob"
# Exact agreement expected: the exports wrote raw_prob = sigmoid(raw_margin) (planning: <= 2e-16).
SIGMOID_TOLERANCE = 1e-12
# Never in a table: the target and D-016 outcome fields, plus dates (in the mapping, not stacker inputs).
BANNED_COLUMNS = OUTCOME_COLUMNS | {"issue_month", "issue_d"}
SHA256_PATTERN = re.compile(rf"^[0-9a-f]{{{provenance.SHA256_HEX_LENGTH}}}$")

REQUIRED_CONFIG_KEYS = {"handoff_version", "status", "fusion_development_version", "role_mapping", "sources", "tables"}
REQUIRED_KEYS = {
    "role_mapping": {"path", "sha256"},
    "tabular": {"model_name", "model_version", "probability_column", "files"},
    "tabular_file": {"path", "sha256", "file_split", "roles"},
    "nlp": {"path", "sha256", "model_name", "model_version", "owner_manifest", "owner_manifest_sha256_key"},
    "table": {"version", "tracks"},
}

# Every ValueError message raised by this module, as str.format templates.
ERROR_MESSAGES: dict[str, str] = {
    # Config (load_handoff_config)
    "config_missing_keys": "Handoff config is missing keys: {missing}",
    "section_missing_keys": "Handoff config [{section}] is missing keys: {missing}",
    "wrong_handoff_version": "Handoff config is for {actual!r}; this module builds {expected!r}",
    "bad_sha256": "Handoff config [{section}] sha256 must be 64 lowercase hex characters; got {value!r}",
    "no_tabular_source": "Handoff config must define [sources.tabular]",
    "no_tabular_files": "[sources.tabular] must list at least one [[sources.tabular.files]] entry",
    "test_split_locked": evaluate.TEST_SPLIT_LOCKED_MESSAGE,
    "tabular_roles_coverage": (
        "[[sources.tabular.files]] roles must cover {expected} exactly once each; got {actual}"
    ),
    "bad_owner_key": "[sources.{track}] owner_manifest_sha256_key must be a non-empty list of strings",
    "no_tables": "Handoff config must define at least one [tables.*] entry",
    "empty_tracks": "[tables.{table}] tracks must be a non-empty list",
    "duplicate_tracks": "[tables.{table}] tracks lists a track more than once: {tracks}",
    "unknown_track": "[tables.{table}] names tracks with no [sources.*] entry: {unknown}",
    "tabular_not_first": "[tables.{table}] tracks must start with 'tabular'; got {tracks}",
    "duplicate_table_versions": "[tables.*] versions must be distinct; got {versions}",
    # Sources (load_role_mapping, load_tabular_scores, load_nlp_scores_v2)
    "source_failed": "{source} ({path}): {error}",
    "tabular_overlap": "Tabular score files overlap on {count} loans",
    "tabular_coverage": "Tabular score files miss {missing} scoring loans and have {extra} unexpected ones",
    "nlp_columns": "Columns must be {expected} in that order; got {actual}",
    "owner_manifest_key": "Owner manifest has no value at key path {key}",
    "owner_manifest_disagrees": "Pinned sha256 {pinned} differs from the owner manifest's {recorded}",
    "nlp_mapping_mismatch": "{count} loans disagree with the role mapping on {column}",
    "not_numeric": "{column} must be numeric; got {dtype}",
    "not_finite": "{column} must be finite on every row; {count} are not",
    "sigmoid_mismatch": (
        "raw_prob differs from sigmoid(raw_margin) by more than {tolerance} on {count} rows (max {max_diff:.3e})"
    ),
    "nlp_version": "model_version must be {expected!r} on every row; got {actual}",
    # Tables (build_handoff_tables)
    "table_failed": "{table} {role}: {error}",
    "banned_columns": "Must not contain target, outcome or date columns {banned}",
    "table_columns": "Must have columns {expected}; got {actual}",
    "table_ids": "Must hold exactly the mapping's {role} loans: {missing} missing, {extra} extra",
    "table_not_sorted": "Rows must be sorted by unique loan_id",
    "table_labels": "{column} must equal the role mapping's on every row; {count} rows differ",
    "score_dtype": "{column} must be float64; got {dtype}",
    "table_version": "{column} must be {expected!r} on every row; got {actual}",
    "versions_disagree": (
        "{table} and {reference} {role} tables differ in {column}; versions of one role must agree exactly"
    ),
    "test_ids_in_tables": "The tables contain {count} locked test IDs, e.g. {sample}",
}


# --------------------------------------------------------------------------- config


def _require_keys(section: str, values: dict[str, Any], required: set[str]) -> None:
    """Raise naming the config section when any required key is absent."""
    if missing := required - set(values):
        raise ValueError(ERROR_MESSAGES["section_missing_keys"].format(section=section, missing=sorted(missing)))


def _require_sha256(section: str, value: Any) -> None:
    """Pins must be full lowercase SHA-256 hex digests, so a typo cannot pass as a pin."""
    if not isinstance(value, str) or not SHA256_PATTERN.match(value):
        raise ValueError(ERROR_MESSAGES["bad_sha256"].format(section=section, value=value))


def nlp_tracks(config: dict[str, Any]) -> list[str]:
    """Every source track except tabular, in config order."""
    return [track for track in config["sources"] if track != TABULAR]


def load_handoff_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the handoff config.

    Raises ValueError for a missing key in any section, a handoff_version other
    than this module's, a malformed sha256, no tabular source or files, a
    tabular file on the test split, tabular roles that do not cover the three
    scoring roles exactly once, a malformed owner-manifest key path, a table
    with no tracks, a repeated or unknown track, a table that does not start
    with tabular, or two tables with the same version.
    """
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)

    if missing := REQUIRED_CONFIG_KEYS - set(config):
        raise ValueError(ERROR_MESSAGES["config_missing_keys"].format(missing=sorted(missing)))
    if config["handoff_version"] != HANDOFF_VERSION:
        raise ValueError(ERROR_MESSAGES["wrong_handoff_version"].format(
            actual=config["handoff_version"], expected=HANDOFF_VERSION,
        ))
    _require_keys("role_mapping", config["role_mapping"], REQUIRED_KEYS["role_mapping"])
    _require_sha256("role_mapping", config["role_mapping"]["sha256"])

    sources = config["sources"]
    if TABULAR not in sources:
        raise ValueError(ERROR_MESSAGES["no_tabular_source"])
    tabular = sources[TABULAR]
    _require_keys("sources.tabular", tabular, REQUIRED_KEYS["tabular"])
    if not tabular["files"]:
        raise ValueError(ERROR_MESSAGES["no_tabular_files"])
    for index, entry in enumerate(tabular["files"]):
        section = f"sources.tabular.files[{index}]"
        _require_keys(section, entry, REQUIRED_KEYS["tabular_file"])
        _require_sha256(section, entry["sha256"])
        if entry["file_split"] == "test":
            raise ValueError(ERROR_MESSAGES["test_split_locked"])
    role_counts = Counter(role for entry in tabular["files"] for role in entry["roles"])
    if role_counts != Counter(SCORING_ROLES):
        raise ValueError(ERROR_MESSAGES["tabular_roles_coverage"].format(
            expected=list(SCORING_ROLES), actual=dict(sorted(role_counts.items())),
        ))
    for track in nlp_tracks(config):
        source = sources[track]
        _require_keys(f"sources.{track}", source, REQUIRED_KEYS["nlp"])
        _require_sha256(f"sources.{track}", source["sha256"])
        key = source["owner_manifest_sha256_key"]
        if not isinstance(key, list) or not key or not all(isinstance(part, str) for part in key):
            raise ValueError(ERROR_MESSAGES["bad_owner_key"].format(track=track))

    if not config["tables"]:
        raise ValueError(ERROR_MESSAGES["no_tables"])
    for table, spec in config["tables"].items():
        _require_keys(f"tables.{table}", spec, REQUIRED_KEYS["table"])
        tracks = spec["tracks"]
        if not isinstance(tracks, list) or not tracks:
            raise ValueError(ERROR_MESSAGES["empty_tracks"].format(table=table))
        if len(set(tracks)) != len(tracks):
            raise ValueError(ERROR_MESSAGES["duplicate_tracks"].format(table=table, tracks=tracks))
        if unknown := [track for track in tracks if track not in sources]:
            raise ValueError(ERROR_MESSAGES["unknown_track"].format(table=table, unknown=unknown))
        if tracks[0] != TABULAR:
            raise ValueError(ERROR_MESSAGES["tabular_not_first"].format(table=table, tracks=tracks))
    versions = [spec["version"] for spec in config["tables"].values()]
    if len(set(versions)) != len(versions):
        raise ValueError(ERROR_MESSAGES["duplicate_table_versions"].format(versions=versions))
    return config


def score_column(track: str) -> str:
    """The stacker input column for one track (same name as the stacker's ``score_column``)."""
    return f"p_default_{track}"


def version_column(track: str) -> str:
    """The model-version metadata column for one track."""
    return f"{track}_model_version"


def table_columns(tracks: list[str]) -> list[str]:
    """Output columns in order: keys and labels, scores in track order, then versions in track order."""
    return ["loan_id", "split", "role", *map(score_column, tracks), *map(version_column, tracks)]


# --------------------------------------------------------------------------- sources


def _role_ids(mapping: pd.DataFrame, roles: list[str] | tuple[str, ...]) -> set[str]:
    """loan_ids of the mapping rows in the given roles."""
    return set(mapping.loc[mapping["role"].isin(roles), "loan_id"])


def load_role_mapping(config: dict[str, Any], *, root: Path = REPO_ROOT) -> pd.DataFrame:
    """The pinned mapping, scoring roles only: loan_id, split, role (issue_month dropped), sorted by loan_id.

    Only the checksum is verified. A file with the pinned SHA-256 is the
    committed mapping, whose columns, unique IDs and role/split values
    test_fusion_roles.py already checks (the committed file equals a fresh
    assignment), so they are not rechecked here. Raises ValueError for a
    checksum mismatch.
    """
    source = config["role_mapping"]
    path = root / source["path"]
    try:
        provenance.verify_file_checksum(path, source["sha256"], allow_crlf=True)
    except ValueError as error:
        raise ValueError(ERROR_MESSAGES["source_failed"].format(
            source="role_mapping", path=source["path"], error=error,
        )) from error
    mapping = pd.read_csv(path, dtype=str)
    scoring = mapping.loc[mapping["role"].isin(SCORING_ROLES), ["loan_id", "split", "role"]]
    return scoring.sort_values("loan_id", ignore_index=True)


def load_tabular_scores(config: dict[str, Any], mapping: pd.DataFrame, *, root: Path = REPO_ROOT) -> pd.DataFrame:
    """loan_id, p_default_tabular, tabular_model_version for exactly the mapping's scoring loans.

    Each file is checksum-verified, then checked by ``fusion_handoff.load_score_file``
    against the mapping IDs of its roles. The files' own ``split`` is checked
    (e.g. ``calibration``) and then dropped: output labels come from the mapping.
    """
    source = config["sources"][TABULAR]
    parts = []
    for entry in source["files"]:
        path = root / entry["path"]
        try:
            provenance.verify_file_checksum(path, entry["sha256"], allow_crlf=True)
            parts.append(load_score_file(
                path,
                probability_column=source["probability_column"],
                split=entry["file_split"],
                expected_ids=_role_ids(mapping, entry["roles"]),
                model_name=source["model_name"],
                allowed_versions=frozenset({source["model_version"]}),
            ))
        except ValueError as error:
            raise ValueError(ERROR_MESSAGES["source_failed"].format(
                source=TABULAR, path=entry["path"], error=error,
            )) from error

    scores = pd.concat(parts, ignore_index=True)
    if count := int(scores["loan_id"].duplicated().sum()):
        raise ValueError(ERROR_MESSAGES["tabular_overlap"].format(count=count))
    expected, actual = set(mapping["loan_id"]), set(scores["loan_id"])
    if expected != actual:
        raise ValueError(ERROR_MESSAGES["tabular_coverage"].format(
            missing=len(expected - actual), extra=len(actual - expected),
        ))
    return pd.DataFrame({
        "loan_id": scores["loan_id"],
        score_column(TABULAR): scores[source["probability_column"]],
        version_column(TABULAR): scores["model_version"].astype(str),
    }).sort_values("loan_id", ignore_index=True)


def owner_manifest_sha256(source: dict[str, Any], *, root: Path = REPO_ROOT) -> str:
    """The score file's SHA-256 as recorded in its owner's manifest, at the configured key path."""
    value: Any = json.loads((root / source["owner_manifest"]).read_text(encoding="utf-8"))
    for key in source["owner_manifest_sha256_key"]:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(ERROR_MESSAGES["owner_manifest_key"].format(key=source["owner_manifest_sha256_key"]))
        value = value[key]
    return value


def _check_nlp_frame(frame: pd.DataFrame, track: str, source: dict[str, Any], mapping: pd.DataFrame) -> None:
    """The v2 six-column contract: shared prediction checks, agreement with the mapping, finite
    margins, sigmoid(raw_margin) = raw_prob, and the single pinned model_version."""
    if frame.columns.tolist() != list(NLP_COLUMNS):
        raise ValueError(ERROR_MESSAGES["nlp_columns"].format(expected=list(NLP_COLUMNS), actual=frame.columns.tolist()))

    # The v2 files carry no model_name, so the shared validator sees a five-column view with the
    # config's constant: duplicates, exact coverage, NaN, numeric and [0, 1].
    score = score_column(track)
    view = pd.DataFrame({
        "loan_id": frame["loan_id"],
        "split": frame["split"],
        score: frame[NLP_PROBABILITY_COLUMN],
        "model_name": source["model_name"],
        "model_version": frame["model_version"],
    })
    evaluate.validate_prediction_frame(view, set(mapping["loan_id"]), probability_column=score)

    # Aligned on loan_id, never on row order.
    aligned = frame[["loan_id", "split", "role"]].merge(
        mapping, on="loan_id", how="left", suffixes=("", "_mapping"), validate="one_to_one",
    )
    for column in ("split", "role"):
        if count := int(aligned[column].astype(str).ne(aligned[f"{column}_mapping"].astype(str)).sum()):
            raise ValueError(ERROR_MESSAGES["nlp_mapping_mismatch"].format(count=count, column=column))

    if not pd.api.types.is_numeric_dtype(frame["raw_margin"]):
        raise ValueError(ERROR_MESSAGES["not_numeric"].format(column="raw_margin", dtype=frame["raw_margin"].dtype))
    margin = frame["raw_margin"].to_numpy(dtype=np.float64)
    if count := int((~np.isfinite(margin)).sum()):
        raise ValueError(ERROR_MESSAGES["not_finite"].format(column="raw_margin", count=count))
    difference = np.abs(expit(margin) - frame[NLP_PROBABILITY_COLUMN].to_numpy(dtype=np.float64))
    if count := int((difference > SIGMOID_TOLERANCE).sum()):
        raise ValueError(ERROR_MESSAGES["sigmoid_mismatch"].format(
            tolerance=SIGMOID_TOLERANCE, count=count, max_diff=float(difference.max()),
        ))

    versions = sorted(frame["model_version"].astype(str).unique().tolist())
    if versions != [source["model_version"]]:
        raise ValueError(ERROR_MESSAGES["nlp_version"].format(expected=source["model_version"], actual=versions))


def load_nlp_scores_v2(
    track: str, config: dict[str, Any], mapping: pd.DataFrame, *, root: Path = REPO_ROOT
) -> pd.DataFrame:
    """loan_id, p_default_<track>, <track>_model_version for exactly the mapping's scoring loans.

    Raises ValueError naming the track and file for a checksum mismatch, a pin
    that differs from the owner manifest's hash, or any failed v2 check. Never
    fills, drops or filters a row.
    """
    source = config["sources"][track]
    path = root / source["path"]
    try:
        provenance.verify_file_checksum(path, source["sha256"])
        recorded = owner_manifest_sha256(source, root=root)
        if recorded != source["sha256"]:
            raise ValueError(ERROR_MESSAGES["owner_manifest_disagrees"].format(
                pinned=source["sha256"], recorded=recorded,
            ))
        frame = read_score_file(path)
        _check_nlp_frame(frame, track, source, mapping)
    except ValueError as error:
        raise ValueError(ERROR_MESSAGES["source_failed"].format(source=track, path=source["path"], error=error)) from error
    return pd.DataFrame({
        "loan_id": frame["loan_id"],
        score_column(track): frame[NLP_PROBABILITY_COLUMN].astype(np.float64),
        version_column(track): frame["model_version"].astype(str),
    }).sort_values("loan_id", ignore_index=True)


def load_all_scores(config: dict[str, Any], mapping: pd.DataFrame, *, root: Path = REPO_ROOT) -> dict[str, pd.DataFrame]:
    """Every configured track's checked scores, keyed by track."""
    scores = {TABULAR: load_tabular_scores(config, mapping, root=root)}
    for track in nlp_tracks(config):
        scores[track] = load_nlp_scores_v2(track, config, mapping, root=root)
    return scores


# --------------------------------------------------------------------------- tables


def _check_table(
    frame: pd.DataFrame, base: pd.DataFrame, tracks: list[str], config: dict[str, Any]
) -> None:
    """Exact columns and loans, labels equal to the mapping, float64 finite scores in [0, 1],
    and each version column equal to its pin."""
    if banned := sorted(set(frame.columns) & BANNED_COLUMNS):
        raise ValueError(ERROR_MESSAGES["banned_columns"].format(banned=banned))
    expected_columns = table_columns(tracks)
    if frame.columns.tolist() != expected_columns:
        raise ValueError(ERROR_MESSAGES["table_columns"].format(expected=expected_columns, actual=frame.columns.tolist()))
    expected, actual = set(base["loan_id"]), set(frame["loan_id"])
    if expected != actual:
        raise ValueError(ERROR_MESSAGES["table_ids"].format(
            role=base["role"].iloc[0], missing=len(expected - actual), extra=len(actual - expected),
        ))
    if not (frame["loan_id"].is_unique and frame["loan_id"].is_monotonic_increasing):
        raise ValueError(ERROR_MESSAGES["table_not_sorted"])
    for column in ("split", "role"):
        if count := int(frame[column].ne(base[column]).sum()):
            raise ValueError(ERROR_MESSAGES["table_labels"].format(column=column, count=count))

    for track in tracks:
        score, version = score_column(track), version_column(track)
        if frame[score].dtype != np.float64:
            raise ValueError(ERROR_MESSAGES["score_dtype"].format(column=score, dtype=frame[score].dtype))
        if count := int((~np.isfinite(frame[score].to_numpy())).sum()):
            raise ValueError(ERROR_MESSAGES["not_finite"].format(column=score, count=count))
        evaluate.validate_prediction_frame(
            frame[["loan_id", "split", score]].assign(
                model_name=track, model_version=frame[version],
            ),
            expected,
            probability_column=score,
        )
        pinned = config["sources"][track]["model_version"]
        if (versions := sorted(frame[version].unique().tolist())) != [pinned]:
            raise ValueError(ERROR_MESSAGES["table_version"].format(column=version, expected=pinned, actual=versions))


def _check_versions_agree(tables: dict[str, dict[str, pd.DataFrame]], config: dict[str, Any]) -> None:
    """Within a role, every table version has the same loan_id column and identical shared scores.

    Exact equality, not a tolerance: the shared columns come from the same join.
    """
    reference, *others = config["tables"]
    for table in others:
        shared = [t for t in config["tables"][table]["tracks"] if t in config["tables"][reference]["tracks"]]
        for role in SCORING_ROLES:
            left, right = tables[reference][role], tables[table][role]
            for column in ["loan_id", *map(score_column, shared), *map(version_column, shared)]:
                if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
                    raise ValueError(ERROR_MESSAGES["versions_disagree"].format(
                        table=table, reference=reference, role=role, column=column,
                    ))


def build_handoff_tables(
    config: dict[str, Any], mapping: pd.DataFrame, scores: dict[str, pd.DataFrame]
) -> dict[str, dict[str, pd.DataFrame]]:
    """{"two_score": {role: frame}, "three_score": {role: frame}} for the three scoring roles.

    Each frame starts from the mapping's rows for the role and left-joins each
    track's scores with ``fusion_handoff.join_scores`` (one-to-one, a missing
    score raises, never an inner join). Raises ValueError naming the table and
    role for any failed check, or if the versions of one role disagree.
    """
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for table, spec in config["tables"].items():
        tracks = spec["tracks"]
        tables[table] = {}
        for role in SCORING_ROLES:
            base = mapping.loc[mapping["role"].eq(role), ["loan_id", "split", "role"]].sort_values(
                "loan_id", ignore_index=True,
            )
            try:
                joined = join_scores(base, {track: scores[track] for track in tracks})
                frame = joined[table_columns(tracks)]
                _check_table(frame, base, tracks, config)
            except ValueError as error:
                raise ValueError(ERROR_MESSAGES["table_failed"].format(table=table, role=role, error=error)) from error
            tables[table][role] = frame
    _check_versions_agree(tables, config)
    return tables


def output_paths(config: dict[str, Any], output_dir: Path, manifest_dir: Path) -> dict[str, dict[str, Any]]:
    """Per table: its role parquet paths and its manifest path."""
    return {
        table: {
            "tables": {role: output_dir / f"fusion_v1_{table}_{role}.parquet" for role in SCORING_ROLES},
            "manifest": manifest_dir / f"fusion_handoff_v1_{table}_manifest.json",
        }
        for table in config["tables"]
    }


# --------------------------------------------------------------------------- manifests


def _source_record(track: str, config: dict[str, Any], frame: pd.DataFrame, root: Path) -> dict[str, Any]:
    """Pinned and observed checksums, version, source column and row count of one track."""
    source = config["sources"][track]
    common = {
        "model_name": source["model_name"],
        "model_version": source["model_version"],
        "output_column": score_column(track),
        "rows": len(frame),
    }
    if track == TABULAR:
        return {
            **common,
            "source_probability_column": source["probability_column"],
            "files": [
                {
                    "path": entry["path"],
                    "pinned_sha256": entry["sha256"],
                    "observed_sha256": sha256_file(root / entry["path"]),
                    "checksum_rule": "LF bytes; CRLF working copy accepted (allow_crlf=True)",
                    "file_split": entry["file_split"],
                    "file_split_used_in_output": False,
                    "roles": entry["roles"],
                }
                for entry in source["files"]
            ],
        }
    return {
        **common,
        "source_probability_column": NLP_PROBABILITY_COLUMN,
        "path": source["path"],
        "pinned_sha256": source["sha256"],
        "observed_sha256": sha256_file(root / source["path"]),
        "owner_manifest": source["owner_manifest"],
        "owner_manifest_sha256": owner_manifest_sha256(source, root=root),
        "sigmoid_check_tolerance": SIGMOID_TOLERANCE,
    }


def _output_record(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Path, checksum, rows, columns and parquet writer of one written table."""
    return {
        "path": provenance.display_path(path),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "parquet_writer": parquet_writer_metadata(path),
    }


def build_table_manifest(
    table: str,
    config: dict[str, Any],
    config_path: Path,
    stats: dict[str, Any],
    scores: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    test_rows: int,
    runtime: dict[str, str],
    root: Path,
) -> dict[str, Any]:
    """One table version's manifest, readable on its own."""
    tracks = config["tables"][table]["tracks"]
    stacker_inputs = [score_column(track) for track in tracks]
    mapping = config["role_mapping"]
    return {
        "table_version": config["tables"][table]["version"],
        "handoff_version": config["handoff_version"],
        "status": config["status"],
        "fusion_development_version": config["fusion_development_version"],
        "cohort": stats["cohort"],
        "split_version": stats["split_version"],
        "dataset_version": stats["dataset_version"],
        "config": {"path": provenance.display_path(config_path), "sha256": sha256_file(config_path)},
        "role_mapping": {
            "path": mapping["path"],
            "pinned_sha256": mapping["sha256"],
            "observed_sha256": sha256_file(root / mapping["path"]),
        },
        "tracks": tracks,
        "sources": {track: _source_record(track, config, scores[track], root) for track in tracks},
        "outputs": {role: _output_record(paths[role], frames[role]) for role in SCORING_ROLES},
        "stacker_inputs": stacker_inputs,
        "metadata_columns": {
            "columns": [column for column in table_columns(tracks) if column not in stacker_inputs],
            "note": "Metadata only, not model inputs. split and role come from the role mapping.",
        },
        "score_semantics": (
            "Raw base-model probabilities for a stacker, not credit PDs. The tabular score is inflated by "
            "balanced class weighting (mean about 0.48 against a 15% default rate); the NLP scores are "
            "unweighted and uncalibrated. Calibrate only the stacked output."
        ),
        "targets_included": False,
        "test_rows_in_tables": test_rows,
        "test_rows_measured_against": "data/test_ids.csv (ID-set check across all six tables)",
        "commands": [
            "python -m pip install -r src/tabular/requirements.txt",
            "python -m src.tabular.fusion_handoff_v1",
            "python -m pytest src/tabular/test_fusion_handoff_v1.py -v -s",
        ],
        "environment": runtime,
    }


# --------------------------------------------------------------------------- run


def run_handoff_v1(
    config_path: str | Path = DEFAULT_CONFIG,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Load, build and check all six tables, then write them and the two manifests.

    Every frame is built and checked, and the test-ID check passes, before the
    first file is written, so a failed check leaves no partial handoff.
    Returns the manifests keyed by table.
    """
    runtime = validate_runtime()
    config_path, data_dir = Path(config_path), Path(data_dir)
    config = load_handoff_config(config_path)
    stats = json.loads((data_dir / provenance.SPLIT_STATISTICS).read_text(encoding="utf-8"))
    mapping = load_role_mapping(config, root=root)
    scores = load_all_scores(config, mapping, root=root)
    tables = build_handoff_tables(config, mapping, scores)

    all_ids = pd.DataFrame({"loan_id": pd.concat([f["loan_id"] for t in tables.values() for f in t.values()])})
    if leaked := _test_ids_in(all_ids, data_dir):
        raise ValueError(ERROR_MESSAGES["test_ids_in_tables"].format(count=len(leaked), sample=leaked[:5]))

    paths = output_paths(config, Path(output_dir), Path(manifest_dir))
    for table, frames in tables.items():
        for role, frame in frames.items():
            path = paths[table]["tables"][role]
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False, engine="pyarrow")

    manifests = {}
    for table, frames in tables.items():
        manifest = build_table_manifest(
            table, config, config_path, stats, scores, frames, paths[table]["tables"], len(leaked), runtime, root,
        )
        manifest_path = paths[table]["manifest"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        manifests[table] = manifest
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the checked fusion tables and their manifests.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR, type=Path)
    args = parser.parse_args()
    manifests = run_handoff_v1(args.config, args.data_dir, output_dir=args.output_dir, manifest_dir=args.manifest_dir)
    for table, manifest in manifests.items():
        print(f"{manifest['table_version']}: stacker inputs {manifest['stacker_inputs']}")
        for role, output in manifest["outputs"].items():
            print(f"  {role}: rows={output['rows']} -> {output['path']}")
        print(f"  test_rows_in_tables={manifest['test_rows_in_tables']}")


if __name__ == "__main__":
    main()

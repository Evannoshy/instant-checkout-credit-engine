"""Late-fusion inputs (fusion-stacker-v1): the stacker config, checked score tables and labels.

``build_role_tables(config, tracks)`` returns one frame per scoring role
(``fusion_fit``, ``fusion_calibration``, ``validation``) with columns
``loan_id, target, p_default_<track>...``, sorted by loan_id. Roles come from
Brandon's shared mapping ``data/fusion/fusion_roles_v1.csv`` (pinned by
SHA-256); labels come from the approved manifest loader
``evaluate.load_manifest``, which removes test rows before returning.

Every source must cover exactly the 48,724 scoring loans. A missing match, a
duplicate ID, a wrong split or role, a non-finite or out-of-range score or an
unpinned model version raises ValueError. Nothing is filled, dropped or
filtered: the joins reuse ``fusion_handoff.join_scores``, which refuses a
missing score.

Two source formats (``[sources.<track>] format``):
- ``tabular_raw_predictions``: the existing raw LR files. The ``calibration``
  file covers fusion_fit + fusion_calibration (tabular_development_v1's
  calibration role = those two fusion roles, checked by test_fusion_roles.py).
- ``nlp_scores_v2``: the shared six-column NLP export. Its ``split`` and
  ``role`` must equal the mapping's on every row.

Only score columns reach the stacker: IDs, roles, dates and version metadata
stay out of the tables' feature columns (``score_columns``).

Known limitations:
- The calibrated tabular reference's Platt mapping was fitted on all 26,940
  Jan-Jul 2013 loans, fusion_fit included. It is a benchmark on validation
  only, not a held-out result on the fusion rows.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tabular import evaluate, provenance
from src.tabular.fusion_handoff import join_scores, load_score_file, read_score_file
from src.tabular.fusion_roles import MAPPING_COLUMNS, SCORING_ROLES
from src.tabular.preprocess import DEFAULT_DATA_DIR, REPO_ROOT

DEFAULT_STACKER_CONFIG = REPO_ROOT / "configs" / "fusion_stacker_v1.toml"

SOURCE_FORMATS = ("tabular_raw_predictions", "nlp_scores_v2")
NLP_SCORE_COLUMNS = ("loan_id", "split", "role", "raw_margin", "raw_prob", "model_version")
# The tabular calibration file's split label for fusion_fit + fusion_calibration loans.
TABULAR_FUSION_SPLIT = "calibration"
TABULAR_FUSION_ROLES = ("fusion_fit", "fusion_calibration")

REQUIRED_CONFIG_KEYS = {
    "stacker_version", "status", "fusion_development_version", "role_mapping", "sources",
    "reference", "variants", "run", "stacker", "calibration", "evaluation",
}

# Every ValueError message raised by this module, as str.format templates.
ERROR_MESSAGES: dict[str, str] = {
    "config_missing_keys": "Stacker config is missing keys: {missing}",
    "bad_setting": "Stacker config {key} must be {requirement}; got {value!r}",
    "unknown_variant": "Run variant {variant!r} is not defined in [variants] {known}",
    "track_without_source": (
        "Variant {variant!r} needs track {track!r}, which has no [sources.{track}] entry yet"
    ),
    "unknown_format": "Source {track} format must be one of {known}; got {value!r}",
    "mapping_columns": "Role mapping must have columns {expected}; got {actual}",
    "mapping_duplicates": "Role mapping has {count} duplicate loan_ids",
    "nlp_columns": "{track} scores must have columns {expected}; got {actual}",
    "nlp_duplicates": "{track} scores have {count} duplicate loan_ids",
    "nlp_coverage": "{track} scores miss {missing} expected loans and have {extra} unexpected ones",
    "nlp_mismatch": "{track} scores disagree with the role mapping on {column} for {count} loans",
    "not_finite": "{track} {column} must be finite on every row; {count} are not",
    "out_of_range": "{track} raw_prob must lie within [0, 1]; {count} rows do not",
    "nlp_version": "{track} model_version must be {expected!r} on every row; got {actual}",
    "label_missing": "{count} {role} loans have no label in the manifest",
    "reference_ids": "Reference rows are not exactly the validation loans",
    "reference_version": "Reference must be {expected!r}; got {actual}",
}


def _require(condition: bool, key: str, requirement: str, value: Any) -> None:
    if not condition:
        raise ValueError(ERROR_MESSAGES["bad_setting"].format(key=key, requirement=requirement, value=value))


def load_stacker_config(path: str | Path = DEFAULT_STACKER_CONFIG) -> dict[str, Any]:
    """Load and validate the stacker config.

    Raises ValueError for missing keys, a weighted or non-probability stacker,
    a calibrator other than Platt, a run variant that is not defined, or a run
    variant that needs a track with no pinned source. Pending tracks may be
    named in [variants] as long as they are not run.
    """
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if missing := REQUIRED_CONFIG_KEYS - set(config):
        raise ValueError(ERROR_MESSAGES["config_missing_keys"].format(missing=sorted(missing)))

    stacker = config["stacker"]
    _require(stacker.get("class_weight") == "none", "stacker.class_weight", '"none"', stacker.get("class_weight"))
    _require(stacker.get("input_space") == "probability", "stacker.input_space", '"probability"',
             stacker.get("input_space"))
    _require(isinstance(stacker.get("C"), (int, float)) and stacker["C"] > 0, "stacker.C", "a positive number",
             stacker.get("C"))
    _require(config["calibration"].get("method") == "platt", "calibration.method", '"platt"',
             config["calibration"].get("method"))
    resamples = config["evaluation"].get("bootstrap_resamples")
    _require(isinstance(resamples, int) and resamples > 0, "evaluation.bootstrap_resamples",
             "a positive integer", resamples)

    for track, source in config["sources"].items():
        if source.get("format") not in SOURCE_FORMATS:
            raise ValueError(ERROR_MESSAGES["unknown_format"].format(
                track=track, known=list(SOURCE_FORMATS), value=source.get("format"),
            ))
    for variant in config["run"]["variants"]:
        if variant not in config["variants"]:
            raise ValueError(ERROR_MESSAGES["unknown_variant"].format(
                variant=variant, known=sorted(config["variants"]),
            ))
        for track in config["variants"][variant]:
            if track not in config["sources"]:
                raise ValueError(ERROR_MESSAGES["track_without_source"].format(variant=variant, track=track))
    return config


def score_column(track: str) -> str:
    """The stacker input column for one track."""
    return f"p_default_{track}"


def score_columns(tracks: list[str]) -> list[str]:
    """Stacker inputs, in variant order. The only columns a stacker may read."""
    return [score_column(track) for track in tracks]


def load_role_mapping(config: dict[str, Any], root: Path = REPO_ROOT) -> pd.DataFrame:
    """The shared role mapping, after its pinned SHA-256 and columns are checked."""
    path = root / config["role_mapping"]["path"]
    provenance.verify_file_checksum(path, config["role_mapping"]["sha256"], allow_crlf=True)
    mapping = pd.read_csv(path, dtype=str)
    if mapping.columns.tolist() != list(MAPPING_COLUMNS):
        raise ValueError(ERROR_MESSAGES["mapping_columns"].format(
            expected=list(MAPPING_COLUMNS), actual=mapping.columns.tolist(),
        ))
    if count := int(mapping["loan_id"].duplicated().sum()):
        raise ValueError(ERROR_MESSAGES["mapping_duplicates"].format(count=count))
    return mapping


def _role_ids(mapping: pd.DataFrame, roles: tuple[str, ...]) -> set[str]:
    return set(mapping.loc[mapping["role"].isin(roles), "loan_id"])


def _load_tabular(track: str, source: dict[str, Any], mapping: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Raw tabular LR scores for the scoring loans, via the handoff's shared score-file checks."""
    parts = []
    for prefix, split, roles in (
        ("fusion", TABULAR_FUSION_SPLIT, TABULAR_FUSION_ROLES),
        ("validation", "validation", ("validation",)),
    ):
        path = root / source[f"{prefix}_path"]
        provenance.verify_file_checksum(path, source[f"{prefix}_sha256"], allow_crlf=True)
        frame = load_score_file(
            path,
            probability_column=source["probability_column"],
            split=split,
            expected_ids=_role_ids(mapping, roles),
            model_name=source["model_name"],
            allowed_versions=frozenset({source["model_version"]}),
        )
        parts.append(frame[["loan_id", source["probability_column"]]])
    return pd.concat(parts, ignore_index=True).rename(
        columns={source["probability_column"]: score_column(track)}
    )


def _check_nlp_scores(track: str, frame: pd.DataFrame, mapping: pd.DataFrame, model_version: str) -> None:
    """The shared NLP export contract: exact columns, coverage, split/role, finite values, one version."""
    if frame.columns.tolist() != list(NLP_SCORE_COLUMNS):
        raise ValueError(ERROR_MESSAGES["nlp_columns"].format(
            track=track, expected=list(NLP_SCORE_COLUMNS), actual=frame.columns.tolist(),
        ))
    if count := int(frame["loan_id"].duplicated().sum()):
        raise ValueError(ERROR_MESSAGES["nlp_duplicates"].format(track=track, count=count))
    expected = mapping.loc[mapping["role"].isin(SCORING_ROLES)].set_index("loan_id")
    ids = set(frame["loan_id"])
    missing, extra = len(set(expected.index) - ids), len(ids - set(expected.index))
    if missing or extra:
        raise ValueError(ERROR_MESSAGES["nlp_coverage"].format(track=track, missing=missing, extra=extra))
    aligned = frame.set_index("loan_id").loc[expected.index]
    for column in ("split", "role"):
        if count := int(aligned[column].astype(str).ne(expected[column]).sum()):
            raise ValueError(ERROR_MESSAGES["nlp_mismatch"].format(track=track, column=column, count=count))
    for column in ("raw_margin", "raw_prob"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if count := int((~np.isfinite(values)).sum()):
            raise ValueError(ERROR_MESSAGES["not_finite"].format(track=track, column=column, count=count))
    probabilities = frame["raw_prob"].to_numpy(dtype=np.float64)
    if count := int(((probabilities < 0) | (probabilities > 1)).sum()):
        raise ValueError(ERROR_MESSAGES["out_of_range"].format(track=track, count=count))
    if (versions := sorted(frame["model_version"].astype(str).unique())) != [model_version]:
        raise ValueError(ERROR_MESSAGES["nlp_version"].format(
            track=track, expected=model_version, actual=versions,
        ))


def _load_nlp(track: str, source: dict[str, Any], mapping: pd.DataFrame, root: Path) -> pd.DataFrame:
    path = root / source["path"]
    provenance.verify_file_checksum(path, source["sha256"])
    frame = read_score_file(path)
    _check_nlp_scores(track, frame, mapping, source["model_version"])
    return frame[["loan_id", "raw_prob"]].astype({"raw_prob": np.float64}).rename(
        columns={"raw_prob": score_column(track)}
    )


LOADERS = {"tabular_raw_predictions": _load_tabular, "nlp_scores_v2": _load_nlp}


def load_track_scores(
    track: str, config: dict[str, Any], mapping: pd.DataFrame, root: Path = REPO_ROOT
) -> pd.DataFrame:
    """``loan_id, p_default_<track>`` for exactly the 48,724 scoring loans, sorted by loan_id."""
    source = config["sources"][track]
    scores = LOADERS[source["format"]](track, source, mapping, root)
    return scores.sort_values("loan_id", ignore_index=True)


def load_labels(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """``loan_id, target`` for train and validation loans, from the approved manifest loader."""
    manifest, _ = evaluate.load_manifest(data_dir)
    return manifest[["loan_id", "target"]].astype({"loan_id": str})


def build_role_tables(
    config: dict[str, Any],
    tracks: list[str],
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    root: Path = REPO_ROOT,
) -> dict[str, pd.DataFrame]:
    """``{role: loan_id, target, p_default_<track>...}`` for each scoring role, checked and sorted.

    The base rows are the mapping's role IDs; every track and the label must
    match each one exactly (``join_scores`` raises on a missing score).
    """
    mapping = load_role_mapping(config, root)
    scores = {track: load_track_scores(track, config, mapping, root) for track in tracks}
    labels = load_labels(data_dir)

    tables = {}
    for role in SCORING_ROLES:
        base = mapping.loc[mapping["role"].eq(role), ["loan_id"]].sort_values("loan_id", ignore_index=True)
        joined = join_scores(base, scores)
        joined = joined.merge(labels, on="loan_id", how="left", validate="one_to_one")
        if count := int(joined["target"].isna().sum()):
            raise ValueError(ERROR_MESSAGES["label_missing"].format(count=count, role=role))
        tables[role] = joined[["loan_id", "target", *score_columns(tracks)]].astype({"target": "int8"})
    return tables


def load_reference(config: dict[str, Any], validation_ids: pd.Series, root: Path = REPO_ROOT) -> pd.DataFrame:
    """The calibrated tabular reference for validation: ``loan_id, p_reference, p_reference_raw``.

    Rows are returned in ``validation_ids`` order so they align with the stacker's.
    """
    reference = config["reference"]
    path = root / reference["path"]
    provenance.verify_file_checksum(path, reference["sha256"])
    frame = read_score_file(path)
    if (versions := sorted(frame["model_version"].unique())) != [reference["model_version"]]:
        raise ValueError(ERROR_MESSAGES["reference_version"].format(
            expected=reference["model_version"], actual=versions,
        ))
    if frame["loan_id"].duplicated().any() or set(frame["loan_id"]) != set(validation_ids):
        raise ValueError(ERROR_MESSAGES["reference_ids"])
    aligned = frame.set_index("loan_id").loc[list(validation_ids)]
    return pd.DataFrame({
        "loan_id": list(validation_ids),
        "p_reference": aligned[reference["probability_column"]].to_numpy(dtype=np.float64),
        "p_reference_raw": aligned[reference["raw_probability_column"]].to_numpy(dtype=np.float64),
    })

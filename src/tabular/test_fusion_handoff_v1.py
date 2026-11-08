"""Fusion handoff v1 tests: a tiny artificial set of sources plus the committed real sources.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_fusion_handoff_v1.py -v -s

Nothing here reads data/raw/loan.csv, so no test is skipped for a missing
LENDINGCLUB_RAW_CSV. The "committed" test needs the six tables and two
manifests written by ``python -m src.tabular.fusion_handoff_v1``.

Fixture sources are written at the real config's relative paths under
``tmp_path`` and re-pinned after every edit, so each bad-file case reaches the
content check it targets instead of stopping at the checksum.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.special import expit

from src.nlp.preprocess import sha256_file
from src.tabular.fusion_handoff_v1 import (
    DEFAULT_CONFIG,
    ERROR_MESSAGES,
    build_handoff_tables,
    load_all_scores,
    load_handoff_config,
    load_role_mapping,
    run_handoff_v1,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCORING_ROLES = ("fusion_fit", "fusion_calibration", "validation")

# Hardcoded on purpose, so a silent change to the contract fails a test instead of being echoed back.
TABLE_COLUMNS = {
    "two_score": [
        "loan_id", "split", "role", "p_default_tabular", "p_default_tfidf",
        "tabular_model_version", "tfidf_model_version",
    ],
    "three_score": [
        "loan_id", "split", "role", "p_default_tabular", "p_default_tfidf", "p_default_distilbert",
        "tabular_model_version", "tfidf_model_version", "distilbert_model_version",
    ],
}
REAL_ROLE_ROWS = {"fusion_fit": 15_587, "fusion_calibration": 11_353, "validation": 21_784}

# Fixture loans: (split, role). The base_fit loan must never reach a table.
FIXTURE_LOANS = {
    "lc_000000001": ("train", "base_fit"),
    "lc_000000002": ("train", "fusion_fit"),
    "lc_000000003": ("train", "fusion_fit"),
    "lc_000000004": ("train", "fusion_calibration"),
    "lc_000000005": ("train", "fusion_calibration"),
    "lc_000000006": ("validation", "validation"),
    "lc_000000007": ("validation", "validation"),
}
SCORED = [loan for loan, (_, role) in FIXTURE_LOANS.items() if role != "base_fit"]
FIXTURE_MARGINS = {"tfidf": [-1.6, -1.2, -2.0, -0.4, -1.9, -1.1], "distilbert": [-1.4, -0.9, -2.2, -0.2, -1.7, -1.3]}
TABULAR_PROBABILITIES = dict(zip(SCORED, [0.41, 0.55, 0.38, 0.62, 0.47, 0.51]))


def error_pattern(key: str) -> str:
    """Regex for the ERROR_MESSAGES[key] text anywhere in a message, any value in each {placeholder}."""
    return ".*".join(re.escape(part) for part in re.split(r"\{[^}]*\}", ERROR_MESSAGES[key]))


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


# --------------------------------------------------------------------------- fixture sources


def mapping_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_id": list(FIXTURE_LOANS),
        "split": [split for split, _ in FIXTURE_LOANS.values()],
        "role": [role for _, role in FIXTURE_LOANS.values()],
        "issue_month": ["2012-12", "2013-01", "2013-04", "2013-05", "2013-07", "2013-08", "2013-12"],
    })


def tabular_frames() -> list[pd.DataFrame]:
    """The two tabular files: fusion_fit + fusion_calibration say split 'calibration', as the real file does."""
    frames = []
    for split, roles in (("calibration", ("fusion_fit", "fusion_calibration")), ("validation", ("validation",))):
        ids = [loan for loan in SCORED if FIXTURE_LOANS[loan][1] in roles]
        frames.append(pd.DataFrame({
            "loan_id": ids,
            "split": split,
            "p_default_tabular": [TABULAR_PROBABILITIES[loan] for loan in ids],
            "model_name": "logistic_regression",
            "model_version": "tabular-logistic-v1-model-fit",
        }))
    return frames


def nlp_frame(track: str, config: dict[str, Any]) -> pd.DataFrame:
    """A valid six-column v2 export, typed like the real files."""
    margins = np.array(FIXTURE_MARGINS[track])
    return pd.DataFrame({
        "loan_id": pd.array(SCORED, dtype="string"),
        "split": pd.array([FIXTURE_LOANS[loan][0] for loan in SCORED], dtype="string"),
        "role": pd.array([FIXTURE_LOANS[loan][1] for loan in SCORED], dtype="string"),
        "raw_margin": margins,
        "raw_prob": expit(margins),
        "model_version": pd.array([config["sources"][track]["model_version"]] * len(SCORED), dtype="string"),
    })


def write_mapping(root: Path, config: dict[str, Any], frame: pd.DataFrame, *, pin: bool = True) -> None:
    path = root / config["role_mapping"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if pin:
        config["role_mapping"]["sha256"] = sha256_file(path)


def write_tabular(root: Path, config: dict[str, Any], index: int, frame: pd.DataFrame, *, pin: bool = True) -> None:
    entry = config["sources"]["tabular"]["files"][index]
    path = root / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if pin:
        entry["sha256"] = sha256_file(path)


def write_owner_manifest(root: Path, source: dict[str, Any], sha256: str) -> None:
    """The owner's manifest with ``sha256`` at the configured key path."""
    value: Any = sha256
    for key in reversed(source["owner_manifest_sha256_key"]):
        value = {key: value}
    path = root / source["owner_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_nlp(root: Path, config: dict[str, Any], track: str, frame: pd.DataFrame, *, pin: bool = True) -> None:
    """Write the export; with pin, re-pin the config and the owner manifest to its new bytes."""
    source = config["sources"][track]
    path = root / source["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")
    if pin:
        source["sha256"] = sha256_file(path)
        write_owner_manifest(root, source, source["sha256"])


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Valid fixture sources at the real config's paths, pinned in a copy of the real config."""
    config = copy.deepcopy(load_handoff_config())
    write_mapping(tmp_path, config, mapping_frame())
    for index, frame in enumerate(tabular_frames()):
        write_tabular(tmp_path, config, index, frame)
    for track in ("tfidf", "distilbert"):
        write_nlp(tmp_path, config, track, nlp_frame(track, config))
    return tmp_path, config


def build(root: Path, config: dict[str, Any]) -> dict[str, dict[str, pd.DataFrame]]:
    """Every loader and the table builder, as run_handoff_v1 calls them (without writing)."""
    mapping = load_role_mapping(config, root=root)
    return build_handoff_tables(config, mapping, load_all_scores(config, mapping, root=root))


# --------------------------------------------------------------------------- real sources


@pytest.fixture(scope="module")
def real_tables() -> dict[str, dict[str, pd.DataFrame]]:
    config = load_handoff_config()
    mapping = load_role_mapping(config)
    return build_handoff_tables(config, mapping, load_all_scores(config, mapping))


def test_real_tables_have_the_agreed_rows_columns_and_labels(real_tables) -> None:
    """Real sources: six tables of 15,587 / 11,353 / 21,784 loans, the agreed columns, IDs and labels equal
    the committed role mapping's, roles disjoint, no base_fit or locked test loan."""
    mapping = pd.read_csv(REPOSITORY_ROOT / "data" / "fusion" / "fusion_roles_v1.csv", dtype=str).set_index("loan_id")
    test_ids = set(pd.read_csv(REPOSITORY_ROOT / "data" / "test_ids.csv", dtype=str)["loan_id"])
    assert set(real_tables) == set(TABLE_COLUMNS)
    for table, frames in real_tables.items():
        assert set(frames) == set(SCORING_ROLES)
        seen: set[str] = set()
        for role, frame in frames.items():
            assert frame.columns.tolist() == TABLE_COLUMNS[table]
            assert len(frame) == REAL_ROLE_ROWS[role]
            ids = set(frame["loan_id"])
            assert ids == set(mapping.index[mapping["role"].eq(role)])
            assert not ids & seen and not ids & test_ids
            seen |= ids
            labels = mapping.loc[frame["loan_id"]]
            assert frame["split"].tolist() == labels["split"].tolist()
            assert frame["role"].eq(role).all()
        assert len(seen) == sum(REAL_ROLE_ROWS.values())
    # The tabular file says "calibration"; the tables carry the original split-v1 label.
    for role in ("fusion_fit", "fusion_calibration"):
        assert real_tables["two_score"][role]["split"].eq("train").all()


def test_real_scores_equal_their_sources_and_versions_agree(real_tables) -> None:
    """Real sources: every score equals its source file's value for that loan_id, each version column is its
    pin, and the two-score and three-score tables of a role share identical loans and tabular / TF-IDF values."""
    config = load_handoff_config()
    tabular = pd.concat(
        pd.read_csv(REPOSITORY_ROOT / entry["path"], dtype={"loan_id": str}) for entry in config["sources"]["tabular"]["files"]
    ).set_index("loan_id")["p_default_tabular"]
    sources = {"tabular": tabular}
    for track in ("tfidf", "distilbert"):
        frame = pd.read_parquet(REPOSITORY_ROOT / config["sources"][track]["path"])
        sources[track] = frame.astype({"loan_id": str}).set_index("loan_id")["raw_prob"]

    for frames in real_tables.values():
        for frame in frames.values():
            for track, values in sources.items():
                if f"p_default_{track}" not in frame:
                    continue
                expected = values.loc[frame["loan_id"]].to_numpy(dtype=np.float64)
                assert np.array_equal(frame[f"p_default_{track}"].to_numpy(), expected)
                assert frame[f"{track}_model_version"].eq(config["sources"][track]["model_version"]).all()

    shared = TABLE_COLUMNS["two_score"]
    for role in SCORING_ROLES:
        assert_frame_equal(real_tables["two_score"][role], real_tables["three_score"][role][shared])


def _without_observed(value: Any) -> Any:
    """Drop observed_sha256 everywhere: a CSV's working-copy hash depends on the checkout's line endings."""
    if isinstance(value, dict):
        return {key: _without_observed(item) for key, item in value.items() if key != "observed_sha256"}
    if isinstance(value, list):
        return [_without_observed(item) for item in value]
    return value


def test_committed_tables_and_manifests_equal_a_fresh_build(tmp_path: Path) -> None:
    """Committed evidence: each manifest's output SHA-256 matches its committed table, and a fresh run from the
    committed sources reproduces every committed table and the recorded source pins."""
    fresh = run_handoff_v1(output_dir=tmp_path / "data", manifest_dir=tmp_path / "reports")
    for table, manifest in fresh.items():
        committed = json.loads(
            (REPOSITORY_ROOT / "reports" / "tabular" / f"fusion_handoff_v1_{table}_manifest.json").read_text(encoding="utf-8")
        )
        assert _without_observed(committed["sources"]) == _without_observed(manifest["sources"])
        assert committed["table_version"] == manifest["table_version"]
        for role in SCORING_ROLES:
            path = REPOSITORY_ROOT / committed["outputs"][role]["path"]
            assert sha256_file(path) == committed["outputs"][role]["sha256"]
            assert_frame_equal(
                pd.read_parquet(path), pd.read_parquet(tmp_path / "data" / f"fusion_v1_{table}_{role}.parquet")
            )


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ('roles = ["fusion_fit", "fusion_calibration"]', 'roles = ["fusion_fit"]', "tabular_roles_coverage"),
        ('roles = ["validation"]', 'roles = ["validation", "fusion_fit"]', "tabular_roles_coverage"),
        ('file_split = "validation"', 'file_split = "test"', "test_split_locked"),
        ('tracks = ["tabular", "tfidf"]', 'tracks = ["tfidf", "tabular"]', "tabular_not_first"),
        ('tracks = ["tabular", "tfidf"]', 'tracks = ["tabular", "bert"]', "unknown_track"),
        ('sha256 = "68599e30', 'sha256 = "68599E30', "bad_sha256"),
    ],
    ids=["role_missing", "role_twice", "test_split", "tabular_not_first", "unknown_track", "bad_sha256"],
)
def test_malformed_config_is_refused(tmp_path: Path, old: str, new: str, key: str) -> None:
    """A tabular role missing or repeated, a tabular file on the test split, a table not led by tabular, an
    unknown track, or a malformed pin is refused before any source is read."""
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "fusion_handoff_v1.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=error_pattern(key)):
        load_handoff_config(path)


# --------------------------------------------------------------------------- sources


def _duplicate(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)


def _set(column: str, value: Any, row: int = 0):
    def edit(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame.loc[row, column] = value
        return frame
    return edit


def _nudge_probability(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.loc[0, "raw_prob"] = frame.loc[0, "raw_prob"] + 1e-6
    return frame


@pytest.mark.parametrize(
    ("edit", "pattern"),
    [
        (_duplicate, re.escape("duplicate loan_id")),
        (lambda frame: frame.iloc[1:].reset_index(drop=True), re.escape("1 missing, 0 extra")),
        (lambda frame: frame.assign(target=0), error_pattern("nlp_columns")),
        (_set("role", "fusion_calibration"), re.escape("1 loans disagree with the role mapping on role")),
        (_set("split", "validation"), re.escape("1 loans disagree with the role mapping on split")),
        (_set("raw_prob", np.nan), re.escape("missing p_default_tfidf values")),
        (_set("raw_prob", 1.1), re.escape("within [0, 1]")),
        (_set("raw_margin", np.inf), error_pattern("not_finite")),
        (_nudge_probability, error_pattern("sigmoid_mismatch")),
        (_set("model_version", "nlp-tfidf-tuned-v1"), error_pattern("nlp_version")),
    ],
    ids=[
        "duplicate_id", "missing_loan", "extra_target_column", "wrong_role", "wrong_split",
        "nan_probability", "probability_above_one", "infinite_margin", "sigmoid_mismatch", "unpinned_version",
    ],
)
def test_bad_nlp_export_is_refused(world, edit, pattern: str) -> None:
    """An NLP export with a duplicate or missing loan, an extra (target) column, a split or role that differs
    from the mapping, a NaN or out-of-range probability, an infinite margin, a probability that is not
    sigmoid(margin), or an unpinned version is refused, naming the track; nothing is filled or dropped."""
    root, config = world
    write_nlp(root, config, "tfidf", edit(nlp_frame("tfidf", config)))
    with pytest.raises(ValueError, match=r"^tfidf \(data/nlp/tfidf_scores_v2\.parquet\): .*" + pattern):
        build(root, config)


def test_tabular_file_missing_a_loan_is_refused(world) -> None:
    """A tabular file that lacks one of its roles' loans is an incomplete join and is refused."""
    root, config = world
    write_tabular(root, config, 0, tabular_frames()[0].iloc[1:])
    with pytest.raises(ValueError, match=r"^tabular \(.*" + re.escape("1 missing, 0 extra")):
        build(root, config)


@pytest.mark.parametrize("source", ["role_mapping", "tabular", "tfidf", "owner_manifest"])
def test_source_that_differs_from_its_pin_is_refused(world, source: str) -> None:
    """A role mapping, tabular file or NLP export edited after it was pinned, or an NLP pin that differs from the
    hash in its owner's manifest, stops the run."""
    root, config = world
    if source == "role_mapping":
        frame = mapping_frame()
        frame.loc[1, "role"] = "validation"
        write_mapping(root, config, frame, pin=False)
        pattern = r"^role_mapping \(.*Checksum mismatch"
    elif source == "tabular":
        write_tabular(root, config, 0, _set("p_default_tabular", 0.9)(tabular_frames()[0]), pin=False)
        pattern = r"^tabular \(.*Checksum mismatch"
    elif source == "tfidf":
        write_nlp(root, config, "tfidf", _set("raw_margin", 0.0)(nlp_frame("tfidf", config)), pin=False)
        pattern = r"^tfidf \(.*Checksum mismatch"
    else:
        write_owner_manifest(root, config["sources"]["tfidf"], "0" * 64)
        pattern = r"^tfidf \(.*" + error_pattern("owner_manifest_disagrees")
    with pytest.raises(ValueError, match=pattern):
        build(root, config)


# --------------------------------------------------------------------------- tables


def test_missing_score_at_the_join_is_never_filled(world) -> None:
    """If a track's checked scores lack one loan at the join, the build stops naming the table and role: no
    filled score and no silently dropped row."""
    root, config = world
    mapping = load_role_mapping(config, root=root)
    scores = load_all_scores(config, mapping, root=root)
    scores["tfidf"] = scores["tfidf"].iloc[1:]
    with pytest.raises(ValueError, match=r"^two_score fusion_fit: 1 loans have no tfidf score; a missing score is never filled"):
        build_handoff_tables(config, mapping, scores)


def test_shuffled_export_gives_identical_tables(world) -> None:
    """Rows are aligned by loan_id, never by position: a shuffled NLP export builds the same six tables."""
    root, config = world
    expected = build(root, config)
    shuffled = nlp_frame("distilbert", config).sample(frac=1, random_state=7).reset_index(drop=True)
    write_nlp(root, config, "distilbert", shuffled)
    actual = build(root, config)
    for table, frames in expected.items():
        for role, frame in frames.items():
            assert_frame_equal(actual[table][role], frame)
    # Sanity check on the fixture: the base_fit loan is in no table, and fusion loans say "train".
    assert "lc_000000001" not in set(pd.concat(f["loan_id"] for t in actual.values() for f in t.values()))
    assert actual["three_score"]["fusion_calibration"]["split"].eq("train").all()

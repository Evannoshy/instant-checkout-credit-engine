"""Fusion handoff (fusion-handoff-v0): calibrated tabular validation scores and
checked base-model score files.

Join, check and record; never compute a new score.

1. Calibrated validation export.
Writes data/tabular/tabular_calibrated_val.parquet: one row per validation loan
with the balanced logistic model's raw probability and the same probability
through the tabular_calibration_v1 Platt mapping.

No new score is computed. The raw scores are read from
reports/tabular/raw_predictions/logistic_regression_validation.csv. The Platt
mapping is refitted with ``calibrate.fit_platt`` on the calibration-role rows
only, exactly as src/tabular/calibrate.py fitted it, and must reproduce the
merged evidence before anything is written:
- slope and intercept equal calibration_metrics.json, and
- the raw and Platt validation metric rows equal calibration_results.csv,
both within ``unweighted_comparison.REPRODUCTION_TOLERANCE`` (1e-6), via the
same ``check_reference_reproduction`` that guards the unweighted comparison.

Output schema (sorted by loan_id):
    loan_id, split="validation", p_default_tabular_raw,
    p_default_tabular_calibrated, model_name, model_version, calibration_version

Validation outcomes are read only to run that reproduction check; they never
enter the output file. ``fit_platt`` receives calibration rows only, so the
mapping cannot be fitted on validation labels. The locked test split is never
read.

2. Score-file loading.
``load_score_file`` reads one track's scores for one split and checks them
against the pins in ``EXPECTED_SOURCES``: the shared prediction-frame rules
(``evaluate.validate_prediction_frame``: columns, duplicates, exact ID coverage
with missing and extra counts, NaN, numeric, [0, 1]), the split label, one
pinned model_name, and every model_version in the pinned set. Not every pinned
version has to appear: DistilBERT train is two models by design. loan_id is
cast to plain str, because the NLP files store it as pandas ``string``.

``load_sources`` loads all six. The NLP files are checked against the full
frozen train (86,293) and validation (21,784) ID sets from the manifest, never
a subset, so a broken NLP file cannot hide behind a later selection. No
frozen list exists for the tabular OOF loans, so the OOF file defines its own
expected IDs after these checks: SHA-256 equal to the OOF manifest's (the
bytes the committed OOF tests verify), row count equal to the manifest's,
every loan in the model_fit role (so no calibration, validation or test loan),
and fold_id counts equal to the manifest's. The folds are not re-derived here;
``test_export_oof_scores.py`` does that and must pass before a handoff run.
The train and validation sets are checked once for overlap with the locked
test IDs. Targets are never read for this step.

Run from the repository root:
    python -m src.tabular.fusion_handoff

Known limitations:
- The Platt mapping belongs to the model fit on all model_fit rows. It must not
  be applied to the out-of-fold scores in tabular_oof_train_v1.parquet, which
  come from five smaller fold models.
- Calibration here is for reporting and later policy work on validation.
  Validation grades and never fits; nothing in this module selects anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tabular import calibrate, development, evaluate, export_oof_scores, provenance, raw_predictions
from src.tabular.preprocess import DEFAULT_DATA_DIR, REPO_ROOT
from src.tabular.unweighted_comparison import (
    BALANCED,
    REFERENCE_METRICS,
    REFERENCE_RESULTS,
    REPRODUCTION_TOLERANCE,
    Entry,
    check_reference_reproduction,
)
from src.tabular.xgboost_baseline import validate_runtime

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "tabular"
CALIBRATED_VAL_PATH = DEFAULT_OUTPUT_DIR / "tabular_calibrated_val.parquet"
OOF_MANIFEST_PATH = export_oof_scores.DEFAULT_MANIFEST_PATH

# Pinned score sources, written verbatim into the handoff manifest. Paths are
# repository-relative. ``id_set`` names the expected loan_id set: the full
# frozen split, or "tabular_oof" for the loans in the checked tabular OOF file.
EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    "tabular_train": {
        "path": "data/tabular/tabular_oof_train_v1.parquet",
        "probability_column": "p_default_tabular_raw",
        "split": "train",
        "id_set": "tabular_oof",
        "model_name": "logistic_regression",
        "allowed_versions": ("tabular-logistic-v1-time-oof",),
        "extra_columns": ("fold_id",),
    },
    "tabular_validation": {
        "path": "reports/tabular/raw_predictions/logistic_regression_validation.csv",
        "probability_column": "p_default_tabular",
        "split": "validation",
        "id_set": "validation",
        "model_name": "logistic_regression",
        "allowed_versions": ("tabular-logistic-v1-model-fit",),
        "extra_columns": (),
    },
    "tfidf_train": {
        "path": "data/nlp/tfidf_oof_train.parquet",
        "probability_column": "prob_default_tfidf",
        "split": "train",
        "id_set": "train",
        "model_name": "tfidf_logistic",
        "allowed_versions": ("nlp-tfidf-tuned-v1",),
        "extra_columns": (),
    },
    "tfidf_validation": {
        "path": "data/nlp/tfidf_oof_validate.parquet",
        "probability_column": "prob_default_tfidf",
        "split": "validation",
        "id_set": "validation",
        "model_name": "tfidf_logistic",
        "allowed_versions": ("nlp-tfidf-tuned-v1",),
        "extra_columns": (),
    },
    "distilbert_train": {
        "path": "data/nlp/distilbert_scores_train.parquet",
        "probability_column": "prob_default_distilbert",
        "split": "train",
        "id_set": "train",
        "model_name": "distilbert-base-uncased",
        "allowed_versions": ("nlp-distilbert-10k-v1", "nlp-distilbert-10k-holdout-v1"),
        "extra_columns": (),
    },
    "distilbert_validation": {
        "path": "data/nlp/distilbert_scores_val.parquet",
        "probability_column": "prob_default_distilbert",
        "split": "validation",
        "id_set": "validation",
        "model_name": "distilbert-base-uncased",
        "allowed_versions": ("nlp-distilbert-10k-v1",),
        "extra_columns": (),
    },
}

TABULAR_MODEL_NAME = "logistic_regression"
TABULAR_FIT_VERSION = "tabular-logistic-v1-model-fit"
CALIBRATED_PROBABILITY_COLUMNS = ("p_default_tabular_raw", "p_default_tabular_calibrated")
CALIBRATED_COLUMNS = (
    "loan_id", "split", *CALIBRATED_PROBABILITY_COLUMNS, "model_name", "model_version", "calibration_version",
)
# The two merged calibration_results.csv rows the export must reproduce.
REPRODUCED_ENTRIES = (
    Entry("balanced_lr_raw", "Balanced LR, raw", BALANCED, "raw"),
    Entry("balanced_lr_platt", "Balanced LR + existing Platt", BALANCED, "platt"),
)


def validate_calibrated_frame(frame: pd.DataFrame, expected_ids: set[str]) -> None:
    """Require the agreed columns, the validation split, and each probability column to pass
    ``evaluate.validate_prediction_frame`` (one row per expected ID, no NaN, numeric, in [0, 1])."""
    if frame.columns.tolist() != list(CALIBRATED_COLUMNS):
        raise ValueError(
            f"Calibrated frame must have columns {list(CALIBRATED_COLUMNS)} in that order; "
            f"got {frame.columns.tolist()}"
        )
    if not frame["split"].eq("validation").all():
        raise ValueError("Calibrated frame must contain only split 'validation'")
    for column in CALIBRATED_PROBABILITY_COLUMNS:
        evaluate.validate_prediction_frame(
            frame[["loan_id", "split", column, "model_name", "model_version"]],
            expected_ids,
            probability_column=column,
        )


def build_calibrated_validation(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    prediction_dir: str | Path = raw_predictions.DEFAULT_PREDICTION_DIR,
    reference_metrics_path: str | Path = REFERENCE_METRICS,
    reference_results_path: str | Path = REFERENCE_RESULTS,
    calibration_config_path: str | Path = calibrate.DEFAULT_CALIBRATION_CONFIG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The calibrated validation frame, and the Platt parameters and metric rows it reproduced.

    Raises ValueError if the prediction files fail calibrate's checks, name a
    model other than the balanced v1 model, or do not reproduce the merged
    Platt parameters and validation metrics.
    """
    config = calibrate.load_calibration_config(calibration_config_path)
    expected_ids, target = calibrate.expected_prediction_ids(data_dir)
    paths = {set_name: Path(prediction_dir) / f"{BALANCED}_{set_name}.csv" for set_name in calibrate.SETS}
    candidate = calibrate.load_candidate(BALANCED, paths, expected_ids, target)
    if (candidate.model_name, candidate.model_version) != (TABULAR_MODEL_NAME, TABULAR_FIT_VERSION):
        raise ValueError(
            f"Expected {TABULAR_MODEL_NAME} {TABULAR_FIT_VERSION} predictions; "
            f"got {candidate.model_name} {candidate.model_version}"
        )

    calibration, validation = candidate.sets["calibration"], candidate.sets["validation"]
    platt = calibrate.fit_platt(calibration.p, calibration.y, config["probability_clip"])
    calibrated = platt.transform(validation.p)

    metric_rows = {
        "balanced_lr_raw": calibrate.metric_row(validation.y, validation.p),
        "balanced_lr_platt": calibrate.metric_row(validation.y, calibrated),
    }
    check_reference_reproduction(
        REPRODUCED_ENTRIES, metric_rows, platt.details,
        json.loads(Path(reference_metrics_path).read_text(encoding="utf-8")),
        pd.read_csv(reference_results_path),
    )

    frame = pd.DataFrame({
        "loan_id": validation.loan_ids,
        "split": "validation",
        "p_default_tabular_raw": validation.p,
        "p_default_tabular_calibrated": np.asarray(calibrated, dtype=np.float64),
        "model_name": candidate.model_name,
        "model_version": candidate.model_version,
        "calibration_version": config["calibration_version"],
    })
    validate_calibrated_frame(frame, expected_ids["validation"])
    details = {
        "platt_mapping": {"fit_on": "calibration role, balanced LR raw scores", **platt.details},
        "validation_metrics": metric_rows,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "calibration_version": config["calibration_version"],
    }
    return frame, details


def run_calibrated_export(
    output_path: str | Path = CALIBRATED_VAL_PATH, data_dir: str | Path = DEFAULT_DATA_DIR
) -> dict[str, Any]:
    """Check the runtime, build the calibrated validation frame and write it as parquet."""
    validate_runtime()
    frame, details = build_calibrated_validation(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return {"rows": len(frame), **details}


def read_score_file(path: str | Path) -> pd.DataFrame:
    """Read a parquet or CSV score file with loan_id as plain str (NLP files store pandas ``string``)."""
    path = Path(path)
    frame = pd.read_csv(path, dtype={"loan_id": str}) if path.suffix == ".csv" else pd.read_parquet(path)
    return frame.astype({"loan_id": str})


def load_score_file(
    path: str | Path,
    *,
    probability_column: str,
    split: str,
    expected_ids: set[str],
    model_name: str,
    allowed_versions: frozenset[str],
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """One track's scores for one split, checked, loan_id normalised to str, sorted by loan_id.

    ``extra_columns`` (the OOF file's fold_id) are set aside for the shared check and
    kept in the result; the caller checks them. Raises ValueError for any
    failed check; it never drops, fills or filters a row.
    """
    frame = read_score_file(path)
    if missing := sorted(set(extra_columns) - set(frame.columns)):
        raise ValueError(f"Score file is missing columns {missing}")
    evaluate.validate_prediction_frame(
        frame.drop(columns=list(extra_columns)), expected_ids, probability_column=probability_column
    )
    if not frame["split"].eq(split).all():
        raise ValueError(f"split must be {split!r} on every row")
    names = frame["model_name"].unique().tolist()
    if names != [model_name]:
        raise ValueError(f"model_name must be {model_name!r} on every row; got {names}")
    if unexpected := sorted(str(version) for version in set(frame["model_version"]) - allowed_versions):
        raise ValueError(f"model_version {unexpected} is not in the pinned set {sorted(allowed_versions)}")
    return frame.sort_values("loan_id", ignore_index=True)


def expected_split_ids(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, set[str]]:
    """The frozen train and validation loan_ids, from the manifest's split column.

    Never built from a score file. Raises ValueError if either set contains a
    locked test ID.
    """
    manifest, _ = evaluate.load_manifest(data_dir)
    loan_ids = manifest["loan_id"].astype(str)
    id_sets = {split: set(loan_ids[manifest["split"].eq(split)]) for split in ("train", "validation")}
    test_ids = set(pd.read_csv(Path(data_dir) / "test_ids.csv", dtype=str)["loan_id"])
    for split, ids in id_sets.items():
        if overlap := len(ids & test_ids):
            raise ValueError(f"Expected {split} IDs include {overlap} locked test IDs")
    return id_sets


def tabular_oof_ids(path: str | Path, oof_manifest: dict[str, Any], data_dir: str | Path) -> set[str]:
    """OOF loan_ids, once their count equals manifest and every one is a model_fit loan.

    model_fit is part of the train split, so this also rules out calibration,
    validation and test loans. The file's bytes are pinned by SHA-256 first.
    """
    loan_ids = read_score_file(path)["loan_id"]
    if len(loan_ids) != oof_manifest["rows"]["oof_scored"]:
        raise ValueError(f"{len(loan_ids)} rows; the OOF manifest records {oof_manifest['rows']['oof_scored']}")
    roles = development.assign_development_roles(data_dir)
    if outside := int((~loan_ids.isin(roles.index[roles["role"].eq("model_fit")])).sum()):
        raise ValueError(f"{outside} loans are outside model_fit (calibration, validation or test)")
    return set(loan_ids)


def check_fold_counts(frame: pd.DataFrame, oof_manifest: dict[str, Any]) -> None:
    """fold_id takes the values 0..n-1 with exactly the holdout counts in the OOF manifest."""
    recorded = [fold["holdout_rows"] for fold in oof_manifest["folds"]]
    counts = frame["fold_id"].value_counts().sort_index()
    if counts.index.tolist() != list(range(len(recorded))) or counts.tolist() != recorded:
        raise ValueError(f"fold_id counts {counts.to_dict()} differ from the OOF manifest's {recorded}")


def load_sources(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    sources: dict[str, dict[str, Any]] = EXPECTED_SOURCES,
    oof_manifest_path: str | Path = OOF_MANIFEST_PATH,
) -> dict[str, pd.DataFrame]:
    """Every pinned score file, checked in full. Raises ValueError naming the source that failed."""
    split_ids = expected_split_ids(data_dir)
    oof_manifest = json.loads(Path(oof_manifest_path).read_text(encoding="utf-8"))
    frames = {}
    for name, source in sources.items():
        path = REPO_ROOT / source["path"]
        is_tabular_oof = source["id_set"] == "tabular_oof"
        try:
            if is_tabular_oof:
                provenance.verify_file_checksum(path, oof_manifest["output"]["sha256"])
                expected_ids = tabular_oof_ids(path, oof_manifest, data_dir)
            else:
                expected_ids = split_ids[source["id_set"]]
            frame = load_score_file(
                path,
                probability_column=source["probability_column"],
                split=source["split"],
                expected_ids=expected_ids,
                model_name=source["model_name"],
                allowed_versions=frozenset(source["allowed_versions"]),
                extra_columns=tuple(source["extra_columns"]),
            )
            if is_tabular_oof:
                check_fold_counts(frame, oof_manifest)
        except ValueError as error:
            raise ValueError(f"{name} ({source['path']}): {error}") from error
        frames[name] = frame
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the calibrated tabular validation scores.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--output", default=CALIBRATED_VAL_PATH, type=Path)
    args = parser.parse_args()
    result = run_calibrated_export(args.output, args.data_dir)
    platt = result["platt_mapping"]
    print(f"rows={result['rows']} platt slope={platt['slope']:.6f} intercept={platt['intercept']:.6f}")
    for key, row in result["validation_metrics"].items():
        print(f"{key}: mean_p={row['mean_p']:.4f} observed={row['prevalence']:.4f} "
              f"roc_auc={row['roc_auc']:.6f} pr_auc={row['pr_auc']:.6f} "
              f"brier={row['brier_score']:.6f} log_loss={row['log_loss']:.6f}")


if __name__ == "__main__":
    main()

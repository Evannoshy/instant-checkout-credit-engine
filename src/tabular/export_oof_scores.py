"""Out-of-fold (OOF) scores of the balanced logistic model on model_fit loans.

An out-of-fold score comes from a model that did not train on that loan. The
folds are the five expanding-window folds declared in
configs/tabular_xgboost_v1.toml [cv]: each fold fits a fresh preprocessing and
logistic pipeline on every model_fit month before its holdout block, then
scores that block. Imputation, scaling and encoding therefore learn only from
the fold's own training rows.

The earliest model_fit months (min_train_share of the rows) are never held
out. Those loans are training-only: no model in this scheme avoided fitting
them, so they get no score and are absent from the export.

Reuses, unchanged: the model_fit loader, fold builder and per-fold fitting
loop (src/tabular/xgboost_baseline.py), the approved logistic pipeline and
settings (src/tabular/logistic_baseline.py), and the shared metrics
(src/tabular/evaluate.py).

Output schema (one row per held-out loan, sorted by loan_id):
    loan_id, split="train", p_default_tabular_raw, fold_id (0-4, the same
    numbering as the XGBoost fold reports), model_name, model_version

The probabilities are raw: class_weight="balanced" lifts their level well
above the observed default rate. Use them for ranking, or calibrate them
before reading them as probabilities. The locked test split is never read.

Run from the repository root:
    python -m src.tabular.export_oof_scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.tabular import development, evaluate, logistic_baseline, preprocess, provenance
from src.tabular.xgboost_baseline import (
    DEFAULT_XGBOOST_CONFIG,
    Fold,
    build_folds,
    count_test_rows_available,
    cross_validate,
    describe_folds,
    load_model_fit,
    load_xgboost_config,
    validate_runtime,
)

OOF_VERSION = "tabular-logistic-v1-time-oof"
OOF_SPLIT = "train"
PROBABILITY_COLUMN = "p_default_tabular_raw"
OOF_COLUMNS = ("loan_id", "split", PROBABILITY_COLUMN, "fold_id", "model_name", "model_version")
FOLD_METRICS = ("positives", "roc_auc", "pr_auc", "brier_score", "log_loss")
EXPECTED_MODEL_FIT_ROWS = 59_353
EXPECTED_OOF_ROWS = 35_234
EXPECTED_TRAINING_ONLY_ROWS = EXPECTED_MODEL_FIT_ROWS - EXPECTED_OOF_ROWS  # 24,119

DEFAULT_OOF_DIR = preprocess.REPO_ROOT / "data" / "tabular"
DEFAULT_OOF_PATH = DEFAULT_OOF_DIR / "tabular_oof_train_v1.parquet"
DEFAULT_MANIFEST_PATH = DEFAULT_OOF_DIR / "tabular_oof_train_v1_manifest.json"


def holdout_ids(folds: tuple[Fold, ...]) -> pd.Index:
    """Every held-out loan_id across the folds, sorted."""
    return pd.Index(np.concatenate([fold.holdout_ids for fold in folds])).unique().sort_values()


def fold_assignment(folds: tuple[Fold, ...]) -> pd.Series:
    """fold_id of each held-out loan, indexed by loan_id and sorted."""
    parts = [pd.Series(fold_id, index=fold.holdout_ids) for fold_id, fold in enumerate(folds)]
    return pd.concat(parts).sort_index().rename("fold_id")


def check_fold_boundaries(folds: tuple[Fold, ...], issue_month: pd.Series) -> None:
    """Require every fold to fit only on earlier months and never on its own holdout loans.

    Also requires the holdout blocks to be disjoint, so each loan is scored once.
    Raises ValueError naming the first fold that breaks a rule.
    """
    for fold_id, fold in enumerate(folds):
        if overlap := fold.train_ids.intersection(fold.holdout_ids).size:
            raise ValueError(f"Fold {fold_id} fits {overlap} of the loans it scores")
        latest_fit_month = issue_month.loc[fold.train_ids].max()
        earliest_holdout_month = issue_month.loc[fold.holdout_ids].min()
        if latest_fit_month >= earliest_holdout_month:
            raise ValueError(
                f"Fold {fold_id} fits on {latest_fit_month}, which is not earlier than its "
                f"first held-out month {earliest_holdout_month}"
            )
    if sum(len(fold.holdout_ids) for fold in folds) != len(holdout_ids(folds)):
        raise ValueError("Holdout blocks overlap, so some loans would be scored twice")


def score_out_of_fold(
    make_pipeline: Callable[[], Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    folds: tuple[Fold, ...],
    issue_month: pd.Series,
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    """Fit a fresh pipeline per fold, score its holdout block, and return the OOF frame.

    Returns (frame in OOF_COLUMNS order sorted by loan_id, per-fold metrics).
    Raises ValueError if the folds break a time or leakage rule.
    """
    check_fold_boundaries(folds, issue_month)
    result = cross_validate(make_pipeline, features, target, folds)
    folds_by_id = fold_assignment(folds)
    frame = pd.DataFrame({
        "loan_id": folds_by_id.index.astype(str),
        "split": OOF_SPLIT,
        PROBABILITY_COLUMN: result.oof_predictions.loc[folds_by_id.index].to_numpy(dtype=np.float64),
        "fold_id": folds_by_id.to_numpy(dtype=np.int64),
        "model_name": logistic_baseline.MODEL_NAME,
        "model_version": OOF_VERSION,
    })
    return frame[list(OOF_COLUMNS)], result.fold_metrics


def validate_oof_frame(
    frame: pd.DataFrame, folds: tuple[Fold, ...], model_fit_ids: set[str]
) -> None:
    """Check the export against the schema, the folds and the model_fit role.

    Raises ValueError, in this order, for: wrong columns; duplicate loan_id;
    any loan outside model_fit (calibration, validation or test); coverage that
    differs from the held-out loans; a wrong split label or fold_id; mixed
    model metadata; or a probability that is missing, infinite or outside [0, 1].
    """
    if frame.columns.tolist() != list(OOF_COLUMNS):
        raise ValueError(f"OOF frame must have columns {list(OOF_COLUMNS)}; got {frame.columns.tolist()}")
    if frame["loan_id"].duplicated().any():
        raise ValueError("OOF frame has duplicate loan_id values")
    if outside := int((~frame["loan_id"].isin(model_fit_ids)).sum()):
        raise ValueError(
            f"OOF frame holds {outside} loans outside model_fit "
            "(calibration, validation or test loans are not allowed)"
        )
    expected = fold_assignment(folds)
    actual = frame.set_index("loan_id")
    missing = expected.index.difference(actual.index).size
    extra = actual.index.difference(expected.index).size
    if missing or extra:
        raise ValueError(f"OOF frame does not cover the held-out loans: {missing} missing, {extra} extra")
    if not frame["split"].eq(OOF_SPLIT).all():
        raise ValueError(f"OOF frame split must be {OOF_SPLIT!r} on every row")
    if not actual["fold_id"].loc[expected.index].eq(expected).all():
        raise ValueError("OOF frame has fold_id values that differ from the fold holding each loan out")
    for column in ("model_name", "model_version"):
        if frame[column].nunique() != 1:
            raise ValueError(f"OOF frame must have one {column}")
    values = pd.to_numeric(frame[PROBABILITY_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{PROBABILITY_COLUMN} must be finite and within [0, 1]")


def require_expected_counts(fit_rows: int, oof_rows: int) -> None:
    """The frozen data must give exactly the agreed model_fit and OOF row counts."""
    if (fit_rows, oof_rows) != (EXPECTED_MODEL_FIT_ROWS, EXPECTED_OOF_ROWS):
        raise ValueError(
            f"Expected {EXPECTED_MODEL_FIT_ROWS} model_fit and {EXPECTED_OOF_ROWS} OOF rows; "
            f"got {fit_rows} and {oof_rows}"
        )


def fold_table(
    folds: tuple[Fold, ...], issue_month: pd.Series, fold_metrics: tuple[dict[str, Any], ...]
) -> list[dict[str, Any]]:
    """Per fold: fit and holdout counts, both month ranges and the holdout metrics."""
    rows = []
    for description, fold, metrics in zip(describe_folds(folds), folds, fold_metrics, strict=True):
        fit_months = issue_month.loc[fold.train_ids]
        rows.append({
            "fold_id": description["fold"],
            "fit_rows": description["train_rows"],
            "fit_first_month": fit_months.min(),
            "fit_last_month": fit_months.max(),
            "holdout_rows": description["holdout_rows"],
            "holdout_first_month": description["holdout_first_month"],
            "holdout_last_month": description["holdout_last_month"],
            "holdout_metrics": {key: metrics[key] for key in FOLD_METRICS},
        })
    return rows


def row_summary(frame: pd.DataFrame, issue_month: pd.Series) -> dict[str, Any]:
    """Counts of scored and training-only model_fit loans, with the training-only months."""
    training_only = issue_month.index.difference(frame["loan_id"])
    return {
        "model_fit": len(issue_month),
        "oof_scored": len(frame),
        "training_only_unscored": len(training_only),
        "training_only_months": [
            issue_month.loc[training_only].min(), issue_month.loc[training_only].max()
        ],
    }


def input_checksums(data_dir: str | Path, raw_csv_path: str | Path, raw_csv_sha256: str) -> dict[str, str]:
    """SHA-256 of the raw CSV, the locked split files and every config the run reads."""
    return {
        provenance.display_path(raw_csv_path): raw_csv_sha256,
        **provenance.file_checksums([
            Path(data_dir) / "split_manifest.csv",
            Path(data_dir) / provenance.SPLIT_STATISTICS,
            preprocess.DEFAULT_FEATURE_CONFIG,
            logistic_baseline.DEFAULT_LOGISTIC_CONFIG,
            DEFAULT_XGBOOST_CONFIG,
            development.DEFAULT_DEVELOPMENT_CONFIG,
        ]),
    }


def build_manifest(
    frame: pd.DataFrame,
    target: pd.Series,
    folds_record: list[dict[str, Any]],
    rows: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    output_path: Path,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Provenance, settings, folds and pooled metrics: everything to trust and rerun the export."""
    _, stats = evaluate.load_manifest(data_dir)
    output = provenance.display_path(output_path)
    return {
        "oof_version": OOF_VERSION,
        "output": {
            "path": output,
            "sha256": provenance.file_checksums([output_path])[output],
            "rows": len(frame),
            "columns": list(OOF_COLUMNS),
        },
        "handoff": {"for": "Brandon", "path": output, "expected_rows": EXPECTED_OOF_ROWS, "key": "loan_id"},
        "base_model": {
            "model_name": logistic_baseline.MODEL_NAME,
            "frozen_version": logistic_baseline.BASELINE_VERSION,
            "config": provenance.display_path(logistic_baseline.DEFAULT_LOGISTIC_CONFIG),
            "settings": configs["logistic"]["model"],
            "feature_set_version": configs["features"]["feature_set_version"],
            "features": configs["features"]["model"],
        },
        "folds_source": {
            "config": provenance.display_path(DEFAULT_XGBOOST_CONFIG),
            "cv": configs["xgboost"]["cv"],
            "fold_id_numbering": "0-based, as in reports/tabular/xgboost_default_comparison.json",
        },
        "dataset_version": stats["dataset_version"],
        "split_version": stats.get("split_version"),
        "cohort": stats["cohort"],
        "rows": rows,
        "folds": folds_record,
        "pooled_oof_metrics": evaluate.evaluate_predictions(
            target.loc[frame["loan_id"]].to_numpy(), frame[PROBABILITY_COLUMN].to_numpy()
        ),
        "test_rows_available_to_model": count_test_rows_available(data_dir),
        "note": (
            "Raw probabilities from class_weight='balanced' models: their level sits well above "
            "the observed default rate. Rank with them, or calibrate before reading them as "
            "probabilities. The v1 Platt mapping belongs to the model_fit model, not these fold models."
        ),
        "reproduce": [
            "python -m pip install -r src/tabular/requirements.txt",
            "python -m src.tabular.export_oof_scores --raw-csv data/raw/loan.csv",
            "LENDINGCLUB_RAW_CSV=data/raw/loan.csv python -m pytest src/tabular/test_export_oof_scores.py -v",
        ],
    }


def run_export(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    output_path: str | Path = DEFAULT_OOF_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Verify inputs, score every fold, validate, then write the parquet and its manifest."""
    runtime = validate_runtime()
    raw_csv_sha256 = provenance.require_locked_raw_csv(data_dir, raw_csv_path)
    configs = {
        "logistic": logistic_baseline.load_logistic_config(),
        "xgboost": load_xgboost_config(),
        "features": preprocess.load_feature_config(),
    }

    features, target, issue_month = load_model_fit(data_dir, raw_csv_path)
    folds = build_folds(issue_month, configs["xgboost"])
    frame, fold_metrics = score_out_of_fold(
        lambda: logistic_baseline.build_pipeline(configs["features"], configs["logistic"]),
        features, target, folds, issue_month,
    )
    roles = development.assign_development_roles(data_dir)
    validate_oof_frame(frame, folds, set(roles.index[roles["role"].eq("model_fit")]))
    require_expected_counts(len(features), len(frame))

    output_path, manifest_path = Path(output_path), Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False, engine="pyarrow")
    manifest = {
        **build_manifest(
            frame, target, fold_table(folds, issue_month, fold_metrics),
            row_summary(frame, issue_month), configs, output_path, data_dir,
        ),
        "inputs_sha256": input_checksums(data_dir, raw_csv_path, raw_csv_sha256),
        "environment": runtime,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export time-aware out-of-fold logistic scores.")
    parser.add_argument("--data-dir", default=preprocess.DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--raw-csv", default=preprocess.DEFAULT_RAW_CSV, type=Path)
    parser.add_argument("--output", default=DEFAULT_OOF_PATH, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, type=Path)
    args = parser.parse_args()
    manifest = run_export(args.data_dir, args.raw_csv, args.output, args.manifest)
    for fold in manifest["folds"]:
        print(f"fold {fold['fold_id']}: fit {fold['fit_rows']} ({fold['fit_first_month']}.."
              f"{fold['fit_last_month']}) -> holdout {fold['holdout_rows']} "
              f"({fold['holdout_first_month']}..{fold['holdout_last_month']}) "
              f"roc_auc={fold['holdout_metrics']['roc_auc']:.4f}")
    print(f"wrote {manifest['output']['rows']} rows to {manifest['output']['path']}")


if __name__ == "__main__":
    main()

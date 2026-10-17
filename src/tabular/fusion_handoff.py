"""Calibrated tabular validation scores for the fusion handoff (fusion-handoff-v0).

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

from src.tabular import calibrate, evaluate, raw_predictions
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

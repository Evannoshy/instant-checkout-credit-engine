"""Constant base-rate baseline: predicts the training-set default rate for
every validation loan.

Establishes the minimum-viable floor prescribed by
docs/tabular-analyst-handbook.md §12.3's "minimum experiment ladder"
("Approve-all, decline-all and prevalence predictor" is step 1) that any
later logistic-regression or XGBoost model must beat. Fits only on train and
scores only validation. The shared loader returns no test rows or test-label
aggregates to this model pipeline.

Run from the repository root:
    python -m src.tabular.baseline
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from src.tabular.evaluate import (
    build_prediction_frame,
    evaluate_predictions,
    get_split_targets,
    load_manifest,
    training_default_rate,
    validate_prediction_frame,
)

BASELINE_VERSION = "tabular-baseline-constant-v1"
MODEL_NAME = "constant_base_rate"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "tabular"


def predict_constant(rate: float, n: int) -> np.ndarray:
    """Return an array of length n, every entry equal to `rate`."""
    return np.full(n, rate, dtype=np.float64)


def run(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Fit the constant baseline on train, score validation, and write artifacts.

    Loads the manifest once, computes the training default rate from train
    targets only, predicts that same rate for every validation loan,
    validates the resulting prediction frame against the agreed schema,
    evaluates it against validation targets, and writes both the prediction
    CSV and the metrics JSON to `output_dir`. Returns
    {"predictions": DataFrame, "metrics": dict}.

    Only train and validation rows are returned by the loader. The metrics JSON
    records the computed number of test rows available to this model pipeline;
    execution fails if that number is not zero.
    """
    manifest, stats = load_manifest(data_dir)
    test_rows_available = int(manifest["split"].eq("test").sum())
    if test_rows_available:
        raise RuntimeError("Locked test rows were exposed to the model-development pipeline")
    train_targets = get_split_targets(manifest, "train")
    validation_targets = get_split_targets(manifest, "validation")

    rate = training_default_rate(manifest)
    probabilities = predict_constant(rate, len(validation_targets))

    predictions = build_prediction_frame(
        loan_ids=validation_targets["loan_id"],
        split_name="validation",
        probabilities=probabilities,
        model_name=MODEL_NAME,
        model_version=BASELINE_VERSION,
    )
    validate_prediction_frame(predictions, expected_loan_ids=set(validation_targets["loan_id"]))

    validation_metrics = evaluate_predictions(
        validation_targets["target"].to_numpy(), predictions["p_default_tabular"].to_numpy()
    )

    metrics = {
        "baseline_version": BASELINE_VERSION,
        "model_name": MODEL_NAME,
        "model_version": BASELINE_VERSION,
        "dataset_version": stats["dataset_version"],
        "split_version": stats.get("split_version"),
        "cohort": stats["cohort"],
        "test_rows_available_to_model": test_rows_available,
        "train": {
            "rows": len(train_targets),
            "positives": int(train_targets["target"].sum()),
            "default_rate": rate,
        },
        "validation": validation_metrics,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "constant_baseline_predictions.csv", index=False)
    (output_dir / "constant_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    return {"predictions": predictions, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the constant base-rate tabular baseline.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

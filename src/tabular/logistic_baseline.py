"""Logistic-regression baseline: the first real tabular model, per
docs/tabular-analyst-handbook.md Section 12.3's minimum experiment ladder
(step 3, after the constant baseline).

Reuses two already-approved modules unchanged rather than re-implementing
either: `src/tabular/preprocess.py` (Brandon's `load_tabular_split`, which
turns raw loan rows into point-in-time-safe, deterministic feature rows for
the approved `tabular_features_v1` contract) and `src/tabular/evaluate.py`
(manifest loading, the shared flat prediction schema, and every ranking/
probability-quality metric). This file only adds what neither of those
already provides: the model pipeline itself, a confusion matrix at a fixed
0.5 threshold, coefficient extraction, and a calibration plot.

Fits only on the 86,293 training loans; scores only the 21,784 validation
loans. The locked test split is never requested: `evaluate.load_manifest`
already returns a manifest with test rows filtered out and test-label
statistics redacted, and `preprocess.load_tabular_split` independently
raises for `split="test"` before any file is opened.

No trained model is ever written to disk: fitting is fast and deterministic
(`random_state=42`, the default `lbfgs` solver), so every run reproduces the
same pipeline in memory rather than needing a saved artifact.

Run from the repository root:
    python -m src.tabular.logistic_baseline
"""

from __future__ import annotations

import argparse
import json
import platform
import tomllib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.tabular import evaluate, preprocess

BASELINE_VERSION = "tabular-logistic-v1"
MODEL_NAME = "logistic_regression"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOGISTIC_CONFIG = REPO_ROOT / "configs" / "tabular_logistic_v1.toml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "tabular"
REQUIRED_LOGISTIC_CONFIG_KEYS = {"C", "class_weight", "max_iter", "random_state"}


def load_logistic_config(path: str | Path = DEFAULT_LOGISTIC_CONFIG) -> dict[str, Any]:
    """Load and validate the logistic hyperparameter config.

    Raises ValueError if the [model] table or any of its four required keys
    (C, class_weight, max_iter, random_state) is missing. Deliberately
    minimal: unlike preprocess.load_feature_config, there is no derived- or
    prohibited-column machinery to validate here, only four fixed
    hyperparameters that Week 2 intentionally does not grid-search.
    """
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if "model" not in config:
        raise ValueError("Logistic config is missing the [model] table")
    if missing := REQUIRED_LOGISTIC_CONFIG_KEYS - set(config["model"]):
        raise ValueError(f"Logistic config [model] is missing keys: {sorted(missing)}")
    return config


def build_pipeline(feature_config: dict[str, Any], logistic_config: dict[str, Any]) -> Pipeline:
    """Build the untrained pipeline for the approved tabular_features_v1 contract.

    Numeric columns: median imputation, then standard scaling. Categorical
    columns: a separate "missing" category, then one-hot encoding with
    handle_unknown="ignore" so a category level absent from training never
    crashes prediction on validation. Both imputers/scaler/encoder are fit
    only when `.fit()` is later called on training rows -- this function
    only assembles the untrained structure. Column lists come directly from
    feature_config["model"], not a hardcoded list, so a future feature-set
    version needs no change here.
    """
    numeric_features = feature_config["model"]["numeric"]
    categorical_features = feature_config["model"]["categorical"]

    numeric_transform = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_transform = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_transform, numeric_features),
        ("categorical", categorical_transform, categorical_features),
    ])
    classifier = LogisticRegression(**logistic_config["model"])
    return Pipeline([("preprocess", preprocessor), ("classify", classifier)])


def extract_coefficients(pipeline: Pipeline, feature_config: dict[str, Any]) -> pd.DataFrame:
    """Return every fitted coefficient, sorted descending by value.

    Positive coefficients are associated with higher predicted default risk
    (target=1); negative with lower risk -- associations from the fitted
    model, not causal effects. Column names come from
    ColumnTransformer.get_feature_names_out() (e.g.
    "categorical__home_ownership_OTHER"), so one-hot-expanded categorical
    levels are named explicitly rather than collapsed. `feature_config` is
    accepted for interface symmetry with the rest of this module but is not
    otherwise needed, since the fitted pipeline already carries every name.
    """
    del feature_config
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = pipeline.named_steps["classify"].coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, output_path: str | Path) -> None:
    """Save a reliability-diagram PNG using scikit-learn's CalibrationDisplay.

    Reuses an already-tested library tool instead of hand-rolling binning
    math. The "Agg" backend is selected at import time since this module runs
    headless (no display in this environment).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax)
    ax.set_title("Logistic baseline calibration (validation)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    output_dir: str | Path,
    *,
    feature_config_path: str | Path = preprocess.DEFAULT_FEATURE_CONFIG,
    logistic_config_path: str | Path = DEFAULT_LOGISTIC_CONFIG,
) -> dict[str, Any]:
    """Fit the logistic pipeline on train, score validation, and write every artifact.

    Only "train" and "validation" are ever requested from preprocess.load_tabular_split.
    The written metrics JSON records "test_rows_available_to_model" -- computed from
    evaluate.load_manifest's already test-filtered manifest, not a hard-coded claim --
    as a written, machine-checkable confirmation that the locked split was never used.
    """
    manifest, stats = evaluate.load_manifest(data_dir)
    feature_config = preprocess.load_feature_config(feature_config_path)
    logistic_config = load_logistic_config(logistic_config_path)

    train_features, train_target = preprocess.load_tabular_split(
        data_dir, raw_csv_path, "train", feature_config_path
    )
    validation_features, validation_target = preprocess.load_tabular_split(
        data_dir, raw_csv_path, "validation", feature_config_path
    )

    pipeline = build_pipeline(feature_config, logistic_config)
    pipeline.fit(train_features, train_target)

    validation_probabilities = pipeline.predict_proba(validation_features)[:, 1]

    predictions = evaluate.build_prediction_frame(
        loan_ids=validation_features.index,
        split_name="validation",
        probabilities=validation_probabilities,
        model_name=MODEL_NAME,
        model_version=BASELINE_VERSION,
    )
    evaluate.validate_prediction_frame(
        predictions, expected_loan_ids=set(validation_features.index)
    )

    validation_metrics = evaluate.evaluate_predictions(
        validation_target.to_numpy(), validation_probabilities
    )
    predicted_labels = (validation_probabilities >= 0.5).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        validation_target.to_numpy(), predicted_labels, labels=[0, 1]
    ).ravel()
    validation_metrics["confusion_matrix_at_0.5"] = {
        "tn": int(true_negatives),
        "fp": int(false_positives),
        "fn": int(false_negatives),
        "tp": int(true_positives),
    }

    coefficients = extract_coefficients(pipeline, feature_config)

    metrics = {
        "baseline_version": BASELINE_VERSION,
        "model_name": MODEL_NAME,
        "model_version": BASELINE_VERSION,
        "dataset_version": stats["dataset_version"],
        "split_version": stats.get("split_version"),
        "cohort": stats["cohort"],
        "test_rows_available_to_model": int(manifest["split"].eq("test").sum()),
        "logistic_config": logistic_config["model"],
        "train": {
            "rows": len(train_target),
            "positives": int(train_target.sum()),
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
    predictions.to_csv(output_dir / "logistic_baseline_predictions.csv", index=False)
    (output_dir / "logistic_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    coefficients.to_csv(output_dir / "logistic_coefficients.csv", index=False)
    plot_calibration(
        validation_target.to_numpy(),
        validation_probabilities,
        output_dir / "logistic_calibration.png",
    )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "coefficients": coefficients,
        "pipeline": pipeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the logistic-regression tabular baseline."
    )
    parser.add_argument("--data-dir", default=preprocess.DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--raw-csv", default=preprocess.DEFAULT_RAW_CSV, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--feature-config", default=preprocess.DEFAULT_FEATURE_CONFIG, type=Path)
    parser.add_argument("--logistic-config", default=DEFAULT_LOGISTIC_CONFIG, type=Path)
    args = parser.parse_args()
    result = run(
        args.data_dir,
        args.raw_csv,
        args.output_dir,
        feature_config_path=args.feature_config,
        logistic_config_path=args.logistic_config,
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

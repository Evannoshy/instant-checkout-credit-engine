"""Shared tabular evaluation utilities: manifest loading, target retrieval,
prediction-frame construction/validation, and evaluation metrics.

Reusable by the constant baseline and by future model pipelines (logistic
regression, XGBoost) so every tabular model shares one manifest loader, one
prediction schema and one metric computation. Intentionally does not read
data/raw/loan.csv: tabular evaluation only needs loan_id/target, not free
text, and the raw CSV is gitignored and may not exist on a fresh checkout.

Mirrors the manifest-integrity checks and fail-loud philosophy of
src/nlp/preprocess.py, including reusing its exact split-lock error message,
but is scoped to structured data and does not inherit any NLP-specific
invariant (text_available, description non-emptiness).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

REQUIRED_MANIFEST_COLUMNS = {
    "loan_id",
    "source_row_number",
    "split",
    "issue_month",
    "target",
    "text_available",
    "cohort",
    "dataset_version",
}
ALLOWED_SPLIT_VALUES = ("train", "validation", "test")
LOADABLE_SPLITS = ("train", "validation")
EXPECTED_COHORT = "real_text_matured_v1"
TEST_SPLIT_LOCKED_MESSAGE = "Only train and validation are available; the final test set is locked."
PREDICTION_COLUMNS = ("loan_id", "split", "p_default_tabular", "model_name", "model_version")


def load_manifest(data_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the model-development portion of the locked split manifest.

    Structural checks cover every split, but target validation and the returned
    frame are restricted to train and validation. Test rows and test-label
    aggregates are removed before anything is returned to model code. Raises
    ValueError with a specific message for the first violation found. Never
    opens data/raw/loan.csv.

    Returns (development_manifest, redacted_stats). The source manifest is the
    repository's locked split registry and currently contains all split rows;
    this boundary prevents its test targets from being exposed downstream.
    """
    data_dir = Path(data_dir)
    manifest = pd.read_csv(data_dir / "split_manifest.csv", dtype={"loan_id": "string"})

    if missing := REQUIRED_MANIFEST_COLUMNS - set(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    structural_columns = list(REQUIRED_MANIFEST_COLUMNS - {"target"})
    if manifest[structural_columns].isna().any().any():
        raise ValueError("Manifest contains missing required metadata")
    if not manifest["split"].isin(ALLOWED_SPLIT_VALUES).all():
        raise ValueError("Manifest has an unknown split")
    if manifest["loan_id"].duplicated().any() or manifest["source_row_number"].duplicated().any():
        raise ValueError("Manifest IDs and source row numbers must be unique across splits")

    numbers = manifest["source_row_number"]
    if (numbers < 1).any() or (numbers % 1 != 0).any():
        raise ValueError("source_row_number must contain positive one-based integers")
    expected_ids = numbers.map(lambda n: f"lc_{int(n):09d}")
    if not manifest["loan_id"].eq(expected_ids).all():
        raise ValueError("loan_id does not match the manifest source-row convention")
    if not manifest["text_available"].isin([0, 1]).all():
        raise ValueError("Manifest text_available must be binary")

    development_mask = manifest["split"].isin(LOADABLE_SPLITS)
    development_manifest = manifest.loc[development_mask].copy()
    development_targets = pd.to_numeric(development_manifest["target"], errors="coerce")
    if development_targets.isna().any() or not development_targets.isin([0, 1]).all():
        raise ValueError("Train and validation targets must be binary")
    development_manifest["target"] = development_targets.astype("int8")

    with (data_dir / "split_statistics.json").open(encoding="utf-8") as stream:
        stats = json.load(stream)
    if len(manifest) != stats["eligible_rows"]:
        raise ValueError("Manifest count differs from split_statistics.json")
    for split_name, split_info in stats["splits"].items():
        if int(manifest["split"].eq(split_name).sum()) != split_info["rows"]:
            raise ValueError(f"Unexpected manifest count for {split_name}")
    if not manifest["cohort"].eq(EXPECTED_COHORT).all():
        raise ValueError(f"Loader expects the {EXPECTED_COHORT} cohort")
    if not manifest["dataset_version"].eq(stats["dataset_version"]).all():
        raise ValueError("Manifest dataset version disagrees with split statistics")

    redacted_stats = copy.deepcopy(stats)
    test_stats = redacted_stats.get("splits", {}).get("test")
    if isinstance(test_stats, dict) and "rows" in test_stats:
        redacted_stats["splits"]["test"] = {"rows": test_stats["rows"]}

    return development_manifest.reset_index(drop=True), redacted_stats


def get_split_targets(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return a loan_id/target frame for one split.

    Only "train" and "validation" are available. "test" raises ValueError
    with the same message used by src/nlp/preprocess.load_original_split, so
    the two loaders fail identically and greppably on the locked split. Any
    other split name is rejected as unknown.
    """
    if split == "test":
        raise ValueError(TEST_SPLIT_LOCKED_MESSAGE)
    if split not in LOADABLE_SPLITS:
        raise ValueError(f"Unknown split: {split!r}")
    selected = manifest.loc[manifest["split"].eq(split), ["loan_id", "target"]]
    return selected.reset_index(drop=True)


def training_default_rate(manifest: pd.DataFrame) -> float:
    """Return the training-set fraction of target == 1 (the base/prevalence rate)."""
    targets = get_split_targets(manifest, "train")
    return float(targets["target"].mean())


def build_prediction_frame(
    loan_ids: pd.Series | np.ndarray,
    split_name: str,
    probabilities: np.ndarray,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    """Assemble the agreed flat prediction schema for any tabular model.

    Generic over the model that produced `probabilities`: the constant
    baseline and future logistic-regression/XGBoost models call this with the
    same signature, so downstream validation/evaluation code needs no
    per-model branching. Columns are always emitted in PREDICTION_COLUMNS
    order, regardless of dict/kwarg insertion order.
    """
    loan_id_series = pd.Series(loan_ids).reset_index(drop=True)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if len(loan_id_series) != len(probability_array):
        raise ValueError(
            f"loan_ids has {len(loan_id_series)} entries but probabilities has "
            f"{len(probability_array)}; they must be the same length"
        )
    frame = pd.DataFrame(
        {
            "loan_id": loan_id_series,
            "split": split_name,
            "p_default_tabular": probability_array,
            "model_name": model_name,
            "model_version": model_version,
        }
    )
    return frame[list(PREDICTION_COLUMNS)]


def validate_prediction_frame(frame: pd.DataFrame, expected_loan_ids: set[str]) -> None:
    """Validate a per-loan prediction table against the agreed schema.

    Raises ValueError, checked in this order, for: a wrong column set/order
    (checked first, since a malformed schema makes every other check
    meaningless); duplicate loan_id values; incomplete id coverage (missing
    and/or extra ids relative to expected_loan_ids, both counts reported);
    any null p_default_tabular value; a non-numeric p_default_tabular column;
    or any p_default_tabular value outside [0, 1].
    """
    if frame.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError(
            f"Prediction frame must have columns {list(PREDICTION_COLUMNS)} in that "
            f"order; got {frame.columns.tolist()}"
        )
    if frame["loan_id"].duplicated().any():
        raise ValueError("Prediction frame has duplicate loan_id values")

    actual_loan_ids = set(frame["loan_id"])
    missing_ids = expected_loan_ids - actual_loan_ids
    extra_ids = actual_loan_ids - expected_loan_ids
    if missing_ids or extra_ids:
        raise ValueError(
            "Prediction frame does not cover the expected loan_id set exactly: "
            f"{len(missing_ids)} missing, {len(extra_ids)} extra"
        )

    if frame["p_default_tabular"].isna().any():
        raise ValueError("Prediction frame has missing p_default_tabular values")
    if not pd.api.types.is_numeric_dtype(frame["p_default_tabular"]):
        raise ValueError("p_default_tabular must be numeric")

    values = frame["p_default_tabular"].to_numpy(dtype=np.float64)
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p_default_tabular values must lie within [0, 1]")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute ranking and probability-quality metrics for one split.

    Mirrors the key shape of the NLP track's baseline evaluate() function
    (rows, positives, prevalence, roc_auc, roc_auc_baseline, pr_auc,
    pr_auc_baseline) and adds brier_score and log_loss, plus
    f1_at_0.5/f1_best/f1_best_threshold, for cross-track consistency.

    Raises ValueError if y_true has fewer than two classes: scikit-learn's
    roc_auc_score would otherwise silently emit a warning and return NaN,
    which conflicts with this repository's fail-loud convention. Passes
    labels=[0, 1] explicitly to log_loss for the same reason.

    For a constant-score y_pred (as produced by the constant baseline),
    roc_auc == 0.5 and pr_auc == prevalence exactly, and f1_best/
    f1_best_threshold degenerate to the single point equal to the constant
    value itself. That is a correct but non-informative result for a model
    with no discrimination, not a bug in this function.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1) to compute ranking metrics")

    rows = len(y_true)
    positives = int(np.sum(y_true == 1))
    prevalence = positives / rows

    roc_auc = float(roc_auc_score(y_true, y_pred))
    pr_auc = float(average_precision_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_pred))
    loss = float(log_loss(y_true, y_pred, labels=[0, 1]))

    f1_at_half = float(f1_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0))

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1_scores = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    best_index = int(np.argmax(f1_scores))

    return {
        "rows": rows,
        "positives": positives,
        "prevalence": prevalence,
        "roc_auc": roc_auc,
        "roc_auc_baseline": 0.5,
        "pr_auc": pr_auc,
        "pr_auc_baseline": prevalence,
        "brier_score": brier,
        "log_loss": loss,
        "f1_at_0.5": f1_at_half,
        "f1_best": float(f1_scores[best_index]),
        "f1_best_threshold": float(thresholds[best_index]),
    }

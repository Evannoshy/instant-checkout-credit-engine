"""XGBoost challenger: pipeline, time-aware cross-validation and experiment runners.

Step 4 of docs/tabular-analyst-handbook.md Section 12.3 (small/default
XGBoost), plus the shared machinery the feature-family ablation, the bounded
search (src/tabular/xgboost_search.py) and the lexical experiment
(src/tabular/lexical_ablation.py) all reuse, so every experiment scores the
same rows with the same folds and the same metrics.

Reuses, unchanged: Brandon's role loader and ablation families
(src/tabular/development.py), the approved preprocessing contract
(src/tabular/preprocess.py), the shared prediction schema and metrics
(src/tabular/evaluate.py) and the logistic pipeline
(src/tabular/logistic_baseline.py).

Data boundary:
- Every model is fitted on ``model_fit`` rows only. Settings are compared with
  expanding-window folds inside ``model_fit``, never on calibration or
  validation rows.
- The locked test split is never requested; the loaders reject it before any
  file is opened, and every written report records
  ``test_rows_available_to_model`` from the test-filtered manifest.

Missing values: numeric columns reach XGBoost with NaN intact (it learns a
default branch for missing values), and categorical columns use the same
"missing" level as the logistic pipeline, so no loan is dropped. The
transformer output is forced dense: XGBoost treats entries absent from a
sparse matrix as missing, which would silently turn one-hot zeros into NaN.
"""

from __future__ import annotations

import platform
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.tabular import development, evaluate, preprocess

MODEL_NAME = "xgboost"
REPO_ROOT = preprocess.REPO_ROOT
DEFAULT_XGBOOST_CONFIG = REPO_ROOT / "configs" / "tabular_xgboost_v1.toml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "tabular"

REQUIRED_XGBOOST_CONFIG_KEYS = {"model_version", "model", "cv", "search", "evaluation"}
REQUIRED_MODEL_KEYS = {
    "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
    "min_child_weight", "reg_lambda", "reg_alpha", "random_state",
}
SUMMARY_METRICS = ("roc_auc", "pr_auc", "brier_score", "log_loss")
METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier_score": brier_score_loss,
    "log_loss": lambda y_true, y_pred: log_loss(y_true, y_pred, labels=[0, 1]),
}
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold: train on every month before the holdout block."""

    train_ids: pd.Index
    holdout_ids: pd.Index
    holdout_first_month: str
    holdout_last_month: str


@dataclass(frozen=True)
class CrossValidationResult:
    """Per-fold metrics, their mean/std summary, and pooled out-of-fold predictions."""

    fold_metrics: tuple[dict[str, Any], ...]
    summary: dict[str, dict[str, float]]
    oof_predictions: pd.Series


def load_xgboost_config(path: str | Path = DEFAULT_XGBOOST_CONFIG) -> dict[str, Any]:
    """Load the XGBoost config, failing loudly on missing tables, keys or bad CV settings."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if missing := REQUIRED_XGBOOST_CONFIG_KEYS - set(config):
        raise ValueError(f"XGBoost config is missing keys: {sorted(missing)}")
    if missing := REQUIRED_MODEL_KEYS - set(config["model"]):
        raise ValueError(f"XGBoost config [model] is missing keys: {sorted(missing)}")
    n_folds = config["cv"].get("n_folds")
    share = config["cv"].get("min_train_share")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError(f"[cv] n_folds must be an integer >= 2; got {n_folds!r}")
    if not isinstance(share, (int, float)) or not 0 < share < 1:
        raise ValueError(f"[cv] min_train_share must lie strictly between 0 and 1; got {share!r}")
    return config


def build_xgboost_pipeline(feature_config: dict[str, Any], params: dict[str, Any]) -> Pipeline:
    """Build the untrained XGBoost pipeline for the configured feature lists.

    Columns outside feature_config["model"] are dropped (remainder="drop"),
    which is what makes the family ablations valid: a removed family cannot
    reach the model through a passthrough remainder.
    """
    categorical_transform = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", list(feature_config["model"]["numeric"])),
            ("categorical", categorical_transform, list(feature_config["model"]["categorical"])),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    classifier = xgboost.XGBClassifier(**params)
    return Pipeline([("preprocess", preprocessor), ("classify", classifier)])


def _month_blocks(issue_month: pd.Series, n_folds: int, min_train_share: float) -> pd.Series:
    """Map each month to a holdout block 0..n_folds-1, or -1 for always-train months.

    A month's position is the cumulative row share at its midpoint, so a month
    is never split and blocks hold roughly equal row counts.
    """
    counts = issue_month.value_counts().sort_index()
    midpoint_share = (counts.cumsum() - counts / 2) / counts.sum()
    scaled = (midpoint_share - min_train_share) / (1 - min_train_share) * n_folds
    blocks = np.floor(scaled).clip(upper=n_folds - 1).astype(int)
    return blocks.where(midpoint_share > min_train_share, -1)


def time_aware_folds(
    issue_month: pd.Series, n_folds: int, min_train_share: float
) -> tuple[Fold, ...]:
    """Expanding-window folds over ``issue_month`` (indexed by loan_id, "YYYY-MM" values).

    Raises ValueError if the index is not unique or the months are too few to
    give every fold a non-empty training set and holdout block.
    """
    if issue_month.index.has_duplicates:
        raise ValueError("issue_month index must hold unique loan_id values")
    month_block = _month_blocks(issue_month, n_folds, min_train_share)
    row_block = issue_month.map(month_block)

    folds: list[Fold] = []
    for block in range(n_folds):
        holdout_months = month_block.index[month_block.eq(block)]
        if holdout_months.empty:
            raise ValueError(f"Too few months for {n_folds} folds: block {block} is empty")
        first, last = holdout_months.min(), holdout_months.max()
        train_ids = issue_month.index[issue_month.lt(first)]
        holdout_ids = issue_month.index[row_block.eq(block)]
        if train_ids.empty:
            raise ValueError(f"Fold {block} has no training months before {first}")
        folds.append(Fold(train_ids.sort_values(), holdout_ids.sort_values(), first, last))
    return tuple(folds)


def _summarise(fold_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation of each summary metric across folds."""
    summary = {}
    for metric in SUMMARY_METRICS:
        values = np.array([fold[metric] for fold in fold_metrics], dtype=np.float64)
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return summary


def cross_validate(
    make_pipeline: Callable[[], Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    folds: tuple[Fold, ...],
) -> CrossValidationResult:
    """Fit a fresh pipeline per fold and score its holdout block with the shared metrics."""
    if not features.index.equals(target.index):
        raise ValueError("Feature and target rows are not aligned on loan_id")
    fold_metrics: list[dict[str, Any]] = []
    oof_parts: list[pd.Series] = []
    for number, fold in enumerate(folds):
        pipeline = make_pipeline()
        pipeline.fit(features.loc[fold.train_ids], target.loc[fold.train_ids])
        probabilities = pipeline.predict_proba(features.loc[fold.holdout_ids])[:, 1]
        metrics = evaluate.evaluate_predictions(
            target.loc[fold.holdout_ids].to_numpy(), probabilities
        )
        fold_metrics.append({
            "fold": number,
            "train_rows": len(fold.train_ids),
            "holdout_first_month": fold.holdout_first_month,
            "holdout_last_month": fold.holdout_last_month,
            **metrics,
        })
        oof_parts.append(pd.Series(probabilities, index=fold.holdout_ids))
    oof = pd.concat(oof_parts).sort_index()
    return CrossValidationResult(tuple(fold_metrics), _summarise(fold_metrics), oof)


def paired_bootstrap_delta(
    y_true: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Point estimate and 95% percentile CI of metric(candidate) - metric(baseline).

    Both models are scored on the same resampled rows each time (paired), so
    the interval reflects the difference between them rather than the noise
    in each. Resamples that contain only one class are redrawn by skipping.
    """
    score = METRIC_FUNCTIONS[metric]
    y_true = np.asarray(y_true)
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    while len(deltas) < resamples:
        rows = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[rows]).size < 2:
            continue
        deltas.append(score(y_true[rows], candidate[rows]) - score(y_true[rows], baseline[rows]))
    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    low, high = np.percentile(deltas, [tail, 100 - tail])
    return {
        "delta": float(score(y_true, candidate) - score(y_true, baseline)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def load_model_fit(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    feature_config_path: str | Path = preprocess.DEFAULT_FEATURE_CONFIG,
    development_config_path: str | Path = development.DEFAULT_DEVELOPMENT_CONFIG,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return model_fit (features, target, issue_month), all indexed by the same loan_id."""
    features, target = development.load_tabular_role(
        data_dir, raw_csv_path, "model_fit", feature_config_path, development_config_path
    )
    roles = development.assign_development_roles(data_dir, development_config_path)
    selected = roles.reindex(features.index)
    if not selected["role"].eq("model_fit").all():
        raise ValueError("Loaded model_fit rows include loans outside the model_fit role")
    return features, target, selected["issue_month"]


def build_folds(issue_month: pd.Series, xgboost_config: dict[str, Any]) -> tuple[Fold, ...]:
    """Folds from the [cv] settings of the XGBoost config."""
    cv = xgboost_config["cv"]
    return time_aware_folds(issue_month, cv["n_folds"], cv["min_train_share"])


def describe_folds(folds: tuple[Fold, ...]) -> list[dict[str, Any]]:
    """JSON-ready description of each fold's months and row counts."""
    return [
        {
            "fold": number,
            "train_rows": len(fold.train_ids),
            "holdout_rows": len(fold.holdout_ids),
            "holdout_first_month": fold.holdout_first_month,
            "holdout_last_month": fold.holdout_last_month,
        }
        for number, fold in enumerate(folds)
    ]


def count_test_rows_available(data_dir: str | Path) -> int:
    """Count test rows in the manifest handed to model code (always 0 by construction)."""
    manifest, _ = evaluate.load_manifest(data_dir)
    return int(manifest["split"].eq("test").sum())


def environment() -> dict[str, str]:
    """Library versions recorded beside every result."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }

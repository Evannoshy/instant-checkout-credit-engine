"""Analyst 2 Week 2: hyperparameter search and out-of-fold probability export.

Run from the repository root:
    python -m src.nlp.evaluate_baseline

Builds on Week 1's ``src/nlp/baseline_tfidf.py`` rather than restating it: the
split loader and the downstream field-label strip are imported, so the text this
tunes on is exactly the text the Week 1 baseline scored. ``preprocess.py`` stays
untouched, keeping the payload byte-identical across the TF-IDF and transformer
tracks.

Metrics come from ``src/tabular/evaluate.py``. That module's
``evaluate_predictions`` already returns ROC-AUC, PR-AUC, Brier, log loss and F1,
and its docstring states it mirrors the NLP baseline's shape for cross-track
consistency, so reusing it keeps one metric implementation for the project
rather than two that can drift. Nothing NLP-specific is imported back into it.

Three properties are load-bearing and easy to get wrong:

* **The vectoriser lives inside the Pipeline.** Fitting TF-IDF on the full
  training split before cross-validating would leak IDF statistics across folds
  and inflate every number here. Because it is a Pipeline step, scikit-learn
  refits it on each fold's training portion only.
* **Selection never sees validation.** The grid is scored by cross-validation on
  training rows. Validation is scored once, after the winner is fixed.
* **Training probabilities are out-of-fold.** Each training row is scored by a
  model that never saw it, which is what decision D-014 requires of training
  predictions consumed by fusion. Validation probabilities come from a single
  fit on all training rows, so the two files are different quantities and are
  written separately.

The exported probability is an **uncalibrated** PD in the sense of decision
D-006. ``class_weight="balanced"`` follows repository precedent and buys ranking
at the cost of probability scale, so the Brier score is reported against the
constant-baseline floor and a like-for-like unweighted diagnostic is recorded.
Consumers doing logistic stacking are unaffected by a monotone rescaling;
consumers reading these as probabilities are not, hence the label.

The locked test split is never requested; ``load_original_split`` refuses it.
"""

from __future__ import annotations

import argparse
import json
import platform
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.nlp.baseline_tfidf import load_baseline_split
from src.tabular.evaluate import evaluate_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "nlp_tfidf_v2.toml"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "nlp"
DEFAULT_PARQUET_DIR = REPO_ROOT / "data" / "nlp"

PROBABILITY_COLUMN = "prob_default_tfidf"
PREDICTION_COLUMNS = ("loan_id", "split", PROBABILITY_COLUMN, "model_name", "model_version")
REQUIRED_CONFIG_TABLES = ("model", "grid", "cv", "week1_reference", "bootstrap")
REQUIRED_MODEL_KEYS = {"class_weight", "max_iter", "random_state", "sublinear_tf"}
REQUIRED_GRID_KEYS = {"max_features", "ngram_range", "C"}


def load_tuning_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the tuning configuration, failing loudly on gaps."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if missing := [table for table in REQUIRED_CONFIG_TABLES if table not in config]:
        raise ValueError(f"Tuning config is missing tables: {missing}")
    if missing_model := REQUIRED_MODEL_KEYS - set(config["model"]):
        raise ValueError(f"Tuning config [model] is missing keys: {sorted(missing_model)}")
    if missing_grid := REQUIRED_GRID_KEYS - set(config["grid"]):
        raise ValueError(f"Tuning config [grid] is missing keys: {sorted(missing_grid)}")
    for key in ("model_version", "model_name"):
        if not config.get(key):
            raise ValueError(f"Tuning config is missing {key}")
    return config


def build_pipeline(
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    C: float,
    sublinear_tf: bool,
    class_weight: str | None,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Assemble the untrained vectoriser-plus-classifier pipeline.

    Keeping TF-IDF as a Pipeline step is what makes cross-validation honest: the
    vocabulary and IDF weights are refit per fold rather than shared across them.
    """
    return Pipeline(
        [
            (
                "vectorize",
                TfidfVectorizer(
                    ngram_range=tuple(ngram_range),
                    max_features=max_features,
                    stop_words="english",
                    sublinear_tf=sublinear_tf,
                ),
            ),
            (
                "classify",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def pipeline_from_params(params: dict[str, Any], model_config: dict[str, Any]) -> Pipeline:
    """Build a pipeline from one grid point plus the fixed model settings."""
    return build_pipeline(
        max_features=params["max_features"],
        ngram_range=params["ngram_range"],
        C=params["C"],
        sublinear_tf=params.get("sublinear_tf", model_config["sublinear_tf"]),
        class_weight=model_config["class_weight"],
        max_iter=model_config["max_iter"],
        random_state=model_config["random_state"],
    )


def make_cv(cv_config: dict[str, Any]) -> StratifiedKFold:
    """Build the one fold definition used for both selection and out-of-fold scoring.

    The same object is reused so the folds that chose the winner are the folds
    that produce its out-of-fold probabilities.
    """
    return StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"] if cv_config["shuffle"] else None,
    )


def run_grid_search(
    train_text: pd.Series,
    y_train: np.ndarray,
    config: dict[str, Any],
    cv: StratifiedKFold,
    *,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every grid point by cross-validation on training rows only.

    ``refit=False``: the winner is rebuilt explicitly below so the fitted object
    and its out-of-fold probabilities are produced by visibly the same code path.
    """
    grid = config["grid"]
    base = build_pipeline(
        max_features=grid["max_features"][0],
        ngram_range=tuple(grid["ngram_range"][0]),
        C=grid["C"][0],
        sublinear_tf=config["model"]["sublinear_tf"],
        class_weight=config["model"]["class_weight"],
        max_iter=config["model"]["max_iter"],
        random_state=config["model"]["random_state"],
    )
    search = GridSearchCV(
        base,
        param_grid={
            "vectorize__max_features": grid["max_features"],
            "vectorize__ngram_range": [tuple(value) for value in grid["ngram_range"]],
            "classify__C": grid["C"],
        },
        scoring=config["cv"]["scoring"],
        cv=cv,
        n_jobs=n_jobs,
        refit=False,
        return_train_score=False,
    )
    search.fit(train_text, y_train)

    results = pd.DataFrame(
        {
            "max_features": [p["vectorize__max_features"] for p in search.cv_results_["params"]],
            "ngram_range": [
                str(p["vectorize__ngram_range"]) for p in search.cv_results_["params"]
            ],
            "C": [p["classify__C"] for p in search.cv_results_["params"]],
            "mean_cv_pr_auc": search.cv_results_["mean_test_score"],
            "std_cv_pr_auc": search.cv_results_["std_test_score"],
            "rank": search.cv_results_["rank_test_score"],
        }
    ).sort_values("rank").reset_index(drop=True)

    winner = search.cv_results_["params"][int(np.argmin(search.cv_results_["rank_test_score"]))]
    best_params = {
        "max_features": int(winner["vectorize__max_features"]),
        "ngram_range": tuple(winner["vectorize__ngram_range"]),
        "C": float(winner["classify__C"]),
        "sublinear_tf": bool(config["model"]["sublinear_tf"]),
    }
    return results, best_params


def out_of_fold_probabilities(
    train_text: pd.Series,
    y_train: np.ndarray,
    pipeline: Pipeline,
    cv: StratifiedKFold,
    *,
    n_jobs: int = -1,
) -> np.ndarray:
    """Score every training row with a model fitted without it.

    ``cross_val_predict`` refits the whole pipeline, vectoriser included, on each
    fold's training portion, so no row contributes to the vocabulary, the IDF
    weights or the coefficients used to score it.
    """
    probabilities = cross_val_predict(
        pipeline, train_text, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs
    )[:, 1]
    if len(probabilities) != len(y_train):
        raise ValueError("Out-of-fold probabilities do not cover every training row")
    if not np.isfinite(probabilities).all():
        raise ValueError("Out-of-fold probabilities contain non-finite values")
    return probabilities


def build_probability_frame(
    loan_ids: pd.Series,
    split_name: str,
    probabilities: np.ndarray,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    """Assemble the versioned per-loan probability table.

    Mirrors the checks and column ordering of
    ``src.tabular.evaluate.build_prediction_frame``. That function is not reused
    directly because it hardcodes ``p_default_tabular``; naming a TF-IDF score
    after the tabular model would be worse than the small duplication. The
    ``model_name``/``model_version`` columns are what decision D-014 means by
    versioned probabilities keyed by locked ``loan_id``.
    """
    ids = pd.Series(loan_ids, dtype="string").reset_index(drop=True)
    values = np.asarray(probabilities, dtype=np.float64)
    if len(ids) != len(values):
        raise ValueError(
            f"loan_ids has {len(ids)} entries but probabilities has {len(values)}; "
            "they must be the same length"
        )
    frame = pd.DataFrame(
        {
            "loan_id": ids,
            "split": split_name,
            PROBABILITY_COLUMN: values,
            "model_name": model_name,
            "model_version": model_version,
        }
    )
    return frame[list(PREDICTION_COLUMNS)]


def validate_probability_frame(frame: pd.DataFrame, expected_loan_ids: set[str]) -> None:
    """Reject a malformed probability table before it reaches another track."""
    if frame.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError(
            f"Probability frame must have columns {list(PREDICTION_COLUMNS)} in that "
            f"order; got {frame.columns.tolist()}"
        )
    if frame["loan_id"].duplicated().any():
        raise ValueError("Probability frame has duplicate loan_id values")
    actual = set(frame["loan_id"])
    missing, extra = expected_loan_ids - actual, actual - expected_loan_ids
    if missing or extra:
        raise ValueError(
            "Probability frame does not cover the expected loan_id set exactly: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    if frame[PROBABILITY_COLUMN].isna().any():
        raise ValueError(f"Probability frame has missing {PROBABILITY_COLUMN} values")
    if not pd.api.types.is_numeric_dtype(frame[PROBABILITY_COLUMN]):
        raise ValueError(f"{PROBABILITY_COLUMN} must be numeric")
    values = frame[PROBABILITY_COLUMN].to_numpy(dtype=np.float64)
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{PROBABILITY_COLUMN} values must lie within [0, 1]")


def export_probabilities(frame: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Write Parquet and read it back, following the lexical-feature precedent."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    written = pd.read_parquet(output_path, engine="pyarrow")
    if written.columns.tolist() != list(PREDICTION_COLUMNS):
        raise RuntimeError("Written Parquet file has unexpected columns")
    if not written["loan_id"].reset_index(drop=True).equals(frame["loan_id"].reset_index(drop=True)):
        raise RuntimeError("Written Parquet file failed its loan_id order verification")
    if not np.allclose(written[PROBABILITY_COLUMN], frame[PROBABILITY_COLUMN]):
        raise RuntimeError("Written Parquet file failed its probability round-trip check")
    return written


def confusion_at(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    """Confusion counts at one explicitly named threshold."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, (probabilities >= threshold).astype(int), labels=[0, 1]
    ).ravel()
    return {"threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def paired_bootstrap_delta(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    *,
    resamples: int,
    random_state: int,
    confidence: float,
) -> dict[str, Any]:
    """Bootstrap the PR-AUC difference between two models on identical rows.

    Paired deliberately: both models are scored on the same resampled rows, so
    the interval describes the difference rather than the sum of two independent
    sampling errors. Comparing whether two separate intervals overlap is a much
    weaker test, and the margins here are a few points at most.
    """
    rng = np.random.default_rng(random_state)
    rows = len(y_true)
    deltas = []
    while len(deltas) < resamples:
        index = rng.integers(0, rows, size=rows)
        sample_y = y_true[index]
        if len(np.unique(sample_y)) < 2:
            continue
        deltas.append(
            average_precision_score(sample_y, probabilities_a[index])
            - average_precision_score(sample_y, probabilities_b[index])
        )
    deltas = np.asarray(deltas)
    tail = (1 - confidence) / 2
    return {
        "observed_delta": float(
            average_precision_score(y_true, probabilities_a)
            - average_precision_score(y_true, probabilities_b)
        ),
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.quantile(deltas, tail)),
        "ci_high": float(np.quantile(deltas, 1 - tail)),
        "share_favouring_a": float(np.mean(deltas > 0)),
        "resamples": int(len(deltas)),
        "confidence": confidence,
    }


def bootstrap_metric_interval(
    y_true: np.ndarray, probabilities: np.ndarray, *, resamples: int, random_state: int, confidence: float
) -> dict[str, float]:
    """Bootstrap interval for a single model's PR-AUC."""
    rng = np.random.default_rng(random_state)
    rows = len(y_true)
    scores = []
    while len(scores) < resamples:
        index = rng.integers(0, rows, size=rows)
        if len(np.unique(y_true[index])) < 2:
            continue
        scores.append(average_precision_score(y_true[index], probabilities[index]))
    scores = np.asarray(scores)
    tail = (1 - confidence) / 2
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "ci_low": float(np.quantile(scores, tail)),
        "ci_high": float(np.quantile(scores, 1 - tail)),
        "resamples": int(len(scores)),
        "confidence": confidence,
    }


def run(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    parquet_dir: str | Path = DEFAULT_PARQUET_DIR,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Tune on training folds, export out-of-fold and validation probabilities."""
    config = load_tuning_config(config_path)
    model_config = config["model"]
    model_name, model_version = config["model_name"], config["model_version"]

    train = load_baseline_split(data_dir, "train")
    validation = load_baseline_split(data_dir, "validation")
    y_train = train["target"].to_numpy()
    y_validation = validation["target"].to_numpy()

    cv = make_cv(config["cv"])
    grid_results, best_params = run_grid_search(
        train["text_tfidf"], y_train, config, cv, n_jobs=n_jobs
    )

    best_pipeline = pipeline_from_params(best_params, model_config)
    oof_probabilities = out_of_fold_probabilities(
        train["text_tfidf"], y_train, best_pipeline, cv, n_jobs=n_jobs
    )

    best_pipeline.fit(train["text_tfidf"], y_train)
    validation_probabilities = best_pipeline.predict_proba(validation["text_tfidf"])[:, 1]

    reference = dict(config["week1_reference"])
    reference_pipeline = pipeline_from_params(
        {
            "max_features": reference["max_features"],
            "ngram_range": tuple(reference["ngram_range"]),
            "C": reference["C"],
            "sublinear_tf": reference["sublinear_tf"],
        },
        model_config,
    )
    reference_pipeline.fit(train["text_tfidf"], y_train)
    reference_probabilities = reference_pipeline.predict_proba(validation["text_tfidf"])[:, 1]

    unweighted_pipeline = pipeline_from_params(best_params, {**model_config, "class_weight": None})
    unweighted_pipeline.fit(train["text_tfidf"], y_train)
    unweighted_probabilities = unweighted_pipeline.predict_proba(validation["text_tfidf"])[:, 1]

    train_metrics = evaluate_predictions(y_train, oof_probabilities)
    validation_metrics = evaluate_predictions(y_validation, validation_probabilities)
    reference_metrics = evaluate_predictions(y_validation, reference_probabilities)
    unweighted_metrics = evaluate_predictions(y_validation, unweighted_probabilities)

    for metrics, y_true, probabilities in (
        (train_metrics, y_train, oof_probabilities),
        (validation_metrics, y_validation, validation_probabilities),
    ):
        metrics["confusion_matrix_at_0.5"] = confusion_at(y_true, probabilities, 0.5)
        metrics["confusion_matrix_at_best_f1"] = confusion_at(
            y_true, probabilities, metrics["f1_best_threshold"]
        )

    bootstrap_config = config["bootstrap"]
    comparison = {
        "validation_pr_auc_interval": bootstrap_metric_interval(
            y_validation,
            validation_probabilities,
            resamples=bootstrap_config["resamples"],
            random_state=bootstrap_config["random_state"],
            confidence=bootstrap_config["confidence"],
        ),
        "tuned_minus_week1_pr_auc": paired_bootstrap_delta(
            y_validation,
            validation_probabilities,
            reference_probabilities,
            resamples=bootstrap_config["resamples"],
            random_state=bootstrap_config["random_state"],
            confidence=bootstrap_config["confidence"],
        ),
    }

    train_frame = build_probability_frame(
        train["loan_id"], "train", oof_probabilities, model_name, model_version
    )
    validation_frame = build_probability_frame(
        validation["loan_id"], "validation", validation_probabilities, model_name, model_version
    )
    validate_probability_frame(train_frame, set(train["loan_id"].astype("string")))
    validate_probability_frame(validation_frame, set(validation["loan_id"].astype("string")))

    parquet_dir = Path(parquet_dir)
    train_path = parquet_dir / "tfidf_oof_train.parquet"
    validation_path = parquet_dir / "tfidf_oof_validate.parquet"
    export_probabilities(train_frame, train_path)
    export_probabilities(validation_frame, validation_path)

    metrics = {
        "model_name": model_name,
        "model_version": model_version,
        "preprocess_version": config["preprocess_version"],
        "feature_set": config["feature_set"],
        "probability_semantics": "uncalibrated_pd",
        "test_rows_used": 0,
        "selection": {
            "scoring": config["cv"]["scoring"],
            "cv": dict(config["cv"]),
            "grid_size": int(len(grid_results)),
            "selected_on": "train_cross_validation_only",
            "best_params": {**best_params, "ngram_range": list(best_params["ngram_range"])},
            "best_mean_cv_pr_auc": float(grid_results.loc[0, "mean_cv_pr_auc"]),
            "best_std_cv_pr_auc": float(grid_results.loc[0, "std_cv_pr_auc"]),
            "fixed_model_settings": dict(model_config),
        },
        "train_out_of_fold": train_metrics,
        "validation": validation_metrics,
        "week1_reference": {"params": reference, "validation": reference_metrics},
        "unweighted_diagnostic": {
            "note": "Winning configuration refitted with class_weight=None; ranking versus calibration trade-off.",
            "validation": unweighted_metrics,
        },
        "comparison": comparison,
        "outputs": {
            "train_parquet": str(train_path.relative_to(REPO_ROOT)),
            "validation_parquet": str(validation_path.relative_to(REPO_ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tfidf_tuning_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    grid_results.to_csv(output_dir / "tfidf_grid_search_results.csv", index=False)

    return {
        "metrics": metrics,
        "grid_results": grid_results,
        "train_frame": train_frame,
        "validation_frame": validation_frame,
        "pipeline": best_pipeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the TF-IDF baseline and export OOF probabilities.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    result = run(
        args.data_dir,
        args.output_dir,
        args.parquet_dir,
        config_path=args.config,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nTop grid configurations:\n{result['grid_results'].head(8).to_string(index=False)}")


if __name__ == "__main__":
    main()

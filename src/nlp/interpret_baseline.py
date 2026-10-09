"""Analyst 2 Week 8: coefficient interpretation, calibration and threshold trade-offs.

Run from the repository root:
    python -m src.nlp.interpret_baseline

Diagnoses the tuned model shipped by ``src/nlp/evaluate_baseline.py`` rather than
proposing a new one. Nothing here changes the exported probabilities.

**Rebuilding, not loading.** No fitted model is persisted anywhere in this
repository: ``*.joblib`` and ``*.pkl`` are gitignored and
``src/tabular/logistic_baseline.py`` states the convention plainly ("No trained
model is ever written to disk: fitting is fast and deterministic"). The winning
hyperparameters are read from ``reports/nlp/tfidf_tuning_metrics.json`` and
refitted under the pinned ``random_state``, which reproduces the same pipeline
without introducing a binary artefact to version.

**Artefact flags.** "Uninformative" is not left to taste; the three classes are
the ones already recorded in ``docs/nlp-baseline-report.md``:

* ``low_document_frequency`` (NB-02) - a coefficient carried by a handful of
  loans is not a finding, however large it is.
* ``cross_field_bigram`` (NB-03) - ``build_text_payload`` concatenates title,
  purpose and description, so a bigram can straddle two fields and describe the
  payload format rather than borrower language. Detected by measuring each term
  inside the individual fields: a term present in the payload but in no single
  field spans a boundary.
* ``period_specific`` (NB-04) - relative frequency swinging by more than
  ``VOLATILITY_THRESHOLD`` across issue-year buckets. Measured as a share of all
  term occurrences in the bucket, because descriptions shortened steadily over
  the cohort period and raw document frequency would fall for almost every term
  for reasons of length alone.

**Calibration.** Four variants are reported on validation, because the brief's
"calibrated probabilities" and decision D-006's "calibrated PD" are claims that
have to be measured rather than asserted:

1. ``balanced`` - the shipped model, uncalibrated.
2. ``unweighted`` - the same configuration with ``class_weight=None``.
3. ``isotonic_oof`` / ``platt_oof`` - calibrators fitted on the out-of-fold
   training probabilities already exported to ``data/nlp/``.
4. ``isotonic_role`` / ``platt_role`` - calibrators fitted on a chronologically
   held-out calibration period, mirroring the role boundaries proposed in
   ``configs/tabular_development_v1.toml`` on the tabular track. Those months are
   read from the manifest here rather than importing that unmerged module.

``CalibratedClassifierCV`` is additionally run as a cross-check. It cannot
consume precomputed out-of-fold probabilities - it wraps an estimator and runs
its own internal cross-validation - so it answers a slightly different question
than variants 3 and 4 and is reported separately.

Platt scaling is strictly monotone, so it leaves ROC-AUC and PR-AUC untouched and
merely re-indexes the thresholds. Isotonic regression is monotone but ties scores
together, so its ranking metrics can move marginally. Either way the
precision/recall trade-off in the threshold sweep is a property of the ranking,
not of the probability scale.

The locked test split is never requested.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from src.nlp.baseline_tfidf import load_baseline_split, strip_field_labels
from src.nlp.evaluate_baseline import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    load_tuning_config,
    make_cv,
    pipeline_from_params,
)
from src.nlp.preprocess import CORE_TEXT_COLUMNS, FIELD_LABELS, clean_text
from src.tabular.evaluate import evaluate_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TUNING_METRICS = DEFAULT_OUTPUT_DIR / "tfidf_tuning_metrics.json"
DEFAULT_OOF_TRAIN = REPO_ROOT / "data" / "nlp" / "tfidf_oof_train.parquet"

INTERPRETABILITY_VERSION = "nlp-tfidf-interpretability-v1"
TOP_K = 30
DOCUMENT_FREQUENCY_FLOOR_PERCENT = 1.0
VOLATILITY_THRESHOLD = 10.0
# A bigram is treated as spanning a field boundary when this fraction or less of
# its payload occurrences can be found inside any single field. Not zero: a term
# like "purchase major" appears within a field in 9 of 492 documents by chance,
# and an exact-zero rule would miss the artefact entirely.
CROSS_FIELD_MAX_WITHIN_RATIO = 0.2
EARLY_BUCKET = "2007-2009"
EARLY_BUCKET_UNTIL = "2010"
# Mirrors configs/tabular_development_v1.toml [roles]; read from the manifest so
# nothing here depends on that unmerged branch.
MODEL_FIT_MONTHS = ("2007-06", "2012-12")
CALIBRATION_MONTHS = ("2013-01", "2013-07")
THRESHOLD_SWEEP = tuple(np.round(np.arange(0.10, 0.6001, 0.05), 2))
RECALL_TARGETS = (0.70, 0.80)
RELIABILITY_BINS = 10


def load_winning_params(metrics_path: str | Path = DEFAULT_TUNING_METRICS) -> dict[str, Any]:
    """Read the selected hyperparameters recorded by the `nlp-tfidf-tuned-v1` run."""
    with Path(metrics_path).open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    selection = metrics["selection"]["best_params"]
    return {
        "max_features": int(selection["max_features"]),
        "ngram_range": tuple(selection["ngram_range"]),
        "C": float(selection["C"]),
        "sublinear_tf": bool(selection["sublinear_tf"]),
    }


def document_frequency(matrix) -> np.ndarray:
    """Count documents containing each feature, not total occurrences."""
    return np.asarray((matrix > 0).sum(axis=0)).ravel()


def field_texts(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """Clean each payload field separately, exactly as build_text_payload would.

    Needed to tell a genuine bigram from one that only exists because the payload
    concatenates fields. ``purpose`` underscores become spaces, matching the
    shared builder.
    """
    texts = {}
    for field in CORE_TEXT_COLUMNS:
        cleaned = rows[field].map(clean_text)
        if field == "purpose":
            cleaned = cleaned.str.replace("_", " ", regex=False).str.split().str.join(" ")
        texts[field] = cleaned
    return texts


def within_field_document_counts(
    terms: np.ndarray, rows: pd.DataFrame, ngram_range: tuple[int, int]
) -> np.ndarray:
    """Document counts for each term measured inside individual fields only."""
    vocabulary = {term: index for index, term in enumerate(terms)}
    counter = CountVectorizer(
        ngram_range=ngram_range, stop_words="english", vocabulary=vocabulary
    )
    totals = np.zeros(len(terms), dtype=np.int64)
    for text in field_texts(rows).values():
        totals += document_frequency(counter.transform(text))
    return totals


def assign_year_buckets(issue_month: pd.Series) -> pd.Series:
    """Year buckets, merging the thin early years into one period.

    2007 contributes 246 training rows and 2008 contributes 1,562, so a term with
    a few hundred documents overall is absent from them by arithmetic rather than
    by era.
    """
    year = issue_month.str[:4]
    return year.where(year >= EARLY_BUCKET_UNTIL, EARLY_BUCKET)


def term_volatility(
    terms: np.ndarray,
    full_terms: np.ndarray,
    vocabulary: dict,
    texts: pd.Series,
    buckets: pd.Series,
    ngram_range: tuple[int, int],
) -> pd.DataFrame:
    """Relative-frequency swing for each term across issue-year buckets.

    The denominator is every occurrence of the full vocabulary in the bucket, so
    the measure is invariant to how long the documents are. ``full_terms`` must
    come from ``get_feature_names_out()``: the ``vocabulary_`` dict is keyed in
    insertion order, not column order, and indexing by its keys would silently
    attribute one term's counts to another.
    """
    counter = CountVectorizer(
        ngram_range=ngram_range, stop_words="english", vocabulary=vocabulary
    )
    positions = pd.Series(range(len(full_terms)), index=full_terms)
    wanted = positions.reindex(terms).to_numpy()
    per_million: dict[str, np.ndarray] = {}
    for name in sorted(buckets.unique()):
        matrix = counter.transform(texts[buckets.eq(name)])
        occurrences = np.asarray(matrix.sum(axis=0)).ravel()
        total = occurrences.sum()
        per_million[name] = 1e6 * occurrences[wanted] / total
    frame = pd.DataFrame(per_million, index=terms)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = frame.max(axis=1) / frame.min(axis=1)
    return pd.DataFrame(
        {
            "volatility_ratio": ratio,
            "peak_bucket": frame.idxmax(axis=1),
            "trough_bucket": frame.idxmin(axis=1),
        }
    )


def extract_coefficients(
    pipeline, train: pd.DataFrame, *, top_k: int = TOP_K
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank the strongest coefficients in each direction and flag artefacts."""
    vectorizer = pipeline.named_steps["vectorize"]
    classifier = pipeline.named_steps["classify"]
    full_terms = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    matrix = vectorizer.transform(train["text_tfidf"])
    frequencies = document_frequency(matrix)
    rows = matrix.shape[0]

    order = np.argsort(coefficients)
    selected = np.concatenate([order[-top_k:][::-1], order[:top_k]])
    terms = full_terms[selected]

    ranked = pd.DataFrame(
        {
            "ngram": terms,
            "coefficient": coefficients[selected],
            "direction": ["higher_risk"] * top_k + ["lower_risk"] * top_k,
            "train_documents": frequencies[selected],
            "train_document_percent": 100 * frequencies[selected] / rows,
        }
    )
    ranked["within_field_documents"] = within_field_document_counts(
        terms, train, vectorizer.ngram_range
    )
    volatility = term_volatility(
        terms,
        full_terms,
        vectorizer.vocabulary_,
        train["text_tfidf"],
        assign_year_buckets(train["issue_month"]),
        vectorizer.ngram_range,
    )
    ranked = ranked.join(volatility, on="ngram")

    ranked["low_document_frequency"] = (
        ranked["train_document_percent"] < DOCUMENT_FREQUENCY_FLOOR_PERCENT
    )
    within_ratio = ranked["within_field_documents"] / ranked["train_documents"]
    ranked["within_field_ratio"] = within_ratio
    ranked["cross_field_bigram"] = ranked["ngram"].str.contains(" ") & within_ratio.le(
        CROSS_FIELD_MAX_WITHIN_RATIO
    )
    ranked["period_specific"] = ranked["volatility_ratio"] > VOLATILITY_THRESHOLD
    flag_columns = ["low_document_frequency", "cross_field_bigram", "period_specific"]
    ranked["artefact_flags"] = ranked[flag_columns].sum(axis=1)
    ranked["retained"] = ranked["artefact_flags"].eq(0)

    summary = {
        "top_k_each_direction": top_k,
        "document_frequency_floor_percent": DOCUMENT_FREQUENCY_FLOOR_PERCENT,
        "volatility_threshold": VOLATILITY_THRESHOLD,
        "cross_field_max_within_ratio": CROSS_FIELD_MAX_WITHIN_RATIO,
        "ranked_terms": int(len(ranked)),
        "flagged_low_document_frequency": int(ranked["low_document_frequency"].sum()),
        "flagged_cross_field_bigram": int(ranked["cross_field_bigram"].sum()),
        "flagged_period_specific": int(ranked["period_specific"].sum()),
        "flagged_any": int((~ranked["retained"]).sum()),
        "retained": int(ranked["retained"].sum()),
        "retained_higher_risk": int(
            (ranked["retained"] & ranked["direction"].eq("higher_risk")).sum()
        ),
        "retained_lower_risk": int(
            (ranked["retained"] & ranked["direction"].eq("lower_risk")).sum()
        ),
        "median_train_document_percent": float(ranked["train_document_percent"].median()),
    }
    return ranked, summary


def logit(probabilities: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Inverse logistic transform, clipped so 0 and 1 do not become infinite.

    For a logistic model this recovers the decision function, so fitting a
    logistic regression on this quantity is Platt scaling in its original sense
    rather than a logistic fit on an already-squashed probability.
    """
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def fit_calibrators(
    reference_probabilities: np.ndarray, reference_targets: np.ndarray
) -> dict[str, Any]:
    """Fit isotonic and Platt calibrators on held-out reference probabilities."""
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(reference_probabilities, reference_targets)
    platt = LogisticRegression()
    platt.fit(logit(reference_probabilities).reshape(-1, 1), reference_targets)
    return {"isotonic": isotonic, "platt": platt}


def apply_calibrator(name: str, calibrator: Any, probabilities: np.ndarray) -> np.ndarray:
    """Map raw probabilities through a fitted calibrator."""
    if name == "isotonic":
        return np.clip(calibrator.predict(probabilities), 0.0, 1.0)
    return calibrator.predict_proba(logit(probabilities).reshape(-1, 1))[:, 1]


def reliability_points(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = RELIABILITY_BINS
) -> dict[str, list[float]]:
    """Quantile-binned reliability curve: observed rate against mean prediction."""
    observed, predicted = calibration_curve(
        y_true, probabilities, n_bins=bins, strategy="quantile"
    )
    return {"mean_predicted": [float(v) for v in predicted], "observed_rate": [float(v) for v in observed]}


def threshold_sweep(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds=THRESHOLD_SWEEP
) -> pd.DataFrame:
    """Precision, recall and false-positive cost at each named threshold."""
    records = []
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(
            y_true, (probabilities >= threshold).astype(int), labels=[0, 1]
        ).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        records.append(
            {
                "threshold": float(threshold),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
                "false_positives_per_true_positive": float(fp / tp) if tp else float("inf"),
                "flagged_rate": float((tp + fp) / len(y_true)),
            }
        )
    return pd.DataFrame(records)


def threshold_for_recall(
    y_true: np.ndarray, probabilities: np.ndarray, target_recall: float
) -> dict[str, Any]:
    """Highest threshold still capturing at least ``target_recall`` of defaults.

    Searched over observed score values rather than the coarse sweep grid, so the
    answer is the real operating point and not the nearest 0.05 step.
    """
    candidates = np.unique(probabilities)[::-1]
    positives = int((y_true == 1).sum())
    for threshold in candidates:
        predicted = probabilities >= threshold
        recall = int((predicted & (y_true == 1)).sum()) / positives
        if recall >= target_recall:
            tn, fp, fn, tp = confusion_matrix(
                y_true, predicted.astype(int), labels=[0, 1]
            ).ravel()
            return {
                "target_recall": target_recall,
                "threshold": float(threshold),
                "recall": float(recall),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "false_positives_per_true_positive": float(fp / tp) if tp else float("inf"),
                "flagged_rate": float((tp + fp) / len(y_true)),
            }
    raise ValueError(f"No threshold reaches recall {target_recall}")


def plot_calibration_curves(
    y_true: np.ndarray, variants: dict[str, np.ndarray], output_path: str | Path
) -> None:
    """Two-panel reliability diagram: weighting effect, then calibrator effect."""
    panels = [
        ("Effect of class weighting", ["balanced", "unweighted"]),
        ("Effect of calibrating the shipped model", ["balanced", "isotonic_oof", "platt_oof"]),
    ]
    # Independent, data-driven limits. A 45-degree diagonal is the usual
    # convention, but at a prevalence near 0.15 the observed rate never exceeds
    # ~0.22 while predictions reach ~0.66, so equal axes leave three quarters of
    # the panel empty. The reference line is still y = x; stretching the vertical
    # axis makes the calibration gap easier to read, not harder.
    shown = [values for _, names in panels for name in names if (values := variants.get(name)) is not None]
    x_upper = min(1.0, 0.05 * np.ceil(max(v.max() for v in shown) / 0.05) + 0.05)
    y_upper = min(1.0, 0.05 * np.ceil(max(max(reliability_points(y_true, v)["observed_rate"]) for v in shown) / 0.05) + 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (title, names) in zip(axes, panels):
        ax.plot([0, max(x_upper, y_upper)], [0, max(x_upper, y_upper)], linestyle=":", color="#888888", label="perfectly calibrated")
        for name in names:
            if name not in variants:
                continue
            points = reliability_points(y_true, variants[name])
            ax.plot(
                points["mean_predicted"],
                points["observed_rate"],
                marker="o",
                markersize=4,
                linewidth=1.4,
                label=name,
            )
        ax.axhline(
            float(np.mean(y_true)),
            color="#C05A27",
            linestyle="--",
            linewidth=1,
            label=f"prevalence {np.mean(y_true):.3f}",
        )
        ax.set(
            title=title,
            xlabel="Mean predicted probability (10 quantile bins)",
            ylabel="Observed default rate",
            xlim=(0, x_upper),
            ylim=(0, y_upper),
        )
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("TF-IDF baseline calibration on validation", fontsize=12)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    metrics_path: str | Path = DEFAULT_TUNING_METRICS,
    oof_path: str | Path = DEFAULT_OOF_TRAIN,
    run_calibrated_classifier_cv: bool = True,
) -> dict[str, Any]:
    """Interpret, calibrate and sweep thresholds for the shipped tuned model."""
    config = load_tuning_config(config_path)
    model_config = config["model"]
    best_params = load_winning_params(metrics_path)

    train = load_baseline_split(data_dir, "train")
    validation = load_baseline_split(data_dir, "validation")
    y_train = train["target"].to_numpy()
    y_validation = validation["target"].to_numpy()

    balanced = pipeline_from_params(best_params, model_config)
    balanced.fit(train["text_tfidf"], y_train)
    probabilities = {
        "balanced": balanced.predict_proba(validation["text_tfidf"])[:, 1]
    }

    unweighted = pipeline_from_params(best_params, {**model_config, "class_weight": None})
    unweighted.fit(train["text_tfidf"], y_train)
    probabilities["unweighted"] = unweighted.predict_proba(validation["text_tfidf"])[:, 1]

    ranked, coefficient_summary = extract_coefficients(balanced, train)

    oof = pd.read_parquet(oof_path)
    aligned = (
        train[["loan_id"]]
        .astype({"loan_id": "string"})
        .merge(oof[["loan_id", "prob_default_tfidf"]], on="loan_id", how="left", validate="one_to_one")
    )
    if aligned["prob_default_tfidf"].isna().any():
        raise ValueError("Exported out-of-fold probabilities do not cover every training row")
    oof_probabilities = aligned["prob_default_tfidf"].to_numpy()

    for name, calibrator in fit_calibrators(oof_probabilities, y_train).items():
        probabilities[f"{name}_oof"] = apply_calibrator(name, calibrator, probabilities["balanced"])

    months = train["issue_month"]
    model_fit_mask = months.between(*MODEL_FIT_MONTHS)
    calibration_mask = months.between(*CALIBRATION_MONTHS)
    role_pipeline = pipeline_from_params(best_params, model_config)
    role_pipeline.fit(train.loc[model_fit_mask, "text_tfidf"], y_train[model_fit_mask.to_numpy()])
    role_reference = role_pipeline.predict_proba(train.loc[calibration_mask, "text_tfidf"])[:, 1]
    probabilities["role_base"] = role_pipeline.predict_proba(validation["text_tfidf"])[:, 1]
    for name, calibrator in fit_calibrators(
        role_reference, y_train[calibration_mask.to_numpy()]
    ).items():
        probabilities[f"{name}_role"] = apply_calibrator(
            name, calibrator, probabilities["role_base"]
        )

    cross_check: dict[str, Any] = {}
    if run_calibrated_classifier_cv:
        for method, label in (("isotonic", "isotonic"), ("sigmoid", "platt")):
            wrapped = CalibratedClassifierCV(
                pipeline_from_params(best_params, model_config),
                method=method,
                cv=make_cv(config["cv"]),
            )
            wrapped.fit(train["text_tfidf"], y_train)
            name = f"{label}_classifier_cv"
            probabilities[name] = wrapped.predict_proba(validation["text_tfidf"])[:, 1]
            cross_check[name] = evaluate_predictions(y_validation, probabilities[name])

    variant_metrics = {
        name: evaluate_predictions(y_validation, values)
        for name, values in probabilities.items()
    }
    reliability = {
        name: reliability_points(y_validation, values) for name, values in probabilities.items()
    }

    sweep = threshold_sweep(y_validation, probabilities["balanced"])
    recall_targets = [
        threshold_for_recall(y_validation, probabilities["balanced"], target)
        for target in RECALL_TARGETS
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "tfidf_calibration_curves.png"
    plot_calibration_curves(y_validation, probabilities, plot_path)

    metrics = {
        "interpretability_version": INTERPRETABILITY_VERSION,
        "diagnoses_model_version": config["model_version"],
        "best_params": {**best_params, "ngram_range": list(best_params["ngram_range"])},
        "test_rows_used": 0,
        "probability_semantics": {
            "balanced": "uncalibrated_pd_as_shipped",
            "isotonic_oof": "calibrated_pd_on_out_of_fold_reference",
            "platt_oof": "calibrated_pd_on_out_of_fold_reference",
            "isotonic_role": "calibrated_pd_on_chronological_calibration_period",
            "platt_role": "calibrated_pd_on_chronological_calibration_period",
        },
        "coefficients": coefficient_summary,
        "calibration": {
            "reference_sets": {
                "out_of_fold": {"rows": int(len(oof_probabilities)), "source": str(Path(oof_path).name)},
                "chronological_role": {
                    "model_fit_months": list(MODEL_FIT_MONTHS),
                    "model_fit_rows": int(model_fit_mask.sum()),
                    "calibration_months": list(CALIBRATION_MONTHS),
                    "calibration_rows": int(calibration_mask.sum()),
                    "mirrors": "configs/tabular_development_v1.toml (proposed, unmerged)",
                },
            },
            "validation_by_variant": variant_metrics,
            "calibrated_classifier_cv_cross_check": cross_check,
            "reliability_curves": {"bins": RELIABILITY_BINS, "strategy": "quantile", "points": reliability},
        },
        "thresholds": {
            "swept_on": "balanced (as shipped, uncalibrated)",
            "sweep": sweep.to_dict(orient="records"),
            "recall_targets": recall_targets,
        },
        "outputs": {
            "calibration_plot": str(plot_path.relative_to(REPO_ROOT)),
            "coefficients_csv": "reports/nlp/tfidf_coefficients.csv",
            "threshold_sweep_csv": "reports/nlp/tfidf_threshold_sweep.csv",
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    ranked.to_csv(output_dir / "tfidf_coefficients.csv", index=False)
    sweep.to_csv(output_dir / "tfidf_threshold_sweep.csv", index=False)
    (output_dir / "tfidf_interpretability_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return {"metrics": metrics, "ranked": ranked, "sweep": sweep, "probabilities": probabilities}


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret and calibrate the tuned TF-IDF baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tuning-metrics", type=Path, default=DEFAULT_TUNING_METRICS)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF_TRAIN)
    parser.add_argument("--skip-classifier-cv", action="store_true")
    args = parser.parse_args()
    result = run(
        args.data_dir,
        args.output_dir,
        config_path=args.config,
        metrics_path=args.tuning_metrics,
        oof_path=args.oof,
        run_calibrated_classifier_cv=not args.skip_classifier_cv,
    )
    metrics = result["metrics"]
    print("Brier by variant (validation):")
    for name, values in metrics["calibration"]["validation_by_variant"].items():
        print(
            f"  {name:24s} brier {values['brier_score']:.5f}  roc {values['roc_auc']:.4f}  pr {values['pr_auc']:.4f}"
        )
    print(f"\nArtefact flags: {json.dumps(metrics['coefficients'], indent=2)}")
    print(f"\nRecall targets:\n{pd.DataFrame(metrics['thresholds']['recall_targets']).to_string(index=False)}")


if __name__ == "__main__":
    main()

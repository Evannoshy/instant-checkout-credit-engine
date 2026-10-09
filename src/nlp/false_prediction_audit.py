"""Helpers for the validation error review in false_prediction_audit.ipynb.

The committed notebook stores aggregate terms and paraphrased case notes. Raw
borrower descriptions are loaded only during a local run and are never exported.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.nlp.baseline_tfidf import load_baseline_split
from src.nlp.evaluate_baseline import load_tuning_config, pipeline_from_params

THRESHOLD = 0.5
SAMPLE_SEED = 42


def load_validation_errors(
    data_dir: str | Path,
    prediction_path: str | Path,
    metrics_path: str | Path,
    *,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Join Yati's saved validation scores to frozen labels and check counts."""
    data_dir = Path(data_dir)
    predictions = pd.read_parquet(prediction_path)
    expected_columns = [
        "loan_id", "split", "prob_default_tfidf", "model_name", "model_version"
    ]
    if predictions.columns.tolist() != expected_columns:
        raise ValueError("Prediction file has an unexpected schema")
    if not predictions["split"].eq("validation").all():
        raise ValueError("Only validation predictions may be audited")
    if predictions["loan_id"].isna().any() or predictions["loan_id"].duplicated().any():
        raise ValueError("Prediction IDs must be present and unique")
    if not predictions["prob_default_tfidf"].between(0, 1).all():
        raise ValueError("Prediction scores must lie within [0, 1]")

    with Path(metrics_path).open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    if not predictions["model_version"].eq(metrics["model_version"]).all():
        raise ValueError("Prediction and metric model versions differ")
    if not predictions["model_name"].eq(metrics["model_name"]).all():
        raise ValueError("Prediction and metric model names differ")

    manifest = pd.read_csv(
        data_dir / "split_manifest.csv", dtype={"loan_id": "string"},
        usecols=["loan_id", "split", "target"],
    )
    labels = manifest.loc[manifest["split"].eq("validation"), ["loan_id", "target"]]
    frozen_ids = pd.read_csv(data_dir / "val_ids.csv", dtype={"loan_id": "string"})
    if frozen_ids.columns.tolist() != ["loan_id"]:
        raise ValueError("val_ids.csv must contain only loan_id")
    if labels["loan_id"].duplicated().any() or frozen_ids["loan_id"].duplicated().any():
        raise ValueError("Validation IDs must be unique")
    if set(predictions["loan_id"]) != set(labels["loan_id"]) or set(labels["loan_id"]) != set(frozen_ids["loan_id"]):
        raise ValueError("Prediction IDs do not match the frozen validation split")
    if not labels["target"].isin([0, 1]).all():
        raise ValueError("Validation target must be binary")

    joined = predictions.merge(labels, on="loan_id", how="inner", validate="one_to_one")
    flagged = joined["prob_default_tfidf"].ge(threshold)
    observed = {
        "tn": int((joined["target"].eq(0) & ~flagged).sum()),
        "fp": int((joined["target"].eq(0) & flagged).sum()),
        "fn": int((joined["target"].eq(1) & ~flagged).sum()),
        "tp": int((joined["target"].eq(1) & flagged).sum()),
    }
    if threshold == THRESHOLD:
        published = metrics["validation"]["confusion_matrix_at_0.5"]
        if any(observed[key] != published[key] for key in observed):
            raise ValueError("Saved scores do not reproduce the published confusion matrix")
    return joined


def select_review_cases(
    predictions: pd.DataFrame,
    *,
    threshold: float = THRESHOLD,
    n_per_error: int = 20,
    n_extreme: int = 10,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """Take the strongest errors and a fixed random draw from the rest."""
    if not 0 < n_extreme < n_per_error:
        raise ValueError("n_extreme must be between zero and n_per_error")
    required = {"loan_id", "target", "prob_default_tfidf"}
    if missing := required - set(predictions.columns):
        raise ValueError(f"Predictions are missing {sorted(missing)}")
    if predictions["loan_id"].duplicated().any():
        raise ValueError("Prediction IDs must be unique")

    rng = np.random.default_rng(seed)
    selected = []
    definitions = (
        ("false_positive", predictions["target"].eq(0) & predictions["prob_default_tfidf"].ge(threshold), True),
        ("false_negative", predictions["target"].eq(1) & predictions["prob_default_tfidf"].lt(threshold), False),
    )
    for error_type, mask, high_scores_first in definitions:
        ranked = predictions.loc[mask].sort_values(
            ["prob_default_tfidf", "loan_id"],
            ascending=[not high_scores_first, True],
        )
        if len(ranked) < n_per_error:
            raise ValueError(f"Only {len(ranked)} {error_type} rows are available")
        strongest = ranked.head(n_extreme).assign(
            error_type=error_type, sample_group="strongest"
        )
        remainder = ranked.iloc[n_extreme:]
        sampled_positions = np.sort(
            rng.choice(len(remainder), size=n_per_error - n_extreme, replace=False)
        )
        sampled = remainder.iloc[sampled_positions].assign(
            error_type=error_type, sample_group="random"
        )
        selected.extend((strongest, sampled))
    return pd.concat(selected, ignore_index=True)


def fit_published_model(
    data_dir: str | Path, metrics_path: str | Path
) -> tuple[Pipeline, pd.DataFrame]:
    """Refit the published winner on training text for term diagnostics."""
    with Path(metrics_path).open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    parameters = dict(metrics["selection"]["best_params"])
    parameters["ngram_range"] = tuple(parameters["ngram_range"])
    config = load_tuning_config()
    if metrics["model_version"] != config["model_version"]:
        raise ValueError("Tuning config and published metrics identify different models")
    model = pipeline_from_params(parameters, config["model"])
    train = load_baseline_split(data_dir, "train")
    model.fit(train["text_tfidf"], train["target"].to_numpy())
    return model, train


def check_refit_scores(
    model: Pipeline, validation_text: pd.DataFrame, predictions: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
) -> float:
    """Confirm the refitted model reproduces Yati's saved validation scores."""
    aligned = predictions[["loan_id", "prob_default_tfidf"]].merge(
        validation_text[["loan_id", "text_tfidf"]],
        on="loan_id", how="left", validate="one_to_one",
    )
    if aligned["text_tfidf"].isna().any():
        raise ValueError("A saved validation prediction has no matching text")
    reproduced = model.predict_proba(aligned["text_tfidf"])[:, 1]
    difference = float(np.max(np.abs(reproduced - aligned["prob_default_tfidf"])))
    if difference > tolerance:
        raise ValueError(
            f"Refitted model differs from saved predictions by {difference:.3g}"
        )
    return difference


def summarize_term_contributions(
    model: Pipeline,
    selected_text: pd.DataFrame,
    train_count: int,
    *,
    terms_per_case: int = 5,
    min_train_documents: int = 50,
) -> pd.DataFrame:
    """Count common n-grams among each error type's leading score contributions.

    A term is counted at most once per loan. Positive contributions are counted
    for false positives and negative contributions for false negatives. The
    document-frequency floor keeps rare, potentially identifying words out of
    the committed notebook output.
    """
    vectorizer = model.named_steps["vectorize"]
    classifier = model.named_steps["classify"]
    terms = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    # TfidfVectorizer uses smooth_idf=True by default. Invert its IDF formula.
    train_document_counts = np.rint(
        (1 + train_count) / np.exp(vectorizer.idf_ - 1) - 1
    ).astype(int)
    matrix = vectorizer.transform(selected_text["text_tfidf"])

    counts: Counter[tuple[str, str]] = Counter()
    contribution_totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row_index, row in enumerate(selected_text.itertuples(index=False)):
        sparse_row = matrix.getrow(row_index)
        contributions = sparse_row.data * coefficients[sparse_row.indices]
        if row.error_type == "false_positive":
            order = np.argsort(contributions)[::-1]
            eligible = [position for position in order if contributions[position] > 0]
        elif row.error_type == "false_negative":
            order = np.argsort(contributions)
            eligible = [position for position in order if contributions[position] < 0]
        else:
            raise ValueError(f"Unknown error type: {row.error_type}")
        chosen = [
            position for position in eligible
            if train_document_counts[sparse_row.indices[position]] >= min_train_documents
        ][:terms_per_case]
        for position in chosen:
            term = str(terms[sparse_row.indices[position]])
            key = (row.error_type, term)
            counts[key] += 1
            contribution_totals[key].append(float(contributions[position]))

    result = pd.DataFrame(
        [
            {
                "error_type": error_type,
                "ngram": term,
                "case_count": count,
                "mean_log_odds_contribution": float(np.mean(contribution_totals[(error_type, term)])),
                "train_document_count": int(train_document_counts[vectorizer.vocabulary_[term]]),
            }
            for (error_type, term), count in counts.items()
        ]
    )
    return result.sort_values(
        ["error_type", "case_count", "ngram"],
        ascending=[True, False, True],
        ignore_index=True,
    )

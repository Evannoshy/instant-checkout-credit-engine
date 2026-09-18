"""Analyst 2: TF-IDF + logistic-regression baseline over the shared text payload.

Run from the repository root:
    python -m src.nlp.baseline_tfidf

This is the classical reference point the transformer in ``dataset.py`` has to
beat. It consumes ``load_original_split``/``build_text_payload`` from
``preprocess.py`` unchanged: the payload is the cross-track contract and is not
modified here. The vectoriser is fitted on training rows only and merely
transforms validation, per the working agreement on fitting preprocessing on
allowed development data. The final test split is locked and never loaded.

Field labels are stripped before vectorising. ``build_text_payload`` prefixes
each field with "Title: ", "Purpose: " or "Description: ", so those words occur
in every document, are not English stop words, and would otherwise consume
vocabulary slots and surface in the ranked coefficients as artefacts of the
payload format rather than borrower language. Stripping is a *downstream*
Analyst 2 transform over the identical shared payload; the labels are read from
``FIELD_LABELS`` so this stays tied to Analyst 1's field set.

Known limitation: bigrams still span field boundaries, so a title's last word
can pair with a purpose's first word. Avoiding that would require vectorising
each field separately, which is a different model, so it is reported rather
than silently worked around.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.nlp.preprocess import (
    FIELD_LABELS,
    PREPROCESS_VERSION,
    clean_text,
    load_original_split,
)

BASELINE_VERSION = "nlp-baseline-tfidf-v1"
MAX_FEATURES = 5_000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42
MAX_ITER = 1_000
TOP_K = 20

# Anchored per line because the payload joins "Label: value" lines with "\n".
_FIELD_LABEL_PREFIX = re.compile(
    r"^(?:" + "|".join(re.escape(label) for label in FIELD_LABELS.values()) + r"): ",
    re.MULTILINE,
)


def strip_field_labels(payload: str) -> str:
    """Remove ``build_text_payload`` field labels, keeping the borrower text.

    Operates only on the label prefixes the shared payload writes. A colon
    inside borrower prose is untouched because the pattern is line-anchored.
    """
    return _FIELD_LABEL_PREFIX.sub("", payload)


def load_baseline_split(data_dir: str | Path, split: str, *, desc_only: bool = False) -> pd.DataFrame:
    """Load a development split and add the label-stripped vectoriser input.

    ``desc_only`` drops title and purpose, retaining the free-text narrative
    alone. Purpose is a 14-level categorical field the tabular track already
    owns, so the ablation shows what the narrative adds beyond it.
    """
    frame = load_original_split(data_dir, split)
    if desc_only:
        # Clean with the shared function so the ablation differs only in fields used.
        text = frame["desc"].map(clean_text)
    else:
        text = frame["text_payload"].map(strip_field_labels)
    frame["text_tfidf"] = text.str.strip()
    return frame


def fit_baseline(train_text: pd.Series, y_train: np.ndarray) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit the vectoriser and classifier on training rows only."""
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        stop_words="english",
    )
    x_train = vectorizer.fit_transform(train_text)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    classifier.fit(x_train, y_train)
    return vectorizer, classifier


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Score ranking quality, plus F1 at two explicitly named thresholds.

    PR-AUC and ROC-AUC are rank-based and unaffected by ``class_weight``. F1 is
    not: balanced weighting moves the decision boundary, so 0.5 is no longer a
    meaningful operating point at a ~15% base rate. Both the 0.5 value and the
    best swept value are reported with their thresholds so neither is read as
    "the" F1. Each metric is stated against the score a trivial model gets:
    PR-AUC's floor is the prevalence, ROC-AUC's is 0.5.
    """
    prevalence = float(np.mean(y_true))
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one more point than thresholds.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_curve = np.nan_to_num(2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]))
    best = int(np.argmax(f1_curve)) if f1_curve.size else 0
    return {
        "rows": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": prevalence,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "pr_auc_baseline": prevalence,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "roc_auc_baseline": 0.5,
        "f1_at_0.5": float(f1_score(y_true, (scores >= 0.5).astype(int))),
        "f1_best": float(f1_curve[best]) if f1_curve.size else 0.0,
        "f1_best_threshold": float(thresholds[best]) if thresholds.size else 0.5,
    }


def top_ngrams(
    vectorizer: TfidfVectorizer, classifier: LogisticRegression, x_train, k: int = TOP_K
) -> pd.DataFrame:
    """Rank n-grams by signed coefficient, with training document frequency.

    Document frequency is reported so a term carried by few loans is not read
    as a finding. Coefficients are descriptive associations: magnitudes depend
    on IDF scaling and split across correlated n-grams, so they do not identify
    risk drivers.
    """
    terms = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    document_frequency = np.asarray((x_train > 0).sum(axis=0)).ravel()
    order = np.argsort(coefficients)
    selected = np.concatenate([order[-k:][::-1], order[:k]])
    return pd.DataFrame(
        {
            "ngram": terms[selected],
            "coefficient": coefficients[selected],
            "train_document_count": document_frequency[selected],
            "train_document_percent": 100 * document_frequency[selected] / x_train.shape[0],
            "direction": ["higher_risk"] * k + ["lower_risk"] * k,
        }
    )


def run(data_dir: str | Path, output_dir: str | Path, *, desc_only: bool = False) -> dict[str, Any]:
    """Fit on train, evaluate on validation, and persist metrics and n-grams."""
    train = load_baseline_split(data_dir, "train", desc_only=desc_only)
    validation = load_baseline_split(data_dir, "validation", desc_only=desc_only)
    y_train = train["target"].to_numpy()
    y_validation = validation["target"].to_numpy()

    vectorizer, classifier = fit_baseline(train["text_tfidf"], y_train)
    x_train = vectorizer.transform(train["text_tfidf"])
    x_validation = vectorizer.transform(validation["text_tfidf"])
    scores = classifier.predict_proba(x_validation)[:, 1]

    metrics = {
        "baseline_version": BASELINE_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "text_input": "desc_only" if desc_only else "payload_labels_stripped",
        "vectorizer": {
            "ngram_range": list(NGRAM_RANGE),
            "max_features": MAX_FEATURES,
            "stop_words": "english",
            "fitted_vocabulary_size": int(len(vectorizer.vocabulary_)),
            "fitted_on": "train",
        },
        "classifier": {
            "class_weight": "balanced",
            "max_iter": MAX_ITER,
            "random_state": RANDOM_STATE,
            "n_iter": int(classifier.n_iter_[0]),
            "converged": bool(classifier.n_iter_[0] < MAX_ITER),
        },
        "train": {"rows": int(len(train)), "positives": int(y_train.sum())},
        "validation": evaluate(y_validation, scores),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    ngrams = top_ngrams(vectorizer, classifier, x_train)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_desc_only" if desc_only else ""
    (output_dir / f"baseline_tfidf_metrics{suffix}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    ngrams.to_csv(output_dir / f"baseline_tfidf_top_ngrams{suffix}.csv", index=False)
    return {"metrics": metrics, "ngrams": ngrams}


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "nlp")
    parser.add_argument(
        "--desc-only",
        action="store_true",
        help="Ablation: vectorise the borrower narrative without title or purpose.",
    )
    args = parser.parse_args()

    result = run(args.data_dir, args.output_dir, desc_only=args.desc_only)
    metrics = result["metrics"]
    validation = metrics["validation"]
    print(json.dumps(metrics, indent=2))
    print(
        f"\nValidation PR-AUC {validation['pr_auc']:.4f} "
        f"(baseline {validation['pr_auc_baseline']:.4f}) | "
        f"ROC-AUC {validation['roc_auc']:.4f} | "
        f"F1@0.5 {validation['f1_at_0.5']:.4f} | "
        f"best F1 {validation['f1_best']:.4f} at {validation['f1_best_threshold']:.4f}"
    )
    print(f"\nTop {TOP_K} each direction:\n{result['ngrams'].to_string(index=False)}")


if __name__ == "__main__":
    main()

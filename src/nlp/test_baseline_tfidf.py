"""Regression tests for the TF-IDF baseline, using tiny artificial examples.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/nlp/test_baseline_tfidf.py -v -s

No borrower data is loaded. ``data/raw/loan.csv`` is deliberately not required,
so these checks run wherever the repository is checked out; the split loader
itself is covered by test_preprocess.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.nlp import baseline_tfidf
from src.nlp.baseline_tfidf import (
    MAX_FEATURES,
    NGRAM_RANGE,
    evaluate,
    fit_baseline,
    load_baseline_split,
    strip_field_labels,
    top_ngrams,
)
from src.nlp.preprocess import FIELD_LABELS, build_text_payload


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def tiny_corpus() -> tuple[pd.Series, np.ndarray]:
    """A separable toy corpus: 'missed' marks positives, 'steady' negatives."""
    positive = ["missed payments again", "missed rent and missed cards"] * 4
    negative = ["steady salary every month", "steady income and steady work"] * 4
    text = pd.Series(positive + negative)
    labels = np.array([1] * len(positive) + [0] * len(negative))
    return text, labels


# --- strip_field_labels: the documented Analyst 2 transform ------------------


@pytest.mark.parametrize("label", sorted(FIELD_LABELS.values()))
def test_strip_removes_every_label_the_shared_contract_defines(label):
    """Every label in preprocess.FIELD_LABELS is stripped, so new fields cannot slip through."""
    assert strip_field_labels(f"{label}: borrower wording") == "borrower wording"


def test_strip_removes_labels_from_a_real_shaped_payload():
    """A full Title/Purpose/Description payload keeps its values and loses its labels."""
    payload = "Title: Paying down cards\nPurpose: credit card\nDescription: I will repay early."
    assert strip_field_labels(payload) == (
        "Paying down cards\ncredit card\nI will repay early."
    )


def test_strip_matches_build_text_payload_output_without_modifying_it():
    """Stripping the shared payload yields exactly the cleaned field values."""
    row = {"title": "Kitchen remodel", "purpose": "home_improvement", "desc": "New cabinets."}
    payload = build_text_payload(row)
    assert payload.startswith("Title: ")  # the contract we must not change
    assert strip_field_labels(payload) == "Kitchen remodel\nhome improvement\nNew cabinets."


def test_strip_preserves_a_colon_inside_borrower_prose():
    """A mid-line colon is borrower wording, not a field label, and survives."""
    payload = "Description: My plan: repay in full by June."
    assert strip_field_labels(payload) == "My plan: repay in full by June."


def test_strip_does_not_remove_a_label_word_occurring_mid_sentence():
    """The pattern is line-anchored, so 'Title:' inside a sentence is left alone."""
    payload = "Description: The form said Title: optional, so I left it blank."
    assert strip_field_labels(payload) == "The form said Title: optional, so I left it blank."


def test_strip_is_idempotent():
    """Stripping already-stripped text changes nothing."""
    once = strip_field_labels("Title: Debt\nPurpose: credit card")
    assert strip_field_labels(once) == once


def test_strip_leaves_unlabelled_text_unchanged():
    """Text with no field labels passes through untouched."""
    assert strip_field_labels("I need to consolidate.") == "I need to consolidate."


# --- evaluate: metrics and their stated baselines ----------------------------


def test_evaluate_reports_perfect_ranking_as_one():
    """Perfectly ordered scores give PR-AUC and ROC-AUC of 1.0."""
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_evaluate_reports_uninformative_scores_at_their_baselines():
    """Constant scores collapse to ROC-AUC 0.5 and PR-AUC at the prevalence."""
    labels = np.array([0, 0, 0, 1])
    metrics = evaluate(labels, np.full(4, 0.5))
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(metrics["prevalence"])


def test_evaluate_records_the_baseline_each_metric_must_beat():
    """Prevalence and 0.5 are carried alongside the scores, so neither is read bare."""
    labels = np.array([0, 0, 0, 1, 1])
    metrics = evaluate(labels, np.array([0.1, 0.2, 0.3, 0.7, 0.9]))
    assert metrics["prevalence"] == pytest.approx(0.4)
    assert metrics["pr_auc_baseline"] == pytest.approx(0.4)
    assert metrics["roc_auc_baseline"] == 0.5
    assert metrics["rows"] == 5
    assert metrics["positives"] == 2


def test_evaluate_swept_f1_is_never_worse_than_f1_at_the_default_threshold():
    """The swept best F1 bounds the 0.5 value, which balanced weighting displaces."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    metrics = evaluate(labels, scores)
    assert metrics["f1_best"] >= metrics["f1_at_0.5"] - 1e-12


def test_evaluate_reports_the_threshold_for_its_best_f1():
    """A swept F1 is meaningless without the threshold it was taken at."""
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.0 <= metrics["f1_best_threshold"] <= 1.0


# --- fit_baseline: the brief's configuration, fitted on train only -----------


def test_fit_baseline_uses_the_specified_vectorizer_configuration(tiny_corpus):
    """Unigrams plus bigrams, 5,000 features and English stop words, as briefed."""
    vectorizer, _ = fit_baseline(*tiny_corpus)
    assert vectorizer.ngram_range == NGRAM_RANGE == (1, 2)
    assert vectorizer.max_features == MAX_FEATURES == 5_000
    assert vectorizer.stop_words == "english"


def test_fit_baseline_removes_english_stop_words(tiny_corpus):
    """Stop words present in the corpus do not reach the vocabulary."""
    vectorizer, _ = fit_baseline(*tiny_corpus)
    assert "and" not in vectorizer.vocabulary_
    assert "every" not in vectorizer.vocabulary_


def test_fit_baseline_learns_vocabulary_from_training_rows_only(tiny_corpus):
    """A word seen only at validation time is absent, so no validation text informs the fit."""
    vectorizer, _ = fit_baseline(*tiny_corpus)
    assert "foreclosure" not in vectorizer.vocabulary_
    unseen = vectorizer.transform(pd.Series(["foreclosure foreclosure"]))
    assert unseen.nnz == 0


def test_fit_baseline_separates_the_two_classes_on_a_separable_corpus(tiny_corpus):
    """A toy corpus with an obvious split is ranked correctly, so the wiring is right."""
    text, labels = tiny_corpus
    vectorizer, classifier = fit_baseline(text, labels)
    scores = classifier.predict_proba(vectorizer.transform(text))[:, 1]
    assert scores[labels == 1].min() > scores[labels == 0].max()


def test_fit_baseline_applies_balanced_class_weights(tiny_corpus):
    """class_weight='balanced' is set, as the brief requires."""
    _, classifier = fit_baseline(*tiny_corpus)
    assert classifier.class_weight == "balanced"


def test_fit_baseline_is_reproducible(tiny_corpus):
    """A pinned random_state gives identical coefficients across runs."""
    _, first = fit_baseline(*tiny_corpus)
    _, second = fit_baseline(*tiny_corpus)
    np.testing.assert_allclose(first.coef_, second.coef_)


# --- top_ngrams: the ranked keyword deliverable ------------------------------


def test_top_ngrams_returns_both_directions_ranked_and_labelled(tiny_corpus):
    """k higher-risk then k lower-risk rows, each block ranked strongest association first."""
    text, labels = tiny_corpus
    vectorizer, classifier = fit_baseline(text, labels)
    x_train = vectorizer.transform(text)
    ranked = top_ngrams(vectorizer, classifier, x_train, k=3)
    assert len(ranked) == 6
    assert list(ranked["direction"]) == ["higher_risk"] * 3 + ["lower_risk"] * 3
    higher = ranked.loc[ranked["direction"].eq("higher_risk"), "coefficient"]
    lower = ranked.loc[ranked["direction"].eq("lower_risk"), "coefficient"]
    # Each block leads with its strongest association: most positive, then most negative.
    assert higher.is_monotonic_decreasing
    assert lower.is_monotonic_increasing
    assert higher.min() > lower.max()


def test_top_ngrams_ranks_the_signal_carrying_terms_first(tiny_corpus):
    """The planted terms surface in the direction they were planted in."""
    text, labels = tiny_corpus
    vectorizer, classifier = fit_baseline(text, labels)
    ranked = top_ngrams(vectorizer, classifier, vectorizer.transform(text), k=3)
    higher = set(ranked.loc[ranked["direction"].eq("higher_risk"), "ngram"])
    lower = set(ranked.loc[ranked["direction"].eq("lower_risk"), "ngram"])
    assert any("missed" in term for term in higher)
    assert any("steady" in term for term in lower)


def test_top_ngrams_reports_document_frequency_so_rare_terms_are_visible(tiny_corpus):
    """Each ranked term carries the document count and percent behind it."""
    text, labels = tiny_corpus
    vectorizer, classifier = fit_baseline(text, labels)
    x_train = vectorizer.transform(text)
    ranked = top_ngrams(vectorizer, classifier, x_train, k=3)
    assert (ranked["train_document_count"] >= 1).all()
    assert (ranked["train_document_count"] <= x_train.shape[0]).all()
    expected = 100 * ranked["train_document_count"] / x_train.shape[0]
    np.testing.assert_allclose(ranked["train_document_percent"], expected)


def test_top_ngrams_counts_documents_not_occurrences(tiny_corpus):
    """A term repeated inside one document still counts once for that document."""
    text, labels = tiny_corpus
    vectorizer, classifier = fit_baseline(text, labels)
    x_train = vectorizer.transform(text)
    ranked = top_ngrams(vectorizer, classifier, x_train, k=5)
    counts = dict(zip(ranked["ngram"], ranked["train_document_count"]))
    # "missed rent and missed cards" repeats 'missed'; it appears in 8 documents.
    assert counts.get("missed", 8) == 8


# --- load_baseline_split: field selection, without touching the raw CSV ------


def test_load_baseline_split_strips_labels_from_the_shared_payload(monkeypatch):
    """The default path vectorises the stripped shared payload, not raw desc."""
    frame = pd.DataFrame(
        {
            "text_payload": ["Title: Cards\nPurpose: credit card\nDescription: Repay early."],
            "desc": ["Repay early."],
            "target": [0],
        }
    )
    monkeypatch.setattr(baseline_tfidf, "load_original_split", lambda *a, **k: frame.copy())
    loaded = load_baseline_split("unused", "train")
    assert loaded["text_tfidf"].iloc[0] == "Cards\ncredit card\nRepay early."


def test_load_baseline_split_desc_only_drops_title_and_purpose(monkeypatch):
    """The ablation keeps the narrative alone and cleans it with the shared function."""
    frame = pd.DataFrame(
        {
            "text_payload": ["Title: Cards\nPurpose: credit card\nDescription: Repay early."],
            "desc": ["Borrower added on 01/02/13 > Repay early.<br />No new debt."],
            "target": [0],
        }
    )
    monkeypatch.setattr(baseline_tfidf, "load_original_split", lambda *a, **k: frame.copy())
    loaded = load_baseline_split("unused", "train", desc_only=True)
    assert loaded["text_tfidf"].iloc[0] == "Repay early. No new debt."

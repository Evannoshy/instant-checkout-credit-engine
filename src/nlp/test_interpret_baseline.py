"""Regression tests for the Week 8 interpretability, calibration and threshold work.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/nlp/test_interpret_baseline.py -v -s

No borrower data is loaded and ``data/raw/loan.csv`` is not required.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.nlp.interpret_baseline import (
    CROSS_FIELD_MAX_WITHIN_RATIO,
    DOCUMENT_FREQUENCY_FLOOR_PERCENT,
    RECALL_TARGETS,
    THRESHOLD_SWEEP,
    VOLATILITY_THRESHOLD,
    apply_calibrator,
    assign_year_buckets,
    document_frequency,
    extract_coefficients,
    field_texts,
    fit_calibrators,
    load_winning_params,
    logit,
    plot_calibration_curves,
    reliability_points,
    term_volatility,
    threshold_for_recall,
    threshold_sweep,
    within_field_document_counts,
)


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def payload_rows() -> pd.DataFrame:
    """Rows whose payload creates a bigram that exists in no single field.

    "cards paying" only ever arises where a title ending in "cards" is
    concatenated with a purpose beginning "paying"; no field contains it.
    """
    return pd.DataFrame(
        {
            "title": ["Consolidate cards"] * 6,
            "purpose": ["paying_debt"] * 6,
            "desc": [
                "I want to consolidate cards and reduce interest.",
                "Paying debt down steadily each month.",
            ] * 3,
            "issue_month": ["2008-05", "2010-03", "2011-07", "2012-02", "2013-04", "2013-06"],
            "text_tfidf": [
                "Consolidate cards\npaying debt\nI want to consolidate cards and reduce interest.",
                "Consolidate cards\npaying debt\nPaying debt down steadily each month.",
            ] * 3,
            "target": [1, 0, 1, 0, 1, 0],
        }
    )


# --- ingesting the tuning result ---------------------------------------------


def test_winning_params_are_read_from_the_recorded_tuning_run(tmp_path):
    """Hyperparameters are ingested from the recorded tuning metrics, not hardcoded here."""
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "selection": {
                    "best_params": {
                        "max_features": 10000,
                        "ngram_range": [1, 2],
                        "C": 0.1,
                        "sublinear_tf": True,
                    }
                }
            }
        )
    )
    params = load_winning_params(path)
    assert params == {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "C": 0.1,
        "sublinear_tf": True,
    }


def test_shipped_tuning_metrics_still_supply_the_winning_parameters():
    """The repository's own metrics file parses, so the run is reproducible."""
    params = load_winning_params()
    assert params["ngram_range"] == (1, 2)
    assert params["max_features"] in (3000, 5000, 10000)
    assert params["C"] > 0


# --- artefact detection ------------------------------------------------------


def test_field_texts_expands_purpose_underscores_like_the_shared_builder(payload_rows):
    """`purpose` underscores become spaces, matching build_text_payload."""
    assert field_texts(payload_rows)["purpose"].iloc[0] == "paying debt"


def test_within_field_counts_find_a_bigram_that_lives_inside_one_field(payload_rows):
    """A real phrase occurring inside the description is counted."""
    counts = within_field_document_counts(
        np.array(["consolidate cards"]), payload_rows, (1, 2)
    )
    assert counts[0] > 0


def test_within_field_counts_miss_a_bigram_that_only_spans_a_boundary(payload_rows):
    """A bigram formed by concatenating two fields is absent from every field.

    This is the signal the cross_field_bigram flag relies on.
    """
    counts = within_field_document_counts(np.array(["cards paying"]), payload_rows, (1, 2))
    assert counts[0] == 0


def test_cross_field_rule_tolerates_incidental_within_field_hits():
    """The flag is a ratio, not an exact zero.

    The real motivating case is "purchase major", which appears inside a single
    field in 9 of 492 documents. An exact-zero rule misses it, so the threshold
    must admit a small incidental share.
    """
    assert 0 < CROSS_FIELD_MAX_WITHIN_RATIO < 1
    assert 9 / 492 <= CROSS_FIELD_MAX_WITHIN_RATIO


def test_year_buckets_merge_the_thin_early_years(payload_rows):
    """2007-2009 become one bucket; later years stay separate."""
    buckets = assign_year_buckets(payload_rows["issue_month"])
    assert buckets.tolist() == ["2007-2009", "2010", "2011", "2012", "2013", "2013"]


def test_volatility_attributes_counts_to_the_right_term():
    """Per-bucket counts align with terms by column index, not dict insertion order.

    ``vocabulary_`` is keyed in insertion order while the matrix columns follow the
    index values, so indexing by its keys silently attributes one term's counts to
    another. A term present in every bucket must come back stable.
    """
    texts = pd.Series(["alpha beta"] * 4 + ["alpha gamma"] * 4)
    buckets = pd.Series(["2010", "2011", "2012", "2013"] * 2)
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
    vectorizer.fit(texts)
    full_terms = vectorizer.get_feature_names_out()
    result = term_volatility(
        np.array(["alpha"]), full_terms, vectorizer.vocabulary_, texts, buckets, (1, 1)
    )
    assert result.loc["alpha", "volatility_ratio"] == pytest.approx(1.0)


def test_flag_thresholds_match_the_baseline_report_findings():
    """The documented artefact criteria are the ones applied, not ad hoc values."""
    assert DOCUMENT_FREQUENCY_FLOOR_PERCENT == 1.0
    assert VOLATILITY_THRESHOLD == 10.0


def test_extract_coefficients_labels_both_directions_and_flags(payload_rows):
    """Ranked output carries direction, frequency and a retained decision per term."""
    pipeline = Pipeline(
        [
            ("vectorize", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("classify", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    pipeline.fit(payload_rows["text_tfidf"], payload_rows["target"].to_numpy())
    ranked, summary = extract_coefficients(pipeline, payload_rows, top_k=3)
    assert len(ranked) == 6
    assert list(ranked["direction"]) == ["higher_risk"] * 3 + ["lower_risk"] * 3
    assert {"low_document_frequency", "cross_field_bigram", "period_specific", "retained"} <= set(
        ranked.columns
    )
    assert summary["ranked_terms"] == 6
    assert summary["retained"] + summary["flagged_any"] == 6


# --- calibration -------------------------------------------------------------


def test_logit_is_the_inverse_of_the_logistic_function():
    """logit recovers the model margin, which is what Platt scaling operates on."""
    margins = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    probabilities = 1 / (1 + np.exp(-margins))
    np.testing.assert_allclose(logit(probabilities), margins, atol=1e-9)


@pytest.mark.parametrize("extreme", [0.0, 1.0])
def test_logit_clips_certainty_instead_of_returning_infinity(extreme):
    """A probability of exactly 0 or 1 must not produce a non-finite margin."""
    assert np.isfinite(logit(np.array([extreme])))[0]


def test_platt_calibration_preserves_ranking_exactly():
    """Platt is strictly monotone, so ROC-AUC is untouched and only thresholds move.

    This is why the threshold sweep is valid regardless of which variant ships.
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    raw = 1 / (1 + np.exp(-(y + rng.normal(0, 1.0, size=400))))
    platt = fit_calibrators(raw, y)["platt"]
    calibrated = apply_calibrator("platt", platt, raw)
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, raw))


def test_isotonic_calibration_is_monotone_but_may_tie_scores():
    """Isotonic never inverts an ordering, though it can collapse distinct scores."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    raw = rng.random(400)
    isotonic = fit_calibrators(raw, y)["isotonic"]
    order = np.argsort(raw)
    calibrated = apply_calibrator("isotonic", isotonic, raw)[order]
    assert np.all(np.diff(calibrated) >= -1e-12)
    assert len(np.unique(calibrated)) <= len(np.unique(raw))


def test_calibrators_return_valid_probabilities():
    """Both calibrators output values inside [0, 1] on unseen input."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=300)
    reference = rng.random(300)
    unseen = rng.random(100)
    for name, calibrator in fit_calibrators(reference, y).items():
        values = apply_calibrator(name, calibrator, unseen)
        assert ((values >= 0) & (values <= 1)).all(), name


def test_calibration_moves_the_mean_prediction_toward_prevalence():
    """A calibrator fitted on an inflated score brings its average back to the base rate."""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.15, size=2000)
    inflated = np.clip(0.45 + 0.25 * (y + rng.normal(0, 1, size=2000)), 0.01, 0.99)
    calibrated = apply_calibrator("platt", fit_calibrators(inflated, y)["platt"], inflated)
    assert abs(calibrated.mean() - y.mean()) < abs(inflated.mean() - y.mean())


def test_reliability_points_use_ten_quantile_bins():
    """The reliability curve is quantile-binned, as the brief specifies."""
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=500)
    points = reliability_points(y, rng.random(500), bins=10)
    assert len(points["mean_predicted"]) == len(points["observed_rate"]) <= 10
    assert points["mean_predicted"] == sorted(points["mean_predicted"])


def test_calibration_plot_is_written(tmp_path):
    """The reliability figure is produced headlessly and lands on disk."""
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=300)
    variants = {"balanced": rng.random(300), "unweighted": rng.random(300)}
    target = tmp_path / "curves.png"
    plot_calibration_curves(y, variants, target)
    assert target.is_file() and target.stat().st_size > 0


# --- threshold trade-offs ----------------------------------------------------


def test_sweep_covers_the_briefed_threshold_range():
    """Thresholds run 0.10 to 0.60 in steps of 0.05."""
    assert THRESHOLD_SWEEP[0] == 0.10
    assert THRESHOLD_SWEEP[-1] == 0.60
    assert len(THRESHOLD_SWEEP) == 11
    assert all(abs(b - a - 0.05) < 1e-9 for a, b in zip(THRESHOLD_SWEEP, THRESHOLD_SWEEP[1:]))


def test_sweep_recall_falls_as_the_threshold_rises():
    """Raising the bar can only catch fewer defaults."""
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=600)
    scores = np.clip(y * 0.2 + rng.random(600), 0, 1)
    sweep = threshold_sweep(y, scores)
    assert sweep["recall"].is_monotonic_decreasing


def test_sweep_counts_reconcile_with_the_row_total():
    """Every row lands in exactly one confusion cell at every threshold."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=200)
    sweep = threshold_sweep(y, rng.random(200))
    assert (sweep[["tn", "fp", "fn", "tp"]].sum(axis=1) == 200).all()


def test_sweep_reports_the_false_positive_cost_of_each_threshold():
    """False positives per true positive is carried, not left to the reader."""
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, size=200)
    sweep = threshold_sweep(y, rng.random(200))
    assert "false_positives_per_true_positive" in sweep.columns
    assert "flagged_rate" in sweep.columns


@pytest.mark.parametrize("target", RECALL_TARGETS)
def test_threshold_for_recall_actually_reaches_its_target(target):
    """The returned operating point captures at least the requested share of defaults."""
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, size=800)
    scores = np.clip(y * 0.3 + rng.random(800), 0, 1)
    result = threshold_for_recall(y, scores, target)
    assert result["recall"] >= target
    assert result["tp"] + result["fn"] == int((y == 1).sum())


def test_threshold_for_recall_searches_beyond_the_sweep_grid():
    """The operating point is a real score value, not snapped to a 0.05 step."""
    rng = np.random.default_rng(10)
    y = rng.integers(0, 2, size=500)
    scores = np.clip(y * 0.25 + rng.random(500), 0, 1)
    result = threshold_for_recall(y, scores, 0.8)
    assert result["threshold"] in set(np.unique(scores))


def test_threshold_for_recall_raises_when_the_target_is_unreachable():
    """An impossible recall target fails loudly instead of returning a wrong point."""
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="recall"):
        threshold_for_recall(y, np.array([0.1, 0.2, 0.3, 0.4]), 1.5)


def test_document_frequency_counts_documents_not_occurrences():
    """A term repeated within one document counts once for that document."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
    matrix = vectorizer.fit_transform(pd.Series(["alpha alpha alpha", "beta"]))
    counts = dict(zip(vectorizer.get_feature_names_out(), document_frequency(matrix)))
    assert counts["alpha"] == 1

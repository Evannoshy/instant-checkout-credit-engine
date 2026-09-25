"""Regression tests for the Week 2 tuning and out-of-fold export, on toy data.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/nlp/test_evaluate_baseline.py -v -s

No borrower data is loaded and ``data/raw/loan.csv`` is not required, so these
run on a fresh checkout and in CI. The split loader is covered by
test_preprocess.py; the Week 1 vectoriser contract by test_baseline_tfidf.py.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.nlp import evaluate_baseline
from src.nlp.evaluate_baseline import (
    DEFAULT_CONFIG,
    PREDICTION_COLUMNS,
    PROBABILITY_COLUMN,
    bootstrap_metric_interval,
    build_pipeline,
    build_probability_frame,
    confusion_at,
    export_probabilities,
    load_baseline_split,
    load_tuning_config,
    make_cv,
    out_of_fold_probabilities,
    paired_bootstrap_delta,
    pipeline_from_params,
    validate_probability_frame,
)

MODEL_SETTINGS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
    "sublinear_tf": True,
}
PARAMS = {"max_features": 500, "ngram_range": (1, 2), "C": 1.0, "sublinear_tf": True}


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def toy_corpus() -> tuple[pd.Series, np.ndarray]:
    """A separable toy corpus: 'missed' marks positives, 'steady' negatives."""
    positive = ["missed payments again", "missed rent and missed cards"] * 10
    negative = ["steady salary every month", "steady income and steady work"] * 10
    return pd.Series(positive + negative), np.array([1] * 20 + [0] * 20)


@pytest.fixture
def memorizable_corpus() -> tuple[pd.Series, np.ndarray]:
    """Each row carries a unique token predicting its label, plus shared filler.

    In-sample a model can memorise these tokens perfectly. Out of fold it cannot,
    because a held-out row's token never appears in the rows used to fit. The gap
    is what proves the folds are genuinely disjoint.
    """
    texts, labels = [], []
    for index in range(60):
        label = index % 2
        texts.append(f"marker{index} filler text")
        labels.append(label)
    return pd.Series(texts), np.array(labels)


# --- configuration -----------------------------------------------------------


def test_shipped_config_loads_and_declares_the_briefed_grid():
    """The repository config parses and matches the brief: 3 x 2 x 4 = 24 points."""
    config = load_tuning_config(DEFAULT_CONFIG)
    assert config["grid"]["max_features"] == [3000, 5000, 10000]
    assert config["grid"]["ngram_range"] == [[1, 1], [1, 2]]
    assert config["grid"]["C"] == [0.01, 0.1, 1.0, 10.0]
    combinations = (
        len(config["grid"]["max_features"])
        * len(config["grid"]["ngram_range"])
        * len(config["grid"]["C"])
    )
    assert combinations == 24
    assert config["model"]["sublinear_tf"] is True
    assert config["cv"]["n_splits"] == 5


def test_config_selection_scoring_is_pr_auc_on_training_folds():
    """Selection uses average precision, the metric the brief names first."""
    assert load_tuning_config(DEFAULT_CONFIG)["cv"]["scoring"] == "average_precision"


@pytest.mark.parametrize("table", ["model", "grid", "cv", "week1_reference", "bootstrap"])
def test_config_rejects_a_missing_table(tmp_path, table):
    """A config missing any required table fails loudly rather than defaulting."""
    config = tmp_path / "broken.toml"
    lines = [
        'model_version = "v"',
        'model_name = "m"',
        'preprocess_version = "p"',
        'feature_set = "f"',
        "[model]",
        'class_weight = "balanced"',
        "max_iter = 10",
        "random_state = 1",
        "sublinear_tf = true",
        "[grid]",
        "max_features = [10]",
        "ngram_range = [[1, 1]]",
        "C = [1.0]",
        "[cv]",
        "n_splits = 2",
        "shuffle = true",
        "random_state = 1",
        'scoring = "average_precision"',
        "[week1_reference]",
        "max_features = 10",
        "ngram_range = [1, 2]",
        "C = 1.0",
        "sublinear_tf = false",
        "[bootstrap]",
        "resamples = 5",
        "random_state = 1",
        "confidence = 0.95",
    ]
    start = lines.index(f"[{table}]")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines)
    )
    config.write_text("\n".join(lines[:start] + lines[end:]) + "\n")
    with pytest.raises(ValueError, match="missing"):
        load_tuning_config(config)


def test_config_rejects_a_missing_model_version(tmp_path):
    """An unversioned model config is rejected; D-014 requires versioned outputs."""
    source = DEFAULT_CONFIG.read_text().replace('model_version = "nlp-tfidf-tuned-v1"', "")
    path = tmp_path / "unversioned.toml"
    path.write_text(source)
    with pytest.raises(ValueError, match="model_version"):
        load_tuning_config(path)


# --- pipeline construction ---------------------------------------------------


def test_pipeline_keeps_the_vectorizer_unfitted_so_folds_refit_it():
    """The vectoriser is a Pipeline step with no vocabulary yet, so CV must refit it."""
    pipeline = build_pipeline(
        max_features=100, ngram_range=(1, 2), C=1.0, sublinear_tf=True,
        class_weight="balanced", max_iter=100, random_state=42,
    )
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["vectorize", "classify"]
    assert not hasattr(pipeline.named_steps["vectorize"], "vocabulary_")


def test_pipeline_applies_every_briefed_hyperparameter():
    """Grid values reach the vectoriser and classifier rather than silently defaulting."""
    pipeline = build_pipeline(
        max_features=3000, ngram_range=(1, 1), C=0.01, sublinear_tf=True,
        class_weight="balanced", max_iter=555, random_state=7,
    )
    vectorizer, classifier = pipeline.named_steps["vectorize"], pipeline.named_steps["classify"]
    assert vectorizer.max_features == 3000
    assert vectorizer.ngram_range == (1, 1)
    assert vectorizer.sublinear_tf is True
    assert vectorizer.stop_words == "english"
    assert classifier.C == 0.01
    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 555
    assert classifier.random_state == 7


def test_pipeline_from_params_allows_the_unweighted_diagnostic():
    """class_weight can be overridden to None for the calibration diagnostic."""
    pipeline = pipeline_from_params(PARAMS, {**MODEL_SETTINGS, "class_weight": None})
    assert pipeline.named_steps["classify"].class_weight is None


def test_week1_reference_disables_sublinear_tf():
    """The Week 1 comparison honours its own sublinear_tf=False, not the grid's True."""
    reference = load_tuning_config(DEFAULT_CONFIG)["week1_reference"]
    pipeline = pipeline_from_params(
        {**reference, "ngram_range": tuple(reference["ngram_range"])}, MODEL_SETTINGS
    )
    assert pipeline.named_steps["vectorize"].sublinear_tf is False


# --- out-of-fold probabilities ----------------------------------------------


def test_out_of_fold_covers_every_row_exactly_once(toy_corpus):
    """Every training row receives one probability in [0, 1]."""
    text, labels = toy_corpus
    probabilities = out_of_fold_probabilities(
        text, labels, pipeline_from_params(PARAMS, MODEL_SETTINGS), make_cv(
            {"n_splits": 5, "shuffle": True, "random_state": 42}
        ), n_jobs=1,
    )
    assert len(probabilities) == len(labels)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_out_of_fold_probabilities_are_not_in_sample(memorizable_corpus):
    """A model that memorises in sample cannot rank its own held-out rows.

    This is the leakage guard: if the vectoriser or classifier were fitted once on
    all rows, the out-of-fold ranking would approach the in-sample ranking instead
    of collapsing toward chance.
    """
    text, labels = memorizable_corpus
    pipeline = pipeline_from_params(PARAMS, MODEL_SETTINGS)
    cv = make_cv({"n_splits": 5, "shuffle": True, "random_state": 42})
    oof = out_of_fold_probabilities(text, labels, pipeline, cv, n_jobs=1)
    pipeline.fit(text, labels)
    in_sample = pipeline.predict_proba(text)[:, 1]
    assert roc_auc_score(labels, in_sample) > 0.95
    assert roc_auc_score(labels, oof) < 0.75


def test_out_of_fold_is_reproducible(toy_corpus):
    """The same fold definition and seed reproduce identical probabilities."""
    text, labels = toy_corpus
    settings = {"n_splits": 5, "shuffle": True, "random_state": 42}
    first = out_of_fold_probabilities(
        text, labels, pipeline_from_params(PARAMS, MODEL_SETTINGS), make_cv(settings), n_jobs=1
    )
    second = out_of_fold_probabilities(
        text, labels, pipeline_from_params(PARAMS, MODEL_SETTINGS), make_cv(settings), n_jobs=1
    )
    np.testing.assert_allclose(first, second)


def test_make_cv_honours_the_configured_folds():
    """Fold count and seed come from config, not a hardcoded value."""
    cv = make_cv({"n_splits": 4, "shuffle": True, "random_state": 7})
    assert cv.n_splits == 4
    assert cv.random_state == 7


def test_make_cv_omits_a_seed_when_not_shuffling():
    """scikit-learn rejects a random_state when shuffle is False."""
    assert make_cv({"n_splits": 3, "shuffle": False, "random_state": 7}).random_state is None


# --- the exported contract ---------------------------------------------------


def test_probability_frame_uses_the_agreed_column_contract():
    """Columns are the briefed probability name plus D-014's versioning, in order."""
    frame = build_probability_frame(
        pd.Series(["lc_000000001", "lc_000000002"]), "train",
        np.array([0.1, 0.9]), "tfidf_logistic", "nlp-tfidf-tuned-v1",
    )
    assert frame.columns.tolist() == list(PREDICTION_COLUMNS)
    assert PROBABILITY_COLUMN == "prob_default_tfidf"
    assert frame["model_version"].eq("nlp-tfidf-tuned-v1").all()
    assert frame["split"].eq("train").all()


def test_probability_frame_rejects_a_length_mismatch():
    """Ids and probabilities of different lengths fail rather than silently align."""
    with pytest.raises(ValueError, match="same length"):
        build_probability_frame(
            pd.Series(["lc_000000001"]), "train", np.array([0.1, 0.2]), "m", "v"
        )


def test_validate_accepts_a_correct_frame():
    """A well-formed frame covering exactly the expected ids passes."""
    frame = build_probability_frame(
        pd.Series(["lc_000000001", "lc_000000002"]), "train", np.array([0.2, 0.4]), "m", "v"
    )
    validate_probability_frame(frame, {"lc_000000001", "lc_000000002"})


def test_validate_rejects_wrong_column_order():
    """A reordered schema is rejected before any value check runs."""
    frame = build_probability_frame(
        pd.Series(["lc_000000001"]), "train", np.array([0.2]), "m", "v"
    )
    with pytest.raises(ValueError, match="columns"):
        validate_probability_frame(frame[list(reversed(PREDICTION_COLUMNS))], {"lc_000000001"})


def test_validate_rejects_duplicate_loan_ids():
    """A duplicated loan_id would break the tabular team's one-to-one merge."""
    frame = build_probability_frame(
        pd.Series(["lc_000000001", "lc_000000001"]), "train", np.array([0.2, 0.3]), "m", "v"
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_probability_frame(frame, {"lc_000000001"})


def test_validate_rejects_incomplete_coverage():
    """Missing or extra ids are reported rather than quietly dropped on merge."""
    frame = build_probability_frame(
        pd.Series(["lc_000000001"]), "train", np.array([0.2]), "m", "v"
    )
    with pytest.raises(ValueError, match="missing"):
        validate_probability_frame(frame, {"lc_000000001", "lc_000000002"})


@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_validate_rejects_probabilities_outside_the_unit_interval(bad_value):
    """A value outside [0, 1] is not a probability and is refused."""
    frame = build_probability_frame(
        pd.Series(["lc_000000001"]), "train", np.array([bad_value]), "m", "v"
    )
    with pytest.raises(ValueError, match="within"):
        validate_probability_frame(frame, {"lc_000000001"})


def test_validate_rejects_missing_probabilities():
    """A null probability fails rather than reaching another track."""
    frame = build_probability_frame(
        pd.Series(["lc_000000001"]), "train", np.array([np.nan]), "m", "v"
    )
    with pytest.raises(ValueError, match="missing"):
        validate_probability_frame(frame, {"lc_000000001"})


def test_export_round_trips_through_parquet(tmp_path):
    """What is written back is what was handed over: ids, order and values."""
    frame = build_probability_frame(
        pd.Series([f"lc_{i:09d}" for i in range(1, 51)]), "validation",
        np.linspace(0.01, 0.99, 50), "tfidf_logistic", "nlp-tfidf-tuned-v1",
    )
    written = export_probabilities(frame, tmp_path / "out.parquet")
    assert written.columns.tolist() == list(PREDICTION_COLUMNS)
    assert written["loan_id"].tolist() == frame["loan_id"].tolist()
    np.testing.assert_allclose(written[PROBABILITY_COLUMN], frame[PROBABILITY_COLUMN])


# --- metrics helpers ---------------------------------------------------------


def test_confusion_at_reports_counts_and_its_threshold():
    """A confusion matrix without its threshold is unreadable, so both are returned."""
    y_true = np.array([0, 0, 1, 1])
    result = confusion_at(y_true, np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert result == {"threshold": 0.5, "tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_paired_bootstrap_finds_no_difference_between_identical_models():
    """Comparing a model with itself gives a zero delta and an interval spanning zero."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    result = paired_bootstrap_delta(
        y_true, scores, scores, resamples=200, random_state=1, confidence=0.95
    )
    assert result["observed_delta"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0 <= result["ci_high"]


def test_paired_bootstrap_detects_a_genuinely_better_model():
    """A model that ranks well beats noise, with an interval clear of zero."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=400)
    informative = y_true + rng.normal(0, 0.3, size=400)
    noise = rng.random(400)
    result = paired_bootstrap_delta(
        y_true, informative, noise, resamples=200, random_state=1, confidence=0.95
    )
    assert result["observed_delta"] > 0
    assert result["ci_low"] > 0
    assert result["share_favouring_a"] > 0.95


def test_bootstrap_interval_brackets_the_point_estimate():
    """The reported PR-AUC lies inside its own confidence interval."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=300)
    scores = y_true + rng.normal(0, 0.5, size=300)
    result = bootstrap_metric_interval(
        y_true, scores, resamples=200, random_state=1, confidence=0.95
    )
    assert result["ci_low"] <= result["pr_auc"] <= result["ci_high"]


# --- the locked split --------------------------------------------------------


def test_test_split_is_refused_without_touching_any_file():
    """The locked split is rejected by the shared loader before data is opened."""
    with pytest.raises(ValueError, match="locked"):
        load_baseline_split("/nonexistent", "test")


def test_module_exposes_the_tracked_parquet_destination():
    """Outputs land in tracked data/nlp/, matching features_lexical, not gitignored data/processed/."""
    assert evaluate_baseline.DEFAULT_PARQUET_DIR.name == "nlp"
    assert evaluate_baseline.DEFAULT_PARQUET_DIR.parent.name == "data"

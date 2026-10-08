"""Regression tests for the lexical logistic-regression experiment.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_lexical_ablation.py -v -s

Uses synthetic frames (no borrower data) with six made-up lexical columns,
against the REAL frozen feature and logistic configs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tabular import development, lexical_ablation, logistic_baseline, preprocess, xgboost_baseline
from src.tabular.test_xgboost_baseline import make_synthetic

LEXICAL_COLUMNS = development.load_development_config()["lexical"]["columns"]
EVALUATION = {"bootstrap_resamples": 20, "seed": 42}


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def with_lexical(features: pd.DataFrame, target: pd.Series, signal: bool) -> pd.DataFrame:
    """Append six lexical columns; with signal=True the first one tracks the target."""
    rng = np.random.default_rng(5)
    lexical = pd.DataFrame(
        rng.normal(size=(len(features), len(LEXICAL_COLUMNS))),
        index=features.index, columns=LEXICAL_COLUMNS,
    )
    if signal:
        lexical[LEXICAL_COLUMNS[0]] += 3 * target.to_numpy()
    return features.join(lexical)


def run(signal: bool) -> pd.DataFrame:
    features, target, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=3, min_train_share=0.4)
    return lexical_ablation.compare_lexical(
        features, with_lexical(features, target, signal), target, folds,
        preprocess.load_feature_config(), LEXICAL_COLUMNS,
        logistic_baseline.load_logistic_config(), EVALUATION,
    )


def test_lexical_config_is_a_copy_with_six_extra_numeric_columns() -> None:
    """The 15-feature config adds the lexical columns without touching the approved config."""
    feature_config = preprocess.load_feature_config()
    before = list(feature_config["model"]["numeric"])
    variant = lexical_ablation.lexical_feature_config(feature_config, LEXICAL_COLUMNS)
    assert variant["model"]["numeric"] == [*before, *LEXICAL_COLUMNS]
    assert feature_config["model"]["numeric"] == before
    assert variant["model"]["categorical"] == feature_config["model"]["categorical"]


def test_results_report_all_four_metrics() -> None:
    """The experiment reports ROC-AUC, PR-AUC, Brier score and log loss for both variants."""
    results = run(signal=False)
    assert results["metric"].tolist() == ["roc_auc", "pr_auc", "brier_score", "log_loss"]
    assert results[["tabular_9_mean", "tabular_9_plus_lexical_6_mean"]].notna().all().all()


def test_informative_lexical_column_is_reported_as_an_improvement() -> None:
    """A lexical column that carries real signal improves ranking metrics with a CI above zero."""
    results = run(signal=True).set_index("metric")
    assert results.loc["roc_auc", "improves"]
    assert results.loc["pr_auc", "improves"]


def test_noise_lexical_columns_are_not_reported_as_an_improvement() -> None:
    """Pure-noise lexical columns do not produce a claimed ROC-AUC improvement."""
    results = run(signal=False).set_index("metric")
    assert not results.loc["roc_auc", "improves"]


def test_misaligned_loan_ids_are_rejected() -> None:
    """A lexical frame in a different loan_id order is refused before any fit."""
    features, target, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=3, min_train_share=0.4)
    shuffled = with_lexical(features, target, signal=False).iloc[::-1]
    with pytest.raises(ValueError, match="not aligned"):
        lexical_ablation.compare_lexical(
            features, shuffled, target, folds, preprocess.load_feature_config(),
            LEXICAL_COLUMNS, logistic_baseline.load_logistic_config(), EVALUATION,
        )


@pytest.mark.parametrize(
    ("metric", "low", "high", "expected"),
    [("roc_auc", 0.001, 0.01, True), ("roc_auc", -0.001, 0.01, False),
     ("log_loss", -0.01, -0.001, True), ("brier_score", -0.01, 0.001, False)],
)
def test_improvement_requires_the_whole_ci_on_the_better_side(
    metric: str, low: float, high: float, expected: bool
) -> None:
    """Higher-is-better metrics need ci_low > 0; lower-is-better metrics need ci_high < 0."""
    assert lexical_ablation._improves(metric, low, high) is expected

"""Regression tests for the bounded XGBoost search and its selection rule.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_xgboost_search.py -v -s

Uses synthetic frames (no borrower data) and the REAL frozen XGBoost config.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.tabular import preprocess, xgboost_baseline, xgboost_search
from src.tabular.test_xgboost_baseline import make_synthetic

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
XGBOOST_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_xgboost_v1.toml"
SEARCH_RULE = {"primary_metric": "roc_auc", "tolerance": 0.002, "max_std_ratio": 1.5}


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def config_with_trials(trials: list[dict]) -> dict:
    """The real config with a replaced trial list, as a new dict."""
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    return {**config, "search": {**config["search"], "trials": trials}}


def trial_frame(rows: list[tuple]) -> pd.DataFrame:
    """Trials as (trial_id, status, roc_auc_mean, roc_auc_std, complexity, n_estimators, max_depth)."""
    return pd.DataFrame(
        rows,
        columns=["trial_id", "status", "roc_auc_mean", "roc_auc_std", "complexity",
                 "param_n_estimators", "param_max_depth"],
    )


def test_real_config_declares_at_most_twenty_valid_trials() -> None:
    """The frozen config has 20 or fewer unique trials, and trial 0 is the Task 1 default."""
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    trials = xgboost_search.validate_trials(config)
    assert len(trials) <= 20 == config["search"]["max_trials"]
    assert xgboost_search.trial_params(config, trials[0]) == config["model"]


def test_over_budget_search_is_rejected_before_running() -> None:
    """A trial list longer than max_trials raises before any model is fitted."""
    config = config_with_trials([{"trial_id": number, "params": {}} for number in range(21)])
    with pytest.raises(ValueError, match="budget is 20"):
        xgboost_search.validate_trials(config)


@pytest.mark.parametrize(
    ("trials", "message"),
    [
        ([{"trial_id": 1, "params": {}}, {"trial_id": 1, "params": {}}], "unique"),
        ([{"trial_id": 1, "params": {"max_dpeth": 4}}], "unknown parameters"),
        ([], "no trials"),
    ],
)
def test_malformed_trial_lists_are_rejected(trials: list[dict], message: str) -> None:
    """Duplicate ids, misspelled parameters and empty lists all fail loudly."""
    with pytest.raises(ValueError, match=message):
        xgboost_search.validate_trials(config_with_trials(trials))


def test_trial_overrides_do_not_mutate_the_default() -> None:
    """Applying a trial's overrides returns a new dict and leaves [model] unchanged."""
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    before = dict(config["model"])
    params = xgboost_search.trial_params(config, {"trial_id": 9, "params": {"max_depth": 5}})
    assert params["max_depth"] == 5
    assert config["model"] == before


def test_failed_trials_are_recorded_not_skipped() -> None:
    """A trial whose fit raises appears in the log with status failed and its error message."""
    features, target, issue_month = make_synthetic(rows=600)
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=3, min_train_share=0.4)
    config = config_with_trials([
        {"trial_id": 0, "params": {"n_estimators": 20}},
        {"trial_id": 1, "params": {"n_estimators": 20, "max_depth": 9}},
    ])

    def factory(feature_config, params):
        if params["max_depth"] == 9:
            raise RuntimeError("deliberate failure")
        return xgboost_baseline.build_xgboost_pipeline(feature_config, params)

    trials = xgboost_search.run_trials(
        features, target, folds, preprocess.load_feature_config(), config, make_pipeline=factory
    )
    assert trials["trial_id"].tolist() == [0, 1]
    assert trials["status"].tolist() == ["ok", "failed"]
    assert "deliberate failure" in trials.loc[1, "error"]
    assert trials.loc[0, "roc_auc_mean"] > 0
    assert {"roc_auc_fold0", "roc_auc_fold2", "log_loss_std"} <= set(trials.columns)


def test_selection_prefers_simpler_model_within_tolerance_and_drops_unstable() -> None:
    """The top but unstable trial is excluded; among near-equal stable trials the simplest wins."""
    trials = trial_frame([
        (0, "ok", 0.720, 0.050, 2400, 300, 3),     # best score but unstable
        (1, "ok", 0.700, 0.010, 6400, 400, 4),     # reference among stable trials
        (2, "ok", 0.699, 0.010, 800, 200, 2),      # within tolerance and simpler
        (3, "ok", 0.690, 0.010, 400, 100, 2),      # simplest but clearly worse
        (4, "failed", None, None, 100, 50, 1),     # failed trials never count
    ])
    annotated, decision = xgboost_search.select_configuration(trials, SEARCH_RULE, n_folds=5)

    assert decision["reference_trial_id"] == 1
    assert decision["selected_trial_id"] == 2
    assert decision["trials_failed"] == 1 and decision["trials_unstable"] == 1
    assert annotated["eligible"].tolist() == [False, True, True, False, False]
    assert annotated["selected"].sum() == 1
    assert decision["selected_params"] == {"n_estimators": 200, "max_depth": 2}
    assert "eligible" not in trials.columns


def test_selection_margin_widens_to_one_standard_error() -> None:
    """When fold noise exceeds the tolerance, trials within one standard error are equivalent."""
    trials = trial_frame([
        (0, "ok", 0.700, 0.030, 6400, 400, 4),
        (1, "ok", 0.690, 0.030, 800, 200, 2),
    ])
    _, decision = xgboost_search.select_configuration(trials, SEARCH_RULE, n_folds=5)
    assert decision["equivalence_margin"] == pytest.approx(0.030 / 5 ** 0.5)
    assert decision["selected_trial_id"] == 1


def test_selection_fails_when_every_trial_failed() -> None:
    """With no successful trial there is nothing to select, so it raises."""
    trials = trial_frame([(0, "failed", None, None, 800, 200, 2)])
    with pytest.raises(ValueError, match="No search trial succeeded"):
        xgboost_search.select_configuration(trials, SEARCH_RULE, n_folds=5)

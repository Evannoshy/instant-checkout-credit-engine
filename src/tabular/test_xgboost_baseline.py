"""Regression tests for the XGBoost pipeline, time-aware folds and experiment runners.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_xgboost_baseline.py -v -s

Most checks use small synthetic frames shaped like the approved
tabular_features_v1 output (no borrower data), against the REAL frozen
feature, ablation and XGBoost configs. Checks that need the real raw CSV are
skipped unless LENDINGCLUB_RAW_CSV is set, matching the other tabular tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tabular import development, logistic_baseline, preprocess, xgboost_baseline

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
XGBOOST_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_xgboost_v1.toml"
REAL_DATA_DIR = REPOSITORY_ROOT / "data"
MONTHS = [f"{year}-{month:02d}" for year in (2010, 2011) for month in range(1, 13)]


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def make_synthetic(rows: int = 1200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic (features, target, issue_month) with the nine model columns and some NaNs."""
    rng = np.random.default_rng(seed)
    index = pd.Index([f"lc_{number:09d}" for number in range(1, rows + 1)], name="loan_id")
    dti = rng.uniform(0, 35, rows)
    features = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 35000, rows),
            "annual_inc_log": np.log1p(rng.uniform(20000, 150000, rows)),
            "dti": dti,
            "revol_util": np.where(rng.random(rows) < 0.05, np.nan, rng.uniform(0, 100, rows)),
            "delinq_2yrs": rng.integers(0, 3, rows).astype(float),
            "inq_last_6mths": rng.integers(0, 5, rows).astype(float),
            "credit_history_months": rng.uniform(24, 400, rows),
            "term": rng.choice(["36", "60"], rows),
            "home_ownership": rng.choice(["MORTGAGE", "RENT", "OWN"], rows),
        },
        index=index,
    )
    probability = 1 / (1 + np.exp(-(-2.5 + 0.08 * dti)))
    target = pd.Series((rng.random(rows) < probability).astype("int8"), index=index, name="target")
    issue_month = pd.Series(np.sort(rng.choice(MONTHS, rows)), index=index, name="issue_month")
    return features, target, issue_month


def default_pipeline():
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    return xgboost_baseline.build_xgboost_pipeline(preprocess.load_feature_config(), config["model"])


def test_real_config_loads_with_conservative_default() -> None:
    """The real XGBoost config loads and its default is shallow, seeded and single-threaded."""
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    assert config["model"]["max_depth"] <= 3
    assert config["model"]["random_state"] == 42
    assert config["model"]["n_jobs"] == 1


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("[cv]\nn_folds = 1\nmin_train_share = 0.4\n", "n_folds"),
        ("[cv]\nn_folds = 5\nmin_train_share = 1.0\n", "min_train_share"),
    ],
)
def test_config_rejects_bad_cv_settings(tmp_path: Path, replacement: str, message: str) -> None:
    """A config with fewer than two folds or a share outside (0, 1) fails loudly."""
    text = XGBOOST_CONFIG.read_text(encoding="utf-8")
    text = text.replace("[cv]\nn_folds = 5\nmin_train_share = 0.4\n", replacement)
    path = tmp_path / "xgboost.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        xgboost_baseline.load_xgboost_config(path)


def test_config_rejects_missing_model_key(tmp_path: Path) -> None:
    """A [model] table missing a required hyperparameter fails loudly."""
    text = XGBOOST_CONFIG.read_text(encoding="utf-8").replace("max_depth = 3\n", "", 1)
    path = tmp_path / "xgboost.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="max_depth"):
        xgboost_baseline.load_xgboost_config(path)


def test_probabilities_cover_every_row_including_missing_and_unseen_values() -> None:
    """Every requested loan gets a probability in [0, 1], even with NaNs or an unseen category."""
    features, target, _ = make_synthetic()
    scoring = features.iloc[:50].copy()
    scoring.loc[scoring.index[:10], ["dti", "revol_util", "loan_amnt"]] = np.nan
    scoring.loc[scoring.index[10:20], "home_ownership"] = "OTHER"
    scoring.loc[scoring.index[20:25], "term"] = np.nan

    probabilities = default_pipeline().fit(features, target).predict_proba(scoring)[:, 1]

    assert len(probabilities) == len(scoring)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_same_seed_gives_identical_predictions() -> None:
    """Two fits with the same seed and data give bit-identical probabilities."""
    features, target, _ = make_synthetic()
    first = default_pipeline().fit(features, target).predict_proba(features)[:, 1]
    second = default_pipeline().fit(features, target).predict_proba(features)[:, 1]
    np.testing.assert_array_equal(first, second)


def test_columns_outside_the_feature_lists_never_reach_the_model() -> None:
    """Dropping a family from the config removes its columns from the fitted model input."""
    features, target, _ = make_synthetic()
    feature_config = preprocess.load_feature_config()
    sets = development.get_ablation_feature_sets(
        feature_config, development.load_ablation_config(feature_config=feature_config)
    )
    variant = sets["without_income_affordability"]
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    pipeline = xgboost_baseline.build_xgboost_pipeline(variant, config["model"]).fit(features, target)

    names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    assert not any(
        removed in name for name in names for removed in ("annual_inc_log", "dti", "home_ownership")
    )
    assert pipeline.named_steps["preprocess"].remainder == "drop"


def test_folds_are_time_ordered_disjoint_and_expanding() -> None:
    """Each fold trains only on earlier months, holdouts never overlap and never split a month."""
    _, _, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=4, min_train_share=0.4)

    assert len(folds) == 4
    seen_holdout: set[str] = set()
    for previous, fold in zip((None, *folds), folds):
        assert issue_month.loc[fold.train_ids].max() < fold.holdout_first_month
        assert set(fold.holdout_ids).isdisjoint(fold.train_ids)
        assert seen_holdout.isdisjoint(fold.holdout_ids)
        seen_holdout.update(fold.holdout_ids)
        if previous is not None:
            assert len(fold.train_ids) > len(previous.train_ids)
            assert fold.holdout_first_month > previous.holdout_last_month
    for month in issue_month.loc[sorted(seen_holdout)].unique():
        assert set(issue_month.index[issue_month.eq(month)]) <= seen_holdout


def test_first_holdout_starts_after_the_minimum_training_share() -> None:
    """Rows before min_train_share of the data are never held out."""
    _, _, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=4, min_train_share=0.4)
    assert len(folds[0].train_ids) >= 0.35 * len(issue_month)


def test_folds_fail_when_months_are_too_few() -> None:
    """Asking for more folds than there are later months raises instead of returning empty folds."""
    index = pd.Index([f"lc_{n:09d}" for n in range(1, 31)], name="loan_id")
    issue_month = pd.Series(["2010-01"] * 20 + ["2010-02"] * 10, index=index)
    with pytest.raises(ValueError, match="Too few months|no training months"):
        xgboost_baseline.time_aware_folds(issue_month, n_folds=5, min_train_share=0.4)


def test_cross_validation_scores_every_holdout_row_once() -> None:
    """Out-of-fold predictions cover each holdout loan exactly once and the summary has all metrics."""
    features, target, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=3, min_train_share=0.4)
    result = xgboost_baseline.cross_validate(default_pipeline, features, target, folds)

    expected = sorted({loan for fold in folds for loan in fold.holdout_ids})
    assert result.oof_predictions.index.tolist() == expected
    assert len(result.fold_metrics) == 3
    assert set(result.summary) == set(xgboost_baseline.SUMMARY_METRICS)


def test_cross_validation_rejects_unaligned_rows() -> None:
    """Features and target with different loan_id order are refused."""
    features, target, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=3, min_train_share=0.4)
    with pytest.raises(ValueError, match="not aligned"):
        xgboost_baseline.cross_validate(default_pipeline, features, target.iloc[::-1], folds)


def test_paired_bootstrap_is_zero_for_identical_models_and_seeded() -> None:
    """Identical predictions give a zero delta and CI; the same seed reproduces the interval."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 500)
    scores = rng.random(500)
    identical = xgboost_baseline.paired_bootstrap_delta(
        y_true, scores, scores, "roc_auc", resamples=50, seed=3
    )
    assert identical == {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    better = np.clip(scores + 0.3 * y_true, 0, 1)
    first = xgboost_baseline.paired_bootstrap_delta(
        y_true, scores, better, "pr_auc", resamples=50, seed=3
    )
    second = xgboost_baseline.paired_bootstrap_delta(
        y_true, scores, better, "pr_auc", resamples=50, seed=3
    )
    assert first == second
    assert first["ci_low"] > 0


def test_test_split_is_rejected_before_any_file_is_read(tmp_path: Path) -> None:
    """Requesting the locked test split through either loader raises the lock message."""
    with pytest.raises(ValueError, match="final test set is locked"):
        development.load_tabular_role(tmp_path, tmp_path / "missing.csv", "test")
    with pytest.raises(ValueError, match="final test set is locked"):
        preprocess.load_tabular_split(tmp_path, tmp_path / "missing.csv", "test")


def fast_xgboost_config() -> dict:
    """The real config with a small bootstrap, as a new dict (the loaded one is not mutated)."""
    config = xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    return {**config, "evaluation": {**config["evaluation"], "bootstrap_resamples": 20}}


def test_default_comparison_scores_both_models_on_the_same_rows() -> None:
    """Task 1 reports LR and XGBoost summaries plus paired deltas for all four metrics."""
    features, target, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=3, min_train_share=0.4)
    report = xgboost_baseline.compare_default_with_logistic(
        features, target, folds, preprocess.load_feature_config(), fast_xgboost_config(),
        logistic_baseline.load_logistic_config(),
    )

    for model in ("logistic_regression", "xgboost_default"):
        assert set(report[model]["summary"]) == set(xgboost_baseline.SUMMARY_METRICS)
        assert [fold["rows"] for fold in report[model]["folds"]] == [
            len(fold.holdout_ids) for fold in folds
        ]
    assert set(report["xgboost_minus_logistic"]) == set(xgboost_baseline.SUMMARY_METRICS)
    for delta in report["xgboost_minus_logistic"].values():
        assert delta["ci_low"] <= delta["ci_high"]


def test_feature_ablation_removes_one_family_at_a_time() -> None:
    """Task 2 gives an all-feature row plus one row per family, with matching removed features."""
    features, target, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=3, min_train_share=0.4)
    feature_config = preprocess.load_feature_config()
    ablation_config = development.load_ablation_config(feature_config=feature_config)

    results = xgboost_baseline.run_feature_ablation_on(
        features, target, folds, feature_config, ablation_config, fast_xgboost_config()
    )

    families = ablation_config["families"]
    assert results["variant"].tolist() == ["all", *(f"without_{name}" for name in families)]
    reference = results.iloc[0]
    assert reference["n_features"] == 9 and reference["removed_features"] == ""
    assert reference["delta_roc_auc"] == 0 and reference["delta_pr_auc_ci_high"] == 0
    for name, spec in families.items():
        row = results.set_index("variant").loc[f"without_{name}"]
        assert row["removed_features"] == ";".join(spec["features"])
        assert row["n_features"] == 9 - len(spec["features"])


def test_paired_deltas_refuse_models_scored_on_different_rows() -> None:
    """Deltas between results with different out-of-fold rows raise instead of misaligning."""
    features, target, issue_month = make_synthetic()
    folds = xgboost_baseline.time_aware_folds(issue_month, n_folds=3, min_train_share=0.4)
    full = xgboost_baseline.cross_validate(default_pipeline, features, target, folds)
    partial = xgboost_baseline.cross_validate(default_pipeline, features, target, folds[1:])
    with pytest.raises(ValueError, match="same out-of-fold rows"):
        xgboost_baseline.paired_deltas(
            target, full, partial, ("roc_auc",), {"bootstrap_resamples": 5, "seed": 0}
        )


@pytest.mark.skipif(
    not os.environ.get("LENDINGCLUB_RAW_CSV"), reason="LENDINGCLUB_RAW_CSV is not set"
)
def test_real_model_fit_folds_stay_inside_model_fit() -> None:
    """On the real data, every fold row is a model_fit loan and the manifest exposes no test rows."""
    raw_csv = Path(os.environ["LENDINGCLUB_RAW_CSV"])
    features, target, issue_month = xgboost_baseline.load_model_fit(REAL_DATA_DIR, raw_csv)
    folds = xgboost_baseline.build_folds(
        issue_month, xgboost_baseline.load_xgboost_config(XGBOOST_CONFIG)
    )
    model_fit = set(features.index)
    for fold in folds:
        assert set(fold.train_ids) <= model_fit and set(fold.holdout_ids) <= model_fit
    assert issue_month.max() <= "2012-12"
    assert target.index.equals(features.index)
    assert xgboost_baseline.count_test_rows_available(REAL_DATA_DIR) == 0

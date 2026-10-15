"""Tests for the unweighted logistic comparison and its Markdown report.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_unweighted_comparison.py -v -s

Unit checks use small synthetic prediction files and fabricated results (no
borrower data). The committed-report checks rebuild the comparison from the
committed prediction files and the split manifest; they do not need the raw CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tabular import calibrate, unweighted_comparison, unweighted_report

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPOSITORY_ROOT / "data"
ENTRIES = unweighted_comparison.compared_entries("xgboost_trial1")


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def synthetic_scores(rows: int, seed: int, shift: float = 0.0) -> calibrate.Scores:
    rng = np.random.default_rng(seed)
    y = (rng.random(rows) < 0.2).astype(np.int64)
    p = np.clip(0.15 + 0.2 * y + shift + rng.normal(0, 0.1, rows), 0.01, 0.99)
    return calibrate.Scores(np.array([f"lc_{i:05d}" for i in range(rows)]), p, y)


def synthetic_candidates() -> dict[str, calibrate.Candidate]:
    """Three candidates on the same synthetic calibration and validation loans."""
    def candidate(name: str, shift: float) -> calibrate.Candidate:
        sets = {"calibration": synthetic_scores(400, 1, shift), "validation": synthetic_scores(300, 2, shift)}
        return calibrate.Candidate(name, "model", f"{name}-v", sets)

    return {
        "logistic_regression": candidate("logistic_regression", 0.3),
        "unweighted_lr_v1": candidate("unweighted_lr_v1", 0.0),
        "xgboost_trial1": candidate("xgboost_trial1", 0.0),
    }


def delta(value: float, low: float, high: float) -> dict[str, float]:
    return {"delta": value, "ci_low": low, "ci_high": high}


def test_four_candidates_are_compared_in_a_fixed_order() -> None:
    """Balanced raw, balanced + Platt, unweighted raw and the selected XGBoost raw."""
    assert [(e.key, e.candidate, e.method) for e in ENTRIES] == [
        ("balanced_lr_raw", "logistic_regression", "raw"),
        ("balanced_lr_platt", "logistic_regression", "platt"),
        ("unweighted_lr_raw", "unweighted_lr_v1", "raw"),
        ("xgboost_selected_raw", "xgboost_trial1", "raw"),
    ]


def test_platt_mapping_is_fitted_on_balanced_calibration_rows_and_applied_only_to_balanced() -> None:
    """The Platt version equals calibrate.fit_platt on balanced calibration rows; other rows stay raw."""
    candidates = synthetic_candidates()
    versions, details = unweighted_comparison.validation_versions(ENTRIES, candidates, 1e-6)
    balanced = candidates["logistic_regression"].sets
    expected = calibrate.fit_platt(balanced["calibration"].p, balanced["calibration"].y, 1e-6)
    np.testing.assert_allclose(versions["balanced_lr_platt"], expected.transform(balanced["validation"].p))
    assert details == expected.details
    np.testing.assert_array_equal(versions["balanced_lr_raw"], balanced["validation"].p)
    np.testing.assert_array_equal(
        versions["unweighted_lr_raw"], candidates["unweighted_lr_v1"].sets["validation"].p
    )


def test_the_balanced_platt_mapping_is_never_applied_to_the_unweighted_model() -> None:
    """Asking for Platt on the unweighted model raises instead of reusing the balanced mapping."""
    wrong = (*ENTRIES, unweighted_comparison.Entry("bad", "Bad", "unweighted_lr_v1", "platt"))
    with pytest.raises(ValueError, match="cannot be applied to unweighted_lr_v1"):
        unweighted_comparison.validation_versions(wrong, synthetic_candidates(), 1e-6)


@pytest.mark.parametrize(
    ("metric", "difference", "expected"),
    [
        ("log_loss", delta(-0.01, -0.012, -0.008), "better"),
        ("log_loss", delta(0.01, 0.008, 0.012), "worse"),
        ("log_loss", delta(-0.001, -0.0015, -0.0005), "no clear difference"),
        ("log_loss", delta(-0.01, -0.02, 0.001), "no clear difference"),
        ("roc_auc", delta(0.01, 0.008, 0.012), "better"),
        ("roc_auc", delta(-0.01, -0.012, -0.008), "worse"),
    ],
    ids=["lower_beyond_margin", "higher_beyond_margin", "inside_margin", "ci_spans_zero",
         "auc_up", "auc_down"],
)
def test_differences_count_only_beyond_the_margin_with_a_ci_excluding_zero(metric, difference, expected) -> None:
    """A verdict needs both the predeclared margin and a 95% CI that excludes zero, in the right direction."""
    assert unweighted_comparison.classify(difference, metric, margin=0.002) == expected


def test_every_reference_is_compared_on_every_metric() -> None:
    """Unweighted LR is compared with the three references on ROC-AUC, PR-AUC, Brier and log loss."""
    candidates = synthetic_candidates()
    versions, _ = unweighted_comparison.validation_versions(ENTRIES, candidates, 1e-6)
    config = {
        "evaluation": {"bootstrap_resamples": 50, "seed": 1},
        "decision": {"margins": {"roc_auc": 0.002, "pr_auc": 0.002, "brier_score": 0.0005, "log_loss": 0.002}},
    }
    y = candidates["logistic_regression"].sets["validation"].y
    differences = unweighted_comparison.paired_differences(ENTRIES, versions, y, config)
    assert set(differences) == {"balanced_lr_raw", "balanced_lr_platt", "xgboost_selected_raw"}
    for metrics in differences.values():
        assert set(metrics) == set(calibrate.METRICS)
        assert all(m["ci_low"] <= m["delta"] <= m["ci_high"] for m in metrics.values())


def write_predictions(path: Path, ids: list[str], split: str, version: str) -> None:
    pd.DataFrame({
        "loan_id": ids, "split": split, "p_default_tabular": 0.2,
        "model_name": "logistic_regression", "model_version": version,
    }).to_csv(path, index=False)


def test_candidates_must_cover_the_same_validation_loans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A prediction file missing one validation loan is refused, so all rows grade the same IDs."""
    ids = {"calibration": ["c1", "c2"], "validation": ["v1", "v2", "v3"]}
    target = pd.Series([0, 1, 0, 1, 0], index=["c1", "c2", "v1", "v2", "v3"])
    monkeypatch.setattr(calibrate, "expected_prediction_ids", lambda _: ({k: set(v) for k, v in ids.items()}, target))
    paths = {}
    for name in ("a", "b"):
        paths[name] = {}
        for set_name, set_ids in ids.items():
            paths[name][set_name] = tmp_path / f"{name}_{set_name}.csv"
            write_predictions(paths[name][set_name], set_ids, set_name, f"{name}-v")
    loaded = unweighted_comparison.load_compared_candidates(paths, tmp_path)
    assert list(loaded["a"].sets["validation"].loan_ids) == list(loaded["b"].sets["validation"].loan_ids)

    write_predictions(paths["b"]["validation"], ["v1", "v2"], "validation", "b-v")
    with pytest.raises(ValueError, match="1 missing"):
        unweighted_comparison.load_compared_candidates(paths, tmp_path)


def test_reference_rows_that_drift_from_merged_evidence_are_refused() -> None:
    """A Platt parameter or reference metric that differs from the merged evidence raises."""
    reference_metrics = json.loads(unweighted_comparison.REFERENCE_METRICS.read_text(encoding="utf-8"))
    reference_results = pd.read_csv(unweighted_comparison.REFERENCE_RESULTS)
    recorded = reference_metrics["candidates"]["logistic_regression"]["calibrators"]["platt"]
    rows = {}
    for entry in ENTRIES:
        merged = reference_results[
            reference_results["candidate"].eq(entry.candidate) & reference_results["method"].eq(entry.method)
            & reference_results["set"].eq("validation")
        ]
        rows[entry.key] = merged.iloc[0].to_dict() if len(merged) else {}
    unweighted_comparison.check_reference_reproduction(ENTRIES, rows, recorded, reference_metrics, reference_results)

    with pytest.raises(ValueError, match="Platt slope"):
        unweighted_comparison.check_reference_reproduction(
            ENTRIES, rows, {**recorded, "slope": recorded["slope"] + 0.01}, reference_metrics, reference_results
        )
    drifted = {**rows, "balanced_lr_platt": {**rows["balanced_lr_platt"], "log_loss": 0.41}}
    with pytest.raises(ValueError, match="log_loss does not reproduce"):
        unweighted_comparison.check_reference_reproduction(
            ENTRIES, drifted, recorded, reference_metrics, reference_results
        )


def fabricated_result(reference_verdict: str) -> dict:
    """A minimal comparison result for rendering checks."""
    def metrics(verdict: str) -> dict:
        return {name: {**delta(-0.01, -0.02, -0.005), "verdict": verdict} for name in calibrate.METRICS}

    rows = [
        {"key": e.key, "label": e.label, "model_version": "v", "roc_auc": 0.66, "pr_auc": 0.25,
         "brier_score": 0.12, "log_loss": 0.40, "mean_p": 0.15, "prevalence": 0.15}
        for e in ENTRIES
    ]
    return {
        "candidates": rows,
        "unweighted_minus_reference": {
            "balanced_lr_raw": metrics("better"),
            "balanced_lr_platt": metrics(reference_verdict),
            "xgboost_selected_raw": metrics("no clear difference"),
        },
        "platt_mapping": {"slope": 1.0, "intercept": -1.7},
        "margins": {"roc_auc": 0.002, "pr_auc": 0.002, "brier_score": 0.0005, "log_loss": 0.002},
        "bootstrap": {"bootstrap_resamples": 1000, "seed": 42},
        "validation_rows": 21784,
        "test_rows_available_to_model": 0,
        "inputs_sha256": {"reports/x.csv": "a" * 64},
    }


def test_report_shows_every_candidate_metric_and_the_observed_default_rate() -> None:
    """The results table names all four candidates and the six requested columns."""
    report = unweighted_report.render_report(fabricated_result("no clear difference"))
    for entry in ENTRIES:
        assert f"| {entry.label} |" in report
    for column in ("ROC-AUC", "PR-AUC", "Brier", "Log loss", "Mean predicted risk", "Observed default rate"):
        assert column in report
    assert "not approval to replace the reference model" in report


@pytest.mark.parametrize(
    ("reference_verdict", "phrase"),
    [
        ("no clear difference", "does not clearly beat the calibrated reference"),
        ("better", "the result is clearly better"),
        ("worse", "the result is clearly worse"),
    ],
)
def test_report_answer_follows_the_computed_verdicts(reference_verdict: str, phrase: str) -> None:
    """The answer sentence changes with the verdicts instead of being written by hand."""
    report = unweighted_report.render_report(fabricated_result(reference_verdict))
    assert "clearly improves probability quality compared with the balanced model's raw scores" in report
    assert phrase in report


def test_mixed_verdicts_are_reported_metric_by_metric() -> None:
    """When Brier and log loss disagree, the summary names each metric's verdict."""
    metrics = {"brier_score": {"verdict": "better"}, "log_loss": {"verdict": "no clear difference"}}
    summary = unweighted_report.verdict_summary(metrics, ("brier_score", "log_loss"))
    assert summary == "mixed (Brier score better, Log loss no clear difference)"


def committed_result() -> dict:
    if not unweighted_comparison.RESULTS_PATH.exists():
        pytest.skip("The comparison has not been run yet")
    return json.loads(unweighted_comparison.RESULTS_PATH.read_text(encoding="utf-8"))


def test_committed_comparison_grades_four_candidates_on_21784_validation_loans() -> None:
    """The committed result covers the four candidates on the same 21,784 validation loans with no test rows."""
    result = committed_result()
    assert result["validation_rows"] == 21_784
    assert result["test_rows_available_to_model"] == 0
    assert [row["key"] for row in result["candidates"]] == [entry.key for entry in ENTRIES]
    assert {row["rows"] for row in result["candidates"]} == {21_784}
    assert len({row["prevalence"] for row in result["candidates"]}) == 1


def test_committed_comparison_and_report_are_reproduced_from_committed_files(tmp_path: Path) -> None:
    """Rebuilding from the committed prediction files reproduces the committed JSON and Markdown."""
    saved = committed_result()
    rebuilt = unweighted_comparison.run_comparison(
        REAL_DATA_DIR, report_path=tmp_path / "report.md", results_path=tmp_path / "result.json"
    )
    assert rebuilt["inputs_sha256"] == saved["inputs_sha256"]
    for actual, expected in zip(rebuilt["candidates"], saved["candidates"], strict=True):
        for key, value in expected.items():
            if isinstance(value, float):
                assert actual[key] == pytest.approx(value, rel=1e-9, abs=1e-12)
            else:
                assert actual[key] == value
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == (
        unweighted_comparison.REPORT_PATH.read_text(encoding="utf-8")
    )

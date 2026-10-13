"""Calibration tests using the committed config, tiny artificial data and the committed files.

Most tests use made-up data: loading tests write a manifest and prediction files to
tmp_path, and calibrator tests use synthetic arrays. The last section checks the
committed evidence: the real prediction files load with the expected counts, and the
committed results match a recomputation within tight numerical tolerances. The full recomputation (bootstrap included,
about 90 seconds) runs only when CALIBRATION_RECOMPUTE is set. No test reads the raw CSV.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_tabular_calibrate.py -v -s

Include the full recomputation (PowerShell):
    $env:CALIBRATION_RECOMPUTE = "1"; python -m pytest src/tabular/test_tabular_calibrate.py -v -s

The -s flag shows print statements; -v shows each PASSED/FAILED result.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.tabular import calibrate, evaluate
from src.tabular.xgboost_baseline import validate_environment
from src.tabular.calibrate import (
    CANDIDATES,
    ERROR_MESSAGES,
    SETS,
    Candidate,
    Scores,
    approval_curve,
    choose_champion,
    choose_method,
    clearly_better,
    default_rate_at_approval,
    evaluate_candidates,
    fit_calibrators,
    fit_isotonic,
    fit_platt,
    load_calibration_config,
    load_candidates,
    policy_table,
    run_calibration,
    threshold_grid,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_calibration_v1.toml"
DEVELOPMENT_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_development_v1.toml"
REAL_DATA_DIR = REPOSITORY_ROOT / "data"
REAL_PREDICTION_DIR = REPOSITORY_ROOT / "reports" / "tabular" / "raw_predictions"
REAL_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "tabular"
EVIDENCE_FILES = ("calibration_metrics.json", "calibration_results.csv", "policy_curve.csv")

# Hardcoded on purpose, from the split manifest and the development roles, so a changed
# input file fails a test instead of being echoed back by it: (rows, defaults).
REAL_COUNTS = {"calibration": (26_940, 4_090), "validation": (21_784, 3_298)}

EPS = 1e-6
FAST_RESAMPLES = 20  # enough to exercise the bootstrap; the real run uses the config's 1,000
METHODS = ["raw", "platt", "isotonic"]

# Fixture rows per role/split and the issue months they cycle through.
FIXTURE_ROWS = {"model_fit": 10, "calibration": 60, "validation": 40, "test": 2}
FIXTURE_MONTHS = {
    "model_fit": ["2012-06", "2012-09", "2012-12"],
    "calibration": ["2013-01", "2013-04", "2013-07"],
    "validation": ["2013-08", "2013-10", "2013-12"],
    "test": ["2014-01"],
}
FIXTURE_COUNTS = [("model_fit = 59353", "model_fit = 10"), ("calibration = 26940", "calibration = 60")]


# Each candidate's raw probability as a function of a loan's true risk:
# inflated like balanced logistic regression, about right, and about right with noise.
def _inflated(risk: np.ndarray) -> np.ndarray:
    return np.clip(3 * risk, 0.01, 0.95)


def _accurate(risk: np.ndarray) -> np.ndarray:
    return np.clip(risk, 0.001, 0.999)


def _noisy(risk: np.ndarray) -> np.ndarray:
    jitter = np.random.default_rng(7).normal(scale=0.02, size=len(risk))
    return np.clip(risk + jitter, 0.001, 0.999)


CANDIDATE_TRANSFORMS = {
    "logistic_regression": _inflated, "xgboost_default": _accurate, "xgboost_trial1": _noisy,
}


def error_pattern(key: str) -> str:
    """Regex for the full ERROR_MESSAGES[key] text, allowing any value in each {placeholder}."""
    literal_parts = re.split(r"\{[^}]*\}", ERROR_MESSAGES[key])
    return "^" + ".*".join(re.escape(part) for part in literal_parts) + "$"


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def write_config_variant(tmp_path: Path, old: str, new: str) -> Path:
    """Copy the committed config with one exact text replacement."""
    text = CALIBRATION_CONFIG.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must appear exactly once in the config"
    path = tmp_path / "calibration.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def fast_config() -> dict:
    """The committed config with a small bootstrap, so decision-rule tests run quickly."""
    config = copy.deepcopy(load_calibration_config(CALIBRATION_CONFIG))
    config["evaluation"]["bootstrap_resamples"] = FAST_RESAMPLES
    return config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_committed_config_loads() -> None:
    """The committed calibration config passes validation."""
    load_calibration_config(CALIBRATION_CONFIG)


def test_threshold_grid_has_101_exact_points_from_zero_to_one() -> None:
    """The grid is 0.00, 0.01, ..., 1.00 with no floating-point drift."""
    grid = threshold_grid(load_calibration_config(CALIBRATION_CONFIG))
    assert len(grid) == 101
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert np.all(np.diff(grid) > 0)
    assert 0.07 in grid and 0.29 in grid


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ('development_version = "tabular_development_v1"', "development_version = 1",
         "development_version"),
        ('methods = ["raw", "platt", "isotonic"]', "methods = []", "methods"),
        ("probability_clip = 1e-6", "probability_clip = 0", "probability_clip"),
        ("probability_clip = 1e-6", "probability_clip = 0.1", "probability_clip"),
        ("calibration_plot_bins = 10", "calibration_plot_bins = 1", "calibration_plot_bins"),
        ("calibration_plot_bins = 10", "calibration_plot_bins = 10.0", "calibration_plot_bins"),
        ('calibration_plot_strategy = "quantile"', 'calibration_plot_strategy = "kmeans"',
         "calibration_plot_strategy"),
        ('approve_rule = "p < threshold"', "approve_rule = true", "[policy] approve_rule"),
        ("stop = 1.00", "stop = 1.50", "[policy] thresholds"),
        ("step = 0.01", "step = 0", "[policy] thresholds"),
        ("step = 0.01", "step = 0.03", "[policy] thresholds"),
        ("illustrative_thresholds = [0.10, 0.15, 0.20, 0.25]",
         "illustrative_thresholds = [0.15, 0.10]", "[policy] illustrative_thresholds"),
        ("approval_rates = [0.50, 0.70, 0.80, 0.90]", "approval_rates = [0.50, 1.20]",
         "[policy] approval_rates"),
        ("bootstrap_resamples = 1000", "bootstrap_resamples = 0", "[evaluation] bootstrap_resamples"),
        ("seed = 42", "seed = -1", "[evaluation] seed"),
        ('method_metric = "log_loss"', 'method_metric = "accuracy"', "[decision] method_metric"),
        ('challenger = "xgboost_trial1"', 'challenger = "xgboost_trial7"', "[decision] challenger"),
        ("pr_auc = 0.002        # same as ROC-AUC\n", "", "[decision.margins]"),
        ("brier_score = 0.0005", "brier_score = -0.0005", "[decision.margins] brier_score"),
    ],
)
def test_bad_value_is_rejected(tmp_path: Path, old: str, new: str, field: str) -> None:
    """A setting of the wrong type or outside its range raises ValueError naming the setting."""
    path = write_config_variant(tmp_path, old, new)
    with pytest.raises(ValueError, match=error_pattern("bad_value")) as error:
        load_calibration_config(path)
    assert str(error.value).startswith(f"Calibration config {field} must be")


@pytest.mark.parametrize(
    ("old", "new", "section"),
    [
        ('calibration_version = "tabular_calibration_v1"\n', "", "top level"),
        ("[evaluation]\nbootstrap_resamples = 1000\n", "[evaluation]\n", "evaluation"),
        (", step = 0.01 }", " }", "policy.thresholds"),
    ],
)
def test_missing_key_is_rejected(tmp_path: Path, old: str, new: str, section: str) -> None:
    """A missing key raises ValueError naming its table."""
    path = write_config_variant(tmp_path, old, new)
    with pytest.raises(ValueError, match=error_pattern("missing_keys")) as error:
        load_calibration_config(path)
    assert f"[{section}]" in str(error.value)


# ---------------------------------------------------------------------------
# Synthetic arrays and candidates (no files)
# ---------------------------------------------------------------------------

def draw_loans(n: int, seed: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(loan_ids, true risk, outcomes) for n made-up loans at ~15% prevalence."""
    rng = np.random.default_rng(seed)
    risk = expit(rng.normal(-1.9, 0.8, n))
    outcomes = rng.binomial(1, risk)
    loan_ids = np.array([f"lc_{offset + number:09d}" for number in range(n)])
    return loan_ids, risk, outcomes


def make_candidates(n: int = 2000, seed: int = 0) -> dict[str, Candidate]:
    """All three candidates on the same made-up calibration and validation loans."""
    sets = {
        "calibration": draw_loans(n, seed, offset=0),
        "validation": draw_loans(n, seed + 1, offset=n),
    }
    return {
        name: Candidate(name, name.split("_")[0], f"{name}-synthetic", {
            set_name: Scores(loan_ids, transform(risk), outcomes)
            for set_name, (loan_ids, risk, outcomes) in sets.items()
        })
        for name, transform in CANDIDATE_TRANSFORMS.items()
    }


def replace_outcomes(
    candidates: dict[str, Candidate], set_name: str, outcomes: np.ndarray
) -> dict[str, Candidate]:
    """Copy of candidates with one set's outcomes swapped for every candidate."""
    return {
        name: Candidate(c.name, c.model_name, c.model_version, {
            **c.sets,
            set_name: Scores(c.sets[set_name].loan_ids, c.sets[set_name].p, outcomes),
        })
        for name, c in candidates.items()
    }


def test_validation_labels_never_reach_a_calibrator() -> None:
    """Permuting validation outcomes leaves every calibrated validation probability unchanged."""
    candidates = make_candidates()
    config = fast_config()
    original = evaluate_candidates(candidates, config)
    validation_y = candidates[CANDIDATES[0]].sets["validation"].y
    shuffled_y = np.random.default_rng(1).permutation(validation_y)
    shuffled = evaluate_candidates(replace_outcomes(candidates, "validation", shuffled_y), config)
    for name in CANDIDATES:
        for method in METHODS:
            np.testing.assert_array_equal(
                original.validation_versions[name][method], shuffled.validation_versions[name][method]
            )


def test_calibration_labels_do_change_the_calibrators() -> None:
    """Changing calibration outcomes changes calibrated validation probabilities (the fit is real)."""
    candidates = make_candidates()
    config = fast_config()
    original = evaluate_candidates(candidates, config)
    calibration_y = candidates[CANDIDATES[0]].sets["calibration"].y
    flipped = evaluate_candidates(replace_outcomes(candidates, "calibration", 1 - calibration_y), config)
    for name in CANDIDATES:
        for method in ("platt", "isotonic"):
            assert not np.allclose(
                original.validation_versions[name][method], flipped.validation_versions[name][method]
            )


@pytest.mark.parametrize("fit", [fit_platt, fit_isotonic])
def test_calibrated_outputs_are_finite_and_within_the_clip(fit) -> None:
    """Platt and isotonic outputs are numeric, finite and in [eps, 1 - eps], even for 0, 1 and
    scores outside the calibration range."""
    _, risk, outcomes = draw_loans(2000, seed=2)
    calibrator = fit(np.clip(risk, 0.05, 0.6), outcomes, EPS)
    scores = np.concatenate([[0.0, 1e-12, 0.01, 0.99, 1.0], risk])
    output = calibrator.transform(scores)
    assert output.dtype == np.float64
    assert np.isfinite(output).all()
    assert output.min() >= EPS and output.max() <= 1 - EPS


def test_platt_preserves_ranking_exactly() -> None:
    """Raw and Platt probabilities have the same ROC-AUC and PR-AUC (to 1e-12)."""
    _, risk, outcomes = draw_loans(3000, seed=3)
    raw = _inflated(risk)
    platt = fit_platt(raw, outcomes, EPS).transform(raw)
    assert roc_auc_score(outcomes, platt) == pytest.approx(roc_auc_score(outcomes, raw), abs=1e-12)
    assert average_precision_score(outcomes, platt) == pytest.approx(
        average_precision_score(outcomes, raw), abs=1e-12
    )


def test_platt_recovers_known_parameters() -> None:
    """Outcomes drawn from sigmoid(0.5 * logit(p) - 1) give a fitted slope ~0.5, intercept ~-1."""
    rng = np.random.default_rng(4)
    raw = rng.uniform(0.02, 0.98, 40_000)
    outcomes = rng.binomial(1, expit(0.5 * logit(raw) - 1.0))
    details = fit_platt(raw, outcomes, EPS).details
    assert details["slope"] == pytest.approx(0.5, abs=0.05)
    assert details["intercept"] == pytest.approx(-1.0, abs=0.05)


def test_isotonic_output_is_non_decreasing() -> None:
    """Isotonic output never goes down as the raw score goes up."""
    _, risk, outcomes = draw_loans(3000, seed=5)
    calibrator = fit_isotonic(_noisy(risk), outcomes, EPS)
    output = calibrator.transform(np.linspace(0, 1, 1001))
    assert np.all(np.diff(output) >= 0)
    assert calibrator.details["steps"] >= 2


def test_miscalibrated_scores_improve_after_calibration() -> None:
    """Scores 3x too high (like balanced LR) get better Brier and log loss on held-out loans."""
    _, cal_risk, cal_y = draw_loans(5000, seed=6)
    _, val_risk, val_y = draw_loans(5000, seed=7)
    calibrators = fit_calibrators(_inflated(cal_risk), cal_y, METHODS, EPS)
    raw = _inflated(val_risk)
    for method in ("platt", "isotonic"):
        calibrated = calibrators[method].transform(raw)
        assert brier_score_loss(val_y, calibrated) < brier_score_loss(val_y, raw)
        assert log_loss(val_y, calibrated) < log_loss(val_y, raw)


def test_isotonic_zero_is_clipped_so_log_loss_stays_finite() -> None:
    """When the lowest scores never default, isotonic gives 0; the clip keeps log loss finite."""
    raw = np.linspace(0.01, 0.99, 200)
    outcomes = (raw > 0.5).astype(int)
    calibrator = fit_isotonic(raw, outcomes, EPS)
    output = calibrator.transform(raw)
    assert output.min() == EPS
    wrong_low_loan = np.array([1, *outcomes[1:]])  # the lowest-score loan defaults after all
    assert np.isfinite(log_loss(wrong_low_loan, output, labels=[0, 1]))


def test_unknown_method_is_rejected() -> None:
    """A method with no fitter raises ValueError before anything is fit."""
    _, risk, outcomes = draw_loans(200, seed=8)
    with pytest.raises(ValueError, match=error_pattern("unknown_methods")):
        fit_calibrators(risk, outcomes, ["raw", "beta"], EPS)


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("metric", "delta", "expected"),
    [
        ("log_loss", {"delta": -0.010, "ci_low": -0.015, "ci_high": -0.005}, True),
        ("log_loss", {"delta": -0.001, "ci_low": -0.002, "ci_high": -0.0005}, False),  # within margin
        ("log_loss", {"delta": -0.010, "ci_low": -0.020, "ci_high": 0.001}, False),    # CI crosses 0
        ("log_loss", {"delta": 0.010, "ci_low": 0.005, "ci_high": 0.015}, False),      # worse
        ("roc_auc", {"delta": 0.010, "ci_low": 0.005, "ci_high": 0.015}, True),
        ("roc_auc", {"delta": 0.001, "ci_low": 0.0005, "ci_high": 0.002}, False),      # within margin
        ("roc_auc", {"delta": -0.010, "ci_low": -0.015, "ci_high": -0.005}, False),    # worse
    ],
)
def test_clearly_better_needs_margin_and_a_ci_excluding_zero(
    metric: str, delta: dict, expected: bool
) -> None:
    """Better means beyond the margin in the right direction, with the CI on the right side of 0."""
    assert clearly_better(delta, metric, margin=0.002) is expected


def _comparison(delta: float) -> dict:
    """A log-loss comparison with a tight CI around delta."""
    return {"log_loss": {"delta": delta, "ci_low": delta - 0.001, "ci_high": delta + 0.001}}


def test_method_rule_moves_on_only_when_the_next_method_is_clearly_better() -> None:
    """Platt clearly beats raw, isotonic does not clearly beat Platt: Platt is chosen."""
    comparisons = {
        "platt_minus_raw": _comparison(-0.05),
        "isotonic_minus_raw": _comparison(-0.051),
        "isotonic_minus_platt": _comparison(-0.001),
    }
    chosen, steps = choose_method(comparisons, METHODS, "log_loss", margin=0.002)
    assert chosen == "platt"
    assert [(s["baseline"], s["candidate"], s["clearly_better"]) for s in steps] == [
        ("raw", "platt", True), ("platt", "isotonic", False),
    ]


def test_method_rule_compares_later_methods_with_the_current_choice() -> None:
    """If Platt does not beat raw, isotonic is compared with raw, not with Platt."""
    comparisons = {
        "platt_minus_raw": _comparison(0.0),
        "isotonic_minus_raw": _comparison(-0.01),
        "isotonic_minus_platt": _comparison(-0.01),
    }
    chosen, steps = choose_method(comparisons, METHODS, "log_loss", margin=0.002)
    assert chosen == "isotonic"
    assert steps[1]["baseline"] == "raw"


def test_identical_challenger_does_not_win() -> None:
    """A challenger with the incumbent's exact probabilities keeps the incumbent."""
    _, risk, outcomes = draw_loans(2000, seed=9)
    margins = fast_config()["decision"]["margins"]
    outcome = choose_champion(outcomes, risk, risk.copy(), margins, fast_config()["evaluation"])
    assert outcome["challenger_wins"] is False
    assert all(d["delta"] == 0 for d in outcome["challenger_minus_incumbent"].values())


def test_clearly_better_challenger_wins() -> None:
    """A well-calibrated challenger beats an incumbent that is 3x too high on probability quality."""
    _, risk, outcomes = draw_loans(4000, seed=10)
    config = fast_config()
    outcome = choose_champion(
        outcomes, _inflated(risk), risk, config["decision"]["margins"], config["evaluation"]
    )
    assert outcome["challenger_clearly_better_on_probability_quality"] is True
    assert outcome["challenger_not_worse_on_roc_auc"] is True
    assert outcome["challenger_wins"] is True


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def test_policy_table_counts_and_rates() -> None:
    """Approval is monotone in the threshold; counts match a direct count; ends behave."""
    _, risk, outcomes = draw_loans(1000, seed=11)
    thresholds = threshold_grid(load_calibration_config(CALIBRATION_CONFIG))
    table = policy_table(risk, outcomes, thresholds)

    assert np.all(np.diff(table["approved"]) >= 0)
    for row in table.itertuples():
        approved = risk < row.threshold
        assert row.approved == approved.sum()
        assert row.defaults_among_approved == outcomes[approved].sum()
    first, last = table.iloc[0], table.iloc[-1]
    assert first["approved"] == 0 and np.isnan(first["default_rate_among_approved"])
    assert last["approval_rate"] == 1.0
    assert last["default_rate_among_approved"] == pytest.approx(outcomes.mean())


def test_policy_threshold_is_strict() -> None:
    """A loan whose probability equals the threshold is not approved (p < threshold)."""
    table = policy_table(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 0]), np.array([0.2]))
    assert table.loc[0, "approved"] == 1


def test_approval_curve_and_table_depend_only_on_ranking() -> None:
    """Raw and Platt give the same threshold-free curve; it ends at the overall default rate."""
    _, risk, outcomes = draw_loans(2000, seed=12)
    raw = _inflated(risk)
    platt = fit_platt(raw, outcomes, EPS).transform(raw)
    raw_curve, platt_curve = approval_curve(raw, outcomes), approval_curve(platt, outcomes)
    np.testing.assert_array_equal(raw_curve[1], platt_curve[1])
    assert raw_curve[0][-1] == 1.0
    assert raw_curve[1][-1] == pytest.approx(outcomes.mean())

    table = default_rate_at_approval(raw, outcomes, [0.5, 1.0])
    lowest_half = np.argsort(raw, kind="stable")[:1000]
    assert table.loc[0, "approved"] == 1000
    assert table.loc[0, "defaults_among_approved"] == outcomes[lowest_half].sum()
    assert table.loc[1, "default_rate_among_approved"] == pytest.approx(outcomes.mean())


# ---------------------------------------------------------------------------
# Whole run on synthetic candidates
# ---------------------------------------------------------------------------

def test_same_inputs_give_identical_outputs() -> None:
    """Two runs on the same inputs give identical results, policy curves and bootstrap CIs."""
    config = fast_config()
    first = evaluate_candidates(make_candidates(), config)
    second = evaluate_candidates(make_candidates(), config)
    assert_frame_equal(first.results, second.results)
    assert_frame_equal(first.policy, second.policy)
    assert first.summary == second.summary


def test_validation_rows_must_be_the_same_loans_for_every_candidate() -> None:
    """Paired comparisons are refused if one candidate's validation loans differ."""
    candidates = make_candidates()
    other = candidates["xgboost_trial1"]
    validation = other.sets["validation"]
    candidates["xgboost_trial1"] = Candidate(other.name, other.model_name, other.model_version, {
        **other.sets,
        "validation": Scores(validation.loan_ids[::-1], validation.p, validation.y),
    })
    with pytest.raises(ValueError, match=error_pattern("rows_not_aligned")):
        evaluate_candidates(candidates, fast_config())


def test_results_cover_every_candidate_method_and_set() -> None:
    """One results row per candidate x method x set, and one policy row per threshold of each."""
    config = fast_config()
    run = evaluate_candidates(make_candidates(), config)
    assert len(run.results) == len(CANDIDATES) * len(METHODS) * len(SETS)
    assert len(run.policy) == len(CANDIDATES) * len(METHODS) * len(threshold_grid(config))
    assert run.summary["champion"]["recommendation"] in (
        config["decision"]["incumbent"], config["decision"]["challenger"]
    )


# ---------------------------------------------------------------------------
# Loading from files: artificial manifest and prediction files in tmp_path
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixtureData:
    data_dir: Path
    development_config: Path
    prediction_dir: Path
    ids: dict[str, list[str]]       # role or split -> loan_ids
    risk: pd.Series                 # true risk by loan_id, used to make predictions
    target: pd.Series               # outcome by loan_id (train and validation only)


def _write_predictions(prediction_dir: Path, ids: dict[str, list[str]], risk: pd.Series) -> None:
    """Shared-schema files for every candidate and set, rows shuffled on purpose."""
    prediction_dir.mkdir(exist_ok=True)
    for name, transform in CANDIDATE_TRANSFORMS.items():
        for set_name in SETS:
            frame = evaluate.build_prediction_frame(
                loan_ids=ids[set_name], split_name=set_name,
                probabilities=transform(risk.loc[ids[set_name]].to_numpy()),
                model_name=name.split("_")[0], model_version=f"{name}-fixture",
            )
            frame.sample(frac=1, random_state=3).to_csv(
                prediction_dir / f"{name}_{set_name}.csv", index=False
            )


@pytest.fixture
def fixture_data(tmp_path: Path) -> FixtureData:
    """112 made-up loans: 10 model_fit, 60 calibration, 40 validation, 2 locked test rows."""
    rng = np.random.default_rng(13)
    groups = [group for group, count in FIXTURE_ROWS.items() for _ in range(count)]
    loan_ids = [f"lc_{number:09d}" for number in range(1, len(groups) + 1)]
    months = [FIXTURE_MONTHS[group][position % len(FIXTURE_MONTHS[group])]
              for position, group in enumerate(groups)]
    risk = pd.Series(expit(rng.normal(-1.2, 0.8, len(groups))), index=loan_ids)
    outcomes = rng.binomial(1, risk.to_numpy())
    splits = ["test" if g == "test" else "validation" if g == "validation" else "train" for g in groups]
    targets = [("LOCKED_TEST_LABEL" if s == "test" else int(o)) for s, o in zip(splits, outcomes)]

    version = "synthetic-calibration-fixture-v1"
    manifest = pd.DataFrame({
        "loan_id": loan_ids,
        "source_row_number": range(1, len(groups) + 1),
        "split": splits,
        "issue_month": months,
        "target": targets,
        "text_available": 1,
        "cohort": "real_text_matured_v1",
        "dataset_version": version,
    })
    manifest.to_csv(tmp_path / "split_manifest.csv", index=False)
    split_rows = manifest["split"].value_counts().to_dict()
    (tmp_path / "split_statistics.json").write_text(json.dumps({
        "dataset_version": version,
        "split_version": "split-v1",
        "cohort": "real_text_matured_v1",
        "raw_rows": len(manifest),
        "eligible_rows": len(manifest),
        "splits": {split: {"rows": rows} for split, rows in split_rows.items()},
    }), encoding="utf-8")

    text = DEVELOPMENT_CONFIG.read_text(encoding="utf-8")
    for old, new in FIXTURE_COUNTS:
        assert old in text, f"Expected development-config text was not found: {old}"
        text = text.replace(old, new, 1)
    development_config = tmp_path / "fixture_development.toml"
    development_config.write_text(text, encoding="utf-8")

    ids = {group: [i for i, g in zip(loan_ids, groups) if g == group] for group in FIXTURE_ROWS}
    trainable = manifest["split"].ne("test")
    target = manifest.loc[trainable].set_index("loan_id")["target"].astype(int)
    for set_name in SETS:
        assert target.loc[ids[set_name]].nunique() == 2, f"fixture {set_name} needs both classes"

    prediction_dir = tmp_path / "raw_predictions"
    _write_predictions(prediction_dir, ids, risk)
    return FixtureData(tmp_path, development_config, prediction_dir, ids, risk, target)


def _load(data: FixtureData) -> dict[str, Candidate]:
    return load_candidates(data.prediction_dir, data.data_dir, data.development_config)


def test_shuffled_prediction_rows_get_their_own_outcomes(fixture_data: FixtureData) -> None:
    """Outcomes are joined by loan_id, not row order: every loan gets its manifest target."""
    candidates = _load(fixture_data)
    assert list(candidates) == list(CANDIDATES)
    for candidate in candidates.values():
        for set_name, scores in candidate.sets.items():
            assert sorted(scores.loan_ids) == list(scores.loan_ids)
            assert set(scores.loan_ids) == set(fixture_data.ids[set_name])
            np.testing.assert_array_equal(
                scores.y, fixture_data.target.loc[scores.loan_ids].to_numpy()
            )
            expected_p = CANDIDATE_TRANSFORMS[candidate.name](
                fixture_data.risk.loc[scores.loan_ids].to_numpy()
            )
            np.testing.assert_allclose(scores.p, expected_p)


def test_calibration_ids_come_from_the_role_not_the_file(fixture_data: FixtureData) -> None:
    """Calibration files are checked against the calibration role: model_fit loans are refused."""
    path = fixture_data.prediction_dir / "xgboost_default_calibration.csv"
    frame = pd.read_csv(path, dtype={"loan_id": str})
    frame.loc[0, "loan_id"] = fixture_data.ids["model_fit"][0]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not cover the expected loan_id set exactly"):
        _load(fixture_data)


def _mutate_missing(frame: pd.DataFrame, data: FixtureData) -> pd.DataFrame:
    return frame.iloc[1:]


def _mutate_extra(frame: pd.DataFrame, data: FixtureData) -> pd.DataFrame:
    extra = frame.iloc[[0]].assign(loan_id=data.ids["test"][0])
    return pd.concat([frame, extra], ignore_index=True)


def _mutate_duplicate(frame: pd.DataFrame, data: FixtureData) -> pd.DataFrame:
    return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_mutate_missing, "does not cover the expected loan_id set exactly: 1 missing, 0 extra"),
        (_mutate_extra, "does not cover the expected loan_id set exactly: 0 missing, 1 extra"),
        (_mutate_duplicate, "duplicate loan_id values"),
    ],
)
@pytest.mark.parametrize("set_name", SETS)
def test_missing_extra_or_duplicate_ids_are_rejected(
    fixture_data: FixtureData, mutate, message: str, set_name: str
) -> None:
    """A prediction file with a missing, extra (here a locked test ID) or duplicated loan fails."""
    path = fixture_data.prediction_dir / f"logistic_regression_{set_name}.csv"
    frame = pd.read_csv(path, dtype={"loan_id": str})
    mutate(frame, fixture_data).to_csv(path, index=False)
    with pytest.raises(ValueError, match=re.escape(message)):
        _load(fixture_data)


@pytest.mark.parametrize("set_name", SETS)
@pytest.mark.parametrize("column", ("model_name", "model_version"))
def test_prediction_metadata_must_match_between_sets(
    fixture_data: FixtureData, set_name: str, column: str,
) -> None:
    """Calibration and validation files must belong to the same model and version."""
    path = fixture_data.prediction_dir / f"logistic_regression_{set_name}.csv"
    frame = pd.read_csv(path, dtype={"loan_id": str})
    frame[column] = "different-model-or-version"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=f"different {column}"):
        _load(fixture_data)


@pytest.mark.parametrize("set_name", SETS)
@pytest.mark.parametrize("column", ("model_name", "model_version"))
def test_prediction_metadata_cannot_mix_values_within_a_file(
    fixture_data: FixtureData, set_name: str, column: str,
) -> None:
    """One file cannot mix predictions labelled as different models or versions."""
    path = fixture_data.prediction_dir / f"logistic_regression_{set_name}.csv"
    frame = pd.read_csv(path, dtype={"loan_id": str})
    frame.loc[0, column] = "different-model-or-version"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=f"one non-empty {column}"):
        _load(fixture_data)


@pytest.mark.parametrize("set_name", SETS)
@pytest.mark.parametrize("column", ("model_name", "model_version"))
@pytest.mark.parametrize("value", (None, "", "   "))
def test_prediction_metadata_cannot_be_missing_or_blank(
    fixture_data: FixtureData, set_name: str, column: str, value: str | None,
) -> None:
    """Every prediction row needs a non-empty model name and version."""
    path = fixture_data.prediction_dir / f"logistic_regression_{set_name}.csv"
    frame = pd.read_csv(path, dtype={"loan_id": str})
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=f"one non-empty {column}"):
        _load(fixture_data)


@pytest.mark.parametrize("set_name", SETS)
@pytest.mark.parametrize("mixed", (False, True))
def test_prediction_split_must_match_the_requested_set(
    fixture_data: FixtureData, set_name: str, mixed: bool,
) -> None:
    """Correct IDs do not excuse a wrong or mixed split label in a prediction file."""
    path = fixture_data.prediction_dir / f"logistic_regression_{set_name}.csv"
    frame = pd.read_csv(path, dtype={"loan_id": str})
    if mixed:
        frame.loc[0, "split"] = "train"
    else:
        frame["split"] = "train"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must contain only split"):
        _load(fixture_data)


def _forbid_raw_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every path to the raw CSV raise, so a test fails if anything tries it."""
    def refuse(*args, **kwargs):
        raise AssertionError("calibration must not load the raw CSV")

    monkeypatch.setattr("src.tabular.preprocess.load_tabular_split", refuse)
    monkeypatch.setattr("src.tabular.development.load_tabular_split", refuse)
    monkeypatch.setattr("src.tabular.development.load_tabular_role", refuse)


@pytest.fixture
def fast_config_path(tmp_path: Path) -> Path:
    return write_config_variant(
        tmp_path, "bootstrap_resamples = 1000", f"bootstrap_resamples = {FAST_RESAMPLES}"
    )


def _run(data: FixtureData, config_path: Path, output_dir: Path, monkeypatch) -> dict:
    """run_calibration on the fixture, with the runtime check stubbed (not under test here)."""
    monkeypatch.setattr(calibrate, "validate_runtime", lambda: {"python": "fixture"})
    return run_calibration(
        data.data_dir, data.prediction_dir, output_dir,
        config_path=config_path, development_config=data.development_config,
    )


def test_full_run_never_reads_the_raw_csv(
    fixture_data: FixtureData, fast_config_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading and the whole run use only the manifest and prediction files (no raw/ folder exists)."""
    _forbid_raw_csv(monkeypatch)
    assert not (fixture_data.data_dir / "raw").exists()
    _run(fixture_data, fast_config_path, tmp_path / "out", monkeypatch)


def test_full_run_writes_every_evidence_file(
    fixture_data: FixtureData, fast_config_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run writes the CSVs, the JSON, three calibration plots and the policy plot."""
    output_dir = tmp_path / "out"
    _run(fixture_data, fast_config_path, output_dir, monkeypatch)
    expected = {
        "calibration_results.csv", "policy_curve.csv", "calibration_metrics.json",
        "policy_curve.png", *(f"calibration_{name}.png" for name in CANDIDATES),
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    policy = pd.read_csv(output_dir / "policy_curve.csv")
    assert len(policy) == len(CANDIDATES) * len(METHODS) * 101


def test_saved_metrics_are_strict_json_and_record_provenance(
    fixture_data: FixtureData, fast_config_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """calibration_metrics.json has no NaN tokens and records versions and zero test rows."""
    output_dir = tmp_path / "out"
    _run(fixture_data, fast_config_path, output_dir, monkeypatch)

    def refuse_constant(token: str):
        raise ValueError(f"non-standard JSON constant {token}")

    saved = json.loads(
        (output_dir / "calibration_metrics.json").read_text(encoding="utf-8"),
        parse_constant=refuse_constant,
    )
    assert saved["test_rows_available_to_model"] == 0
    assert saved["dataset_version"] == "synthetic-calibration-fixture-v1"
    assert saved["calibrator_fit_rows"] == "calibration role only"
    assert set(saved["candidates"]) == set(CANDIDATES)
    for name in CANDIDATES:
        assert saved["candidates"][name]["model_version"] == f"{name}-fixture"
        assert saved["candidates"][name]["rows"] == {
            "calibration": {"rows": 60, "positives": int(fixture_data.target.loc[
                fixture_data.ids["calibration"]].sum())},
            "validation": {"rows": 40, "positives": int(fixture_data.target.loc[
                fixture_data.ids["validation"]].sum())},
        }


def test_saved_metrics_agree_with_a_fresh_recomputation(
    fixture_data: FixtureData, fast_config_path: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every saved validation metric equals one recomputed here from the loaded files."""
    output_dir = tmp_path / "out"
    _run(fixture_data, fast_config_path, output_dir, monkeypatch)
    saved = pd.read_csv(output_dir / "calibration_results.csv")

    for name, candidate in _load(fixture_data).items():
        calibration, validation = candidate.sets["calibration"], candidate.sets["validation"]
        calibrators = fit_calibrators(calibration.p, calibration.y, METHODS, EPS)
        for method, calibrator in calibrators.items():
            expected = evaluate.evaluate_predictions(validation.y, calibrator.transform(validation.p))
            row = saved.query("candidate == @name and method == @method and set == 'validation'")
            assert len(row) == 1
            for metric in ("roc_auc", "pr_auc", "brier_score", "log_loss"):
                assert row[metric].item() == pytest.approx(expected[metric], rel=1e-12)


# ---------------------------------------------------------------------------
# Committed evidence: real prediction files and reports/tabular outputs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_candidates() -> dict[str, Candidate]:
    """Allan's committed prediction files, loaded once for this section."""
    return load_candidates(REAL_PREDICTION_DIR, REAL_DATA_DIR)


def test_committed_candidates_load_with_the_expected_counts(
    real_candidates: dict[str, Candidate],
) -> None:
    """Every committed candidate has 26,940 calibration (4,090 defaults) and 21,784 validation
    (3,298 defaults) loans, aligned loan for loan across candidates."""
    assert list(real_candidates) == list(CANDIDATES)
    for candidate in real_candidates.values():
        counts = {
            set_name: (len(scores.y), int(scores.y.sum()))
            for set_name, scores in candidate.sets.items()
        }
        assert counts == REAL_COUNTS, candidate.name
    reference = real_candidates[CANDIDATES[0]].sets["validation"]
    for candidate in real_candidates.values():
        np.testing.assert_array_equal(candidate.sets["validation"].loan_ids, reference.loan_ids)


def test_committed_results_match_recomputed_metrics(
    real_candidates: dict[str, Candidate],
) -> None:
    """Every row of the committed calibration_results.csv equals a recomputation from the
    committed prediction files (no bootstrap, so this runs in seconds)."""
    config = load_calibration_config(CALIBRATION_CONFIG)
    saved = pd.read_csv(REAL_OUTPUT_DIR / "calibration_results.csv")
    assert len(saved) == len(CANDIDATES) * len(config["methods"]) * len(SETS)

    for name, candidate in real_candidates.items():
        calibration = candidate.sets["calibration"]
        calibrators = fit_calibrators(
            calibration.p, calibration.y, config["methods"], config["probability_clip"]
        )
        for method, calibrator in calibrators.items():
            for set_name, scores in candidate.sets.items():
                expected = calibrate.metric_row(scores.y, calibrator.transform(scores.p))
                row = saved.query("candidate == @name and method == @method and set == @set_name")
                assert len(row) == 1, (name, method, set_name)
                assert row["model_version"].item() == candidate.model_version
                for key, value in expected.items():
                    assert row[key].item() == pytest.approx(value, rel=1e-12), (
                        name, method, set_name, key,
                    )


def _assert_evidence_matches(expected, actual, path: str = "evidence") -> None:
    """Compare floats within tolerance, but keep counts, types and decisions exact."""
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_evidence_matches(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (first, second) in enumerate(zip(expected, actual)):
            _assert_evidence_matches(first, second, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-14), path
    else:
        assert actual == expected, path


def test_evidence_comparison_accepts_floating_point_roundoff() -> None:
    """A one-bit floating-point difference does not invalidate otherwise identical evidence."""
    expected = {"rows": 12, "chosen_method": "platt", "metrics": [0.65], "chosen": True}
    actual = copy.deepcopy(expected)
    actual["metrics"][0] = float(np.nextafter(0.65, np.inf))
    _assert_evidence_matches(expected, actual)


@pytest.mark.parametrize(("field", "value"), [
    ("rows", 13), ("rows", 12.0), ("chosen_method", "isotonic"),
    ("metrics", [0.650001]), ("metrics", [0.65, 0.66]), ("chosen", False),
])
def test_evidence_comparison_rejects_changed_results(field: str, value) -> None:
    """Numerical tolerance cannot hide changed counts, methods, decisions or meaningful scores."""
    expected = {"rows": 12, "chosen_method": "platt", "metrics": [0.65], "chosen": True}
    actual = {**expected, field: value}
    with pytest.raises(AssertionError):
        _assert_evidence_matches(expected, actual)


@pytest.mark.skipif(
    not os.environ.get("CALIBRATION_RECOMPUTE"),
    reason="CALIBRATION_RECOMPUTE is not set (full rerun takes about 90 seconds)",
)
def test_committed_evidence_equals_a_fresh_recomputation(tmp_path: Path) -> None:
    """A pinned-runtime rerun matches numerical evidence without changing any recorded provenance."""
    run_calibration(REAL_DATA_DIR, REAL_PREDICTION_DIR, tmp_path)
    saved, fresh = (
        json.loads((directory / EVIDENCE_FILES[0]).read_text(encoding="utf-8"))
        for directory in (REAL_OUTPUT_DIR, tmp_path)
    )
    validate_environment(saved["environment"])
    validate_environment(fresh["environment"])
    assert set(fresh) == set(saved), (
        f"Keys differ: only committed {sorted(set(saved) - set(fresh))}, "
        f"only fresh {sorted(set(fresh) - set(saved))}; rerun python -m src.tabular.calibrate"
    )
    for key in saved:
        if key == "environment":
            continue
        if key in ("candidates", "champion", "results"):
            _assert_evidence_matches(saved[key], fresh[key], key)
        else:
            assert fresh[key] == saved[key], key
    for filename in EVIDENCE_FILES[1:]:
        assert_frame_equal(
            pd.read_csv(tmp_path / filename), pd.read_csv(REAL_OUTPUT_DIR / filename),
            check_exact=False, rtol=1e-12, atol=1e-14, obj=filename,
        )

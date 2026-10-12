"""Calibration tests using the committed config and tiny, artificial variants of it.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_tabular_calibrate.py -v -s

The -s flag shows print statements; -v shows each PASSED/FAILED result.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.tabular.calibrate import ERROR_MESSAGES, load_calibration_config, threshold_grid


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_calibration_v1.toml"


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

"""Late-fusion stacker tests: synthetic tables for the fitting boundaries, plus one real end-to-end run.

From the repository root:
    python -m pytest src/fusion/test_stacker.py -v

The real run uses the committed sources, writes only to pytest's temporary
directory and lowers the bootstrap to 20 resamples to stay fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.fusion import inputs, stacker

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "fusion_stacker_v1.toml"
VARIANTS = ("tabular_only", "tabular_tfidf", "tabular_tfidf_distilbert")
TRACKS = ["tabular", "tfidf"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def _table(rows: int, seed: int) -> pd.DataFrame:
    """Two noisy scores of a shared latent risk, with labels drawn from it."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=rows)
    return pd.DataFrame({
        "loan_id": [f"lc_{seed}_{index:05d}" for index in range(rows)],
        "target": (rng.random(rows) < _sigmoid(risk - 1.7)).astype("int8"),
        "p_default_tabular": _sigmoid(risk + rng.normal(scale=0.8, size=rows) - 1),
        "p_default_tfidf": _sigmoid(risk + rng.normal(scale=1.5, size=rows) - 1),
    })


@pytest.fixture()
def tables() -> dict[str, pd.DataFrame]:
    return {"fusion_fit": _table(3000, 1), "fusion_calibration": _table(2000, 2), "validation": _table(2000, 3)}


@pytest.fixture(scope="module")
def config() -> dict:
    return inputs.load_stacker_config(CONFIG_PATH)


def test_stacker_is_unweighted_and_reads_only_score_columns(tables, config):
    model = stacker.fit_stacker(tables["fusion_fit"], TRACKS, config["stacker"])
    assert model.class_weight is None
    assert model.n_features_in_ == len(TRACKS)


def test_validation_labels_never_change_the_stacker_or_calibrator(tables, config):
    before = stacker.fit_variant("tabular_tfidf", TRACKS, tables, config)
    flipped = {**tables, "validation": tables["validation"].assign(target=1 - tables["validation"]["target"])}
    after = stacker.fit_variant("tabular_tfidf", TRACKS, flipped, config)
    np.testing.assert_array_equal(before.stacker.coef_, after.stacker.coef_)
    np.testing.assert_array_equal(before.validation_platt, after.validation_platt)


def test_calibration_labels_change_only_the_calibrator(tables, config):
    before = stacker.fit_variant("tabular_tfidf", TRACKS, tables, config)
    calibration = tables["fusion_calibration"]
    relabelled = {**tables, "fusion_calibration": calibration.assign(target=1 - calibration["target"])}
    after = stacker.fit_variant("tabular_tfidf", TRACKS, relabelled, config)
    np.testing.assert_array_equal(before.stacker.coef_, after.stacker.coef_)
    assert before.platt.details != after.platt.details


def test_single_input_stacker_keeps_the_ranking_of_its_input(tables, config):
    fitted = stacker.fit_variant("tabular_only", ["tabular"], tables, config)
    y, raw = tables["validation"]["target"], tables["validation"]["p_default_tabular"]
    assert roc_auc_score(y, fitted.validation_platt) == pytest.approx(roc_auc_score(y, raw), abs=1e-12)


def test_comparisons_cover_reference_and_text_versus_tabular_only():
    names = ["reference", "tabular_only_raw", "tabular_only_platt", "tabular_tfidf_raw", "tabular_tfidf_platt"]
    pairs = stacker.comparison_pairs({name: np.zeros(1) for name in names})
    assert ("tabular_only_platt", "reference") in pairs
    assert ("tabular_tfidf_platt", "tabular_only_platt") in pairs
    assert ("tabular_tfidf_raw", "tabular_only_platt") not in pairs
    assert len(pairs) == 5


def test_comparisons_include_distilbert_over_tfidf_when_both_text_variants_run():
    names = ["reference", "tabular_only_platt", "tabular_tfidf_platt", "tabular_tfidf_distilbert_platt"]
    pairs = stacker.comparison_pairs({name: np.zeros(1) for name in names})
    assert ("tabular_tfidf_distilbert_platt", "tabular_tfidf_platt") in pairs
    assert len(pairs) == 6


def test_artifact_records_fit_roles_and_unweighted_settings(tables, config):
    fitted = stacker.fit_variant("tabular_only", ["tabular"], tables, config)
    artifact = stacker.variant_artifact(fitted, config)
    assert artifact["stacker"]["fit_role"] == "fusion_fit"
    assert artifact["stacker"]["class_weight"] is None
    assert artifact["calibrator"]["fit_role"] == "fusion_calibration"
    assert artifact["input_columns"] == ["p_default_tabular"]
    assert set(artifact["calibrator"]) >= {"slope", "intercept", "probability_clip"}


# --- committed real sources --------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("stacker")
    fast = out / "stacker.toml"
    fast.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace("bootstrap_resamples = 1000", "bootstrap_resamples = 20"),
        encoding="utf-8",
    )
    metrics = stacker.run_stacker(fast, report_dir=out / "reports", artifact_dir=out / "artifacts")
    return metrics, out


def test_real_run_fits_and_grades_on_the_agreed_rows(real_run):
    metrics, _ = real_run
    assert metrics["rows"] == {"fusion_fit": 15_587, "fusion_calibration": 11_353, "validation": 21_784}
    assert metrics["test_rows_read"] == 0
    assert set(metrics["validation_metrics"]) == {
        "reference", *(f"{variant}_{kind}" for variant in VARIANTS for kind in ("raw", "platt")),
    }
    assert "tabular_tfidf_distilbert_platt_minus_tabular_tfidf_platt" in metrics["comparisons"]
    assert metrics["decision"]["status"] == "pending lead confirmation"


def test_real_tabular_only_stacker_cannot_change_ranking(real_run):
    metrics, _ = real_run
    rows = metrics["validation_metrics"]
    for name in ("tabular_only_raw", "tabular_only_platt"):
        assert rows[name]["roc_auc"] == pytest.approx(rows["reference"]["roc_auc"], abs=1e-9)


def test_real_run_writes_versioned_predictions_and_checksummed_outputs(real_run):
    metrics, out = real_run
    validation = pd.read_parquet(out / "artifacts" / "predictions_validation.parquet")
    calibration = pd.read_parquet(out / "artifacts" / "predictions_fusion_calibration.parquet")
    assert len(validation) == 21_784 and len(calibration) == 11_353
    assert validation["stacker_version"].eq("fusion-stacker-v1").all()
    assert not validation["loan_id"].duplicated().any()
    written = json.loads((out / "reports" / "stacker_v1_metrics.json").read_text(encoding="utf-8"))
    assert sorted(Path(path).name for path in written["outputs_sha256"]) == [
        "predictions_fusion_calibration.parquet", "predictions_validation.parquet",
        "stacker_v1_calibration.png", "stacker_v1_results.csv",
        "tabular_only.json", "tabular_tfidf.json", "tabular_tfidf_distilbert.json",
    ]

"""Regression tests for the logistic-regression tabular baseline.

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/tabular/test_logistic_baseline.py -v -s

Most checks use a tiny, synthetic 14-row manifest + raw CSV written under
tmp_path (no borrower data), reused against the REAL, frozen
configs/tabular_features_v1.toml and configs/tabular_logistic_v1.toml, exactly
as src/tabular/test_tabular_preprocess.py already does for load_tabular_split.
The synthetic fixture deliberately gives validation a home_ownership level
("OWN") that never appears in training, to exercise the
handle_unknown="ignore" path. One check at the end runs the real pipeline
against the real data/ directory and the real raw CSV; it is skipped unless
LENDINGCLUB_RAW_CSV is set, matching src/tabular/test_tabular_preprocess.py's
own convention, since the raw CSV is gitignored and may not exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tabular import evaluate, logistic_baseline, preprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_features_v1.toml"
LOGISTIC_CONFIG = REPOSITORY_ROOT / "configs" / "tabular_logistic_v1.toml"
REAL_DATA_DIR = REPOSITORY_ROOT / "data"
REAL_RAW_CSV = REPOSITORY_ROOT / "data" / "raw" / "loan.csv"


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest) -> None:
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def artificial_data_dir(tmp_path: Path) -> Path:
    """Fourteen made-up loans: 8 train, 4 validation, 2 locked test, no borrower data.

    Training home_ownership is only MORTGAGE/RENT; validation additionally
    includes OWN, which training never sees -- this exercises the model
    pipeline's handle_unknown="ignore" path end to end.
    """
    (tmp_path / "raw").mkdir()
    raw = pd.DataFrame(
        {
            "loan_amnt": [
                "10000", "12000", "15000", "9000", "11000", "13000", "10500", "9500",
                "10200", "14000", "9800", "12500", "10000", "11000",
            ],
            "term": [
                " 36 months", " 36 months", " 60 months", " 36 months",
                " 36 months", " 60 months", " 36 months", " 36 months",
                " 36 months", " 36 months", " 60 months", " 36 months",
                " 36 months", " 60 months",
            ],
            "annual_inc": [
                "50000", "40000", "70000", "35000", "55000", "38000", "52000", "36000",
                "51000", "42000", "37000", "39000", "45000", "48000",
            ],
            "dti": [
                "15.0", "20.0", "10.0", "25.0", "14.0", "22.0", "13.0", "24.0",
                "15.5", "18.0", "21.0", "23.0", "16.0", "17.0",
            ],
            "revol_util": [
                "30.0", "60.0", "25.0", "70.0", "28.0", "65.0", "27.0", "68.0",
                "29.0", "55.0", "62.0", "66.0", "33.0", "31.0",
            ],
            "delinq_2yrs": ["0", "1", "0", "2", "0", "1", "0", "2", "0", "1", "0", "1", "0", "0"],
            "inq_last_6mths": ["1", "2", "0", "3", "1", "2", "1", "2", "1", "2", "1", "3", "1", "1"],
            "earliest_cr_line": [
                "Jan-2000", "Feb-2001", "Mar-2002", "Apr-2003", "May-2004", "Jun-2005",
                "Jul-2006", "Aug-2007", "Sep-2003", "Oct-2002", "Nov-2004", "Dec-2005",
                "Jan-2006", "Feb-2007",
            ],
            "home_ownership": [
                "MORTGAGE", "RENT", "MORTGAGE", "RENT", "MORTGAGE", "RENT", "MORTGAGE", "RENT",
                "MORTGAGE", "OWN", "RENT", "RENT", "OTHER", "MORTGAGE",
            ],
            "issue_d": [
                "Jan-2013", "Feb-2013", "Mar-2013", "Apr-2013", "May-2013", "Jun-2013",
                "Jul-2013", "Aug-2013", "Sep-2013", "Oct-2013", "Nov-2013", "Dec-2013",
                "Jan-2014", "Feb-2014",
            ],
            "loan_status": [
                "Fully Paid", "Charged Off", "Fully Paid", "Charged Off",
                "Fully Paid", "Charged Off", "Fully Paid", "Charged Off",
                "Fully Paid", "Charged Off", "Fully Paid", "Charged Off",
                "Fully Paid", "Charged Off",
            ],
        }
    )
    raw.to_csv(tmp_path / "raw" / "loan.csv", index=False)

    version = "synthetic-logistic-fixture-v1"
    splits = ["train"] * 8 + ["validation"] * 4 + ["test"] * 2
    targets: list[object] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, "LOCKED_TEST_LABEL", "LOCKED_TEST_LABEL"]
    manifest = pd.DataFrame(
        {
            "loan_id": [f"lc_{n:09d}" for n in range(1, 15)],
            "source_row_number": list(range(1, 15)),
            "split": splits,
            "issue_month": [
                "2013-01", "2013-02", "2013-03", "2013-04", "2013-05", "2013-06",
                "2013-07", "2013-08", "2013-09", "2013-10", "2013-11", "2013-12",
                "2014-01", "2014-02",
            ],
            "target": targets,
            "text_available": [1] * 14,
            "cohort": ["real_text_matured_v1"] * 14,
            "dataset_version": [version] * 14,
        }
    )
    manifest.to_csv(tmp_path / "split_manifest.csv", index=False)
    manifest.loc[manifest["split"].eq("train"), ["loan_id"]].to_csv(
        tmp_path / "train_ids.csv", index=False
    )
    manifest.loc[manifest["split"].eq("validation"), ["loan_id"]].to_csv(
        tmp_path / "val_ids.csv", index=False
    )

    statistics = {
        "dataset_version": version,
        "split_version": "split-v1",
        "cohort": "real_text_matured_v1",
        "raw_rows": 14,
        "eligible_rows": 14,
        "splits": {
            "train": {"rows": 8},
            "validation": {"rows": 4},
            "test": {"rows": 2},
        },
    }
    (tmp_path / "split_statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    return tmp_path


def _raw_csv(data_dir: Path) -> Path:
    return data_dir / "raw" / "loan.csv"


# --- config loading ---


def test_load_logistic_config_returns_expected_hyperparameters():
    """The real logistic config loads with exactly the four approved hyperparameters."""
    config = logistic_baseline.load_logistic_config(LOGISTIC_CONFIG)
    assert config["model"] == {
        "C": 1.0, "class_weight": "balanced", "max_iter": 1000, "random_state": 42,
    }


def test_load_logistic_config_rejects_missing_required_key(tmp_path):
    """A config missing one of the four required hyperparameters is rejected."""
    path = tmp_path / "bad_logistic.toml"
    path.write_text('[model]\nC = 1.0\nclass_weight = "balanced"\nmax_iter = 1000\n')
    with pytest.raises(ValueError):
        logistic_baseline.load_logistic_config(path)


# --- pipeline structure ---


def test_pipeline_uses_the_approved_feature_columns_in_order():
    """The pipeline's numeric/categorical column lists exactly match tabular_features_v1."""
    feature_config = preprocess.load_feature_config(FEATURE_CONFIG)
    logistic_config = logistic_baseline.load_logistic_config(LOGISTIC_CONFIG)
    pipeline = logistic_baseline.build_pipeline(feature_config, logistic_config)
    transformers = {name: columns for name, _, columns in pipeline.named_steps["preprocess"].transformers}
    assert transformers["numeric"] == feature_config["model"]["numeric"]
    assert transformers["categorical"] == feature_config["model"]["categorical"]


def test_pipeline_hyperparameters_match_the_config_file():
    """The fitted classifier's hyperparameters come from the config, not hardcoded defaults."""
    feature_config = preprocess.load_feature_config(FEATURE_CONFIG)
    logistic_config = logistic_baseline.load_logistic_config(LOGISTIC_CONFIG)
    pipeline = logistic_baseline.build_pipeline(feature_config, logistic_config)
    classifier = pipeline.named_steps["classify"]
    assert classifier.C == 1.0
    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42


def test_categorical_encoder_ignores_unknown_categories_by_configuration():
    """The one-hot encoder is configured to ignore, not crash on, an unseen category."""
    feature_config = preprocess.load_feature_config(FEATURE_CONFIG)
    logistic_config = logistic_baseline.load_logistic_config(LOGISTIC_CONFIG)
    pipeline = logistic_baseline.build_pipeline(feature_config, logistic_config)
    transformers = dict(
        (name, transformer) for name, transformer, _ in pipeline.named_steps["preprocess"].transformers
    )
    encoder = transformers["categorical"].named_steps["encode"]
    assert encoder.handle_unknown == "ignore"


# --- run() behaviour ---


def test_run_only_requests_train_and_validation_splits(artificial_data_dir, tmp_path, monkeypatch):
    """run() never requests the locked test split from preprocess.load_tabular_split."""
    requested_splits: list[str] = []
    real_load_tabular_split = preprocess.load_tabular_split

    def spy(data_dir, raw_csv_path, split, feature_config=preprocess.DEFAULT_FEATURE_CONFIG, **kwargs):
        requested_splits.append(split)
        return real_load_tabular_split(data_dir, raw_csv_path, split, feature_config, **kwargs)

    monkeypatch.setattr(logistic_baseline.preprocess, "load_tabular_split", spy)
    logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    assert "test" not in requested_splits
    assert requested_splits == ["train", "validation"]


def test_run_trains_on_the_documented_number_of_training_rows(artificial_data_dir, tmp_path):
    """The model is fit on exactly the 8 synthetic training rows, not validation or test."""
    result = logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    assert result["metrics"]["train"]["rows"] == 8
    assert result["metrics"]["validation"]["rows"] == 4
    assert result["metrics"]["test_rows_available_to_model"] == 0


def test_unknown_validation_category_does_not_crash_and_still_predicts(artificial_data_dir, tmp_path):
    """A home_ownership level ('OWN') absent from training still yields a valid probability."""
    result = logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    row = result["predictions"].set_index("loan_id").loc["lc_000000010"]
    assert np.isfinite(row["p_default_tabular"])
    assert 0.0 <= row["p_default_tabular"] <= 1.0


def test_every_validation_loan_id_receives_exactly_one_prediction(artificial_data_dir, tmp_path):
    """Every validation loan_id appears exactly once in the prediction frame."""
    result = logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    expected_ids = {f"lc_{n:09d}" for n in range(9, 13)}
    assert set(result["predictions"]["loan_id"]) == expected_ids
    assert result["predictions"]["loan_id"].is_unique


def test_predictions_pass_the_shared_validator(artificial_data_dir, tmp_path):
    """The prediction frame passes evaluate.validate_prediction_frame unchanged."""
    result = logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    expected_ids = {f"lc_{n:09d}" for n in range(9, 13)}
    evaluate.validate_prediction_frame(result["predictions"], expected_loan_ids=expected_ids)


def test_output_matches_the_agreed_prediction_schema(artificial_data_dir, tmp_path):
    """Columns, model_name, model_version and split match the agreed schema exactly."""
    result = logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    predictions = result["predictions"]
    assert predictions.columns.tolist() == list(evaluate.PREDICTION_COLUMNS)
    assert set(predictions["model_name"]) == {"logistic_regression"}
    assert set(predictions["model_version"]) == {"tabular-logistic-v1"}
    assert set(predictions["split"]) == {"validation"}


def test_same_input_and_config_produce_identical_predictions(artificial_data_dir, tmp_path):
    """Running run() twice on unchanged input produces identical probabilities."""
    first = logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out1",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    second = logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out2",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    pd.testing.assert_frame_equal(
        first["predictions"].sort_values("loan_id").reset_index(drop=True),
        second["predictions"].sort_values("loan_id").reset_index(drop=True),
    )


def test_saved_metrics_agree_with_recomputed_metrics(artificial_data_dir, tmp_path):
    """Metrics recomputed from the written predictions CSV match the written metrics JSON."""
    output_dir = tmp_path / "out"
    logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), output_dir,
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    saved_metrics = json.loads((output_dir / "logistic_baseline_metrics.json").read_text(encoding="utf-8"))
    saved_predictions = pd.read_csv(output_dir / "logistic_baseline_predictions.csv", dtype={"loan_id": "string"})

    _, validation_target = preprocess.load_tabular_split(
        artificial_data_dir, _raw_csv(artificial_data_dir), "validation", FEATURE_CONFIG
    )
    aligned = saved_predictions.set_index("loan_id").loc[validation_target.index]
    recomputed = evaluate.evaluate_predictions(
        validation_target.to_numpy(), aligned["p_default_tabular"].to_numpy()
    )
    for key, value in recomputed.items():
        assert saved_metrics["validation"][key] == pytest.approx(value)


def test_metrics_include_confusion_matrix_and_test_row_sentinel(artificial_data_dir, tmp_path):
    """Validation metrics include a 0.5-threshold confusion matrix and the test-row sentinel."""
    result = logistic_baseline.run(
        artificial_data_dir, _raw_csv(artificial_data_dir), tmp_path / "out",
        feature_config_path=FEATURE_CONFIG, logistic_config_path=LOGISTIC_CONFIG,
    )
    confusion = result["metrics"]["validation"]["confusion_matrix_at_0.5"]
    assert set(confusion) == {"tn", "fp", "fn", "tp"}
    assert sum(confusion.values()) == 4
    assert result["metrics"]["test_rows_available_to_model"] == 0


# --- real-data regression check ---


@pytest.mark.skipif(
    not os.getenv("LENDINGCLUB_RAW_CSV"),
    reason="Set LENDINGCLUB_RAW_CSV to run the locked real-data regression check.",
)
def test_real_logistic_baseline_produces_full_validation_coverage(tmp_path):
    """The real pipeline against the real data/ directory covers every validation loan."""
    raw_csv_path = Path(os.environ["LENDINGCLUB_RAW_CSV"])
    result = logistic_baseline.run(REAL_DATA_DIR, raw_csv_path, tmp_path / "out")
    assert len(result["predictions"]) == 21_784
    assert result["predictions"]["p_default_tabular"].between(0.0, 1.0).all()
    assert result["metrics"]["test_rows_available_to_model"] == 0

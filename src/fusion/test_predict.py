"""Prediction handoff tests: fallback behaviour, refusal of missing scores, and reproduction of the
committed stacker and reference predictions.

From the repository root:
    python -m pytest src/fusion/test_predict.py -v
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fusion import predict
from src.fusion.predict import ERROR_MESSAGES, FusionPredictor, PlattMapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _message(key: str) -> str:
    """A regex for one whole ERROR_MESSAGES template: literal text escaped, each placeholder any text."""
    return "^" + ".*".join(re.escape(part) for part in re.split(r"\{[^}]*\}", ERROR_MESSAGES[key])) + "$"


@pytest.fixture(scope="module")
def tabular_only() -> FusionPredictor:
    return predict.load_predictor("tabular_only")


@pytest.fixture()
def text_variant() -> FusionPredictor:
    """A synthetic two-track predictor, for the rules that need a text track."""
    identity = PlattMapping(slope=1.0, intercept=0.0, clip=1e-6)
    return FusionPredictor(
        stacker_version="fusion-stacker-v1", variant="tabular_tfidf", tracks=("tabular", "tfidf"),
        coefficients=(3.0, 1.0), intercept=-3.0, calibrator=identity,
        fallback=PlattMapping(slope=1.0, intercept=-1.0, clip=1e-6), fallback_version="tabular_calibration_v1",
    )


# --- fallback ----------------------------------------------------------------------------------

def test_unusable_text_returns_the_calibrated_tabular_score_with_a_fallback_flag(text_variant):
    result = predict.predict(text_variant, {"tabular": 0.3}, text_usable=False)
    assert result["fallback"] is True
    assert result["source"] == "tabular_fallback"
    assert result["tracks_used"] == ["tabular"]
    assert result["model_version"] == "tabular_calibration_v1"
    assert result["p_default"] == pytest.approx(text_variant.fallback.apply(0.3))


def test_unusable_text_with_an_explicit_none_text_score_still_falls_back(text_variant):
    result = predict.predict(text_variant, {"tabular": 0.3, "tfidf": None}, text_usable=False)
    assert result["fallback"] is True


def test_unusable_text_refuses_a_text_score(text_variant):
    with pytest.raises(ValueError, match=_message("text_score_without_text")):
        predict.predict(text_variant, {"tabular": 0.3, "tfidf": 0.2}, text_usable=False)


def test_usable_text_uses_the_stacker_without_a_fallback_flag(text_variant):
    result = predict.predict(text_variant, {"tabular": 0.3, "tfidf": 0.2}, text_usable=True)
    assert result["fallback"] is False
    assert result["source"] == "fusion_stacker"
    assert result["tracks_used"] == ["tabular", "tfidf"]
    assert result["p_default"] == pytest.approx(1 / (1 + math.exp(-(3 * 0.3 + 0.2 - 3))))


# --- never fill a missing score --------------------------------------------------------------

@pytest.mark.parametrize("scores, key", [
    ({"tabular": 0.3}, "missing_score"),
    ({"tabular": 0.3, "tfidf": None}, "missing_score"),
    ({"tfidf": 0.2}, "missing_score"),
    ({"tabular": 0.3, "tfidf": float("nan")}, "bad_score"),
    ({"tabular": 0.3, "tfidf": float("inf")}, "bad_score"),
    ({"tabular": 1.3, "tfidf": 0.2}, "bad_score"),
    ({"tabular": 0.3, "tfidf": "0.2"}, "bad_score"),
    ({"tabular": True, "tfidf": 0.2}, "bad_score"),
    ({"tabular": 0.3, "tfidf": 0.2, "distilbert": 0.1}, "unknown_tracks"),
])
def test_usable_text_refuses_missing_or_invalid_scores(text_variant, scores, key):
    with pytest.raises(ValueError, match=_message(key)):
        predict.predict(text_variant, scores, text_usable=True)


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_text_usable_must_be_an_explicit_boolean(text_variant, flag):
    with pytest.raises(ValueError, match=_message("text_flag")):
        predict.predict(text_variant, {"tabular": 0.3, "tfidf": 0.2}, text_usable=flag)


def test_fallback_still_requires_a_tabular_score(text_variant):
    with pytest.raises(ValueError, match=_message("missing_score")):
        predict.predict(text_variant, {}, text_usable=False)


# --- committed artifacts -----------------------------------------------------------------------

def test_loaded_predictor_reproduces_committed_validation_predictions(tabular_only):
    committed = pd.read_parquet(REPOSITORY_ROOT / "data" / "fusion" / "stacker_v1" / "predictions_validation.parquet")
    raw = pd.read_csv(
        REPOSITORY_ROOT / "reports" / "tabular" / "raw_predictions" / "logistic_regression_validation.csv",
        dtype={"loan_id": str},
    ).set_index("loan_id").loc[committed["loan_id"], "p_default_tabular"]
    rebuilt = [predict.predict(tabular_only, {"tabular": p}, text_usable=True)["p_default"] for p in raw]
    np.testing.assert_allclose(rebuilt, committed["p_tabular_only_platt"], rtol=0, atol=1e-12)


def test_fallback_reproduces_the_approved_calibrated_tabular_reference(tabular_only):
    reference = pd.read_parquet(REPOSITORY_ROOT / "data" / "tabular" / "tabular_calibrated_val.parquet")
    rebuilt = [
        predict.predict(tabular_only, {"tabular": p}, text_usable=False)["p_default"]
        for p in reference["p_default_tabular_raw"]
    ]
    np.testing.assert_allclose(rebuilt, reference["p_default_tabular_calibrated"], rtol=0, atol=1e-12)


def test_artifact_for_another_variant_is_refused(tmp_path):
    source = REPOSITORY_ROOT / "data" / "fusion" / "stacker_v1" / "tabular_only.json"
    (tmp_path / "tabular_tfidf.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match=_message("artifact_mismatch")):
        predict.load_predictor("tabular_tfidf", artifact_dir=tmp_path)

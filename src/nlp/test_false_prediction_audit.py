"""Small data-contract checks for the false-prediction audit."""

from __future__ import annotations

import pandas as pd
import pytest

from src.nlp.false_prediction_audit import select_review_cases


def test_selection_is_reproducible_and_contains_both_error_types() -> None:
    rows = pd.DataFrame(
        {
            "loan_id": [f"loan_{index}" for index in range(24)],
            "target": [0] * 12 + [1] * 12,
            "prob_default_tfidf": [0.51 + 0.01 * index for index in range(12)]
            + [0.49 - 0.01 * index for index in range(12)],
        }
    )
    first = select_review_cases(rows, n_per_error=6, n_extreme=3)
    second = select_review_cases(rows, n_per_error=6, n_extreme=3)

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby(["error_type", "sample_group"]).size().to_dict() == {
        ("false_negative", "random"): 3,
        ("false_negative", "strongest"): 3,
        ("false_positive", "random"): 3,
        ("false_positive", "strongest"): 3,
    }
    assert first["loan_id"].is_unique
    assert first.loc[first["error_type"].eq("false_positive"), "target"].eq(0).all()
    assert first.loc[first["error_type"].eq("false_negative"), "target"].eq(1).all()


def test_selection_rejects_insufficient_errors() -> None:
    rows = pd.DataFrame(
        {"loan_id": ["a", "b"], "target": [0, 1], "prob_default_tfidf": [0.8, 0.2]}
    )
    with pytest.raises(ValueError, match="Only 1 false_positive"):
        select_review_cases(rows, n_per_error=2, n_extreme=1)

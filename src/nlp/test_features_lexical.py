"""Tests for deterministic lexical features and frozen-ID alignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.nlp.features_lexical import (
    FEATURE_COLUMNS,
    align_to_split_ids,
    build_lexical_feature_table,
    extract_lexical_features,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_extract_lexical_features_has_expected_readability_and_density():
    """A simple sentence gives reproducible Flesch and keyword values."""
    features = extract_lexical_features("The cat sat behind the medical van.")
    assert features["flesch_reading_ease"] == pytest.approx(78.8729, abs=1e-4)
    assert features["flesch_kincaid_grade"] == pytest.approx(3.9971, abs=1e-4)
    assert features["lexical_diversity_ttr"] == pytest.approx(6 / 7)
    assert features["financial_distress_keyword_density"] == pytest.approx(2 / 7)


def test_sentiment_and_subjectivity_use_textblob_scores():
    """Clearly positive and negative prose maps to signed polarity and subjectivity."""
    positive = extract_lexical_features("I love this excellent plan.")
    negative = extract_lexical_features("I hate this terrible emergency.")
    assert positive["sentiment_polarity"] > 0
    assert negative["sentiment_polarity"] < 0
    assert positive["sentiment_subjectivity"] > 0
    assert negative["sentiment_subjectivity"] > 0


def test_empty_text_returns_finite_zero_features():
    """Empty text has a defined, division-safe representation."""
    assert extract_lexical_features("") == {name: 0.0 for name in FEATURE_COLUMNS}


def test_feature_builder_preserves_rows_and_rejects_test_split():
    """Feature extraction keeps input order and cannot access held-out rows."""
    rows = pd.DataFrame(
        {
            "loan_id": ["loan_b", "loan_a"],
            "split": ["validation", "train"],
            "text_payload": ["Medical emergency.", "A routine consolidation."],
        }
    )
    result = build_lexical_feature_table(rows)
    assert result["loan_id"].tolist() == ["loan_b", "loan_a"]
    assert list(result.columns) == ["loan_id", "split", *FEATURE_COLUMNS]
    assert result[list(FEATURE_COLUMNS)].notna().all().all()

    rows.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="Only train and validation"):
        build_lexical_feature_table(rows)


def test_alignment_matches_train_then_validation_id_order(tmp_path):
    """The exported contract follows frozen train/validation IDs exactly."""
    pd.DataFrame({"loan_id": ["train_2", "train_1"]}).to_csv(
        tmp_path / "train_ids.csv", index=False
    )
    pd.DataFrame({"loan_id": ["val_1"]}).to_csv(tmp_path / "val_ids.csv", index=False)
    features = pd.DataFrame(
        {
            "loan_id": ["val_1", "train_1", "train_2"],
            "split": ["validation", "train", "train"],
            **{name: [0.1, 0.2, 0.3] for name in FEATURE_COLUMNS},
        }
    )
    aligned = align_to_split_ids(features, tmp_path)
    assert aligned["loan_id"].tolist() == ["train_2", "train_1", "val_1"]
    assert aligned["split"].tolist() == ["train", "train", "validation"]


def test_exported_parquet_matches_frozen_ids():
    """The Parquet artifact exactly matches frozen train then validation IDs."""
    data_dir = REPOSITORY_ROOT / "data"
    artifact = data_dir / "nlp" / "text_lexical_features.parquet"
    actual = pd.read_parquet(artifact)
    train_ids = pd.read_csv(data_dir / "train_ids.csv", dtype={"loan_id": "string"})
    validation_ids = pd.read_csv(data_dir / "val_ids.csv", dtype={"loan_id": "string"})
    expected = pd.concat(
        [
            train_ids.assign(split="train"),
            validation_ids.assign(split="validation"),
        ],
        ignore_index=True,
    )

    assert list(actual.columns) == ["loan_id", "split", *FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(actual[["loan_id", "split"]], expected)
    assert actual["loan_id"].is_unique
    assert actual[list(FEATURE_COLUMNS)].notna().all().all()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda frame: frame.assign(loan_id=["train_1", "wrong"]),
        lambda frame: frame.assign(split=["validation", "validation"]),
    ],
)
def test_alignment_rejects_missing_extra_or_misassigned_ids(tmp_path, mutation):
    """Missing, extra and cross-split IDs fail instead of silently joining."""
    pd.DataFrame({"loan_id": ["train_1"]}).to_csv(tmp_path / "train_ids.csv", index=False)
    pd.DataFrame({"loan_id": ["val_1"]}).to_csv(tmp_path / "val_ids.csv", index=False)
    features = pd.DataFrame(
        {
            "loan_id": ["train_1", "val_1"],
            "split": ["train", "validation"],
            **{name: [0.1, 0.2] for name in FEATURE_COLUMNS},
        }
    )
    with pytest.raises(ValueError, match="do not match"):
        align_to_split_ids(mutation(features), tmp_path)

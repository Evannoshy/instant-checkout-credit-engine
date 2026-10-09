from __future__ import annotations

import pandas as pd
import pytest

from src.nlp.features_lexical import FEATURE_COLUMNS, extract_lexical_features
from src.nlp.lexical_eda import (
    add_keyword_source_metrics,
    compute_outcome_summary,
    compute_point_biserial_correlations,
    compute_purpose_summary,
    join_training_features,
)
from src.nlp.preprocess import build_text_payload


def _training_rows() -> pd.DataFrame:
    raw = [
        {
            "loan_id": "a",
            "split": "train",
            "target": 0,
            "title": "Kitchen update",
            "purpose": "home_improvement",
            "desc": "I am updating my kitchen.",
        },
        {
            "loan_id": "b",
            "split": "train",
            "target": 1,
            "title": "Medical bill",
            "purpose": "medical",
            "desc": "I need help with an unexpected bill.",
        },
        {
            "loan_id": "c",
            "split": "train",
            "target": 1,
            "title": "Card refinance",
            "purpose": "debt_consolidation",
            "desc": "I am behind and have overdue payments.",
        },
        {
            "loan_id": "d",
            "split": "train",
            "target": 0,
            "title": "Car repair",
            "purpose": "car",
            "desc": "I am repairing my reliable family vehicle.",
        },
    ]
    for row in raw:
        row["text_payload"] = build_text_payload(row)
    return pd.DataFrame(raw)


def _features(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in rows.itertuples(index=False):
        records.append(
            {
                "loan_id": row.loan_id,
                "split": "train",
                **extract_lexical_features(row.text_payload),
            }
        )
    return pd.DataFrame(records)


def test_join_requires_exact_frozen_training_membership() -> None:
    rows = _training_rows()
    features = _features(rows).iloc[:-1]

    with pytest.raises(ValueError, match="exactly match"):
        join_training_features(features, rows)


def test_correlations_are_ranked_by_absolute_effect_size() -> None:
    rows = _training_rows()
    joined = join_training_features(_features(rows), rows)

    result = compute_point_biserial_correlations(joined)

    assert set(result["feature"]) == set(FEATURE_COLUMNS)
    assert result["absolute_r"].is_monotonic_decreasing
    assert result["point_biserial_r"].between(-1, 1).all()


def test_outcome_summary_has_one_row_per_feature_and_outcome() -> None:
    rows = _training_rows()
    joined = join_training_features(_features(rows), rows)

    result = compute_outcome_summary(joined)

    assert len(result) == 2 * len(FEATURE_COLUMNS)
    assert set(result["outcome"]) == {"non_default", "default"}
    assert result["count"].eq(2).all()


def test_keyword_metrics_identify_purpose_label_without_desc_hit() -> None:
    rows = _training_rows()
    joined = join_training_features(_features(rows), rows)

    result = add_keyword_source_metrics(joined).set_index("loan_id")

    assert result.loc["b", "purpose_distress_count"] == 1
    assert result.loc["b", "desc_distress_count"] == 0
    assert bool(result.loc["b", "purpose_label_only_hit"])
    assert result.loc["c", "purpose_distress_count"] == 0
    assert result.loc["c", "desc_distress_count"] == 2


def test_purpose_summary_separates_full_payload_and_desc_language() -> None:
    rows = _training_rows()
    joined = join_training_features(_features(rows), rows)
    enriched = add_keyword_source_metrics(joined)

    result = compute_purpose_summary(enriched).set_index("purpose")

    assert result.loc["medical", "purpose_label_only_hit_rate"] == 1.0
    assert result.loc["medical", "desc_distress_mention_rate"] == 0.0
    assert result.loc["debt_consolidation", "desc_distress_mention_rate"] == 1.0
    assert result.loc["debt_consolidation", "empirical_default_rate"] == 1.0
    assert (
        result["payload_distress_hits"]
        == result["purpose_label_distress_hits"]
        + result["title_distress_hits"]
        + result["description_distress_hits"]
    ).all()

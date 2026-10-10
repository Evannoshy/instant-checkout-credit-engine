"""Tests for the DistilBERT score export: offline inference and the Parquet contract.

The contract tests read the Parquet files produced by python -m src.nlp.export_distilbert_scores.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast, set_seed

from src.nlp.export_distilbert_scores import (
    HOLDOUT_MODEL_VERSION,
    MODEL_VERSION,
    OUTPUTS,
    PROBABILITY_COLUMN,
    score,
)
from src.nlp.train_distilbert_prototype import SCALED

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
WORDS = ["title", "description", ":", "missed", "payments", "steady", "salary", "loan"]


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


def test_score_returns_rows_in_input_order_despite_length_sorting():
    """Batched, length-sorted scores equal one-at-a-time scores in the original row order."""
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    tokenizer = DistilBertTokenizerFast(vocab={t: i for i, t in enumerate(tokens)})
    set_seed(0)
    model = DistilBertForSequenceClassification(
        DistilBertConfig(vocab_size=len(tokens), dim=32, hidden_dim=64, n_layers=1, n_heads=2)
    )
    # Lengths deliberately out of order, so sorting reorders the rows.
    repeats = [7, 1, 4, 9, 2, 6, 3, 8, 5, 1]
    loans = pd.DataFrame({
        "text_payload": [f"Title: loan\nDescription: {'missed payments ' * n}" for n in repeats],
        "target": [i % 2 for i in range(len(repeats))],
    })

    batched = score(model, tokenizer, loans, batch_size=4)
    one_by_one = [score(model, tokenizer, loans.iloc[[i]], batch_size=1)[0] for i in range(len(loans))]

    np.testing.assert_allclose(batched, one_by_one, atol=1e-6)
    assert ((batched >= 0) & (batched <= 1)).all()


@pytest.mark.parametrize(
    ("split", "id_file", "holdout_scored"),
    [("train", "train_ids.csv", SCALED.train_rows), ("validation", "val_ids.csv", 0)],
)
def test_exported_parquet_matches_frozen_ids(split, id_file, holdout_scored):
    """The Parquet artifact matches the frozen IDs in order, with no NaNs, duplicates or test loans."""
    actual = pd.read_parquet(DATA_DIR / "nlp" / OUTPUTS[split])
    expected_ids = pd.read_csv(DATA_DIR / id_file, dtype={"loan_id": "string"})["loan_id"]
    test_ids = pd.read_csv(DATA_DIR / "test_ids.csv", dtype={"loan_id": "string"})["loan_id"]

    assert actual.columns.tolist() == [
        "loan_id", "split", PROBABILITY_COLUMN, "model_name", "model_version"
    ]
    pd.testing.assert_series_equal(actual["loan_id"].astype("string"), expected_ids)
    assert actual["split"].eq(split).all()
    assert actual["loan_id"].is_unique
    assert actual.notna().all().all()
    assert actual[PROBABILITY_COLUMN].between(0, 1).all()
    assert not actual["loan_id"].isin(test_ids).any()
    # The main model's 10,000 training loans are scored by the holdout model instead.
    assert actual["model_version"].isin([MODEL_VERSION, HOLDOUT_MODEL_VERSION]).all()
    assert actual["model_version"].eq(HOLDOUT_MODEL_VERSION).sum() == holdout_scored

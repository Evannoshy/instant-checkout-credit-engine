"""Tests for the DistilBERT score export and wrapper: offline inference, export checks and the Parquet contracts.

The contract tests read the Parquet files produced by python -m src.nlp.export_distilbert_scores.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast, set_seed

import src.nlp.export_distilbert_scores as export
from src.nlp.export_distilbert_scores import (
    MANIFEST,
    MODEL_DIR_V2,
    MODEL_VERSION,
    OUTPUT,
    check_export,
    score,
    score_borrower_text,
)
from src.nlp.preprocess import PREPROCESS_VERSION, build_text_payload, load_original_split, sha256_file
from src.nlp.train_distilbert_prototype import RANDOM_STATE, SCALED

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
          "title", "purpose", "description", ":", "missed", "payments", "steady", "salary", "car", "loan"]


def frozen_ids(name: str) -> pd.Series:
    return pd.read_csv(DATA_DIR / name, dtype={"loan_id": "string"})["loan_id"]


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def tiny():
    """A one-layer DistilBERT and a word-level tokenizer, fully offline."""
    set_seed(0)
    model = DistilBertForSequenceClassification(
        DistilBertConfig(vocab_size=len(TOKENS), dim=32, hidden_dim=64, n_layers=1, n_heads=2)
    )
    return model, DistilBertTokenizerFast(vocab={t: i for i, t in enumerate(TOKENS)})


@pytest.fixture
def saved_model(tiny, tmp_path):
    """The tiny model and tokenizer saved like a trained run, for the wrapper to load."""
    for artifact in tiny:
        artifact.save_pretrained(tmp_path)
    return tmp_path


def test_score_returns_rows_in_input_order_despite_length_sorting(tiny):
    """Batched, length-sorted (margin, prob) equal one-at-a-time values, and sigmoid(margin) equals prob."""
    repeats = [7, 1, 4, 9, 2, 6, 3, 8, 5, 1]  # out of order, so sorting reorders the rows
    texts = [f"Title: loan\nDescription: {'missed payments ' * n}" for n in repeats]

    model, tokenizer = tiny
    margin, prob = score(model, tokenizer, texts, batch_size=4)
    singles = [score(model, tokenizer, [text], batch_size=1) for text in texts]

    np.testing.assert_allclose(margin, [m[0] for m, _ in singles], atol=1e-5)
    np.testing.assert_allclose(prob, [p[0] for _, p in singles], atol=1e-6)
    np.testing.assert_allclose(prob, 1 / (1 + np.exp(-margin)), atol=1e-6)


@pytest.fixture
def export_frame(monkeypatch) -> pd.DataFrame:
    """A four-loan export that passes every check once ROLE_ROWS is scaled down to match it."""
    monkeypatch.setattr(export, "ROLE_ROWS", {"fusion_fit": 2, "fusion_calibration": 1, "validation": 1})
    frame = pd.DataFrame({
        "loan_id": ["a", "b", "c", "d"],
        "split": ["train", "train", "train", "validation"],
        "role": ["fusion_fit", "fusion_fit", "fusion_calibration", "validation"],
        "raw_margin": [0.0, 1.0, -1.0, 2.0],
        "raw_prob": [0.5, 0.73, 0.27, 0.88],
        "model_version": MODEL_VERSION,
    })
    mapping = frame[["loan_id", "split", "role"]].copy()
    monkeypatch.setattr(export, "load_role_mapping", lambda: mapping.copy())
    return frame


def test_check_export_accepts_a_valid_frame(export_frame):
    """A frame meeting the shared contract passes."""
    check_export(export_frame, {"z"})


def test_check_export_accepts_reordered_rows(export_frame):
    """Alignment is checked by loan ID, not by incidental row order."""
    check_export(export_frame.iloc[::-1].reset_index(drop=True), set())


def test_role_mapping_matches_the_published_contract():
    """The committed mapping has the pinned checksum and all expected roles."""
    mapping = export.load_role_mapping()
    assert mapping["loan_id"].notna().all() and mapping["loan_id"].is_unique
    assert mapping["role"].value_counts().to_dict() == {"base_fit": 59_353, **export.ROLE_ROWS}


def test_role_mapping_rejects_a_changed_file(tmp_path):
    """Changing the role mapping stops validation before a different population can be scored."""
    path = tmp_path / "roles.csv"
    path.write_text("loan_id,split,role,issue_month\na,train,fusion_fit,2013-01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        export.load_role_mapping(path)


@pytest.mark.parametrize(
    ("broken", "model_ids"),
    [
        pytest.param(lambda f: f.assign(loan_id=["a", "a", "c", "d"]), set(), id="duplicate-id"),
        pytest.param(lambda f: f.assign(raw_margin=[np.nan, 1.0, -1.0, 2.0]), set(), id="nan-margin"),
        pytest.param(lambda f: f.assign(raw_prob=[1.5, 0.73, 0.27, 0.88]), set(), id="prob-above-1"),
        pytest.param(lambda f: f.assign(role=["base_fit", "fusion_fit", "fusion_calibration", "validation"]),
                     set(), id="base-fit-role"),
        pytest.param(lambda f: f.assign(split=["validation", "train", "train", "validation"]), set(), id="wrong-split"),
        pytest.param(lambda f: f.drop(columns="model_version"), set(), id="dropped-column"),
        pytest.param(lambda f: f, {"b"}, id="model-saw-loan"),
        pytest.param(lambda f: f.assign(loan_id=["unknown", "b", "c", "d"]), set(), id="unknown-id"),
        pytest.param(lambda f: f.assign(loan_id=[None, "b", "c", "d"]), set(), id="missing-id"),
        pytest.param(lambda f: f.assign(role=["fusion_calibration", "fusion_fit", "fusion_fit", "validation"]),
                     set(), id="swapped-roles-same-counts"),
        pytest.param(lambda f: f.assign(model_version="another-model"), set(), id="wrong-version"),
        pytest.param(lambda f: f.assign(model_version=[None, MODEL_VERSION, MODEL_VERSION, MODEL_VERSION]),
                     set(), id="missing-version"),
    ],
)
def test_check_export_rejects_a_broken_frame(export_frame, broken, model_ids):
    """Each contract or leakage violation stops the export before anything is written."""
    with pytest.raises(ValueError):
        check_export(broken(export_frame), model_ids)


def test_wrapper_matches_the_export_scoring_path(tiny, saved_model):
    """score_borrower_text builds the shared payload and returns exactly what score() gives for it."""
    payload = build_text_payload({"title": "Car loan", "purpose": "car", "desc": "Missed payments"})
    margin, prob = score(*tiny, [payload])

    assert score_borrower_text("Car loan", "car", "Missed payments", model=saved_model) == pytest.approx({
        "text_usable": True,
        "desc_usable": True,
        "raw_margin": margin[0],
        "raw_prob": prob[0],
        "model_version": MODEL_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
    }, abs=1e-4)


@pytest.mark.parametrize(("title", "purpose", "desc"), [("", "", ""), (None, "  ", None), ("n/a", "NULL", "<br>")])
def test_wrapper_returns_no_score_for_unusable_text(title, purpose, desc):
    """An empty cleaned payload gives text_usable False and no score, without ever loading a model."""
    result = score_borrower_text(title, purpose, desc, model="no-model-is-loaded")
    assert (result["text_usable"], result["desc_usable"], result["raw_margin"], result["raw_prob"]) == (
        False, False, None, None
    )


def test_wrapper_scores_a_wordless_description_and_flags_it(saved_model):
    """A purpose with a missing-marker description is scored, with desc_usable False."""
    result = score_borrower_text("Car loan", "car", "n/a", model=saved_model)
    assert result["text_usable"] and not result["desc_usable"]
    assert 0 <= result["raw_prob"] <= 1


def test_v2_export_meets_the_shared_contract():
    """The v2 Parquet holds the 48,724 role loans in frozen ID order, matches its manifest and scores no model loan."""
    if not OUTPUT.exists():
        pytest.skip("Run python -m src.nlp.export_distilbert_scores on the GPU node first")
    frame = pd.read_parquet(OUTPUT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    check_export(frame, {*manifest["training_ids"]["ids"], *manifest["selection_ids"]["ids"]})
    frozen = pd.concat([frozen_ids("train_ids.csv"), frozen_ids("val_ids.csv")])
    assert frame["loan_id"].tolist() == frozen[frozen.isin(frame["loan_id"])].tolist()
    assert not frame["loan_id"].isin(frozen_ids("test_ids.csv")).any()
    assert frame["model_version"].eq(MODEL_VERSION).all()
    assert manifest["checksums"][OUTPUT.name] == sha256_file(OUTPUT)


def test_wrapper_reproduces_the_v2_export_on_the_trained_model():
    """On the real model, score_borrower_text from raw fields matches the exported raw_prob for 200 validation loans."""
    if not (OUTPUT.exists() and (MODEL_DIR_V2 / "config.json").exists()):
        pytest.skip("Needs the trained distilbert_fusion_v2 model and its v2 export (GPU node)")
    exported = pd.read_parquet(OUTPUT).set_index("loan_id")["raw_prob"]
    loans = load_original_split(DATA_DIR, "validation").sample(n=200, random_state=RANDOM_STATE)

    fields = loans[["title", "purpose", "desc"]].itertuples(index=False, name=None)
    wrapper = [score_borrower_text(*row)["raw_prob"] for row in fields]

    # Match the documented absolute tolerance for fp16 export versus wrapper inference.
    np.testing.assert_allclose(wrapper, exported[loans["loan_id"]], atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    ("filename", "split", "id_file", "holdout_scored"),
    [
        ("distilbert_scores_train.parquet", "train", "train_ids.csv", SCALED.train_rows),
        ("distilbert_scores_val.parquet", "validation", "val_ids.csv", 0),
    ],
)
def test_v1_parquet_matches_frozen_ids(filename, split, id_file, holdout_scored):
    """The v1 Parquets Brandon's v0 handoff pins stay unchanged: frozen ID order, no NaNs, duplicates or test loans."""
    actual = pd.read_parquet(DATA_DIR / "nlp" / filename)

    assert actual.columns.tolist() == ["loan_id", "split", "prob_default_distilbert", "model_name", "model_version"]
    pd.testing.assert_series_equal(actual["loan_id"].astype("string"), frozen_ids(id_file))
    assert actual["split"].eq(split).all()
    assert actual["loan_id"].is_unique
    assert actual.notna().all().all()
    assert actual["prob_default_distilbert"].between(0, 1).all()
    assert not actual["loan_id"].isin(frozen_ids("test_ids.csv")).any()
    assert actual["model_version"].isin(["nlp-distilbert-10k-v1", "nlp-distilbert-10k-holdout-v1"]).all()
    assert actual["model_version"].eq("nlp-distilbert-10k-holdout-v1").sum() == holdout_scored

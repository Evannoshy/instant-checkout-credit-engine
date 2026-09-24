"""Regression tests using tiny, artificial examples (no borrower data).

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/nlp/test_train_distilbert_prototype.py -v -s

A seeded one-layer DistilBERT and a hand-written vocabulary replace the pretrained
download, so every test runs offline in seconds.
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EvalPrediction,
    Trainer,
    set_seed,
)

from src.nlp import train_distilbert_prototype as proto
from src.nlp.dataset import MAX_LENGTH, LoanTextDataset

WORDS = ["title", "description", ":", "missed", "payments", "steady", "salary", "loan"]


@pytest.fixture(autouse=True)
def describe_test(request: pytest.FixtureRequest):
    """Announce the check; pytest reports success only after it actually passes."""
    description = request.function.__doc__ or request.node.name
    print(f"\n[CHECK] {description.strip()}")


@pytest.fixture
def tokenizer() -> DistilBertTokenizerFast:
    """A tokenizer over a hand-written vocabulary."""
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    tok = DistilBertTokenizerFast(vocab={t: i for i, t in enumerate(tokens)})
    # transformers 5 silently ignores vocab_file=, guard against an all-[UNK] vocabulary.
    assert tok.vocab_size == len(tokens)
    return tok


@pytest.fixture
def tiny_model(monkeypatch, tokenizer):
    """Replace the pretrained download with a seeded one-layer DistilBERT."""

    def build() -> DistilBertForSequenceClassification:
        set_seed(proto.RANDOM_STATE)
        config = DistilBertConfig(
            vocab_size=tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2
        )  # num_labels defaults to 2
        return DistilBertForSequenceClassification(config)

    monkeypatch.setattr(proto, "new_model", build)
    return build


@pytest.fixture
def loans() -> pd.DataFrame:
    """80 separable loans at a 20% default rate: 'missed payments' marks defaults."""
    target = [int(i % 5 == 0) for i in range(80)]
    return pd.DataFrame({
        "loan_id": [f"lc_{i:09d}" for i in range(80)],
        "text_payload": [
            f"Title: loan\nDescription: {'missed payments' if t else 'steady salary'}" for t in target
        ],
        "target": target,
    })


def test_prototype_uses_the_specified_hyperparameters():
    """Prototype: 1,000 train / 500 validation loans, 2 epochs, lr 2e-5, AdamW, batch 16, max_length 384."""
    assert (proto.PROTOTYPE.train_rows, proto.VAL_ROWS, proto.PROTOTYPE.epochs) == (1_000, 500, 2)
    assert proto.PROTOTYPE.learning_rate == 2e-5
    assert proto.OPTIMIZER == "adamw_torch"
    assert (proto.BATCH_SIZE, MAX_LENGTH) == (16, 384)


def test_overfit_uses_the_specified_sample_and_epochs():
    """Overfit gate: 100 train loans for 10 epochs."""
    assert (proto.OVERFIT.train_rows, proto.OVERFIT.epochs) == (100, 10)


def test_build_trainer_passes_the_run_config_to_the_trainer(tiny_model, tokenizer, loans, tmp_path):
    """RunConfig, batch size, optimizer, weight decay and seed reach TrainingArguments unchanged."""
    ds = LoanTextDataset(loans, tokenizer)
    args = proto.build_trainer(proto.PROTOTYPE, tiny_model(), tokenizer, ds, ds, tmp_path).args
    assert args.learning_rate == proto.PROTOTYPE.learning_rate
    assert args.num_train_epochs == proto.PROTOTYPE.epochs
    assert args.per_device_train_batch_size == args.per_device_eval_batch_size == proto.BATCH_SIZE
    assert args.optim == proto.OPTIMIZER
    assert args.weight_decay > 0  # AdamW, not Adam
    assert args.eval_strategy == "epoch"
    assert args.seed == proto.RANDOM_STATE


def test_stratified_sample_keeps_the_default_rate_exactly(loans):
    """A 40-row sample of a 20% default-rate frame holds exactly 8 defaults."""
    sample = proto.stratified_sample(loans, 40)
    assert len(sample) == 40
    assert sample["target"].sum() == 8


def test_stratified_sample_is_reproducible(loans):
    """The same frame and size always draw the same loans."""
    first, second = (proto.stratified_sample(loans, 40)["loan_id"].tolist() for _ in range(2))
    assert first == second


def test_compute_metrics_scores_the_positive_class_probability():
    """Logits favouring class 1 exactly on the defaults rank perfectly, including fp16 logits."""
    labels = np.array([0, 0, 1, 1])
    logits = np.array([[2.0, -2.0], [1.0, 0.0], [0.0, 1.0], [-2.0, 2.0]], dtype=np.float16)
    metrics = proto.compute_metrics(EvalPrediction(predictions=logits, label_ids=labels))
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


@pytest.mark.parametrize(("cuda", "native_bf16", "expected"), [
    (False, False, "fp32"),
    (True, True, "bf16"),
    (True, False, "fp16"),
])
def test_precision_uses_bf16_only_where_the_gpu_supports_it_natively(monkeypatch, cuda, native_bf16, expected):
    """CPU runs fp32; GPUs use bf16 only without emulation, otherwise fp16."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda including_emulation=True: native_bf16 and not including_emulation
    )
    assert proto.precision() == expected


def test_overfit_gate_passes_when_the_model_memorises(monkeypatch, tiny_model, tokenizer, loans):
    """A learnable task at a working learning rate clears every check."""
    monkeypatch.setattr(proto, "OVERFIT", dataclasses.replace(proto.OVERFIT, train_rows=16, epochs=30, learning_rate=1e-2))
    proto.overfit(loans, tokenizer)


@pytest.mark.parametrize(("learning_rate", "failed_check"), [
    (0.0, "optimizer updated"),  # Gradients flow but weights never move.
    (1e-6, "train loss ~0"),  # Weights move but too little to memorise.
])
def test_overfit_gate_stops_the_run_when_training_is_broken(monkeypatch, tiny_model, tokenizer, loans, learning_rate, failed_check):
    """A frozen optimizer or a model that cannot memorise raises before the prototype runs."""
    monkeypatch.setattr(proto, "OVERFIT", dataclasses.replace(proto.OVERFIT, train_rows=16, epochs=2, learning_rate=learning_rate))
    with pytest.raises(RuntimeError, match=failed_check):
        proto.overfit(loans, tokenizer)


@pytest.fixture
def small_prototype(monkeypatch, tiny_model, tmp_path):
    """A 32-train / 16-validation prototype that saves under tmp_path."""
    monkeypatch.setattr(proto, "PROTOTYPE", dataclasses.replace(proto.PROTOTYPE, train_rows=32))
    monkeypatch.setattr(proto, "VAL_ROWS", 16)
    monkeypatch.setattr(proto, "MODEL_DIR", tmp_path / "models" / "distilbert_prototype")
    return proto.MODEL_DIR


def test_prototype_saves_a_loadable_model_with_its_run_record(small_prototype, tokenizer, loans):
    """The checkpoint reloads, and metrics.json holds config, validation metrics and the sampled loan IDs."""
    proto.prototype(loans, loans, tokenizer)
    DistilBertForSequenceClassification.from_pretrained(small_prototype)
    DistilBertTokenizerFast.from_pretrained(small_prototype)
    record = json.loads((small_prototype / "metrics.json").read_text(encoding="utf-8"))
    assert record["config"]["learning_rate"] == proto.PROTOTYPE.learning_rate
    assert {"eval_roc_auc", "eval_pr_auc"} <= set(record["validation"])
    assert len(record["sample_loan_ids"]["train"]) == 32
    assert len(record["sample_loan_ids"]["validation"]) == 16
    assert not small_prototype.with_name(small_prototype.name + ".partial").exists()


def test_prototype_is_reproducible(small_prototype, tokenizer, loans):
    """Two runs with the same seed report identical validation metrics."""
    scores = []
    for _ in range(2):
        proto.prototype(loans, loans, tokenizer)
        scores.append(json.loads((small_prototype / "metrics.json").read_text(encoding="utf-8"))["validation"])
    assert scores[0]["eval_loss"] == scores[1]["eval_loss"]
    assert scores[0]["eval_roc_auc"] == scores[1]["eval_roc_auc"]


def test_a_crashed_run_leaves_the_previous_checkpoint_untouched(monkeypatch, small_prototype, tokenizer, loans):
    """Training fails mid-run: the last good model and metrics.json stay exactly as they were."""
    proto.prototype(loans, loans, tokenizer)
    before = {p.name: p.read_bytes() for p in small_prototype.iterdir()}

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(Trainer, "train", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        proto.prototype(loans, loans, tokenizer)
    assert {p.name: p.read_bytes() for p in small_prototype.iterdir()} == before

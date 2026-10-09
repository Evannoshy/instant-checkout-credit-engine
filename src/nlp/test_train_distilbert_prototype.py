"""Regression tests using tiny, artificial examples (no borrower data).

From the repository root, show each check and its actual pytest result with:
    python -m pytest src/nlp/test_train_distilbert_prototype.py -v -s

A seeded one-layer DistilBERT and a hand-written vocabulary replace the pretrained
download, so every test runs offline in seconds.
"""

import dataclasses
import json
from types import SimpleNamespace

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


def test_scaled_uses_the_specified_configuration():
    """10k scaling: 10,000 train / 5,000 validation loans, 3 epochs, the prototype's lr and batch size."""
    assert (proto.SCALED.train_rows, proto.SCALED_VAL_ROWS, proto.SCALED.epochs) == (10_000, 5_000, 3)
    assert proto.SCALED.learning_rate == proto.PROTOTYPE.learning_rate
    assert proto.ROC_AUC_TARGET == 0.53


def test_overfit_uses_the_specified_sample_and_epochs():
    """Overfit gate: 100 train loans for 10 epochs at the recommended lr 2e-5."""
    assert (proto.OVERFIT.train_rows, proto.OVERFIT.epochs, proto.OVERFIT.learning_rate) == (100, 10, 2e-5)


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


def test_build_trainer_keeps_the_best_epoch_only_when_asked(tiny_model, tokenizer, loans, tmp_path):
    """Best-epoch selection by ROC-AUC is opt-in; otherwise nothing is checkpointed mid-run."""
    ds = LoanTextDataset(loans, tokenizer)
    default = proto.build_trainer(proto.PROTOTYPE, tiny_model(), tokenizer, ds, ds, tmp_path).args
    assert default.save_strategy == "no"
    assert not default.load_best_model_at_end
    best = proto.build_trainer(proto.SCALED, tiny_model(), tokenizer, ds, ds, tmp_path, keep_best_epoch=True).args
    assert best.save_strategy == best.eval_strategy == "epoch"
    assert best.load_best_model_at_end
    assert best.metric_for_best_model == "roc_auc"
    assert best.greater_is_better


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


def test_roc_auc_interval_brackets_the_point_estimate_reproducibly():
    """Perfect ranking gives [1, 1], random scores straddle 0.5, the same scores repeat, one class gives NaN."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=400)
    assert proto.roc_auc_interval(labels, labels.astype(float)) == (1.0, 1.0)
    scores = rng.random(400)
    ci_low, ci_high = proto.roc_auc_interval(labels, scores)
    assert ci_low < 0.5 < ci_high
    assert proto.roc_auc_interval(labels, scores) == (ci_low, ci_high)
    assert all(np.isnan(proto.roc_auc_interval(np.zeros(10, dtype=int), scores[:10])))


@pytest.mark.parametrize(("roc_auc", "ci_low", "expected"), [
    (0.56, 0.54, "PASS"),  # Whole interval above 0.53
    (0.54, 0.52, "INCONCLUSIVE"),  # Point estimate above, interval still includes 0.53
    (0.52, 0.50, "FAIL"),
])
def test_target_verdict_passes_only_when_the_whole_interval_clears_the_target(roc_auc, ci_low, expected):
    """A borderline ROC-AUC above 0.53 is inconclusive until its confidence interval clears 0.53."""
    assert proto.target_verdict(roc_auc, ci_low) == expected


def test_class_weights_follow_the_default_rate():
    """At a 15.2% default rate, class weights are [1.0, 848/152]."""
    weights = proto.class_weights(pd.Series([1] * 152 + [0] * 848))
    assert weights.tolist() == pytest.approx([1.0, 848 / 152])


def test_weighted_trainer_penalises_a_missed_default_more_than_a_false_alarm(tiny_model, tokenizer, loans, tmp_path):
    """An equally wrong missed default gets N_non-default / N_default times a false alarm's gradient."""
    ds = LoanTextDataset(loans, tokenizer)  # 20% defaults -> weight 64 / 16 = 4
    assert type(proto.build_trainer(proto.PROTOTYPE, tiny_model(), tokenizer, ds, ds, tmp_path)) is Trainer
    trainer = proto.build_trainer(proto.PROTOTYPE, tiny_model(), tokenizer, ds, ds, tmp_path, class_weighted=True)
    assert isinstance(trainer, proto.WeightedTrainer)

    logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]], requires_grad=True)
    trainer.compute_loss(lambda **_: SimpleNamespace(logits=logits), {"labels": torch.tensor([1, 0])}).backward()
    missed_default, false_alarm = logits.grad.abs().sum(dim=1)
    assert (missed_default / false_alarm).item() == pytest.approx(4.0)


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


@pytest.fixture
def small_scaled(monkeypatch, tiny_model, tmp_path):
    """A 32-train / 16-validation scaled run that saves under tmp_path.

    At lr 3e-2 validation ROC-AUC peaks before the last epoch, so "best" and "last" differ.
    """
    monkeypatch.setattr(proto, "SCALED", dataclasses.replace(proto.SCALED, train_rows=32, learning_rate=3e-2))
    monkeypatch.setattr(proto, "SCALED_VAL_ROWS", 16)
    monkeypatch.setattr(proto, "MODEL_DIR", tmp_path / "models" / "distilbert_prototype")
    monkeypatch.setattr(proto, "REPORT_DIR", tmp_path / "reports")
    return tmp_path


@pytest.mark.parametrize(("weighted", "name"), [(True, "distilbert_10k"), (False, "distilbert_10k_unweighted")])
def test_scaled_saves_the_best_epoch_and_a_slim_report(small_scaled, tokenizer, loans, weighted, name):
    """The saved model is the best-ROC-AUC epoch, and the report lists every epoch under the run's own name."""
    proto.scaled(loans, loans, tokenizer, weighted=weighted)
    report = json.loads((small_scaled / "reports" / f"{name}_metrics.json").read_text(encoding="utf-8"))
    epochs = report["validation_by_epoch"]
    assert [e["epoch"] for e in epochs] == [1, 2, 3]
    best = next(e for e in epochs if e["roc_auc"] == max(x["roc_auc"] for x in epochs))  # Trainer keeps the first maximum
    assert best is not epochs[-1]  # Otherwise this test could not tell "best" from "last"
    assert report["best_epoch"] == best["epoch"]
    assert report["validation"] == {k: v for k, v in best.items() if k != "epoch"}
    assert (report["config"]["class_weights"] is not None) == weighted
    assert "peak_gpu_memory_gb" in report["environment"]
    # Every epoch carries a ROC-AUC interval, and the verdict follows from the best epoch's.
    assert all(e["roc_auc_ci_low"] <= e["roc_auc"] <= e["roc_auc_ci_high"] for e in epochs)
    verdict = proto.target_verdict(best["roc_auc"], best["roc_auc_ci_low"])
    assert report["roc_auc_target"] == {"target": 0.53, "verdict": verdict}

    # Re-evaluating the saved weights reproduces the best epoch's loss, not the last epoch's.
    model_dir = small_scaled / "models" / name
    assert not list(model_dir.glob("checkpoint-*"))
    train_ds = LoanTextDataset(proto.stratified_sample(loans, 32), tokenizer)
    val_ds = LoanTextDataset(proto.stratified_sample(loans, 16), tokenizer)
    saved = DistilBertForSequenceClassification.from_pretrained(model_dir)
    trainer = proto.build_trainer(proto.SCALED, saved, tokenizer, train_ds, val_ds, small_scaled, class_weighted=weighted)
    assert trainer.evaluate()["eval_loss"] == pytest.approx(best["loss"])
    assert best["loss"] != pytest.approx(epochs[-1]["loss"])


def test_holdout_trains_only_on_loans_the_main_model_never_saw(small_scaled, tokenizer, loans):
    """The holdout model trains on a full sample that shares no loan with the main model's."""
    proto.scaled(loans, loans, tokenizer)
    proto.scaled(loans, loans, tokenizer, holdout=True)
    main_ids, holdout_ids = (
        set(json.loads((small_scaled / "models" / name / "metrics.json").read_text(encoding="utf-8"))["sample_loan_ids"]["train"])
        for name in ("distilbert_10k", "distilbert_10k_holdout")
    )
    assert len(holdout_ids) == proto.SCALED.train_rows
    assert not main_ids & holdout_ids
    assert (small_scaled / "reports" / "distilbert_10k_holdout_metrics.json").exists()

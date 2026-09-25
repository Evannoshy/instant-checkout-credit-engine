"""DistilBERT Training Prototype.

Run from the repository root:
    python -m src.nlp.train_distilbert_prototype [--stage overfit|prototype|all]

Stage 1 (overfit): train on 100 loans for 10 epochs and stop unless the loss
reaches ~0 and the optimizer actually updates the weights.
Stage 2 (prototype): fine-tune on 1,000 train loans for 2 epochs, report ROC-AUC
and PR-AUC on 500 validation loans, and save the model with metrics.json to
models/distilbert_prototype/.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.nlp.baseline_tfidf import RANDOM_STATE, evaluate
from src.nlp.dataset import MAX_LENGTH, LoanTextDataset
from src.nlp.preprocess import PREPROCESS_VERSION, load_original_split

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "distilbert_prototype"
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
OPTIMIZER = "adamw_torch"
WEIGHT_DECAY = 0.01
VAL_ROWS = 500
OVERFIT_MAX_LOSS = 0.15
# Head + first-layer biases: no weight decay, so only gradients can change them.
WATCHED_PARAMS = ("classifier.bias", "distilbert.transformer.layer.0.attention.q_lin.bias")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class RunConfig:
    """Settings that differ between the two stages."""

    train_rows: int
    epochs: int
    learning_rate: float
    lr_scheduler_type: str
    logging_steps: int

# Overfit: memorise 100 loans fast to verify training loss goes to approximately 0.
OVERFIT = RunConfig(train_rows=100, epochs=10, learning_rate=2e-5, lr_scheduler_type="constant", logging_steps=1)
# Prototype Fine-Tuning: train on 1,000 loans and evaluate on 500 each of 2 epochs to test the Trainer end-to-end.
PROTOTYPE = RunConfig(train_rows=1_000, epochs=2, learning_rate=2e-5, lr_scheduler_type="linear", logging_steps=10)


def precision() -> str:
    """fp32 on CPU; on GPU bf16 where supported natively (no loss scaling, no skipped steps), else fp16."""
    if not torch.cuda.is_available():
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample n rows keeping the default rate exact (PR-AUC depends on it)."""
    sample, _ = train_test_split(df, train_size=n, stratify=df["target"], random_state=RANDOM_STATE)
    return sample.reset_index(drop=True)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, Any]:
    """Score the class-1 (default) probability with the TF-IDF baseline's metrics."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if not isinstance(logits, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("Expected a single logits array and a single label array")
    scores = torch.from_numpy(logits).float().softmax(-1)[:, 1].numpy()
    return evaluate(labels, scores)


def new_model() -> DistilBertForSequenceClassification:
    """Pretrained DistilBERT with a freshly initialised 2-class head."""
    # The new head starts with random weights; seed now so every run starts the same.
    # Trainer's own seed comes too late: it is applied after this model already exists.
    set_seed(RANDOM_STATE)
    return DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def build_trainer(
    cfg: RunConfig,
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    train_ds: LoanTextDataset,
    eval_ds: LoanTextDataset | None,
    output_dir: Path,
) -> Trainer:
    """Build the Trainer both stages share (AdamW, batch size 16, seeded).

    Args:
        eval_ds: Evaluated after every epoch; None skips evaluation.
        output_dir: Trainer working directory; no mid-run checkpoints are written.
    """
    mixed = precision()
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        eval_strategy="epoch" if eval_ds is not None else "no",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        bf16=mixed == "bf16",
        fp16=mixed == "fp16",
        train_sampling_strategy="group_by_length",  # Batches similar lengths -> less padding
        save_strategy="no",  # Only the final model is saved, explicitly
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=RANDOM_STATE,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )


def overfit(train_df: pd.DataFrame, tokenizer: DistilBertTokenizerFast) -> None:
    """Verify gradient descent works by overfitting a small sample."""
    cfg = OVERFIT
    print(f"\n=== Stage 1: overfit {cfg.train_rows} loans x {cfg.epochs} epochs (lr={cfg.learning_rate}) ===")
    ds = LoanTextDataset(stratified_sample(train_df, cfg.train_rows), tokenizer)
    model = new_model()
    before = {name: model.get_parameter(name).detach().cpu().clone() for name in WATCHED_PARAMS}

    with tempfile.TemporaryDirectory() as tmp:
        trainer = build_trainer(cfg, model, tokenizer, ds, None, Path(tmp))
        trainer.train()
        eval_loss = trainer.evaluate(ds)["eval_loss"]

    history = trainer.state.log_history
    # Mean loss of the first and last epoch, since single-step losses are noisy.
    first = np.mean([h["loss"] for h in history if "loss" in h and h["epoch"] <= 1])
    last = np.mean([h["loss"] for h in history if "loss" in h and h["epoch"] > cfg.epochs - 1])
    # Per-step gradient norms, dropping the inf/nan that fp16 logs for skipped steps.
    grad_norms = [g for h in history if "grad_norm" in h and math.isfinite(g := h["grad_norm"])]

    checks = {
        f"backward: {len(grad_norms)} finite grad norms, all > 0": bool(grad_norms) and min(grad_norms) > 0,
        **{
            f"optimizer updated {name}": not torch.equal(before[name], model.get_parameter(name).detach().cpu())
            for name in WATCHED_PARAMS
        },
        f"train loss ~0: epoch 1 mean {first:.4f} -> epoch {cfg.epochs} mean {last:.4f} < {OVERFIT_MAX_LOSS}": last < OVERFIT_MAX_LOSS,
        f"eval loss on the same {cfg.train_rows} rows {eval_loss:.4f} < {OVERFIT_MAX_LOSS}": eval_loss < OVERFIT_MAX_LOSS,
    }
    for label, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if failed := [label for label, ok in checks.items() if not ok]:
        raise RuntimeError(f"Overfit gate failed, the training loop is not learning: {failed}")


def prototype(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer: DistilBertTokenizerFast) -> None:
    """Fine-tune on 1,000 loans, evaluate on 500, and save the model to MODEL_DIR."""
    cfg = PROTOTYPE
    print(
        f"\n=== Stage 2: prototype {cfg.train_rows} train / {VAL_ROWS} val "
        f"x {cfg.epochs} epochs (lr={cfg.learning_rate}) ==="
    )
    train_sample = stratified_sample(train_df, cfg.train_rows)
    val_sample = stratified_sample(val_df, VAL_ROWS)

    # Build in a sibling dir and swap in at the end: a crash never leaves a mixed checkpoint.
    staging = MODEL_DIR.with_name(MODEL_DIR.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    trainer = build_trainer(
        cfg, new_model(), tokenizer,
        LoanTextDataset(train_sample, tokenizer), LoanTextDataset(val_sample, tokenizer), staging,
    )
    trainer.train()

    evals = [h for h in trainer.state.log_history if "eval_roc_auc" in h]
    for h in evals:
        print(
            f"epoch {h['epoch']:.0f}: eval_loss {h['eval_loss']:.4f} | "
            f"ROC-AUC {h['eval_roc_auc']:.4f} (baseline 0.5) | "
            f"PR-AUC {h['eval_pr_auc']:.4f} (baseline {h['eval_pr_auc_baseline']:.4f})"
        )

    trainer.save_model(str(staging))
    record = {
        "model_name": MODEL_NAME,
        "preprocess_version": PREPROCESS_VERSION,
        "config": {
            **asdict(cfg),
            "val_rows": VAL_ROWS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "precision": precision(),
            "seed": RANDOM_STATE,
        },
        "train_positives": int(train_sample["target"].sum()),
        "validation": evals[-1],
        "sample_loan_ids": {
            "train": train_sample["loan_id"].tolist(),
            "validation": val_sample["loan_id"].tolist(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": DEVICE_NAME,
        },
        "log_history": trainer.state.log_history,
    }
    (staging / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    staging.rename(MODEL_DIR)
    print(f"Saved model and metrics.json to {MODEL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["overfit", "prototype", "all"], default="all")
    stage = parser.parse_args().stage

    print(f"Device: {DEVICE_NAME} ({precision()})")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_df = load_original_split(DATA_DIR, "train")

    if stage in {"overfit", "all"}:
        overfit(train_df, tokenizer)
    if stage in {"prototype", "all"}:
        prototype(train_df, load_original_split(DATA_DIR, "validation"), tokenizer)


if __name__ == "__main__":
    main()

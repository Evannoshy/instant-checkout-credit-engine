"""DistilBERT Training Prototype.

Run from the repository root:
    python -m src.nlp.train_distilbert_prototype [--stage overfit|prototype|scaled|all] [--unweighted | --holdout]

Stage 1 (overfit): train on 100 loans for 10 epochs and stop unless the loss
reaches ~0 and the optimizer actually updates the weights.
Stage 2 (prototype): fine-tune on 1,000 train loans for 2 epochs, report ROC-AUC
and PR-AUC on 500 validation loans, and save the model with metrics.json to
models/distilbert_prototype/.
Stage 3 (scaled, not part of "all"): fine-tune on 10,000 train loans for 3 epochs
with class-weighted loss (--unweighted for the comparison run), evaluate on 5,000
validation loans, keep the epoch with the best ROC-AUC in models/distilbert_10k/,
and write reports/nlp/distilbert_10k_metrics.json. --holdout trains the same model on
10,000 loans distilbert_10k never saw, so the export can score every train loan without leakage.
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
from sklearn.metrics import roc_auc_score
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
REPORT_DIR = ROOT / "reports" / "nlp"
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
OPTIMIZER = "adamw_torch"
WEIGHT_DECAY = 0.01
VAL_ROWS = 500
SCALED_VAL_ROWS = 5_000 # 5,000 rows (760 defaults) give a 95% ROC-AUC interval of about +/-0.02; 500 rows give about +/-0.07.
ROC_AUC_TARGET = 0.53
BOOTSTRAP_RESAMPLES = 1_000
CONFIDENCE = 0.95
OVERFIT_MAX_LOSS = 0.15
WATCHED_PARAMS = ("classifier.bias", "distilbert.transformer.layer.0.attention.q_lin.bias")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class RunConfig:
    """Settings that differ between the stages."""

    train_rows: int
    epochs: int
    learning_rate: float
    lr_scheduler_type: str
    logging_steps: int

# Overfit: memorise 100 loans fast to verify training loss goes to approximately 0.
OVERFIT = RunConfig(train_rows=100, epochs=10, learning_rate=2e-5, lr_scheduler_type="constant", logging_steps=1)
# Prototype Fine-Tuning: train on 1,000 loans and evaluate on 500 each of 2 epochs to test the Trainer end-to-end.
PROTOTYPE = RunConfig(train_rows=1_000, epochs=2, learning_rate=2e-5, lr_scheduler_type="linear", logging_steps=10)
# 10k scaling: 10x the prototype's data with the same lr and batch size, so data size (and weighting) is the only change.
SCALED = RunConfig(train_rows=10_000, epochs=3, learning_rate=2e-5, lr_scheduler_type="linear", logging_steps=50)


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
    scores = default_probability(logits)
    ci_low, ci_high = roc_auc_interval(labels, scores)
    return {**evaluate(labels, scores), "roc_auc_ci_low": ci_low, "roc_auc_ci_high": ci_high}


def default_probability(logits: np.ndarray) -> np.ndarray:
    """Softmax probability of class 1 (default); float() first, since fp16 logits lose precision."""
    return torch.from_numpy(logits).float().softmax(-1)[:, 1].numpy()


def roc_auc_interval(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ROC-AUC over resampled loans; NaN if one class only."""
    if labels.min() == labels.max():
        return math.nan, math.nan
    rng = np.random.default_rng(RANDOM_STATE)
    resampled = []
    while len(resampled) < BOOTSTRAP_RESAMPLES:
        index = rng.integers(0, len(labels), size=len(labels))
        if labels[index].min() != labels[index].max():
            resampled.append(roc_auc_score(labels[index], scores[index]))
    tail = (1 - CONFIDENCE) / 2
    return float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1 - tail))


def target_verdict(roc_auc: float, ci_low: float) -> str:
    """PASS only if the whole confidence interval clears ROC_AUC_TARGET."""
    if ci_low > ROC_AUC_TARGET:
        return "PASS"
    return "INCONCLUSIVE" if roc_auc > ROC_AUC_TARGET else "FAIL"


def class_weights(targets: pd.Series) -> torch.Tensor:
    """Loss weights [1.0, N_non-default / N_default]; ~[1.0, 5.58] at the 15.2% default rate."""
    positives = int(targets.sum())
    return torch.tensor([1.0, (len(targets) - positives) / positives])


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy, so a missed default costs more than a false alarm.

    Unweighted, the 1k prototype predicted the base rate for every loan (F1@0.5 = 0.0).
    eval_loss is weighted too, so it is not comparable with unweighted runs.
    """

    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # compute_loss ignores num_items_in_batch, so Trainer must scale for gradient accumulation.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Class-weighted cross-entropy for one batch."""
        labels = inputs.pop("labels")  # pop: otherwise the model also computes an unweighted loss
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


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
    class_weighted: bool = False,
    keep_best_epoch: bool = False,
) -> Trainer:
    """Build the Trainer all stages share (AdamW, batch size 16, seeded).

    Args:
        eval_ds: Evaluated after every epoch; None skips evaluation.
        output_dir: Trainer working directory; no mid-run checkpoints are written
            unless keep_best_epoch is set.
        class_weighted: Return a WeightedTrainer weighted by train_ds labels.
        keep_best_epoch: Checkpoint each epoch and reload the one with the best
            validation ROC-AUC when training ends. Requires eval_ds.
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
        save_strategy="epoch" if keep_best_epoch else "no",  # Otherwise only the final model is saved, explicitly
        load_best_model_at_end=keep_best_epoch,
        metric_for_best_model="roc_auc" if keep_best_epoch else None,
        save_total_limit=1,
        save_only_model=True,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=RANDOM_STATE,
    )
    common = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    if class_weighted:
        return WeightedTrainer(**common, class_weights=class_weights(train_ds.df["target"]))
    return Trainer(**common)


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


def fine_tune(
    cfg: RunConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: DistilBertTokenizerFast,
    val_rows: int,
    model_dir: Path,
    class_weighted: bool = False,
    keep_best_epoch: bool = False,
) -> dict[str, Any]:
    """Fine-tune on stratified samples, save the model with metrics.json to model_dir, and return that record.

    Args:
        class_weighted: Train with WeightedTrainer.
        keep_best_epoch: Save the epoch with the best validation ROC-AUC instead of the last one.
    """
    print(
        f"\n=== {model_dir.name}: {cfg.train_rows} train / {val_rows} val "
        f"x {cfg.epochs} epochs (lr={cfg.learning_rate}, class_weighted={class_weighted}) ==="
    )
    train_sample = stratified_sample(train_df, cfg.train_rows)
    val_sample = stratified_sample(val_df, val_rows)

    # Build in a sibling dir and swap in at the end: a crash never leaves a mixed checkpoint.
    staging = model_dir.with_name(model_dir.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    trainer = build_trainer(
        cfg, new_model(), tokenizer,
        LoanTextDataset(train_sample, tokenizer), LoanTextDataset(val_sample, tokenizer), staging,
        class_weighted=class_weighted, keep_best_epoch=keep_best_epoch,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_memory = {
        "allocated": torch.cuda.max_memory_allocated() / 2**30,
        "reserved": torch.cuda.max_memory_reserved() / 2**30,
    } if torch.cuda.is_available() else None

    evals = [h for h in trainer.state.log_history if "eval_roc_auc" in h]
    for h in evals:
        print(
            f"epoch {h['epoch']:.0f}: eval_loss {h['eval_loss']:.4f} | "
            f"ROC-AUC {h['eval_roc_auc']:.4f} (baseline 0.5) | "
            f"PR-AUC {h['eval_pr_auc']:.4f} (baseline {h['eval_pr_auc_baseline']:.4f})"
        )
    validation = max(evals, key=lambda h: h["eval_roc_auc"]) if keep_best_epoch else evals[-1]

    for checkpoint in staging.glob("checkpoint-*"):
        shutil.rmtree(checkpoint)
    trainer.save_model(str(staging))
    record = {
        "model_name": MODEL_NAME,
        "preprocess_version": PREPROCESS_VERSION,
        "config": {
            **asdict(cfg),
            "val_rows": val_rows,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "precision": precision(),
            "seed": RANDOM_STATE,
            "roc_auc_ci": {"resamples": BOOTSTRAP_RESAMPLES, "confidence": CONFIDENCE},
            "class_weights": trainer.class_weights.tolist() if isinstance(trainer, WeightedTrainer) else None,
        },
        "train_positives": int(train_sample["target"].sum()),
        "validation": validation,
        "sample_loan_ids": {
            "train": train_sample["loan_id"].tolist(),
            "validation": val_sample["loan_id"].tolist(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": DEVICE_NAME,
            "peak_gpu_memory_gb": peak_memory,
        },
        "log_history": trainer.state.log_history,
    }
    (staging / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    shutil.rmtree(model_dir, ignore_errors=True)
    staging.rename(model_dir)
    print(f"Saved model and metrics.json to {model_dir}")
    return record


def prototype(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer: DistilBertTokenizerFast) -> None:
    """Fine-tune on 1,000 loans, evaluate on 500, and save the model to MODEL_DIR."""
    fine_tune(PROTOTYPE, train_df, val_df, tokenizer, VAL_ROWS, MODEL_DIR)


def slim_report(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce a run record to the layout of reports/nlp/distilbert_prototype_metrics.json."""

    def metrics(h: dict[str, Any]) -> dict[str, Any]:
        return {k.removeprefix("eval_"): v for k, v in h.items()
                if k.startswith("eval_") and not k.endswith(("runtime", "per_second"))}

    history = record["log_history"]
    summary = next(h for h in history if "train_loss" in h)  # Trainer's end-of-training entry
    return {
        "model_name": record["model_name"],
        "preprocess_version": record["preprocess_version"],
        "config": record["config"],
        "train": {
            "rows": record["config"]["train_rows"],
            "positives": record["train_positives"],
            "steps": summary["step"],
            "mean_loss": summary["train_loss"],
            "runtime_seconds": summary["train_runtime"],
        },
        "best_epoch": round(record["validation"]["epoch"]),
        "best_epoch_selected_by": "ROC-AUC on this validation sample (so slightly optimistic)",
        "validation": metrics(record["validation"]),
        "validation_by_epoch": [{"epoch": round(h["epoch"]), **metrics(h)} for h in history if "eval_roc_auc" in h],
        "environment": record["environment"],
    }


def scaled(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: DistilBertTokenizerFast,
    weighted: bool = True,
    holdout: bool = False,
) -> None:
    """Fine-tune on 10,000 loans, keep the best epoch, and write the slim report to REPORT_DIR.

    Args:
        weighted: Use class-weighted loss; False is the comparison run, saved under *_unweighted names.
        holdout: Train on 10,000 loans distilbert_10k never saw, so it can score that model's
            training loans without leakage; saved under *_holdout names.
    """
    name = "distilbert_10k_holdout" if holdout else "distilbert_10k" if weighted else "distilbert_10k_unweighted"
    if holdout:
        main_run = json.loads((MODEL_DIR.with_name("distilbert_10k") / "metrics.json").read_text(encoding="utf-8"))
        train_df = train_df[~train_df["loan_id"].isin(main_run["sample_loan_ids"]["train"])]
    record = fine_tune(
        SCALED, train_df, val_df, tokenizer, SCALED_VAL_ROWS, MODEL_DIR.with_name(name),
        class_weighted=weighted, keep_best_epoch=True,
    )
    report = slim_report(record)
    best = report["validation"]
    verdict = target_verdict(best["roc_auc"], best["roc_auc_ci_low"])
    report["roc_auc_target"] = {"target": ROC_AUC_TARGET, "verdict": verdict}
    path = REPORT_DIR / f"{name}_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"{verdict}: best epoch {report['best_epoch']} ROC-AUC {best['roc_auc']:.4f}, "
        f"{CONFIDENCE:.0%} CI [{best['roc_auc_ci_low']:.4f}, {best['roc_auc_ci_high']:.4f}] "
        f"vs target {ROC_AUC_TARGET} (1k prototype: 0.505). Saved report to {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["overfit", "prototype", "scaled", "all"], default="all")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--unweighted", action="store_true", help="With --stage scaled: the unweighted comparison run")
    variant.add_argument("--holdout", action="store_true", help="With --stage scaled: train on loans distilbert_10k never saw")
    args = parser.parse_args()
    stage = args.stage
    if (args.unweighted or args.holdout) and stage != "scaled":
        parser.error("--unweighted and --holdout only apply to --stage scaled")

    print(f"Device: {DEVICE_NAME} ({precision()})")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_df = load_original_split(DATA_DIR, "train")

    if stage in {"overfit", "all"}:
        overfit(train_df, tokenizer)
    if stage in {"prototype", "all"}:
        prototype(train_df, load_original_split(DATA_DIR, "validation"), tokenizer)
    if stage == "scaled":
        scaled(
            train_df, load_original_split(DATA_DIR, "validation"), tokenizer,
            weighted=not args.unweighted, holdout=args.holdout,
        )


if __name__ == "__main__":
    main()

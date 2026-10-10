"""Export DistilBERT default probabilities for the tabular track.

Run from the repository root on the GPU node:
    python -m src.nlp.export_distilbert_scores [--model-dir DIR] [--holdout-model-dir DIR] [--output-dir DIR]

No loan is scored by a model that trained on it: distilbert_10k scores every loan except its
own 10,000 training loans, which distilbert_10k_holdout scores; model_version records which.
Probabilities are uncalibrated (class weighting inflates them).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from transformers import DataCollatorWithPadding, DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.nlp.dataset import LoanTextDataset
from src.nlp.preprocess import load_original_split
from src.nlp.train_distilbert_prototype import DATA_DIR, MODEL_DIR, MODEL_NAME, default_probability, precision

SCALED_MODEL_DIR = MODEL_DIR.with_name("distilbert_10k")
HOLDOUT_MODEL_DIR = MODEL_DIR.with_name("distilbert_10k_holdout")
PROBABILITY_COLUMN = "prob_default_distilbert"
MODEL_VERSION = "nlp-distilbert-10k-v1"
HOLDOUT_MODEL_VERSION = "nlp-distilbert-10k-holdout-v1"
INFERENCE_BATCH_SIZE = 128  # 8x training: no gradients or optimizer state to hold
OUTPUTS = {"train": "distilbert_scores_train.parquet", "validation": "distilbert_scores_val.parquet"}


@torch.no_grad()
def score(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    df: pd.DataFrame,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    """P(default) for every row of df, returned in df's row order."""
    model.eval()
    device = model.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision())
    # Batch similar lengths together to cut padding; probs[order] restores the input order.
    order = np.argsort(df["text_payload"].str.len().to_numpy(), kind="stable")
    loader = DataLoader(
        Subset(LoanTextDataset(df, tokenizer), order.tolist()),
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    logits = []
    for batch in loader:
        batch.pop("labels")
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits.append(model(**batch.to(device)).logits.float().cpu())
    probs = np.empty(len(df))
    probs[order] = default_probability(torch.cat(logits).numpy())
    return probs


def load(model_dir: Path, device: torch.device) -> tuple[DistilBertForSequenceClassification, set[str]]:
    """A saved model and the loan IDs it was trained on."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    record = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    return model, set(record["sample_loan_ids"]["train"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=SCALED_MODEL_DIR)
    parser.add_argument("--holdout-model-dir", type=Path, default=HOLDOUT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "nlp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({precision()})")
    model, trained_ids = load(args.model_dir, device)
    holdout_model, holdout_ids = load(args.holdout_model_dir, device)
    if trained_ids & holdout_ids:
        raise ValueError("The holdout model trained on some of the main model's training loans")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in OUTPUTS.items():
        df = load_original_split(DATA_DIR, split)
        seen = df["loan_id"].isin(trained_ids).to_numpy()
        if split == "train" and seen.sum() != len(trained_ids):
            raise ValueError("Some of the main model's training loans are missing from the train split")
        probs = np.empty(len(df))
        probs[~seen] = score(model, tokenizer, df[~seen])
        if seen.any():
            probs[seen] = score(holdout_model, tokenizer, df[seen])
        frame = pd.DataFrame({
            "loan_id": df["loan_id"].astype("string"),
            "split": split,
            PROBABILITY_COLUMN: probs,
            "model_name": MODEL_NAME,
            "model_version": np.where(seen, HOLDOUT_MODEL_VERSION, MODEL_VERSION),
        })
        path = args.output_dir / filename
        frame.to_parquet(path, index=False)
        print(f"{split}: {len(frame)} rows -> {path}")
        for version, rows in ((MODEL_VERSION, ~seen), (HOLDOUT_MODEL_VERSION, seen)):
            if rows.any():
                print(
                    f"  {version}: {rows.sum()} rows, ROC-AUC {roc_auc_score(df['target'][rows], probs[rows]):.4f}, "
                    f"mean P(default) {probs[rows].mean():.3f}"
                )


if __name__ == "__main__":
    main()

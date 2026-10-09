"""Export DistilBERT default probabilities for the tabular track.

Run from the repository root on the GPU node:
    python -m src.nlp.export_distilbert_scores [--model-dir DIR] [--output-dir DIR]

Probabilities are uncalibrated (class weighting inflates them); in_distilbert_train
marks the 10,000 train loans the model saw, whose scores are optimistic.
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
PROBABILITY_COLUMN = "prob_default_distilbert"
MODEL_VERSION = "nlp-distilbert-10k-v1"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=SCALED_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "nlp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({precision()})")
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    record = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))
    trained_ids = set(record["sample_loan_ids"]["train"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in OUTPUTS.items():
        df = load_original_split(DATA_DIR, split)
        probs = score(model, tokenizer, df)
        frame = pd.DataFrame({
            "loan_id": df["loan_id"].astype("string"),
            "split": split,
            PROBABILITY_COLUMN: probs,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "in_distilbert_train": df["loan_id"].isin(trained_ids),
        })
        if split == "train" and frame["in_distilbert_train"].sum() != len(trained_ids):
            raise ValueError("Some of the model's training loans are missing from the train split")
        path = args.output_dir / filename
        frame.to_parquet(path, index=False)
        print(
            f"{split}: {len(frame)} rows, {frame['in_distilbert_train'].sum()} seen in training, "
            f"ROC-AUC {roc_auc_score(df['target'], probs):.4f} -> {path}"
        )


if __name__ == "__main__":
    main()

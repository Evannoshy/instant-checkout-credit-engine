"""Minimal executable check: one forward/backward pass on a dummy batch.

Run from the repo root: python -m src.nlp.sanity_check
Confirms the LoanTextDataset -> DataLoader -> DistilBertForSequenceClassification
loop works end to end, with dynamic per-batch padding (see dataset.py).
"""

from pathlib import Path

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, DistilBertForSequenceClassification

from src.nlp.dataset import LoanTextDataset, tokenizer
from src.nlp.preprocess import load_original_split

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BATCH_SIZE = 4


def main() -> None:
    df = load_original_split(DATA_DIR, "train").head(BATCH_SIZE)
    dataset = LoanTextDataset(df, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),  # pad each batch to its own longest row
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    batch = next(iter(loader))
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)

    outputs = model(**batch)
    outputs.loss.backward()
    print("loss:", outputs.loss.item())


if __name__ == "__main__":
    main()

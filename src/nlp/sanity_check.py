"""Minimal executable check: one forward/backward pass on a real batch.

Run from the repo root: python -m src.nlp.sanity_check
Confirms the LoanTextDataset -> DataLoader -> DistilBertForSequenceClassification
loop works end to end, with dynamic per-batch padding (see dataset.py).

Runs two batches:
1. The first SAMPLE_ROWS training rows, in whatever order they come in.
   These are usually short, so this mostly checks the pipeline wiring.
2. A batch of real long borrower descriptions, picked so that MAX_LENGTH
   truncation/padding is actually exercised. This is the one that answers
   "does max_length=384 at batch_size=16 run without crashing?"
"""

from pathlib import Path

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, DistilBertForSequenceClassification

from src.nlp.dataset import MAX_LENGTH, LoanTextDataset, tokenizer
from src.nlp.preprocess import load_original_split

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SAMPLE_ROWS = 500
BATCH_SIZE = 16


def run_batch(label: str, df, model) -> None:
    """Tokenize df as one batch and run it through the model forward + backward."""
    dataset = LoanTextDataset(df, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),  # pad every row to this batch's longest row
    )
    batch = next(iter(loader))

    print(f"\n=== {label} ===")
    num_rows, num_tokens = batch["input_ids"].shape
    print(f"Batch shape: {num_rows} loan descriptions, each padded/truncated to {num_tokens} tokens")

    model.zero_grad()
    outputs = model(**batch)
    outputs.loss.backward()
    print(f"Forward + backward pass succeeded. Loss: {outputs.loss.item():.4f}")


def main() -> None:
    df = load_original_split(DATA_DIR, "train")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    run_batch(f"Check the first {SAMPLE_ROWS} rows", df.head(SAMPLE_ROWS), model)

    # A word count above MAX_LENGTH is well over the token count so they are guaranteed 
    # to need truncation.
    word_counts = df["text_payload"].str.split().str.len()
    long_rows = df[word_counts > MAX_LENGTH]
    long_batch = long_rows.sample(BATCH_SIZE, random_state=0)

    run_batch(
        f"Check {BATCH_SIZE} longest real descriptions (tests whether MAX_LENGTH={MAX_LENGTH} triggers an OOM)",
        long_batch, model,
    )
    print(f"\nResult: batch_size={BATCH_SIZE} at max_length={MAX_LENGTH} ran with no crash, using real long borrower text.")


if __name__ == "__main__":
    main()

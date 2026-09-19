"""PyTorch Dataset wrapping tokenized loan text payloads.

Uses the shared build_text_payload()/load_original_split() from preprocess.py.
Padding is deferred to batch time per text_eda.ipynb: truncation=True at 
max_length=384, with dynamic padding per batch rather than a fixed length per row.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

MAX_LENGTH = 384 # Set to 128 in-case of OOM error


class LoanTextDataset(Dataset):
    """Tokenizes loan text payloads on the fly. Labels come from ``target``."""

    def __init__(
        self, df: pd.DataFrame, tokenizer: DistilBertTokenizerFast, max_length: int = MAX_LENGTH
    ) -> None:
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        """Encodes a row's text into token IDs, attention mask and a label.
        Returns unpadded Python lists/int. DataCollatorWithPadding converts to
        padded tensors at batch time.

        Args:
            idx: Positional row index requested by the DataLoader.

        Returns:
            A dict with input_ids, attention_mask, and labels for one row.
        """
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["text_payload"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": int(row["target"]),
        }

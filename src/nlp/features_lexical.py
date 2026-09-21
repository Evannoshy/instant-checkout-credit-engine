"""Build deterministic borrower-level lexical features for tabular models.

The input text is the approved payload produced by :mod:`src.nlp.preprocess`:
title, purpose and original borrower description. The held-out test split is
never loaded. Run from the repository root with::

    python -m src.nlp.features_lexical

The output contains train rows followed by validation rows, in exactly the
order specified by ``train_ids.csv`` and ``val_ids.csv``.
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from textblob import TextBlob

from src.nlp.preprocess import load_original_split

DISTRESS_KEYWORDS = frozenset(
    {
        "arrears",
        "bankruptcy",
        "behind",
        "delinquent",
        "emergency",
        "eviction",
        "foreclosure",
        "medical",
        "overdue",
        "payday",
        "unemployed",
    }
)
FEATURE_COLUMNS = (
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "lexical_diversity_ttr",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "financial_distress_keyword_density",
)

_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?:[.!?]+|\n+)")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def tokenize_words(text: str) -> list[str]:
    """Return case-folded, Unicode-aware word tokens without numbers."""
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str; got {type(text).__name__}")
    return [token.casefold().replace("’", "'") for token in _WORD.findall(text)]


def count_sentences(text: str, word_count: int | None = None) -> int:
    """Count non-empty sentence-like spans, treating payload lines as spans."""
    if word_count is None:
        word_count = len(tokenize_words(text))
    if word_count == 0:
        return 0
    spans = [span for span in _SENTENCE_BOUNDARY.split(text) if tokenize_words(span)]
    return max(1, len(spans))


def count_syllables(word: str) -> int:
    """Estimate English syllables deterministically without downloaded models."""
    ascii_word = (
        unicodedata.normalize("NFKD", word.casefold())
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    ascii_word = re.sub(r"[^a-z]", "", ascii_word)
    if not ascii_word:
        return 1
    if len(ascii_word) <= 3:
        return 1

    syllables = len(_VOWEL_GROUP.findall(ascii_word))
    if ascii_word.endswith("e") and not ascii_word.endswith(("le", "ye")):
        syllables -= 1
    if (
        ascii_word.endswith("le")
        and len(ascii_word) > 2
        and ascii_word[-3] not in "aeiouy"
    ):
        syllables += 1
    return max(1, syllables)


def extract_lexical_features(text: str) -> dict[str, float]:
    """Compute the six numerical lexical features for one borrower payload."""
    words = tokenize_words(text)
    word_count = len(words)
    if word_count == 0:
        return {name: 0.0 for name in FEATURE_COLUMNS}

    sentence_count = count_sentences(text, word_count)
    syllable_count = sum(count_syllables(word) for word in words)
    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count
    sentiment = TextBlob(text).sentiment
    distress_count = sum(word in DISTRESS_KEYWORDS for word in words)

    values = {
        "flesch_reading_ease": (
            206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        ),
        "flesch_kincaid_grade": (
            0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
        ),
        "lexical_diversity_ttr": len(set(words)) / word_count,
        "sentiment_polarity": float(sentiment.polarity),
        "sentiment_subjectivity": float(sentiment.subjectivity),
        "financial_distress_keyword_density": distress_count / word_count,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Lexical feature calculation produced a non-finite value")
    return values


def build_lexical_feature_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Transform preprocessed borrower rows while preserving their order."""
    required = {"loan_id", "split", "text_payload"}
    if missing := required - set(rows.columns):
        raise ValueError(f"Input rows are missing columns: {sorted(missing)}")
    if rows["loan_id"].isna().any() or rows["loan_id"].duplicated().any():
        raise ValueError("Input loan_id values must be present and unique")
    if not rows["split"].isin(["train", "validation"]).all():
        raise ValueError("Only train and validation rows may be featurized")
    if rows["text_payload"].isna().any():
        raise ValueError("Input text_payload values must be present")

    metrics = pd.DataFrame(
        [extract_lexical_features(text) for text in rows["text_payload"]],
        columns=FEATURE_COLUMNS,
        index=rows.index,
    )
    keys = rows[["loan_id", "split"]].copy()
    keys["loan_id"] = keys["loan_id"].astype("string")
    return pd.concat([keys, metrics], axis="columns").reset_index(drop=True)


def _load_expected_keys(data_dir: Path) -> pd.DataFrame:
    pieces = []
    for split, filename in (("train", "train_ids.csv"), ("validation", "val_ids.csv")):
        ids = pd.read_csv(data_dir / filename, dtype={"loan_id": "string"})
        if list(ids.columns) != ["loan_id"]:
            raise ValueError(f"{filename} must contain exactly one loan_id column")
        if ids["loan_id"].isna().any() or ids["loan_id"].duplicated().any():
            raise ValueError(f"{filename} contains missing or duplicate loan IDs")
        ids["split"] = split
        pieces.append(ids)
    expected = pd.concat(pieces, ignore_index=True)
    if expected["loan_id"].duplicated().any():
        raise ValueError("Loan IDs overlap between train_ids.csv and val_ids.csv")
    return expected


def align_to_split_ids(features: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    """Validate membership and reorder features to the frozen ID files."""
    data_dir = Path(data_dir)
    required = {"loan_id", "split", *FEATURE_COLUMNS}
    if missing := required - set(features.columns):
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    if features["loan_id"].isna().any() or features["loan_id"].duplicated().any():
        raise ValueError("Feature loan_id values must be present and unique")
    expected = _load_expected_keys(data_dir)
    actual_keys = features[["loan_id", "split"]].copy()
    actual_keys["loan_id"] = actual_keys["loan_id"].astype("string")
    if set(map(tuple, actual_keys.itertuples(index=False, name=None))) != set(
        map(tuple, expected.itertuples(index=False, name=None))
    ):
        raise ValueError("Feature IDs or split assignments do not match the frozen ID files")

    aligned = expected.merge(
        features,
        on=["loan_id", "split"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Aligned feature table contains missing numerical values")
    return aligned[["loan_id", "split", *FEATURE_COLUMNS]]


def export_lexical_features(
    data_dir: str | Path,
    output_path: str | Path,
    *,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Build, align, validate and write the train/validation Parquet table."""
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    source_rows = pd.concat(
        [
            load_original_split(data_dir, split, chunksize=chunksize)
            for split in ("train", "validation")
        ],
        ignore_index=True,
    )
    features = build_lexical_feature_table(source_rows)
    aligned = align_to_split_ids(features, data_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    written = pd.read_parquet(output_path, engine="pyarrow")
    expected = _load_expected_keys(data_dir)
    if not written[["loan_id", "split"]].reset_index(drop=True).equals(expected):
        raise RuntimeError("Written Parquet file failed its ID-order verification")
    if written[list(FEATURE_COLUMNS)].isna().any().any():
        raise RuntimeError("Written Parquet file contains missing numerical features")
    return written


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repository_root / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "data" / "nlp" / "text_lexical_features.parquet",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    table = export_lexical_features(args.data_dir, args.output, chunksize=args.chunksize)
    counts = table["split"].value_counts().to_dict()
    print(f"Wrote {len(table):,} aligned rows to {args.output} ({counts})")


if __name__ == "__main__":
    main()

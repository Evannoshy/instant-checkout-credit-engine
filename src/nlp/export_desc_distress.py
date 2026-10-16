"""Export description-only distress density for the frozen development IDs.

Run ``python -m src.nlp.export_desc_distress`` from the repository root.
Only ``desc`` is read from loan.csv. No labels, titles or purpose values enter
the feature path, and no held-out test features are computed or exported.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.nlp.features_lexical import DISTRESS_KEYWORDS, _load_expected_keys, tokenize_words
from src.nlp.preprocess import PREPROCESS_VERSION, clean_text, sha256_file

FEATURE_VERSION = "nlp-desc-distress-v1"
FEATURE = "desc_distress_keyword_density"
OUTPUT_COLUMNS = ("loan_id", "split", FEATURE)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("loan_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field(FEATURE, pa.float64(), nullable=False),
    ],
    metadata={b"feature_version": FEATURE_VERSION.encode()},
)


def parquet_writer_metadata(path: str | Path) -> dict[str, str]:
    """Read the writer details from the file, not the current Python environment."""
    footer = pq.read_metadata(path)
    pandas_metadata = json.loads(footer.metadata[b"pandas"])
    return {
        "created_by": footer.created_by,
        "pandas_version": pandas_metadata["pandas_version"],
    }


def description_distress_density(description: object) -> float:
    """Count exact keyword tokens / all word tokens in the cleaned description.

    Missing, marker-only and zero-word descriptions have density zero. Repeated
    keyword tokens each count; negation does not change an exact-match hit.
    """
    words = tokenize_words(clean_text(description))
    return sum(word in DISTRESS_KEYWORDS for word in words) / len(words) if words else 0.0


def build_description_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Ignore every input field except loan_id, split and desc."""
    if missing := {"loan_id", "split", "desc"} - set(rows.columns):
        raise ValueError(f"Description rows are missing columns: {sorted(missing)}")
    keys = rows[["loan_id", "split"]].astype("string").reset_index(drop=True)
    if keys["loan_id"].isna().any() or keys["loan_id"].str.strip().eq("").any():
        raise ValueError("loan_id must be present and nonempty")
    if keys["loan_id"].duplicated().any():
        raise ValueError("loan_id must be unique")
    if keys["split"].isna().any() or not keys["split"].isin(["train", "validation"]).all():
        raise ValueError("Only train and validation rows may be featurized")
    keys[FEATURE] = rows["desc"].map(description_distress_density).to_numpy(dtype=float)
    return keys


def _load_descriptions(data_dir: Path, *, chunksize: int) -> tuple[pd.DataFrame, dict]:
    """Map frozen IDs to CSV data records without reading outcome columns."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    with (data_dir / "split_statistics.json").open(encoding="utf-8") as stream:
        stats = json.load(stream)
    raw_path = data_dir / "raw" / "loan.csv"
    expected_hash = stats.get("raw_csv_sha256")
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("Missing or invalid raw_csv_sha256 benchmark")
    if sha256_file(raw_path) != expected_hash:
        raise ValueError("Raw CSV SHA-256 does not match the approved benchmark")

    expected = _load_expected_keys(data_dir).astype("string")
    manifest = pd.read_csv(
        data_dir / "split_manifest.csv",
        usecols=["loan_id", "split", "source_row_number"],
        dtype={"loan_id": "string", "split": "string", "source_row_number": "Int64"},
    )
    selected = manifest.loc[manifest["split"].isin(["train", "validation"])].copy()
    if selected["loan_id"].isna().any() or selected["loan_id"].duplicated().any():
        raise ValueError("Development manifest IDs must be present and unique")
    numbers = selected["source_row_number"]
    if numbers.isna().any() or numbers.duplicated().any() or not numbers.between(1, stats["raw_rows"]).all():
        raise ValueError("Development source row numbers must be unique valid CSV record numbers")
    if not selected["loan_id"].eq(numbers.map(lambda n: f"lc_{int(n):09d}")).all():
        raise ValueError("loan_id does not match the one-based source-row convention")
    actual_keys = set(selected[["loan_id", "split"]].itertuples(index=False, name=None))
    if actual_keys != set(expected.itertuples(index=False, name=None)):
        raise ValueError("Development manifest does not match the frozen ID files")
    for split in ("train", "validation"):
        if int(expected["split"].eq(split).sum()) != stats["splits"][split]["rows"]:
            raise ValueError(f"Frozen {split} count differs from split_statistics.json")

    wanted = set(numbers)
    pieces = []
    records_seen = 0
    with pd.read_csv(
        raw_path, usecols=["desc"], dtype="string",
        keep_default_na=False, chunksize=chunksize,
    ) as reader:
        for chunk in reader:
            chunk["source_row_number"] = range(records_seen + 1, records_seen + len(chunk) + 1)
            records_seen += len(chunk)
            retained = chunk.loc[chunk["source_row_number"].isin(wanted)]
            if not retained.empty:
                pieces.append(retained.copy())
    if records_seen != stats["raw_rows"]:
        raise ValueError("Raw CSV record count differs from split_statistics.json")
    if not pieces:
        raise ValueError("No frozen development descriptions found")
    rows = expected.merge(selected, on=["loan_id", "split"], validate="one_to_one", sort=False)
    rows = rows.merge(
        pd.concat(pieces, ignore_index=True), on="source_row_number",
        how="left", validate="one_to_one", sort=False, indicator=True,
    )
    if not rows["_merge"].eq("both").all():
        raise ValueError("Some frozen IDs have no source description record")
    provenance = {
        "dataset_version": stats["dataset_version"],
        "cohort": stats["cohort"],
        "split_version": stats["split_version"],
        "preprocess_version": PREPROCESS_VERSION,
        "raw_csv_sha256": expected_hash,
        "manifest_sha256": sha256_file(data_dir / "split_manifest.csv"),
        "train_ids_sha256": sha256_file(data_dir / "train_ids.csv"),
        "val_ids_sha256": sha256_file(data_dir / "val_ids.csv"),
    }
    return rows[["loan_id", "split", "desc"]], provenance


def export_description_distress(
    data_dir: str | Path,
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Write an aligned three-column Parquet and optional aggregate provenance."""
    data_dir, output_path = Path(data_dir), Path(output_path)
    if metadata_path is not None and Path(metadata_path).resolve() == output_path.resolve():
        raise ValueError("Metadata and Parquet paths must differ")
    rows, provenance = _load_descriptions(data_dir, chunksize=chunksize)
    features = build_description_features(rows)
    expected = _load_expected_keys(data_dir).astype("string")
    pd.testing.assert_frame_equal(features[["loan_id", "split"]], expected)
    if not features[FEATURE].between(0, 1).all():
        raise ValueError("Description density must be finite and within [0, 1]")

    table = pa.Table.from_pandas(features, schema=OUTPUT_SCHEMA, preserve_index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        pq.write_table(table, temporary_path, compression="snappy")
        written = pd.read_parquet(temporary_path)
        pd.testing.assert_frame_equal(written, features)
        # Replace only after schema, values and frozen row order survive a round trip.
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "feature_version": FEATURE_VERSION,
            "columns": list(OUTPUT_COLUMNS),
            "source_fields": ["desc"],
            "density_definition": "exact distress-keyword tokens / all cleaned description word tokens",
            "zero_word_density": 0.0,
            "keywords": sorted(DISTRESS_KEYWORDS),
            "rows": len(features),
            "rows_by_split": {name: int(count) for name, count in features["split"].value_counts().items()},
            "positive_density_rows_by_split": {
                name: int(group[FEATURE].gt(0).sum()) for name, group in features.groupby("split")
            },
            "test_rows_exported": 0,
            "outcome_columns_read": [],
            "parquet_sha256": sha256_file(output_path),
            "parquet_writer": parquet_writer_metadata(output_path),
            "provenance": provenance,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return features


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument(
        "--output", type=Path, default=root / "data/nlp/desc_distress_keyword_density.parquet",
    )
    parser.add_argument(
        "--metadata", type=Path, default=root / "reports/nlp/desc_distress_export_metrics.json",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    features = export_description_distress(
        args.data_dir, args.output, metadata_path=args.metadata, chunksize=args.chunksize,
    )
    print(f"Exported {len(features):,} rows and {len(features.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()

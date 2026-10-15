"""Synthetic contract tests and checks against the committed aggregate evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.nlp import export_desc_distress as export
from src.nlp.preprocess import sha256_file


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Medical emergency medical bills", 3 / 4),
        ("Borrower added on 07/12/13 > <p>MEDICAL bills overdue</p>", 2 / 3),
        ("medicalized medical-bills bills", 0.0),
        ("Not unemployed", 1 / 2),
        ("Ordinary debt consolidation", 0.0),
        ("N/A", 0.0),
        (None, 0.0),
        (pd.NA, 0.0),
        ("123 !!!", 0.0),
    ],
)
def test_density_uses_existing_cleaning_and_exact_token_semantics(description, expected):
    assert export.description_distress_density(description) == pytest.approx(expected)


def test_features_are_invariant_to_title_purpose_and_outcome():
    rows = pd.DataFrame({
        "loan_id": ["a", "b"], "split": ["train", "validation"],
        "desc": ["A routine purchase", "Medical bills"],
        "title": ["medical emergency", "ordinary"],
        "purpose": ["medical", "debt_consolidation"], "target": [1, 0],
    }, index=[9, 2])
    original = export.build_description_features(rows)
    altered = export.build_description_features(rows.assign(
        purpose="medical", title="unemployed bankruptcy", target=[0, 1],
    ))
    pd.testing.assert_frame_equal(original, altered)
    assert list(original.columns) == list(export.OUTPUT_COLUMNS)
    assert original[export.FEATURE].tolist() == [0.0, 0.5]


@pytest.mark.parametrize("mutation", [
    {"split": ["train", "test"]}, {"split": ["train", None]},
    {"loan_id": ["a", "a"]}, {"loan_id": ["a", None]}, {"loan_id": ["a", ""]},
])
def test_builder_rejects_test_rows_or_bad_keys(mutation):
    rows = pd.DataFrame({"loan_id": ["a", "b"], "split": ["train", "validation"], "desc": ["", ""]})
    with pytest.raises(ValueError):
        export.build_description_features(rows.assign(**mutation))


@pytest.fixture
def source(tmp_path):
    """Small synthetic source, including a quoted newline and excluded records."""
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    pd.DataFrame({
        "desc": [
            "Medical emergency", "EXCLUDED TEST DESCRIPTION",
            "<p>Overdue\nbills</p>", "N/A", "Outside cohort",
        ],
        "purpose": ["car"] * 5, "title": ["medical"] * 5,
        "loan_status": ["OUTCOME MUST NOT BE READ"] * 5,
    }).to_csv(data / "raw/loan.csv", index=False)
    pd.DataFrame({
        "loan_id": [f"lc_{n:09d}" for n in [1, 2, 3, 4]],
        "source_row_number": [1, 2, 3, 4],
        "split": ["train", "test", "validation", "train"],
        "target": ["OUTCOME MUST NOT BE READ"] * 4,
    }).to_csv(data / "split_manifest.csv", index=False)
    pd.DataFrame({"loan_id": ["lc_000000004", "lc_000000001"]}).to_csv(data / "train_ids.csv", index=False)
    pd.DataFrame({"loan_id": ["lc_000000003"]}).to_csv(data / "val_ids.csv", index=False)
    stats = {
        "raw_csv_sha256": sha256_file(data / "raw/loan.csv"),
        "raw_rows": 5, "dataset_version": "synthetic-fixture", "cohort": "synthetic",
        "split_version": "synthetic-v1", "splits": {"train": {"rows": 2}, "validation": {"rows": 1}},
    }
    (data / "split_statistics.json").write_text(json.dumps(stats), encoding="utf-8")
    return data


def test_export_round_trip_order_isolation_and_chunk_independence(source, tmp_path, monkeypatch):
    reads = []
    descriptions = []
    original_read = pd.read_csv
    original_density = export.description_distress_density

    def tracked_read(path, *args, **kwargs):
        reads.append((Path(path).name, kwargs.get("usecols")))
        return original_read(path, *args, **kwargs)

    def tracked_density(description):
        descriptions.append(description)
        return original_density(description)

    monkeypatch.setattr(pd, "read_csv", tracked_read)
    monkeypatch.setattr(export, "description_distress_density", tracked_density)
    output = tmp_path / "density.parquet"
    metadata_path = tmp_path / "metadata.json"
    first = export.export_description_distress(source, output, metadata_path=metadata_path, chunksize=1)
    first_hash = sha256_file(output)
    second = export.export_description_distress(source, output, chunksize=3)
    pd.testing.assert_frame_equal(first, second)
    assert sha256_file(output) == first_hash
    assert first["loan_id"].tolist() == ["lc_000000004", "lc_000000001", "lc_000000003"]
    assert first[export.FEATURE].tolist() == [0.0, 1.0, 0.5]
    assert "EXCLUDED TEST DESCRIPTION" not in descriptions
    assert "Outside cohort" not in descriptions
    assert all(columns == ["desc"] for name, columns in reads if name == "loan.csv")
    assert all(
        columns == ["loan_id", "split", "source_row_number"]
        for name, columns in reads if name == "split_manifest.csv"
    )
    schema = pq.read_schema(output)
    assert schema.equals(export.OUTPUT_SCHEMA, check_metadata=False)
    assert schema.metadata[b"feature_version"] == export.FEATURE_VERSION.encode()
    assert "__index_level_0__" not in schema.names
    metadata = json.loads(metadata_path.read_text())
    assert metadata["rows_by_split"] == {"train": 2, "validation": 1}
    assert metadata["outcome_columns_read"] == []
    assert metadata["test_rows_exported"] == 0
    assert metadata["parquet_sha256"] == first_hash


@pytest.mark.parametrize("defect", [
    "checksum", "wrong_id", "missing_id", "duplicate_id", "wrong_split", "source_row", "row_count",
])
def test_export_fails_closed_on_source_or_membership_mismatch(source, tmp_path, defect):
    if defect in {"checksum", "row_count"}:
        path = source / "split_statistics.json"
        stats = json.loads(path.read_text())
        field = "raw_csv_sha256" if defect == "checksum" else "raw_rows"
        stats[field] = "0" * 64 if defect == "checksum" else 6
        path.write_text(json.dumps(stats))
    elif defect in {"wrong_id", "missing_id", "duplicate_id"}:
        path = source / "train_ids.csv"
        ids = pd.read_csv(path)
        if defect == "wrong_id":
            ids.loc[0, "loan_id"] = "lc_000000005"
        elif defect == "missing_id":
            ids = ids.iloc[:1]
        else:
            ids.loc[0, "loan_id"] = ids.loc[1, "loan_id"]
        ids.to_csv(path, index=False)
    else:
        path = source / "split_manifest.csv"
        manifest = pd.read_csv(path)
        field = "split" if defect == "wrong_split" else "source_row_number"
        manifest.loc[0, field] = "validation" if defect == "wrong_split" else 3
        manifest.to_csv(path, index=False)
    output = tmp_path / "density.parquet"
    with pytest.raises(ValueError):
        export.export_description_distress(source, output, chunksize=2)
    assert not output.exists()


def test_committed_export_matches_frozen_ids_and_training_eda():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/nlp/desc_distress_keyword_density.parquet"
    actual = pd.read_parquet(path)
    expected = pd.concat([
        pd.read_csv(root / "data/train_ids.csv", dtype="string").assign(split="train"),
        pd.read_csv(root / "data/val_ids.csv", dtype="string").assign(split="validation"),
    ], ignore_index=True).astype("string")
    assert list(actual.columns) == list(export.OUTPUT_COLUMNS)
    assert pq.read_schema(path).equals(export.OUTPUT_SCHEMA, check_metadata=False)
    pd.testing.assert_frame_equal(actual[["loan_id", "split"]], expected)
    assert actual["loan_id"].is_unique
    assert actual[export.FEATURE].between(0, 1).all()
    summary = pd.read_csv(root / "reports/nlp/lexical_purpose_summary.csv")
    train = actual.loc[actual["split"].eq("train"), export.FEATURE]
    expected_mean = (
        (summary["n_loans"] * summary["mean_desc_distress_keyword_density"]).sum()
        / summary["n_loans"].sum()
    )
    assert train.mean() == pytest.approx(expected_mean, abs=1e-14)
    assert train.gt(0).sum() == round((summary["n_loans"] * summary["desc_distress_mention_rate"]).sum())
    metadata = json.loads((root / "reports/nlp/desc_distress_export_metrics.json").read_text())
    assert metadata["parquet_sha256"] == sha256_file(path)

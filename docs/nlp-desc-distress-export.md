# Description-only distress density handoff

**Feature version:** `nlp-desc-distress-v1`<br>
**Owner:** Zheng Wei, NLP track<br>
**Requested by:** Evan, via the Week 9 task<br>
**Data / split:** `kaggle-adarshsng-local-2026-09-09` / `split-v1`

The export at `data/nlp/desc_distress_keyword_density.parquet` gives Evan one value
per borrower using the original description. It excludes title and purpose from
the calculation, so a medical category label alone cannot create a keyword hit.

## Contract

| Column | Parquet type | Meaning |
|---|---|---|
| `loan_id` | string, non-null | Unique ID from the frozen cohort |
| `split` | string, non-null | `train` or `validation` |
| `desc_distress_keyword_density` | float64, non-null | Keyword-token count divided by word-token count in cleaned `desc` |

There are exactly three columns and 108,077 rows: 86,293 train rows followed by
21,784 validation rows. Each split follows its frozen ID-file order. No test rows,
labels, description text, title text or purpose values are included. Join on
`loan_id` and `split`; do not join by row position.

The original `train` split includes the tabular track's 59,353 `model_fit` rows and
26,940 `calibration` rows. Those roles remain downstream assignments under
`configs/tabular_development_v1.toml`. The export keeps the shared split labels.
The keyword calculation is deterministic and fits no vocabulary, threshold or
other parameter on either group.

## Calculation and source checks

The script uses `clean_text`, `tokenize_words`, and the existing eleven-word
`DISTRESS_KEYWORDS` set. Cleaning removes known entry wrappers and HTML and
normalizes whitespace and Unicode. Every exact keyword-token occurrence counts,
including repeats; all description word tokens count in the denominator.
Stop words are retained. There is no stemming, substring matching, or negation
handling. For example, `not unemployed` still contains one keyword hit.

An empty, missing, marker-only, or zero-word description receives `0.0`. Zero can
therefore mean either no recognized keyword or no usable words. The three-column
contract does not distinguish those cases. Values are finite and lie in `[0, 1]`.

The loader verifies the source CSV checksum, frozen IDs, split assignments and
one-based CSV record mapping before exporting. It reads only `desc` from the raw
CSV and only ID, split and record-number columns from the manifest. It scans the
source in chunks, discards records outside the requested development IDs, and
never reads outcome columns. The Parquet is read back to check its schema, values
and order before replacing the destination.

## Reproduce and join

From the repository root with the approved raw source available:

```sh
python -m src.nlp.export_desc_distress
```

The command writes the Parquet and `reports/nlp/desc_distress_export_metrics.json`.
The metadata records input hashes, the keyword set, split counts, the output
checksum and the writer details read from the Parquet footer. The committed file
was written with pandas 2.2.2 and Arrow 16.1.0. Rebuilding with the current pins
(pandas 3.0.5 and Arrow 23.0.1) gives identical IDs, values and row order, but a
different file checksum because the serialization and embedded metadata differ.
Compare those contents separately from byte identity when changing writer versions.
Use `--data-dir`, `--output`, `--metadata`, or `--chunksize` to override the defaults.

```python
import pandas as pd

desc_features = pd.read_parquet("data/nlp/desc_distress_keyword_density.parquet")
joined = tabular_rows.merge(
    desc_features,
    on=["loan_id", "split"],
    how="left",
    validate="one_to_one",
)
assert len(joined) == len(tabular_rows)
assert joined["desc_distress_keyword_density"].notna().all()
```

`tabular_rows` must use the shared `train` and `validation` labels. Preserve any
separate development-role column during the join. A missing match is a contract
error; filling it with zero would disguise a failed join as a no-keyword result.

## Evidence and use

The training export has 4,661 descriptions with at least one distress keyword;
validation has 814. The training mean density and mention count agree with the
previous purpose-level EDA aggregates. The tests cover metadata independence,
cleaning, missing descriptions, exact token semantics, frozen ID alignment,
rejection of source mismatches, Parquet types and repeatability across chunk sizes.

Use this column instead of the composite `financial_distress_keyword_density`
when evaluating description-only keyword information. The existing
`text_lexical_features.parquet` is preserved for reproducing earlier experiments.
Its hash-pinned tabular configuration is unchanged. Including the new column in
a production or selected model still needs a versioned feature-contract update
and model evidence. Description-only density can remain correlated with purpose
through the prose; this export removes the direct metadata contribution without
claiming independence or improved accuracy.

The [pipeline limitations log](nlp-pipeline-limitations.md) records Ari's review
points, including the correct percentage denominators and the error-audit limits.

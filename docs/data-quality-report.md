# Data Quality Report

**Document status:** Evidence record for G1 review  
**Version:** 0.1  
**Date:** 16 September 2026  
**Accountable owner:** Evan (Tabular Track Tech Lead)  
**Primary author:** Brandon (Tabular Track Analyst 1)   
**Reviewers:** Tabular Track Lead, Fusion/API Lead and project co-leads  
**Dataset version under review:** `kaggle-adarshsng-local-2026-09-09`  
**Split version under review:** `split-v1`

## 1. Purpose

This report records the evidence for the frozen LendingClub data used by the NLP and tabular tracks. It confirms the eligible cohort, the locked split counts and identifier integrity, and it documents one unresolved provenance discrepancy in the source archive checksum.

Every claim below is reproducible from the repository by running the test suite named in section 6. Numbers stated here are the observed values from that run, not values copied from `split_statistics.json`.

## 2. Artefacts under review

| Artefact | Path | SHA-256 |
|---|---|---|
| Raw source CSV | `data/raw/loan.csv` | `23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c` |
| Split manifest | `data/split_manifest.csv` | `35e3545ee7a8a1221e1936ec22a4ef7951deea0936cc6b645035da2c4486a01f` |
| Split statistics | `data/split_statistics.json` | Read as the locked reference |
| Identifier files | `data/train_ids.csv`, `data/val_ids.csv`, `data/test_ids.csv` | Checked in full, see section 5 |
| Source archive | `archive.zip` (held outside the repository) | `e5de55438920e393443d6212946cd6a0954664e952cdb3683c5607c8beb192f1` |

**Environment:** Python 3.13.9 with pandas 3.0.5, numpy 2.5.3 and pytest 9.1.1 in the project virtual environment. Note that `src/nlp/requirements.txt` records verification under Python 3.14.6; the checks in this report are insensitive to that difference, but the discrepancy is worth aligning before results are published.

## 3. Source version and checksum verification

**Status: partially verified.** The source archive checksum does not match the locked value, but the raw CSV content has been independently cross-verified.

`split_statistics.json` locks the source archive to a SHA-256 of `c6255f6a8099b25303360976597fd8f86ae9087176598692a5a37dd8dc3339a1`. The archive held locally hashes to a different value.

| Property | Value |
|---|---|
| Expected `archive_sha256` | `c6255f6a8099b25303360976597fd8f86ae9087176598692a5a37dd8dc3339a1` |
| Observed archive SHA-256 | `e5de55438920e393443d6212946cd6a0954664e952cdb3683c5607c8beb192f1` |
| Result | **Mismatch** |

The raw CSV itself was independently cross-checked with Evan: its SHA-256 (`23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c`) was compared against his copy of `loan.csv` and confirmed identical.

## 4. Split counts and composition

**Status: verified.** All three locked counts match, in both the identifier files and the manifest.

| Split | Locked count | `*_ids.csv` | `split_manifest.csv` | Defaults | Default rate | Issue-month window |
|---|---:|---:|---:|---:|---:|---|
| Train | 86,293 | 86,293 | 86,293 | 13,197 | 0.152932 | 2007-06 to 2013-07 |
| Validation | 21,784 | 21,784 | 21,784 | 3,298 | 0.151396 | 2013-08 to 2013-12 |
| Test | 14,922 | 14,922 | 14,922 | 2,390 | 0.160166 | 2014-01 to 2014-03 |
| **Total** | **122,999** | **122,999** | **122,999** | 18,885 | 0.153538 | 2007-06 to 2014-03 |

Supporting observations:

- The split totals sum to 122,999, equal to `eligible_rows` in `split_statistics.json`.
- Per-split row counts, default counts and default rates all reconcile with `split_statistics.json` to six decimal places.
- The issue-month windows are contiguous and strictly ordered, consistent with the declared `chronological_by_issue_month` method. No month appears in more than one split, so the splits carry no temporal overlap.
- `invalid_eligible_dates` is 0, and every cohort row parses to a valid issue month.
- `cohort` is `real_text_matured_v1` and `dataset_version` is `kaggle-adarshsng-local-2026-09-09` on all 122,999 rows, with no stray values.

## 5. Identifier integrity

**Status: verified.** No missing, duplicate, overlapping or disagreeing identifiers were found.

| Check | Result | Evidence |
|---|---|---|
| Missing `loan_id` values | None | 0 null or blank values across all three identifier files and the manifest |
| Malformed `loan_id` values | None | All 122,999 identifiers match `lc_NNNNNNNNN` |
| Duplicate `loan_id` within a file | None | Unique counts equal row counts: 86,293, 21,784 and 14,922 |
| Identifiers in more than one split | None | All three pairwise intersections are empty |
| Differences against `split_manifest.csv` | None | Every identifier resolves to a manifest row carrying the matching split label |
| Missing required manifest metadata | None | No nulls in any required manifest column |

Two additional structural checks were run because the identifier scheme depends on them:

- **Identifier convention.** `loan_id` is `lc_` followed by the zero-padded `source_row_number`, a one-based index into the data records of `loan.csv` excluding the header. This holds for all 122,999 manifest rows.
- **Merge integrity.** Joining the manifest to the raw record index passes a pandas `validate="one_to_one"` merge with no unmatched rows. Each requested identifier therefore resolves to exactly one raw record, and no two identifiers claim the same record.

Row-level agreement with the raw source was also confirmed for both frozen label fields. `issue_month` matches `issue_d` for every row, and `target` matches `loan_status` for every row under the mapping `Fully Paid` to 0 and `Charged Off` to 1. No other status value occurs in the cohort.

## 6. Automated verification

The checks in this report are implemented as a pytest suite at `scripts/test_data_integrity.py` so that they can be re-run whenever the data or splits change. From the repository root:

```
python -m pytest scripts/test_data_integrity.py -v -s
```

The suite contains 14 checks and all 14 pass against the artefacts in section 2. The `-s` flag prints each check description; `-v` reports each result individually. It follows the conventions of the existing `src/nlp/test_preprocess.py` suite, and the expensive scans of `loan.csv` are module-scoped fixtures so the raw file is read once per run rather than once per check.

| Check | Report section |
|---|---|
| `test_split_manifest_row_counts_match_ids_files` | 4 |
| `test_ids_files_have_no_missing_or_malformed_values` | 5 |
| `test_ids_files_have_no_internal_duplicates` | 5 |
| `test_ids_files_do_not_overlap` | 5 |
| `test_ids_file_matches_split_manifest_classification` | 5 |
| `test_loan_id_suffix_matches_source_row_number` | 5 |
| `test_each_loan_id_merges_to_exactly_one_raw_row` | 5 |
| `test_target_values_are_binary_with_no_nulls` | 5 |
| `test_issue_month_matches_raw_issue_d` | 5 |
| `test_target_matches_raw_loan_status` | 5 |


## 7. Findings and required follow-up

| ID | Finding | Severity | Required action | Owner |
|---|---|---|---|---|
| DQ-01 | Source archive SHA-256 does not match `split_statistics.json`, and no local file matches the expected value | Medium | Data owner to either re-pin the archive hash against the copy in use or publish a `raw_csv_sha256` for `loan.csv` so provenance can be verified at content level | Data owner |

## 8. Definition-of-done status

| Criterion | Status | Evidence |
|---|---|---|
| The source dataset checksum has been verified | **Not met** | Section 3; mismatch recorded as DQ-01 |
| The 122,999-loan cohort can be reproduced | Met | Section 4; `train_ids.csv`, `val_ids.csv` and `test_ids.csv` sum to 122,999 and agree with `split_manifest.csv`, matching `eligible_rows` in `split_statistics.json` |
| All three split counts match the locked values | Met | Section 4; 86,293, 21,784 and 14,922 confirmed in both sources |
| No missing, duplicate or overlapping IDs are found | Met | Section 5; all identifier checks clean |



## 9. Cohort reproduction and DQ-01 resolution

**Resolution date:** 19 September 2026  
**Status:** Resolved

The original v1 eligibility and split rules are now frozen in
`configs/cohort_v1.toml`, documented in `docs/cohort-definition.md` and implemented
by the read-only `scripts/reproduce_locked_cohort.py` audit.

Two local ZIP packages were examined:

- `loan.csv.zip` has SHA-256
  `c6255f6a8099b25303360976597fd8f86ae9087176598692a5a37dd8dc3339a1`.
- `archive.zip` has SHA-256
  `e5de55438920e393443d6212946cd6a0954664e952cdb3683c5607c8beb192f1`
  because it also contains `LCDataDictionary.xlsx`.

Both archives contain a byte-identical `loan.csv` with SHA-256
`23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c`.
The raw CSV hash is now recorded in `data/split_statistics.json` as the
content-level source identity. DQ-01 is therefore closed; the earlier archive
mismatch reflected packaging, not different loan data.

The reproduction audit scanned 2,260,668 raw rows and produced exactly 122,999
eligible rows: 86,293 train, 21,784 validation and 14,922 test. Comparison against
`data/split_manifest.csv` found zero missing IDs, zero extra IDs and zero mismatches
in source row, split, issue month, target, text availability, cohort or dataset
version. Machine-readable evidence is stored in
`reports/data/cohort_reproduction_v1.json`.

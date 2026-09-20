# LendingClub Cohort Definition v1

**Status:** Accepted and executable  
**Cohort version:** `real_text_matured_v1`  
**Split version:** `split-v1`  
**Dataset version:** `kaggle-adarshsng-local-2026-09-09`  
**Owner:** Tabular and NLP Track Leads  
**Effective date:** 19 September 2026

## 1. Purpose

This document is the authoritative, human-readable definition of the 122,999-loan
LendingClub evidence cohort. The machine-readable contract is
`configs/cohort_v1.toml`; `scripts/reproduce_locked_cohort.py` is its executable
implementation. A change to any rule below requires a new cohort or split version.

The existing `data/split_manifest.csv` remains the canonical row registry. The
reproduction script audits it and never overwrites it.

## 2. Frozen source

| Property | Frozen value |
|---|---|
| Source file | `loan.csv` from the Kaggle Lending Club Loan Data CSV archive |
| Raw data rows | 2,260,668, excluding the header |
| Raw CSV SHA-256 | `23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c` |
| Canonical archive SHA-256 | `c6255f6a8099b25303360976597fd8f86ae9087176598692a5a37dd8dc3339a1` |
| Alternate archive SHA-256 | `e5de55438920e393443d6212946cd6a0954664e952cdb3683c5607c8beb192f1` |

The two observed archives have different ZIP hashes because one also contains
`LCDataDictionary.xlsx`. Both contain a byte-identical `loan.csv` with the raw CSV
hash above. The raw CSV hash is therefore the content-level identity used by the
cohort audit. Raw data is restricted and must not be committed.

## 3. Exact eligibility rule

A source row is eligible if and only if all of these conditions are true:

1. `loan_status` is exactly `Fully Paid` or `Charged Off`.
2. The raw `desc` CSV field contains at least one non-whitespace character.
3. `issue_d` parses exactly with the `%b-%Y` format.
4. The parsed issue month is from June 2007 through March 2014, inclusive.

No other repayment status is eligible. In particular, `Default`, `Current`, late,
grace-period and credit-policy exception statuses are excluded from cohort v1.

### 3.1 Raw description parsing is part of the contract

The cohort builder reads the three audit columns with `keep_default_na=False`.
This preserves source strings such as `None` and `n/a` instead of converting them
to null values before the eligibility check.

Four canonical training rows contain those literal values: two `None` values, one
lowercase `none` value and one `n/a` value. They satisfy the original
raw-non-whitespace rule and have usable
`title` or `purpose` text. Downstream NLP cleaning may correctly treat the literal
description markers as missing; that later cleaning rule does not change cohort
membership. Using pandas' default NA parsing during cohort construction would
incorrectly remove three of these locked IDs and would make eligibility depend on
pandas' evolving default missing-value vocabulary.

## 4. Stable identifier

`source_row_number` is the one-based position of a CSV data record after the
header. Physical line numbers must not be used because quoted descriptions can
span physical lines.

The stable identifier is:

```python
loan_id = f"lc_{source_row_number:09d}"
```

Original LendingClub `id` and `member_id` values are not used because they are
blank in this source.

## 5. Target

| Raw `loan_status` | Target |
|---|---:|
| `Fully Paid` | 0 |
| `Charged Off` | 1 |

The target is the ultimate LendingClub outcome used for this historical evidence
lane. It is not the project's proposed BNPL 30+ DPD product outcome and must not be
described as representative of Singapore BNPL applicants.

## 6. Chronological split

| Split | Inclusive issue-month range | Locked rows |
|---|---|---:|
| Train | June 2007 through July 2013 | 86,293 |
| Validation | August 2013 through December 2013 | 21,784 |
| Test | January 2014 through March 2014 | 14,922 |
| **Total** | June 2007 through March 2014 | **122,999** |

The final test set remains locked for model selection. The audit utility may
mechanically compare its IDs, issue months and labels with the frozen source, but
model-development loaders must not expose test features, labels or performance.

## 7. Required reproduction result

From the repository root, with the restricted raw CSV available locally:

```text
python scripts/reproduce_locked_cohort.py --raw-csv data/raw/loan.csv
```

The command succeeds only when all of the following are true:

- The raw CSV SHA-256 matches the frozen value.
- The raw file contains 2,260,668 data rows.
- The reconstructed cohort contains 122,999 rows.
- The reconstructed and locked `loan_id` sets are identical.
- Every `source_row_number`, split, issue month and target agrees.
- Split counts equal 86,293 / 21,784 / 14,922.
- Cohort and dataset-version fields agree.

To run the exact-match integration test:

```text
set LENDINGCLUB_RAW_CSV=data/raw/loan.csv
python -m unittest scripts/test_cohort_reproduction.py -v
```

On PowerShell, set the variable with:

```powershell
$env:LENDINGCLUB_RAW_CSV = "data/raw/loan.csv"
python -m unittest scripts/test_cohort_reproduction.py -v
```

## 8. Change control

- Do not edit the locked ID files or manifest to make a failed audit pass.
- Investigate and report every missing, extra or mismatched ID.
- Do not alter a v1 rule in place. Propose `cohort_v2` and a new split version.
- Record the reason, evidence, downstream impact and migration plan in the decision log.
- Tabular and NLP leads must approve any cohort-version change before model work uses it.

## 9. Verified evidence

The v1 reproduction was run independently on 19 September 2026 against both local
archives. Their `loan.csv` members were byte-identical. The executable rules
reproduced all 122,999 manifest rows with zero missing IDs, zero extra IDs and zero
field mismatches.

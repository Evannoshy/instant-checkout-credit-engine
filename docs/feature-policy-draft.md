# Tabular Feature Policy — Draft

**Document status:** Draft for Tabular Track Tech Lead review — proposes `tabular_features_v1`  
**Version:** 0.2  
**Date:** 22 September 2026  
**Accountable owner:** Evan (Tabular Track Tech Lead)  
**Primary author:** Brandon (Tabular Analyst 1)  
**Reviewers:** Tabular Track Tech Lead; NLP Track Lead (text-field scope); Project co-leads (population and fairness questions)  

---

## 1. Purpose

This document proposes which structured columns from the LendingClub evidence-lane cohort may be
used as features for tabular baseline modeling, which are prohibited, and which require a
decision from the Tabular Track Tech Lead before use.

It records a per-column classification with rationale, the observed data type and training-set
missingness of each reviewed column, and the proposed feature set `tabular_features_v1` for the
logistic baseline (§7). It does not decide the champion model, the serving schema, or fairness
policy.

All classifications below are proposals until reviewed.

### 1.1 Changes in v0.2

- **Training-only evidence.** v0.1 measured missingness and category levels on the train,
  validation and test split files and cited test-split facts in its reasoning. That used the
  locked test set to inform feature decisions. v0.2 removes every validation and test figure and
  every cross-split comparison; all statistics below are recomputed on the 86,293 training rows
  only (§2.1). The nine-feature recommendation does not change: each of the nine is at or near
  0% missing on training rows, so the selection never depended on test data.
- **`tabular_features_v1`.** §7 now names the nine-feature set as the proposed
  `tabular_features_v1`, defined in `configs/tabular_features_v1.toml` and implemented by
  `src/tabular/preprocess.py`, and states the agreed transformations and exclusions.
- **Correction.** §8 previously counted 38 post-origination/outcome columns in §6.1; the table
  lists 37.

## 2. Scope

- **Cohort:** 122,999-loan evidence-lane cohort `real_text_matured_v1` / `split-v1`
  (train 86,293 / validation 21,784 / test 14,922)
- **Source dictionary:** `LCDataDictionary.xlsx`, `LoanStats` sheet, unless otherwise noted
- **Columns reviewed:** 27 candidate columns, plus the post-origination field families in §6
- **Not reviewed in this pass:** the ~90 additional bureau-detail columns (`num_*`, `mo_sin_*`,
  `sec_app_*`, `open_il_*`, `bc_*`). Deferred to a second pass; not approved by omission, and
  excluded from `tabular_features_v1`.
- **Out of scope:** free-text handling, which belongs to the NLP track; see §6.3

### 2.1 Profiling basis

Data type and missingness were measured on the **training split only** (86,293 rows). Data types
are as loaded from CSV by pandas. Two conventions apply throughout:

- Several bureau counts (`delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `total_acc`) load
  as `float64` despite being integer counts. This is an artifact of the upstream file, not a sign
  of missing values; all five are fully populated in training.
- Validation and test missingness are not reported. Validation is reserved for evaluating models;
  test is locked (§2.3).

### 2.2 Column-name corrections

Three names in the reviewed list did not match the dataset schema and are corrected here:

| As listed | Actual column |
|---|---|
| `verification` | `verification_status` |
| `inqLast6Mths` | `inq_last_6mths` (camelCase form appears only in the `browseNotes` sheet) |
| `intialial_list_status` | `initial_list_status` |

### 2.3 Data-use rules for feature decisions

- **Training statistics only.** Feature selection, missing-value rules, category handling and
  every preprocessing decision use training-set statistics only. Any learned preprocessing step
  (imputation, scaling, encoding, calibration) is fitted on training rows only, inside the model
  pipeline.
- **No test-set evidence.** Test-set missingness, category levels and default rates are not used
  to choose, reject or transform features.
- **Validation is for evaluation.** Validation data measures model performance; it is not used to
  choose features in this document.
- **Test data remains locked.** Per decision D-015, the shared loaders
  (`src/tabular/evaluate.py`, `src/tabular/preprocess.py`, `src/nlp/preprocess.py`) reject
  `split="test"`, and test features, outcomes and performance are unavailable for model
  selection.

## 3. Classification method

Each column was assigned one of three buckets.

- **Proposed** — available at origination, has a stated financial concept, has a definable
  missing-value rule, and is adequately populated in the training split.
- **Discuss** — available at origination, but carries a judgment question: high missingness,
  fairness risk, platform artifact, or redundancy. Each entry states a reason for and against.
- **Prohibited** — unavailable at origination, derived from the target, carries no information, or
  is out of scope for the tabular track. Each entry names the exclusion type.

Exclusion types used in §6:

| Type | Meaning |
|---|---|
| `POST_OUTCOME` | Value exists only because the loan ran after origination |
| `REFRESHED_SOURCE` | Value is updated after origination; the at-origination value is not recoverable |
| `TARGET_DERIVED` | Column is a source field for the label itself |
| `IDENTIFIER` | Identifier or link field with no risk meaning |
| `NO_VARIANCE` | Constant in the training split; carries no information |
| `OUT_OF_SCOPE_TEXT` | High-cardinality free text; not a leakage concern |

## 4. Proposed features

Six of these fifteen are not in `tabular_features_v1` (§7): `emp_length`, `purpose`, `open_acc`,
`pub_rec`, `revol_bal` and `total_acc`. They remain Proposed — usable in a later feature-set
version — not prohibited.

| Column | Description (LC dictionary) | Reason for decision | Data type | Miss % train |
|---|---|---|---|---|
| `loan_amnt` | The listed amount of the loan applied for by the borrower | Core exposure size, fixed at application. | `int64` | 0.00 |
| `term` | Number of payments on the loan, in months | Product term, known at application. Two values only; parse to a category (`36` / `60`). | `string` | 0.00 |
| `emp_length` | Employment length in years, 0 to 10, where 10 is ten or more | Employment stability proxy. Ordinal once parsed; top value censored at `10+ years`. Missing in 3.49% of training rows — low enough for a treat-as-own-category rule. | `string` | 3.49 |
| `home_ownership` | Home ownership status provided by the borrower or obtained from the credit report | Housing-cost and asset proxy. Five values in training, not the dictionary's four: `MORTGAGE`, `RENT`, `OWN`, plus `OTHER` (124 rows) and `NONE` (29 rows). Collapse the two rare levels to `OTHER` before encoding. | `string` | 0.00 |
| `annual_inc` | Self-reported annual income provided by the borrower during registration | Primary affordability input. Severely right-skewed in training (4,000 to 7,141,778); needs a log transform or winsorisation. Self-reported — see `verification_status` in §5. | `float64` | 0.00 |
| `purpose` | A category provided by the borrower for the loan request | Borrower-stated use of funds, fixed at application. 14 levels in training, low enough to one-hot. Levels can appear or disappear over time, so the encoder needs an unseen-category rule. | `string` | 0.00 |
| `dti` | Ratio of the borrower's monthly debt payments, excluding mortgage and the requested LC loan, to self-reported monthly income | Core indebtedness ratio. Two limits: the definition excludes both the mortgage and the requested loan, and observed training values stop at 34.99, which is an LC underwriting cut-off rather than the true distribution. | `float64` | 0.00 |
| `delinq_2yrs` | Number of 30+ days past-due delinquencies in the credit file in the past 2 years | Prior delinquency count from the bureau file. Bounded lookback, so it does not carry indefinite history. | `float64` | 0.00 |
| `earliest_cr_line` | The month the borrower's earliest reported credit line was opened | Credit-file age. Must be derived to tenure in months relative to `issue_d`; the raw `Mon-YYYY` string is not itself the feature. | `string` | 0.00 |
| `inq_last_6mths` | Number of inquiries in the past 6 months, excluding auto and mortgage | Recent credit-seeking intensity, known at application. | `float64` | 0.00 |
| `open_acc` | Number of open credit lines in the borrower's credit file | Active credit relationships at application. | `float64` | 0.00 |
| `pub_rec` | Number of derogatory public records | Severe derogatory history. Overlaps with `pub_rec_bankruptcies`, which was not reviewed in this pass. | `float64` | 0.00 |
| `revol_bal` | Total credit revolving balance | Absolute revolving debt. Interpretable only alongside a limit, which is why `revol_util` is also proposed. | `int64` | 0.00 |
| `revol_util` | Revolving line utilization rate: credit used relative to available revolving credit | Normalised revolving pressure. Already numeric percentage points (0 to 113.9 in training); no percent-sign parsing needed. | `float64` | 0.08 |
| `total_acc` | Total number of credit lines currently in the credit file | Credit-file depth. Related to `open_acc` but not equivalent. | `float64` | 0.00 |

## 5. Discuss features

These are available at origination. They are held for a decision because using them carries a cost
that a validation metric will not show. **All twelve remain excluded from the Week 2 logistic
model (`tabular_features_v1`).** Including any of them requires a Tech Lead decision recorded in
`docs/decisions/decision-log.md` and a new feature-set version.

| Column | Description (LC dictionary) | Reason for | Reason against | Data type | Miss % train |
|---|---|---|---|---|---|
| `verification_status` | Indicates if income was verified by LC, not verified, or if the income source was verified | A data-quality flag on `annual_inc`, not a risk conclusion. Fully populated, three levels. | The three levels are defined by LC's own verification process and have no equivalent in an SG checkout product, so a coefficient learned here will not transfer to serving. | `string` | 0.00 |
| `int_rate` | Interest rate on the loan | Priced before funding, so it passes the timing rule. Numeric percentage points, no parsing needed, fully populated. | LC's own pricing output, not borrower behaviour. If it dominates the lift, the model is mostly decoding LC's scorecard. It also has no serving-time equivalent — in the target system the price is set after the risk decision, so the feature could not be computed at inference. | `float64` | 0.00 |
| `grade` | LC assigned loan grade | Compact and informative. | An underwriting-model output with no meaning outside LC's own assessment. Fully determined by `sub_grade` (its first character). | `string` | 0.00 |
| `sub_grade` | LC assigned loan subgrade | Finer-grained than `grade`. | Same as above. | `string` | 0.00 |
| `installment` | The monthly payment owed by the borrower if the loan originates | Direct affordability quantity; combines with `annual_inc` into a payment-to-income ratio. | Arithmetically derived: the standard annuity formula on `funded_amnt`, `term` and `int_rate` reproduces it to within $1 for 99.96% of training rows. It adds no information beyond `loan_amnt`, `term` and `int_rate`. | `float64` | 0.00 |
| `zip_code` | First 3 numbers of the zip code provided by the borrower | Geography carries genuine risk signal and supports subgroup analysis. | Established proxy-discrimination risk, and 846 levels in training is too high-cardinality to one-hot safely. | `string` | 0.00 |
| `addr_state` | The state provided by the borrower | Coarser than `zip_code`; supports geographic robustness checks. | Same proxy concern at lower resolution. Same recommendation. | `string` | 0.00 |
| `initial_list_status` | The initial listing status of the loan: W or F | Trivially encodable, fully populated. | A platform mechanic (whole vs fractional funding) with no borrower-risk meaning; `w` is 8.3% of training rows. Any predictive power would reflect LC's listing policy, which can change over time, rather than borrower risk. | `string` | 0.00 |
| `funded_amnt` | The total amount committed to that loan at that point in time | Captures partial-funding cases; it differs from `loan_amnt` in 1.7% of training rows. | "At that point in time" means the value accumulates during the listing period, after the application. `loan_amnt` covers the same concept without the timing ambiguity. | `int64` | 0.00 |
| `funded_amnt_inv` | The total amount committed by investors at that point in time | Same as above. | Same objection, more strongly: it equals `loan_amnt` in only 68.7% of training rows, so it largely records investor demand during listing. | `float64` | 0.00 |
| `mths_since_last_delinq` | Number of months since the borrower's last delinquency | Recency complements the count in `delinq_2yrs`. Missing is structural, not unknown: every null row has `delinq_2yrs == 0`, so a null means no delinquency on file. | Missing in 61% of training rows. Any imputation constant would define the majority of the column. | `float64` | 61.28 |
| `mths_since_last_record` | Number of months since the last public record | Same structure; pairs with `pub_rec`. Every null row has `pub_rec == 0`. | Missing in 93% of training rows. At that rate the column is effectively a rare-event indicator, and `pub_rec` already carries that with full coverage. | `float64` | 92.86 |

## 6. Prohibited features

### 6.1 Post-origination and outcome fields

Every column below records what happened to the loan after money changed hands. None was available
at origination.

| Column | Description (LC dictionary) | Exclusion type |
|---|---|---|
| `loan_status` | Current status of the loan | `TARGET_DERIVED` — source field for the label |
| `out_prncp` | Remaining outstanding principal for total amount funded | `POST_OUTCOME` |
| `out_prncp_inv` | Remaining outstanding principal for the investor-funded portion | `POST_OUTCOME` |
| `total_pymnt` | Payments received to date for total amount funded | `POST_OUTCOME` |
| `total_pymnt_inv` | Payments received to date for the investor-funded portion | `POST_OUTCOME` |
| `total_rec_prncp` | Principal received to date | `POST_OUTCOME` |
| `total_rec_int` | Interest received to date | `POST_OUTCOME` |
| `total_rec_late_fee` | Late fees received to date | `POST_OUTCOME` — late fees imply delinquency |
| `recoveries` | Post charge-off gross recovery | `POST_OUTCOME` — non-zero only after charge-off |
| `collection_recovery_fee` | Post charge-off collection fee | `POST_OUTCOME` |
| `last_pymnt_d` | Last month payment was received | `POST_OUTCOME` |
| `last_pymnt_amnt` | Last total payment amount received | `POST_OUTCOME` |
| `next_pymnt_d` | Next scheduled payment date | `POST_OUTCOME` — exists only for a live loan |
| `last_credit_pull_d` | The most recent month LC pulled credit for this loan | `REFRESHED_SOURCE` — updated through the life of the loan |
| `pymnt_plan` | Indicates if a payment plan has been put in place | `POST_OUTCOME` — a payment plan follows repayment difficulty |
| `hardship_flag`, `hardship_type`, `hardship_reason`, `hardship_status`, `hardship_amount`, `hardship_start_date`, `hardship_end_date`, `hardship_length`, `hardship_dpd`, `hardship_loan_status`, `hardship_payoff_balance_amount`, `hardship_last_payment_amount`, `deferral_term`, `payment_plan_start_date`, `orig_projected_additional_accrued_interest` | Hardship-plan fields, populated only where the borrower entered a hardship arrangement | `POST_OUTCOME` |
| `debt_settlement_flag`, `debt_settlement_flag_date`, `settlement_status`, `settlement_date`, `settlement_amount`, `settlement_percentage`, `settlement_term` | Debt-settlement fields; the dictionary defines the flag as applying to a borrower "who has charged-off" | `POST_OUTCOME` — the definition states the loan has already charged off |

### 6.2 Identifiers, constants and link fields

| Column | Description (LC dictionary) | Exclusion type |
|---|---|---|
| `id` | A unique LC assigned ID for the loan listing | `IDENTIFIER` — and 100% null in training |
| `member_id` | A unique LC assigned ID for the borrower member | `IDENTIFIER` — 100% null in training, so it cannot support grouped splitting either |
| `url` | URL for the LC page with listing data | `IDENTIFIER` — 100% null in training |
| `loan_id` | Project-assigned identifier; not a LendingClub field | `IDENTIFIER` — join key to the split manifest; unique per row |
| `issue_d` | The month which the loan was funded | Timestamp, not a feature. Required for splitting and for deriving credit-file tenure. |
| `policy_code` | publicly available policy_code=1; new products not publicly available policy_code=2 | `NO_VARIANCE` — constant at 1 in training. Excluded for lack of information, not for leakage; the dictionary gives no further detail on what distinguishes the two codes. |
| `application_type` | Indicates whether the loan is an individual or joint application | `NO_VARIANCE` — constant at `Individual` in training, which also empties `annual_inc_joint`, `dti_joint` and `verification_status_joint`. |

### 6.3 Out-of-scope text fields

Excluded for a different reason than the rows above. These are available at decision time and carry
no leakage. They are unusable here because they are high-cardinality free text.

| Column | Description (LC dictionary) | Exclusion type |
|---|---|---|
| `emp_title` | The job title supplied by the borrower when applying | `OUT_OF_SCOPE_TEXT` — 59,658 unstandardised values in training |
| `title` | The loan title provided by the borrower | `OUT_OF_SCOPE_TEXT` — 31,629 borrower-typed values in training, largely redundant with `purpose` |
| `desc` | Loan description provided by the borrower | `OUT_OF_SCOPE_TEXT` — free-text narrative; the natural input for the text track |

The synthetic descriptions in `data/synthetic_descriptions_v2.csv` are also excluded from the
tabular track. They are systems-lane data (D-013) and must not back any real-world performance
claim.

## 7. Proposed feature set: `tabular_features_v1`

The nine features below are the proposed `tabular_features_v1` for the Week 2 logistic
regression baseline. The set is defined executably in `configs/tabular_features_v1.toml` and
loaded by `src/tabular/preprocess.py` (`load_tabular_split`), which rejects any configuration
that uses a prohibited column.

Selection rules applied, in order:

1. One feature per financial concept; drop near-duplicates, because correlated inputs split the
   coefficient between them and destabilise both.
2. No column from §5 pending a decision.
3. Missingness under 10% on training rows.
4. Every feature explainable to a borrower in plain language.

| # | Source column | Model feature | Concept covered | Transformation |
|---|---|---|---|---|
| 1 | `loan_amnt` | `loan_amnt` | Exposure size | Numeric |
| 2 | `term` | `term` | Product term | `" 36 months"` → `"36"`, `" 60 months"` → `"60"`; categorical |
| 3 | `annual_inc` | `annual_inc_log` | Repayment capacity | `log(1 + annual_inc)` |
| 4 | `dti` | `dti` | Existing debt burden relative to income | Numeric |
| 5 | `revol_util` | `revol_util` | Revolving credit pressure | Numeric |
| 6 | `delinq_2yrs` | `delinq_2yrs` | Recent repayment failure | Numeric |
| 7 | `inq_last_6mths` | `inq_last_6mths` | Credit-seeking intensity | Numeric |
| 8 | `earliest_cr_line` | `credit_history_months` | Credit-file maturity | Whole months from `earliest_cr_line` to `issue_d` |
| 9 | `home_ownership` | `home_ownership` | Housing cost and asset proxy | `NONE` / `OTHER` / `ANY` → `OTHER`; categorical |

Preprocessing rules:

- Every transformation above is fixed and row-wise; none uses statistics from any split.
  `ANY` does not occur in training; it appears in later LendingClub loans outside the cohort
  window and is mapped defensively.
- Missing values are preserved as missing and no loan is dropped for a missing feature. Median
  imputation, scaling and one-hot encoding are not part of preprocessing; the model pipeline
  learns them from training rows only.
- Unexpected values (non-numeric text, an unknown `term` or `home_ownership` level, an
  unparseable date) stop the load with an error instead of becoming missing.

Excluded from `tabular_features_v1`:

- All twelve §5 Discuss features, including `grade`, `sub_grade`, `int_rate`, `installment`,
  `verification_status`, `zip_code` and `addr_state`.
- The six Proposed features not selected (§4).
- The ~90 unreviewed bureau-detail columns (§2).
- Every post-origination, outcome and identifier field (§6.1, §6.2).
- Borrower descriptions and all other free text, and the synthetic descriptions (§6.3).

## 8. Definition of done

| Requirement | Status | Evidence |
|---|---|---|
| ~20–30 potential baseline features reviewed | Met | 27 candidate columns carried through to a Proposed/Discuss decision (§4: 15, §5: 12) |
| Table of ~20–30 potential baseline features | Met | §4 + §5 tables, 27 rows combined |
| Data type and missingness recorded per reviewed feature | Met | Measured on the training split only; populated in §4 and §5 for every row (§2.1) |
| Obvious post-loan and outcome-related leakage fields identified and prohibited | Met | §6.1: 37 post-origination/outcome columns across 17 entries, each tagged `POST_OUTCOME`, `REFRESHED_SOURCE` or `TARGET_DERIVED` |
| Every reviewed feature has a preliminary decision and explanation | Met | Every row in §4–§6 carries a decision bucket and a stated reason |
| Nine-feature list stated as the proposed `tabular_features_v1` | Met | §7; `configs/tabular_features_v1.toml` |
| Feature and preprocessing decisions use training-set statistics only | Met | §2.1, §2.3; validation and test figures removed in v0.2 (§1.1) |
| Test-set missingness and default rates not used to choose features | Met | §2.3; no test statistic remains in §4–§7 |
| The 12 Discuss features excluded from the Week 2 model | Met | §5, §7 |
| The ~90 unreviewed bureau columns excluded | Met | §2, §7 |
| Test data remains locked | Met | §2.3; `load_tabular_split` rejects `split="test"`, covered by `src/tabular/test_preprocess.py` |
| No raw LendingClub data committed to GitHub | Met | `data/raw/` is gitignored; only `split_manifest.csv`, `*_ids.csv` and `synthetic_descriptions_v2.csv` are tracked, none of which carry raw LC feature columns |
| Tech Lead approval of `tabular_features_v1` | Not met | Pending Evan's review. On approval, the document status changes to "Approved for logistic baseline v1" and the decision is recorded in `docs/decisions/decision-log.md` |

Not yet closed — carried forward as decisions for the Tabular Track Tech Lead, not blockers to
`tabular_features_v1`:

- Whether `int_rate` (and by extension `grade`, `sub_grade`, `installment`) enters a later
  feature-set version, or stays excluded per §5/§7.
- Audit-only treatment for `zip_code` and `addr_state`.
- The missing-value rule for `mths_since_last_delinq` and `mths_since_last_record`.
- Whether `verification_status` is usable given it has no serving-time equivalent outside LC.
- Review of the ~90 deferred bureau-detail columns (`num_*`, `mo_sin_*`, `sec_app_*`, `open_il_*`,
  `bc_*`) not covered in this pass.

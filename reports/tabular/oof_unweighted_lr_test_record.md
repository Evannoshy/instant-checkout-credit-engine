# Test Record: OOF Scores and Unweighted LR Experiment

**Date run:** 15 October 2026 · **Code commit tested:** `75b1e95` (branch `model/tabular-oof-unweighted-lr`)
**Author:** Allan
**Covers:** `src/tabular/export_oof_scores.py`, `src/tabular/logistic_unweighted.py`,
`src/tabular/unweighted_comparison.py`, `src/tabular/unweighted_report.py`, `src/tabular/provenance.py`,
and the `calibrate.py` loader refactor.

## 1. Environment

| Item | Value |
|---|---|
| Machine | macOS 26.6.2, arm64 (Darwin) |
| Python | 3.13.9 |
| Packages | pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, pyarrow 23.0.1, xgboost 3.4.1 (all equal to `src/tabular/requirements.txt`) |
| Raw CSV | `data/raw/loan.csv`, SHA-256 `23783ef320e4df24ac113d6e5b830edb909912b7783d49b89aacd5690dc9120c`, equal to `raw_csv_sha256` in `data/split_statistics.json` (checked before both runs, and by every runner before it reads data) |

## 2. Results

| Run | Command | Passed | Failed | Skipped |
|---|---|---:|---:|---:|
| A. Without the raw CSV | `python -m pytest src/tabular -q -rs` | 334 | 0 | 9 |
| B. Real data | `LENDINGCLUB_RAW_CSV=data/raw/loan.csv CALIBRATION_RECOMPUTE=1 python -m pytest src/tabular -q -rfs` | 342 | 1 | 0 |

**Run A skips (9).** Each needs the raw CSV or the opt-in calibration rerun, and every one of them ran in
run B:
- `LENDINGCLUB_RAW_CSV` not set (8): `test_export_oof_scores.py:247`, `test_logistic_unweighted.py:229`,
  `test_logistic_baseline.py:335`, `test_raw_predictions.py:129`, `test_tabular_development.py:303`,
  `test_tabular_development.py:562`, `test_tabular_preprocess.py:558`, `test_xgboost_baseline.py:283`.
- `CALIBRATION_RECOMPUTE` not set (1): `test_tabular_calibrate.py:902`.

**Run B failure (1).** `test_raw_predictions.py::test_real_candidate_refit_reproduces_committed_predictions`
raised `ValueError: Search selection used a different environment`. This existing test refits the
frozen v1 shortlist and requires the exact environment recorded in `reports/tabular/xgboost_selection.json`
(Windows, AMD64, Python 3.13.15). This machine matches every package pin but runs Python 3.13.9 on macOS,
so the guard refuses by design. The test is unchanged on this branch and fails the same way on `main` on this
machine. It does not involve the new code, and the new experiments write no v1 file.

## 3. New tests (all passed in run B)

| File | Tests | What it proves |
|---|---:|---|
| `test_provenance.py` | 6 | Raw CSV accepted only when its SHA-256 equals the locked benchmark; checksums keyed by repository path |
| `test_export_oof_scores.py` | 23 | Per OOF row: ID absent from its fold's fitting IDs and all fitting months earlier, on synthetic folds **and on the 35,234 committed rows with the real folds**. Also: exactly 35,234 unique rows, correct fold IDs, finite probabilities in [0, 1], no calibration/validation/test IDs, a fresh pipeline per fold whose scaler learns only its fold's rows, and a refit that reproduces the committed parquet |
| `test_logistic_unweighted.py` | 11 | `class_weight=None` reaches the fitted classifier; only `class_weight` differs from v1; one fit on exactly the `model_fit` rows; the test split is never requested; v1 config and prediction files unchanged; 26,940 calibration and 21,784 validation loans covered; a refit reproduces the committed files |
| `test_unweighted_comparison.py` | 19 | All four candidates graded on the same 21,784 validation IDs; Platt fitted on balanced calibration rows only and never applied to the unweighted model; reference rows reproduce the merged results; verdict rule; report wording follows the verdicts; committed JSON and Markdown rebuild exactly |

The `calibrate.py` refactor is covered by the existing `test_tabular_calibrate.py`: 97 passed in run B,
including the full `CALIBRATION_RECOMPUTE` rerun against the committed calibration results.

## 4. Test labels

Neither pipeline can read test labels. `evaluate.load_manifest` drops test rows before returning, the loaders
reject `split="test"` before opening any file, and every manifest records `test_rows_available_to_model: 0`.
Only one test reads `data/test_ids.csv`, and it reads loan IDs only, to prove that none appear in the OOF
export.

## 5. Windows review and portability checks (17 October 2026)

The full suite was rerun on Windows AMD64, Python 3.13.15, with the package pins
above. Code tested: `d3accde`. The checkout used `core.autocrlf=true`, and
`LENDINGCLUB_RAW_CSV` pointed to the verified CSV outside the repository.
`CALIBRATION_RECOMPUTE=1` was enabled.

**Result: 366 passed, 0 failed, 0 skipped, in 473.93 seconds.** This includes the
environment-locked refit that could not run on the author's Mac, both new
real-data refits, the full calibration rerun and the comparison-report rebuild.

The follow-up adds 23 checksum tests. Verification now accepts LF/CRLF conversion
in UTF-8 CSV, JSON and TOML files, but still rejects changed content, missing or
extra inputs and incorrect current hashes. The raw CSV must match the locked
byte checksum even when stored elsewhere. Binary files, including Parquet, keep
strict byte checks.

Runners still record actual file-byte hashes. A regenerated report keeps its own
input hash table; the reproduction test verifies those hashes separately before
comparing the report's results and wording. Prediction comparisons retain their
original numerical tolerances. No model settings, split files, feature contracts,
committed predictions or reference calibration results changed in this follow-up.

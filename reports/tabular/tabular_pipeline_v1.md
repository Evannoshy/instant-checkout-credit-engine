# Tabular pipeline handoff

The saved model is the balanced logistic-regression baseline fitted on the
59,353 `model_fit` loans. It supplies the raw tabular score used by the fusion
stacker. The separate Platt mapping supplies the calibrated tabular fallback.

## Files and rebuild command

- Pipeline: `artifacts/tabular/tabular_logistic_v1_model_fit.joblib`.
- Manifest: `reports/tabular/tabular_pipeline_v1_manifest.json`.
- Feature rules: `configs/tabular_features_v1.toml`.

The pipeline is gitignored. Use a copy shared by its owner with the matching
manifest, or rebuild it locally. From the repository root, using Python 3.13:

```powershell
python -m pip install -r src/tabular/requirements.txt
python -m src.tabular.export_tabular_pipeline --raw-csv data/raw/loan.csv
```

Use the checksum-verified LendingClub CSV. The command fits only on `model_fit`,
checks all 48,724 committed calibration and validation scores, checks the Platt
mapping, and saves the pipeline and its manifest. A local rebuild may have a
different artifact checksum; use the manifest written by that same rebuild.
Do not commit the raw CSV or joblib file.

## Build the model input

The saved pipeline accepts these nine built columns, in this order:

`loan_amnt, annual_inc_log, dti, revol_util, delinq_2yrs, inq_last_6mths,
credit_history_months, term, home_ownership`.

It does not accept raw borrower fields. First run the row-wise transformations
from `src/tabular/preprocess.py` through `_build_features`. This is the same
Stage A code used during training. Supply LendingClub-format strings, including
`issue_d` as the decision month in `Mon-YYYY` format. For a historical loan, use
its recorded issue month. For a new demo request, use its decision month, not
the current date when replaying an older request.

The example below scores one fictional historical borrower. It reads no labels
or test data.

```python
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.tabular import preprocess, provenance

manifest = json.loads(
    Path("reports/tabular/tabular_pipeline_v1_manifest.json").read_text(encoding="utf-8")
)
artifact = manifest["artifact"]
provenance.verify_file_checksum(artifact["path"], artifact["sha256"])
pipeline = joblib.load(artifact["path"])

rows = pd.DataFrame([{
    "loan_amnt": "10000", "annual_inc": "60000", "dti": "15.2",
    "revol_util": "35.0", "delinq_2yrs": "0", "inq_last_6mths": "1",
    "term": "36 months", "home_ownership": "RENT",
    "earliest_cr_line": "Jan-2005", "issue_d": "Nov-2013",
}], dtype="string")
config = preprocess.load_feature_config()
features = preprocess._build_features(
    rows, config, pd.Index(["demo_borrower"], name="loan_id")
)
raw_prob = float(pipeline.predict_proba(features)[0, 1])

platt = manifest["reference_calibrator"]
clip = platt["probability_clip"]
fallback_pd = float(expit(
    platt["slope"] * logit(np.clip(raw_prob, clip, 1 - clip)) + platt["intercept"]
))
```

Verify the checksum before loading joblib, which can execute code. Use the
scikit-learn and joblib versions recorded in the manifest. Missing feature
values are imputed inside the fitted pipeline. Stage A rejects unsupported
categories and malformed numeric or date values; follow its existing rules
rather than adding a second set of transformations in the app.
Pass `revol_util` in percentage points without a percent sign, as in the source
CSV: `"35.0"` means 35%, not 0.35%.

## Pass the correct score downstream

- Fusion input: `{"tabular": raw_prob}` plus the required raw NLP scores.
- Missing-text fallback: `fallback_pd`, tagged as `tabular_calibration_v1`.
- Never pass `fallback_pd` as the stacker's tabular input. That would calibrate
  the tabular score twice and differ from the research pipeline.

The fallback mapping was fitted on all 26,940 January to July 2013 loans. It
must not be used as a held-out feature when fitting the stacker on those loans.
The model and its metrics describe the historical LendingClub cohort. The
single-borrower example is a demo, not evidence of present-day credit accuracy.

The app still needs to connect Stage A, the saved pipeline, the text scorers
and the fusion predictor. Evan and Ariel must confirm the missing-text rule
and the model variant before that integration is released.

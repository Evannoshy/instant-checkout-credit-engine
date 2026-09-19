"""Tests for the frozen cohort contract.

The small-fixture tests run without the restricted raw dataset. Set
``LENDINGCLUB_RAW_CSV`` to enable the exact 2.26-million-row integration test.
Optional ``COHORT_MANIFEST`` and ``COHORT_STATISTICS`` variables override the
repository defaults for independent local verification.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from reproduce_locked_cohort import (  # noqa: E402
    load_config,
    reconstruct_cohort,
    reproduce,
)


CONFIG_PATH = REPO_ROOT / "configs" / "cohort_v1.toml"


class CohortRuleTests(unittest.TestCase):
    def test_exact_eligibility_boundaries_and_raw_description_rule(self) -> None:
        rows = pd.DataFrame(
            [
                {"issue_d": "Jun-2007", "loan_status": "Fully Paid", "desc": " purpose "},
                {"issue_d": "Mar-2014", "loan_status": "Charged Off", "desc": "n/a"},
                {"issue_d": "Jan-2010", "loan_status": "Default", "desc": "text"},
                {"issue_d": "Jan-2010", "loan_status": "Fully Paid", "desc": "   "},
                {"issue_d": "Apr-2014", "loan_status": "Fully Paid", "desc": "text"},
                {"issue_d": "Jan-2010", "loan_status": "Current", "desc": "text"},
                {"issue_d": "not-a-date", "loan_status": "Charged Off", "desc": "text"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            raw_csv = Path(directory) / "loan.csv"
            rows.to_csv(raw_csv, index=False)
            reconstructed, diagnostics = reconstruct_cohort(
                raw_csv,
                load_config(CONFIG_PATH),
                chunksize=2,
            )

        self.assertEqual(reconstructed["loan_id"].tolist(), ["lc_000000001", "lc_000000002"])
        self.assertEqual(reconstructed["split"].tolist(), ["train", "test"])
        self.assertEqual(reconstructed["target"].tolist(), [0, 1])
        self.assertEqual(reconstructed["issue_month"].tolist(), ["2007-06", "2014-03"])
        self.assertEqual(diagnostics["raw_rows"], 7)
        self.assertEqual(diagnostics["eligible_rows"], 2)
        self.assertEqual(diagnostics["literal_missing_marker_descriptions"], 1)

    def test_source_row_number_is_one_based_across_chunks(self) -> None:
        rows = pd.DataFrame(
            [
                {"issue_d": "Jan-2010", "loan_status": "Current", "desc": "skip"},
                {"issue_d": "Jan-2010", "loan_status": "Fully Paid", "desc": "keep"},
                {"issue_d": "Jan-2010", "loan_status": "Charged Off", "desc": "keep"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            raw_csv = Path(directory) / "loan.csv"
            rows.to_csv(raw_csv, index=False)
            reconstructed, _ = reconstruct_cohort(
                raw_csv,
                load_config(CONFIG_PATH),
                chunksize=1,
            )

        self.assertEqual(reconstructed["source_row_number"].tolist(), [2, 3])
        self.assertEqual(reconstructed["loan_id"].tolist(), ["lc_000000002", "lc_000000003"])


@unittest.skipUnless(
    os.environ.get("LENDINGCLUB_RAW_CSV"),
    "Set LENDINGCLUB_RAW_CSV to run the restricted-data integration test",
)
class RealDataIntegrationTest(unittest.TestCase):
    def test_raw_source_reproduces_locked_manifest_exactly(self) -> None:
        raw_csv = Path(os.environ["LENDINGCLUB_RAW_CSV"])
        manifest = Path(os.environ.get("COHORT_MANIFEST", REPO_ROOT / "data" / "split_manifest.csv"))
        statistics = Path(
            os.environ.get("COHORT_STATISTICS", REPO_ROOT / "data" / "split_statistics.json")
        )
        config = Path(os.environ.get("COHORT_CONFIG", CONFIG_PATH))
        report = reproduce(
            raw_csv=raw_csv,
            manifest_path=manifest,
            statistics_path=statistics,
            config_path=config,
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["missing_id_count"], 0)
        self.assertEqual(report["extra_id_count"], 0)
        self.assertFalse(any(report["field_mismatch_counts"].values()))


if __name__ == "__main__":
    unittest.main()

"""Offline smoke test for the DistilBERT latency profiler on a tiny model."""

import pandas as pd

from src.nlp.profile_distilbert_latency import profile
from src.nlp.test_export_distilbert_scores import saved_model, tiny  # noqa: F401 (pytest fixtures)


def test_profile_reports_load_single_loan_and_batch_timings(saved_model):
    """Load time, single-loan percentiles and batch throughput are all measured and sensible."""
    loans = pd.DataFrame({
        "title": ["Car loan"] * 6,
        "purpose": ["car", "debt_consolidation"] * 3,
        "desc": ["Missed payments " * n for n in range(6)],
    })

    report = profile(loans, model=saved_model, single=4, warmup=2, batch=6)

    assert report["load_seconds"] > 0
    single = report["single_loan_ms"]
    assert single["loans"] == 4 and 0 < single["p50"] <= single["p95"]
    assert report["batch"]["loans"] == 6 and report["batch"]["loans_per_second"] > 0

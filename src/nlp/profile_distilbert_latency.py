"""Profile DistilBERT v2 inference: model-loading time, single-loan latency and batch throughput.

Run from the repository root, once on the GPU node and once on CPU (CUDA_VISIBLE_DEVICES=""):
    python -m src.nlp.profile_distilbert_latency [--model DIR_OR_HUB_REPO] [--revision SHA]

Every timing starts from raw title, purpose and desc, so preprocessing and tokenisation are included.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers

from src.nlp.export_distilbert_scores import MODEL_DIR_V2, load_model, score, score_borrower_text
from src.nlp.preprocess import build_text_payload, load_original_split
from src.nlp.train_distilbert_prototype import DATA_DIR, DEVICE_NAME, RANDOM_STATE, REPORT_DIR, precision

FIELDS = ["title", "purpose", "desc"]
SINGLE_LOANS = 200
WARMUP_CALLS = 10
BATCH_LOANS = 2_000


def profile(
    loans: pd.DataFrame,
    *,
    model: str | Path,
    revision: str | None = None,
    single: int = SINGLE_LOANS,
    warmup: int = WARMUP_CALLS,
    batch: int = BATCH_LOANS,
) -> dict[str, Any]:
    """Time a cold model load, single-loan score_borrower_text calls and batched score() on raw loan text."""
    source = str(model)
    records = list(loans[FIELDS].itertuples(index=False, name=None))

    load_model.cache_clear()
    start = time.perf_counter()
    loaded, tokenizer = load_model(source, revision)
    load_seconds = time.perf_counter() - start

    for row in records[:warmup]:
        score_borrower_text(*row, model=source, revision=revision)
    latencies_ms = []
    for row in records[:single]:
        start = time.perf_counter()
        score_borrower_text(*row, model=source, revision=revision)
        latencies_ms.append((time.perf_counter() - start) * 1_000)

    batch_size = 128 if loaded.device.type == "cuda" else 32
    rows = records[:batch]
    start = time.perf_counter()
    score(loaded, tokenizer, [build_text_payload(dict(zip(FIELDS, row))) for row in rows], batch_size=batch_size)
    batch_seconds = time.perf_counter() - start

    return {
        "device": DEVICE_NAME,
        "precision": precision(),
        "torch_threads": torch.get_num_threads(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "model": {"source": source, "revision": revision},
        "load_seconds": load_seconds,
        "single_loan_ms": {
            "loans": len(latencies_ms),
            "warmup_calls": warmup,
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "mean": float(np.mean(latencies_ms)),
        },
        "batch": {
            "loans": len(rows),
            "batch_size": batch_size,
            "seconds": batch_seconds,
            "loans_per_second": len(rows) / batch_seconds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=str(MODEL_DIR_V2), help="Local model dir or private Hub repo id")
    parser.add_argument("--revision", help="Hub commit hash when --model is a Hub repo")
    args = parser.parse_args()

    loans = load_original_split(DATA_DIR, "validation").sample(
        n=max(SINGLE_LOANS, BATCH_LOANS), random_state=RANDOM_STATE
    ).reset_index(drop=True)
    report = profile(loans, model=args.model, revision=args.revision)

    path = REPORT_DIR / f"distilbert_latency_{'cuda' if torch.cuda.is_available() else 'cpu'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    single = report["single_loan_ms"]
    print(
        f"{report['device']} ({report['precision']}): load {report['load_seconds']:.2f} s | single loan "
        f"p50 {single['p50']:.1f} ms, p95 {single['p95']:.1f} ms | batch "
        f"{report['batch']['loans_per_second']:.0f} loans/s. Saved {path}"
    )


if __name__ == "__main__":
    main()

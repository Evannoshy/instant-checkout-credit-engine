"""Measure sequential loopback HTTP requests through the complete scoring path."""
import argparse
import json
import platform
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from src.serving.api import make_server
from src.serving.prepare_bundle import DEFAULT_BUNDLE
from src.serving.score_application import ApplicationScorer, TEXT_POLICIES, VARIANTS
from src.tabular.preprocess import REPO_ROOT


def benchmark(scorer, requests=200, warmup=10):
    if requests < 1 or warmup < 0:
        raise ValueError("Use at least one request and a nonnegative warmup")
    body = (REPO_ROOT / "data/fixtures/demo_application.json").read_bytes()
    server = make_server(scorer, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    latencies, processing = [], []
    try:
        for index in range(warmup + requests):
            start = time.perf_counter()
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/score", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.load(response)
            elapsed = (time.perf_counter() - start) * 1000
            if index >= warmup:
                latencies.append(elapsed)
                processing.append(result["timing_ms"]["total"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return {
        "variant": scorer.variant, "text_policy": scorer.text_policy,
        "scope": "sequential loopback HTTP: parsing, validation, features, models, fusion and response",
        "requests": requests, "warmup": warmup, "concurrency": 1,
        "http_ms": dict(zip(("p50", "p95", "p99"), map(float, np.percentile(latencies, [50, 95, 99])))),
        "mean_model_processing_ms": float(np.mean(processing)),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor()},
        "limitations": "One fictional payload on this machine. Not a deployment SLA, concurrency or GPU benchmark.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--text-policy", choices=TEXT_POLICIES, required=True)
    parser.add_argument("--distilbert-dir", type=Path)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    scorer = ApplicationScorer.load(args.bundle, variant=args.variant,
                                    text_policy=args.text_policy, distilbert_dir=args.distilbert_dir)
    result = {"model_load_ms": (time.perf_counter() - start) * 1000,
              **benchmark(scorer, args.requests, args.warmup)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

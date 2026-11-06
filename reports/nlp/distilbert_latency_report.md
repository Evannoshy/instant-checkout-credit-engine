# DistilBERT v2 latency report

Model: `nlp-distilbert-fusion-v2` (Hub revision `699f04cfdb6849fb17f560628a31008f50e074bf`).
Measured by `src/nlp/profile_distilbert_latency.py` on the SoC cluster node `xgpe1`.

Every timing starts from raw title, purpose and description, so preprocessing and tokenisation are included.

| Device | Model load | Single loan p50 / p95 | Batch throughput |
|---|---:|---:|---:|
| GPU, NVIDIA TITAN RTX, fp16 | 0.47 s | 6.0 ms / 6.7 ms | 2,702 loans/s |
| CPU, fp32 (same node, CUDA hidden) | 0.45 s | 96.6 ms / 169.9 ms | 12 loans/s |

- **Single loan:** 200 calls after 10 warm-up calls.
- **Batch:** 2,000 loans in one `score()` pass. Batch size is 128 on GPU and 32 on CPU.
- **Raw results:** `reports/nlp/distilbert_latency_cuda.json` and `reports/nlp/distilbert_latency_cpu.json`.

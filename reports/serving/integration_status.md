# Local integration check, 13 November 2026

## Verified path

The local form and HTTP API accept raw borrower fields and call the existing
Stage A transformer, balanced tabular LR, saved tabular-only stacker and Platt
mapping. Empty text follows the explicitly selected rule and uses the approved
tabular fallback. No approval threshold is applied.

The bundle preparation command rebuilt the tabular pipeline and reproduced all
48,724 existing calibration and validation scores. The serving parity check
then scored 50 real validation loans from raw fields and matched the committed
tabular-only fusion probabilities within absolute tolerance 1e-10.

The relevant serving, fusion, text-preprocessing and provenance checks passed:
183 passed, 1 skipped. The skipped check is live two-score parity, which needs
the original TF-IDF artifact. The browser form was checked with a fictional
borrower, followed by a blank-text request that returned the tabular fallback.

## Model artifacts still needed

An attempted rebuild of the unweighted TF-IDF model used the recorded winning
settings and the original population loader and pipeline builder. It fitted
59,353 base-fitting loans and scored the 48,724 shared fusion-role loans.
Its maximum absolute difference from the committed v2 probabilities was
0.07809525133798267. This is not acceptable inference parity. Its outputs were
not substituted into the fusion model and no committed score file was changed.

The original pipeline is identified by SHA-256
`db4ba92f7bbd71c1b91098667993bbfc1e7b8975306ed0d229b665d717c92177`
in `reports/nlp/tfidf_scores_v2_manifest.json`. Obtain that file from the NLP
owner and investigate the reproduction difference before accepting a rebuild.
The cause is not established by this check.

Vincent's private DistilBERT v2 weights are also absent. The adapter verifies
the recorded weights, config and tokenizer hashes, but its live path has not
been exercised here. Both text variants refuse startup without the required
verified artifacts. They do not silently run as tabular-only.

Offline fusion is separate from these artifact gaps: its experiment still
reproduces from the committed, hash-pinned score exports.

## Local benchmark

`local_cpu_benchmark.json` records 200 sequential loopback HTTP requests after
10 warm-up calls, using one fictional payload and the tabular-only preview.
Recorded HTTP p50/p95/p99 were approximately 40.9/54.4/64.4 ms on the machine
listed in that file. This includes request parsing, feature transformation,
model inference, fusion and JSON serialization. Model loading is separate.
The recorded model-load time measures artifact loading after Python imports;
it does not include starting Python or importing the dependencies.

This is a single-machine, single-concurrency CPU measurement. It is not evidence
for the complete three-model route, production load, external network latency,
GPU serving or a service-level guarantee.

## Remaining release decisions

The demo variant and text rule are explicit preview settings. The final variant,
missing-text rule, hardware and evaluation gate still require the co-leads'
agreement. The final test was not evaluated. See `docs/final-evaluation-plan.md`.

"""Rebuild the tabular model and optionally package the original verified TF-IDF artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import sklearn

from src.tabular import provenance
from src.tabular.export_tabular_pipeline import run_export
from src.tabular.preprocess import REPO_ROOT

DEFAULT_BUNDLE = REPO_ROOT / "artifacts" / "serving" / "demo_v1"
BUNDLE_VERSION = "checkout-demo-bundle-v1"


def prepare(raw_csv: Path, output: Path = DEFAULT_BUNDLE, tfidf_artifact: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    print("Rebuilding and checking the tabular pipeline...", flush=True)
    tabular = run_export(
        REPO_ROOT / "data", raw_csv, output / "tabular.joblib", output / "tabular_manifest.json"
    )
    owner_path = REPO_ROOT / "reports/nlp/tfidf_scores_v2_manifest.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if tfidf_artifact is not None:
        provenance.verify_file_checksum(tfidf_artifact, owner["pipeline_artifact"]["sha256"])
        shutil.copyfile(tfidf_artifact, output / "tfidf.joblib")
    stackers = output / "stackers"
    stackers.mkdir(exist_ok=True)
    for name in ("tabular_only", "tabular_tfidf", "tabular_tfidf_distilbert"):
        shutil.copyfile(REPO_ROOT / f"data/fusion/stacker_v1/{name}.json", stackers / f"{name}.json")
    files = [output / "tabular.joblib", *sorted(stackers.glob("*.json"))]
    if tfidf_artifact is not None:
        files.append(output / "tfidf.joblib")
    pinned_sources = [
        "configs/fusion_stacker_v1.toml", "configs/tabular_features_v1.toml",
        "reports/tabular/calibration_metrics.json", "reports/nlp/tfidf_scores_v2_manifest.json",
        "reports/nlp/distilbert_scores_v2_manifest.json",
    ]
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "research_only": True,
        "available_tracks": ["tabular", "tfidf"] if tfidf_artifact is not None else ["tabular"],
        "environment": {"scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "files": {p.relative_to(output).as_posix(): provenance.sha256_file(p) for p in files},
        "source_files": {p: provenance.sha256_file(REPO_ROOT / p) for p in pinned_sources},
        "models": {"tabular": tabular["model_version"], "tfidf": owner["model_version"],
                   "distilbert": "nlp-distilbert-fusion-v2"},
        "reproduction": {"tabular_rows": 48_724, "tfidf_original_artifact_verified": tfidf_artifact is not None},
        "distilbert": "Not bundled. Supply the private v2 checkpoint separately; its recorded hashes are checked.",
        "test_evaluated": False,
    }
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified local bundle: {output}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--tfidf-artifact", type=Path, help="Yati's original pipeline; its published hash must match")
    args = parser.parse_args()
    prepare(args.raw_csv, args.output, args.tfidf_artifact)

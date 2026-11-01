"""Export DistilBERT v2 scores for the fusion stacker, and score one borrower's text.

Run from the repository root on the GPU node, after --stage fusion of train_distilbert_prototype:
    python -m src.nlp.export_distilbert_scores [--model-dir DIR] [--hub-repo REPO --hub-revision SHA]

distilbert_fusion_v2 trained and chose its checkpoint on base_fit loans only, so one model scores all
48,724 fusion_fit, fusion_calibration and validation loans (fusion task sheet). raw_margin is the
default logit minus the non-default logit; raw_prob = sigmoid(raw_margin) is uncalibrated.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import roc_auc_score
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.nlp.dataset import MAX_LENGTH
from src.nlp.features_lexical import tokenize_words
from src.nlp.preprocess import PREPROCESS_VERSION, build_text_payload, clean_text, load_original_split, sha256_file
from src.nlp.train_distilbert_prototype import (
    DATA_DIR,
    FUSION_NAME,
    FUSION_POOL_ROWS,
    FUSION_SELECTION_MONTHS,
    FUSION_TRAIN_MONTHS,
    MODEL_DIR,
    MODEL_NAME,
    REPORT_DIR,
    precision,
)

MODEL_DIR_V2 = MODEL_DIR.with_name(FUSION_NAME)
MODEL_VERSION = "nlp-distilbert-fusion-v2"
OUTPUT = DATA_DIR / "nlp" / "distilbert_scores_v2.parquet"
MANIFEST = REPORT_DIR / "distilbert_scores_v2_manifest.json"
COLUMNS = ["loan_id", "split", "role", "raw_margin", "raw_prob", "model_version"]
ROLE_MONTHS = {
    "fusion_fit": ("2013-01", "2013-04"),
    "fusion_calibration": ("2013-05", "2013-07"),
    "validation": ("2013-08", "2013-12"),
}
ROLE_ROWS = {"fusion_fit": 15_587, "fusion_calibration": 11_353, "validation": 21_784}
ROLE_SPLIT = {"fusion_fit": "train", "fusion_calibration": "train", "validation": "validation"}
INFERENCE_BATCH_SIZE = 128  # 8x training: no gradients or optimizer state to hold


@functools.cache
def load_model(
    source: str, revision: str | None = None
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    """A saved model and its tokenizer from a local dir or Hub repo, loaded once per process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(source, revision=revision).to(device)
    return model, DistilBertTokenizerFast.from_pretrained(source, revision=revision)


@torch.inference_mode()
def score(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    texts: Sequence[str],
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """(raw_margin, raw_prob) for each text, in input order."""
    model.eval()
    device = model.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision())
    texts = list(texts)
    order = torch.from_numpy(np.argsort([len(t) for t in texts], kind="stable"))  # similar lengths cut padding
    logits = torch.empty(len(texts), 2, dtype=torch.float64)
    for rows in order.split(batch_size):
        batch = tokenizer(
            [texts[i] for i in rows.tolist()], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        )
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits[rows] = model(**batch.to(device)).logits.cpu().double()
    return (logits[:, 1] - logits[:, 0]).numpy(), logits.softmax(-1)[:, 1].numpy()


def score_borrower_text(
    title: str | None,
    purpose: str | None,
    desc: str | None,
    *,
    model: str | Path = MODEL_DIR_V2,
    revision: str | None = None,
) -> dict[str, Any]:
    """Score one borrower with the export's preprocessing and model; no score when the text is unusable.

    text_usable: the cleaned title + purpose + description payload is non-empty (the confirmed definition).
    desc_usable: the description has at least one word, the rule behind desc_distress density 0.0.
    """
    payload = build_text_payload({"title": title, "purpose": purpose, "desc": desc})
    result = {
        "text_usable": bool(payload),
        "desc_usable": bool(tokenize_words(clean_text(desc))),
        "raw_margin": None,
        "raw_prob": None,
        "model_version": MODEL_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
    }
    if payload:
        margin, prob = score(*load_model(str(model), revision), [payload], batch_size=1)
        result.update(raw_margin=float(margin[0]), raw_prob=float(prob[0]))
    return result


def development_rows() -> pd.DataFrame:
    """The loans to score, in frozen train-then-validation ID order, with their fusion role; base_fit is dropped."""
    rows = pd.concat([load_original_split(DATA_DIR, split) for split in ("train", "validation")], ignore_index=True)
    rows["role"] = pd.Series(pd.NA, index=rows.index, dtype="string")
    for role, (first, last) in ROLE_MONTHS.items():
        rows.loc[rows["issue_month"].between(first, last), "role"] = role
    return rows[rows["role"].notna()].reset_index(drop=True)


def check_export(frame: pd.DataFrame, model_ids: set[str]) -> None:
    """Raise unless frame meets the shared six-column contract and scores no loan the model trained or selected on."""
    if frame.columns.tolist() != COLUMNS:
        raise ValueError(f"Columns {frame.columns.tolist()} differ from {COLUMNS}")
    counts = frame["role"].value_counts().to_dict()
    checks = {
        f"role counts {counts} differ from {ROLE_ROWS}": counts != ROLE_ROWS,
        "split does not match role": not frame["split"].eq(frame["role"].map(ROLE_SPLIT)).all(),
        "duplicate loan_id": frame["loan_id"].duplicated().any(),
        "non-finite score": not np.isfinite(frame[["raw_margin", "raw_prob"]].to_numpy(dtype=float)).all(),
        "raw_prob outside [0, 1]": not frame["raw_prob"].between(0, 1).all(),
        "scores a loan the model trained or selected on": frame["loan_id"].isin(model_ids).any(),
    }
    if failures := [message for message, failed in checks.items() if failed]:
        raise ValueError("Export check failed: " + "; ".join(failures))


def id_set(ids: Sequence[str], months: tuple[str, str]) -> dict[str, Any]:
    """An ID list with its month rule, count and SHA-256 (over the sorted IDs, newline-joined)."""
    ids = sorted(ids)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return {"months": list(months), "count": len(ids), "sha256": digest, "ids": ids}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR_V2)
    parser.add_argument("--hub-repo", help="Private Hub repo holding the uploaded model, recorded in the manifest")
    parser.add_argument("--hub-revision", help="Hub commit hash of that upload")
    args = parser.parse_args()

    record = json.loads((args.model_dir / "metrics.json").read_text(encoding="utf-8"))
    used = record["sample_loan_ids"]
    if (len(used["train"]), len(used["validation"])) != FUSION_POOL_ROWS:
        raise ValueError(f"{args.model_dir} was not trained and selected on the full fusion pools {FUSION_POOL_ROWS}")
    model, tokenizer = load_model(str(args.model_dir))
    print(f"Device: {model.device} ({precision()})")

    rows = development_rows()
    margin, prob = score(model, tokenizer, rows["text_payload"].tolist())
    frame = pd.DataFrame({
        "loan_id": rows["loan_id"],
        "split": rows["split"],
        "role": rows["role"],
        "raw_margin": margin,
        "raw_prob": prob,
        "model_version": MODEL_VERSION,
    }).astype({"loan_id": "string", "split": "string", "role": "string", "model_version": "string"})
    check_export(frame, {*used["train"], *used["validation"]})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(OUTPUT), frame)
    print(f"{len(frame)} rows -> {OUTPUT}")
    for role, group in rows.groupby("role", sort=False):
        print(f"  {role}: {len(group)} loans, ROC-AUC {roc_auc_score(group['target'], prob[group.index]):.4f}")

    manifest = {
        "model_version": MODEL_VERSION,
        "model_name": MODEL_NAME,
        "preprocess_version": PREPROCESS_VERSION,
        "hub": {"repo": args.hub_repo, "revision": args.hub_revision},
        "training_ids": id_set(used["train"], FUSION_TRAIN_MONTHS),
        "selection_ids": id_set(used["validation"], FUSION_SELECTION_MONTHS),
        "roles": {role: {"months": list(ROLE_MONTHS[role]), "count": n} for role, n in ROLE_ROWS.items()},
        "settings": record["config"],
        "environment": {
            "training": record["environment"],
            "export": {"torch": torch.__version__, "transformers": transformers.__version__,
                       "device": str(model.device), "precision": precision()},
        },
        "checksums": {
            **{f"{args.model_dir.name}/{p.name}": sha256_file(p) for p in sorted(args.model_dir.iterdir()) if p.is_file()},
            OUTPUT.name: sha256_file(OUTPUT),
            "split_manifest.csv": sha256_file(DATA_DIR / "split_manifest.csv"),
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()

"""Single-borrower inference using the existing feature, text and fusion code."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
import sklearn

from src.fusion.predict import load_predictor, predict
from src.nlp.baseline_tfidf import strip_field_labels
from src.nlp.preprocess import build_text_payload, clean_text
from src.tabular import preprocess, provenance

VARIANTS = ("tabular_only", "tabular_tfidf", "tabular_tfidf_distilbert")
TEXT_POLICIES = ("description", "payload")
NUMERIC_FIELDS = ("loan_amnt", "annual_inc", "dti", "revol_util", "delinq_2yrs", "inq_last_6mths")
TABULAR_FIELDS = (*NUMERIC_FIELDS, "term", "home_ownership", "earliest_cr_line")
TEXT_FIELDS = ("title", "purpose", "desc")
MAX_TEXT_CHARS = 12_000


def validate_application(request: Any) -> dict:
    if not isinstance(request, dict) or set(request) - {"application_id", "decision_month", "tabular", "text"}:
        raise ValueError("Expected application_id, decision_month, tabular and optional text")
    identifier = request.get("application_id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier):
        raise ValueError("application_id must contain 1 to 80 letters, digits, underscores or hyphens")
    month = request.get("decision_month")
    if not isinstance(month, str) or not re.fullmatch(r"[A-Z][a-z]{2}-[0-9]{4}", month):
        raise ValueError("decision_month must use Mon-YYYY, for example Nov-2013")
    pd.to_datetime(month, format="%b-%Y", errors="raise")
    fields = request.get("tabular")
    if not isinstance(fields, dict) or set(fields) != set(TABULAR_FIELDS):
        raise ValueError("tabular must contain exactly: " + ", ".join(TABULAR_FIELDS))
    fields = dict(fields)
    for name, value in fields.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int, float))):
            raise ValueError(f"{name} must be a string, number or null")
        if name in NUMERIC_FIELDS and value not in (None, ""):
            number = float(value)
            if not math.isfinite(number) or number < 0 or (name == "loan_amnt" and number == 0):
                raise ValueError(f"{name} must be finite and nonnegative; loan_amnt must be positive")
            if name in ("delinq_2yrs", "inq_last_6mths") and not number.is_integer():
                raise ValueError(f"{name} must be a whole number")
        fields[name] = None if value in (None, "") else str(value)
    if fields["loan_amnt"] is None:
        raise ValueError("loan_amnt is required")
    text = request.get("text")
    if text is None:
        text = {}
    if not isinstance(text, dict) or set(text) - set(TEXT_FIELDS):
        raise ValueError("text may contain only title, purpose and desc")
    text = {name: text.get(name) for name in TEXT_FIELDS}
    if any(value is not None and not isinstance(value, str) for value in text.values()):
        raise ValueError("Text fields must be strings or null")
    if sum(len(value or "") for value in text.values()) > MAX_TEXT_CHARS:
        raise ValueError(f"Combined text must not exceed {MAX_TEXT_CHARS} characters")
    return {"application_id": identifier, "decision_month": month, "tabular": fields, "text": text}


class ApplicationScorer:
    def __init__(self, tabular, tfidf, predictor, *, variant: str, text_policy: str,
                 feature_config: dict, versions: dict, distilbert: Callable | None = None):
        if variant not in VARIANTS or text_policy not in TEXT_POLICIES:
            raise ValueError("Choose an explicit supported variant and text policy")
        if variant != "tabular_only" and tfidf is None:
            raise ValueError("This variant requires a verified TF-IDF pipeline")
        if variant == "tabular_tfidf_distilbert" and distilbert is None:
            raise ValueError("This variant requires the verified private DistilBERT checkpoint")
        self.tabular, self.tfidf, self.predictor = tabular, tfidf, predictor
        self.variant, self.text_policy = variant, text_policy
        self.feature_config, self.versions, self.distilbert = feature_config, versions, distilbert

    @classmethod
    def load(cls, bundle_dir: Path, *, variant: str, text_policy: str, distilbert_dir: Path | None = None):
        bundle_dir = Path(bundle_dir).resolve()
        record = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
        if record["bundle_version"] != "checkout-demo-bundle-v1":
            raise ValueError("Unsupported bundle version")
        if record["environment"] != {"scikit_learn": sklearn.__version__, "joblib": joblib.__version__}:
            raise ValueError("Use the scikit-learn and joblib versions recorded in the bundle")
        available = record.get("available_tracks", [])
        if available not in (["tabular"], ["tabular", "tfidf"]):
            raise ValueError("Bundle must declare its verified available tracks")
        if variant != "tabular_only" and "tfidf" not in available:
            raise ValueError("TF-IDF is unavailable: supply the original checksum-verified v2 pipeline")
        required_files = {"tabular.joblib", *{f"stackers/{name}.json" for name in VARIANTS}}
        if "tfidf" in available:
            required_files.add("tfidf.joblib")
        if set(record["files"]) != required_files:
            raise ValueError("Bundle must pin every available pipeline and all three stacker files")
        required_sources = {"configs/fusion_stacker_v1.toml", "configs/tabular_features_v1.toml",
                            "reports/tabular/calibration_metrics.json", "reports/nlp/tfidf_scores_v2_manifest.json",
                            "reports/nlp/distilbert_scores_v2_manifest.json"}
        if set(record["source_files"]) != required_sources:
            raise ValueError("Bundle source pins are incomplete")
        expected_versions = {"tabular": "tabular-logistic-v1-model-fit", "tfidf": "nlp-tfidf-fusion-v2",
                             "distilbert": "nlp-distilbert-fusion-v2"}
        if record["models"] != expected_versions:
            raise ValueError("Bundle model versions differ from the fusion inputs")
        for name, expected in record["files"].items():
            path = (bundle_dir / name).resolve()
            if not path.is_relative_to(bundle_dir):
                raise ValueError("Bundle file must stay inside its directory")
            provenance.verify_file_checksum(path, expected)
        for name, expected in record["source_files"].items():
            provenance.verify_file_checksum(preprocess.REPO_ROOT / name, expected, allow_crlf=True)
        # All artifact hashes are checked before either pickle is opened.
        tabular = joblib.load(bundle_dir / "tabular.joblib")
        tfidf = joblib.load(bundle_dir / "tfidf.joblib") if variant != "tabular_only" else None
        transformer = None
        if variant == "tabular_tfidf_distilbert":
            if distilbert_dir is None:
                raise ValueError("Supply --distilbert-dir for the three-score variant")
            owner = json.loads((preprocess.REPO_ROOT / "reports/nlp/distilbert_scores_v2_manifest.json").read_text())
            checks = {Path(name).name: digest for name, digest in owner["checksums"].items()
                      if "/" in name}
            for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
                if name not in checks:
                    raise ValueError(f"Checkpoint manifest is missing {name}")
                provenance.verify_file_checksum(distilbert_dir / name, checks[name])
            from src.nlp.export_distilbert_scores import load_model, score_borrower_text
            load_model(str(distilbert_dir))
            transformer = lambda text: score_borrower_text(**text, model=distilbert_dir)["raw_prob"]
        return cls(tabular, tfidf, load_predictor(variant, artifact_dir=bundle_dir / "stackers"),
                   variant=variant, text_policy=text_policy, feature_config=preprocess.load_feature_config(),
                   versions=record["models"], distilbert=transformer)

    def score(self, request: dict) -> dict:
        start = time.perf_counter()
        data = validate_application(request)
        fields = {**data["tabular"], "issue_d": data["decision_month"], "loan_id": data["application_id"]}
        rows = pd.DataFrame([fields], dtype="string")
        features = preprocess._build_features(
            rows, self.feature_config, pd.Index([data["application_id"]], name="loan_id")
        )
        if features["credit_history_months"].lt(0).any():
            raise ValueError("earliest_cr_line cannot be later than decision_month")
        after_features = time.perf_counter()
        scores = {"tabular": float(self.tabular.predict_proba(features)[0, 1])}
        after_tabular = time.perf_counter()
        payload = build_text_payload(data["text"])
        usable = bool(clean_text(data["text"]["desc"])) if self.text_policy == "description" else bool(payload)
        if usable and "tfidf" in self.predictor.tracks:
            scores["tfidf"] = float(self.tfidf.predict_proba([strip_field_labels(payload)])[0, 1])
        after_tfidf = time.perf_counter()
        if usable and "distilbert" in self.predictor.tracks:
            scores["distilbert"] = self.distilbert(data["text"])
        after_distilbert = time.perf_counter()
        result = predict(self.predictor, scores, text_usable=usable)
        end = time.perf_counter()
        return {
            "application_id": data["application_id"], **result,
            "research_only": True, "variant": self.variant, "text_policy": self.text_policy,
            "probability_semantics": "calibrated historical LendingClub outcome; not an approval decision",
            "base_scores": scores, "base_model_versions": {name: self.versions[name] for name in scores},
            "quality_flags": [f"missing_{name}" for name, value in data["tabular"].items() if value is None],
            "timing_ms": {
                "features": (after_features - start) * 1000,
                "tabular": (after_tabular - after_features) * 1000,
                "tfidf_and_text": (after_tfidf - after_tabular) * 1000,
                "distilbert": (after_distilbert - after_tfidf) * 1000,
                "fusion": (end - after_distilbert) * 1000, "total": (end - start) * 1000,
            },
        }

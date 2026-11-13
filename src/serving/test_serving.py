"""Inference, failure handling and optional real-data parity. Final test remains locked."""
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fusion.predict import FusionPredictor, PlattMapping
from src.serving.api import make_server
from src.serving.score_application import ApplicationScorer, TABULAR_FIELDS
from src.tabular import evaluate, preprocess
from src.tabular import provenance


class ConstantModel:
    def __init__(self, probability):
        self.probability, self.calls = probability, 0

    def predict_proba(self, rows):
        self.calls += 1
        return np.tile([1 - self.probability, self.probability], (len(rows), 1))


@pytest.fixture
def sample():
    return json.loads((preprocess.REPO_ROOT / "data/fixtures/demo_application.json").read_text())


@pytest.fixture
def scorer():
    identity = PlattMapping(1.0, 0.0, 1e-6)
    predictor = FusionPredictor("fixture", "tabular_tfidf", ("tabular", "tfidf"), (2.0, 1.0), -2.0,
                                identity, identity, "fixture-fallback")
    return ApplicationScorer(ConstantModel(0.3), ConstantModel(0.2), predictor,
                              variant="tabular_tfidf", text_policy="payload",
                              feature_config=preprocess.load_feature_config(),
                              versions={"tabular": "fixture", "tfidf": "fixture"})


def test_score_uses_raw_probabilities_and_existing_stacker(scorer, sample):
    result = scorer.score(sample)
    assert result["p_default"] == pytest.approx(1 / (1 + np.exp(1.2)))
    assert result["base_scores"] == {"tabular": 0.3, "tfidf": 0.2}
    assert result["research_only"] and not result["fallback"]
    assert not {"decision", "approved", "target"} & result.keys()


def test_empty_text_calls_no_text_model(scorer, sample):
    sample["text"] = None
    result = scorer.score(sample)
    assert result["fallback"] and result["p_default"] == pytest.approx(0.3)
    assert scorer.tfidf.calls == 0


@pytest.mark.parametrize("policy,fallback", [("description", True), ("payload", False)])
def test_text_policy_is_explicit(scorer, sample, policy, fallback):
    scorer.text_policy = policy
    sample["text"]["desc"] = None
    assert scorer.score(sample)["fallback"] is fallback
    assert scorer.tfidf.calls == int(not fallback)


@pytest.mark.parametrize("edit", [
    lambda x: x.update(target=1),
    lambda x: x["tabular"].update(loan_status="Charged Off"),
    lambda x: x["tabular"].update(loan_amnt=-1),
    lambda x: x["tabular"].update(loan_amnt=None),
    lambda x: x["tabular"].update(dti=float("nan")),
    lambda x: x["tabular"].update(revol_util=True),
    lambda x: x["tabular"].update(inq_last_6mths=1.5),
    lambda x: x["tabular"].update(home_ownership="UNKNOWN"),
    lambda x: x["tabular"].update(earliest_cr_line="Jan-2020"),
    lambda x: x.update(decision_month="2013-11"),
    lambda x: x.update(text=[]),
    lambda x: x.update(text={"desc": 42}),
    lambda x: x.update(text={"desc": "x" * 12001}),
])
def test_invalid_requests_are_rejected(scorer, sample, edit):
    edit(sample)
    with pytest.raises((ValueError, TypeError)):
        scorer.score(sample)


def test_missing_optional_numeric_values_use_training_imputation(scorer, sample):
    sample["tabular"]["annual_inc"] = None
    assert "missing_annual_inc" in scorer.score(sample)["quality_flags"]


def test_missing_transformer_cannot_silently_change_the_variant(scorer):
    with pytest.raises(ValueError, match="DistilBERT"):
        ApplicationScorer(scorer.tabular, scorer.tfidf, scorer.predictor,
                          variant="tabular_tfidf_distilbert", text_policy="payload",
                          feature_config=scorer.feature_config, versions=scorer.versions)


def test_missing_bundle_pins_stop_loading_before_joblib(tmp_path, monkeypatch):
    import joblib
    import sklearn
    record = {"bundle_version": "checkout-demo-bundle-v1", "available_tracks": ["tabular"], "environment": {
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}, "files": {}}
    (tmp_path / "bundle.json").write_text(json.dumps(record))
    monkeypatch.setattr(joblib, "load", lambda p: pytest.fail("An unchecked pickle was loaded"))
    with pytest.raises(ValueError, match="pin every available pipeline"):
        ApplicationScorer.load(tmp_path, variant="tabular_only", text_policy="payload")


def test_modified_artifact_is_rejected_before_joblib(tmp_path, monkeypatch):
    import joblib
    import sklearn
    files = ["tabular.joblib", *[f"stackers/{variant}.json" for variant in
                                  ("tabular_only", "tabular_tfidf", "tabular_tfidf_distilbert")]]
    digests = {}
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture only; never deserialized")
        digests[name] = provenance.sha256_file(path)
    sources = ["configs/fusion_stacker_v1.toml", "configs/tabular_features_v1.toml",
               "reports/tabular/calibration_metrics.json", "reports/nlp/tfidf_scores_v2_manifest.json",
               "reports/nlp/distilbert_scores_v2_manifest.json"]
    record = {"bundle_version": "checkout-demo-bundle-v1", "available_tracks": ["tabular"],
              "environment": {"scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
              "files": digests, "source_files": {name: "unused" for name in sources},
              "models": {"tabular": "tabular-logistic-v1-model-fit", "tfidf": "nlp-tfidf-fusion-v2",
                         "distilbert": "nlp-distilbert-fusion-v2"}}
    (tmp_path / "bundle.json").write_text(json.dumps(record))
    (tmp_path / "tabular.joblib").write_bytes(b"changed after pinning")
    monkeypatch.setattr(joblib, "load", lambda p: pytest.fail("An unchecked pickle was loaded"))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        ApplicationScorer.load(tmp_path, variant="tabular_only", text_policy="payload")


@pytest.fixture
def endpoint(scorer):
    server = make_server(scorer, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_score_and_health(endpoint, sample):
    with urllib.request.urlopen(endpoint + "/api/health") as response:
        assert json.load(response)["ready"]
    request = urllib.request.Request(endpoint + "/api/score", data=json.dumps(sample).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        assert json.load(response)["research_only"]


@pytest.mark.parametrize("body,content_type,status", [
    (b"{}", "text/plain", 415), (b"not json", "application/json", 400),
    (b'{"x":NaN}', "application/json", 400), (b"x" * 32769, "application/json", 413),
], ids=["wrong-media-type", "invalid-json", "nonfinite-json", "oversized-body"])
def test_http_rejects_invalid_bodies(endpoint, body, content_type, status):
    request = urllib.request.Request(endpoint + "/api/score", data=body, headers={"Content-Type": content_type})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == status


def test_failed_model_does_not_return_a_probability(endpoint, scorer, sample, monkeypatch):
    monkeypatch.setattr(scorer.tabular, "predict_proba", lambda _: (_ for _ in ()).throw(RuntimeError("private path")))
    request = urllib.request.Request(endpoint + "/api/score", data=json.dumps(sample).encode(),
                                     headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    body = error.value.read().decode()
    assert error.value.code == 503 and "private path" not in body and "p_default" not in body


@pytest.mark.skipif(not (os.environ.get("SERVING_BUNDLE") and os.environ.get("LENDINGCLUB_RAW_CSV")),
                    reason="Needs local verified bundle and restricted raw CSV")
@pytest.mark.parametrize("variant", ["tabular_only", "tabular_tfidf"])
def test_real_borrowers_reproduce_committed_fusion_predictions(variant):
    record = json.loads((Path(os.environ["SERVING_BUNDLE"]) / "bundle.json").read_text())
    if variant != "tabular_only" and "tfidf" not in record["available_tracks"]:
        pytest.skip("Original verified TF-IDF artifact is not available")
    scorer = ApplicationScorer.load(Path(os.environ["SERVING_BUNDLE"]),
                                     variant=variant, text_policy="payload")
    manifest, _ = evaluate.load_manifest(preprocess.REPO_ROOT / "data")
    selected = manifest.loc[manifest["split"].eq("validation")].sample(50, random_state=42)
    columns = [*scorer.feature_config["source"]["features"], "issue_d", "title", "purpose", "desc"]
    records, count = [], 0
    for chunk in pd.read_csv(os.environ["LENDINGCLUB_RAW_CSV"], usecols=columns, dtype="string",
                             keep_default_na=False, chunksize=200_000):
        chunk["source_row_number"] = range(count + 1, count + len(chunk) + 1)
        count += len(chunk)
        records.append(chunk.loc[chunk["source_row_number"].isin(selected["source_row_number"])])
    rows = selected[["loan_id", "source_row_number"]].merge(pd.concat(records), on="source_row_number", validate="one_to_one")
    expected = pd.read_parquet(preprocess.REPO_ROOT / "data/fusion/stacker_v1/predictions_validation.parquet").set_index("loan_id")
    for row in rows.to_dict("records"):
        request = {"application_id": row["loan_id"], "decision_month": row["issue_d"],
                   "tabular": {name: row[name] for name in TABULAR_FIELDS},
                   "text": {name: row[name] for name in ("title", "purpose", "desc")}}
        assert scorer.score(request)["p_default"] == pytest.approx(expected.loc[row["loan_id"], f"p_{variant}_platt"], abs=1e-10)

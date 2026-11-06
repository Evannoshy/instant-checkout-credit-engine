"""Analyst 2: unweighted TF-IDF scores for the fusion experiment (fusion_development_v1).

Run from the repository root:
    python -m src.nlp.export_scores_v2

Produces ``data/nlp/tfidf_scores_v2.parquet`` for Allan's late-fusion stacker:
one row for each of the 48,724 loans whose fusion role is not ``base_fit``,
carrying both the decision margin and the probability from the same fitted
model.

**Role mapping is read, not imported.** ``src/tabular/fusion_roles.py`` states
that the CSV is the cross-track contract and other tracks read the file rather
than importing that module. ``data/fusion/fusion_roles_v1.csv`` is therefore
loaded directly and its SHA-256 verified against the pin in
``configs/nlp_tfidf_fusion_v2.toml`` before anything is fitted.

**Everything is fitted on base_fit only.** Vocabulary, IDF weights,
hyperparameter selection and the final coefficients all come from the 59,353
loans issued 2007-06 to 2012-12. The three scoring roles are transformed and
scored, never fitted on. Settings are chosen with expanding-window
time-ordered folds reused from ``src.tabular.xgboost_baseline.time_aware_folds``
under the same ``n_folds``/``min_train_share`` the tabular track uses, so both
tracks select under comparable conditions. This replaces the random folds used
for ``nlp-tfidf-tuned-v1``, whose optimism was recorded as NT-02 in
``docs/nlp-tuning-report.md``.

**The probability is unweighted, not calibrated.** ``class_weight=None`` gives a
probability whose level already matches the base rate, which
``reports/nlp/baseline_tfidf_interpretability.md`` (NI-01) measured as better
than every post-hoc calibration route tried. It is nevertheless a *raw* model
output: nothing here is a post-hoc calibrator and the export must not be
described as calibrated.

**Vocabulary reach is measured, not assumed.** base_fit ends in December 2012
while every scored loan is from 2013, and borrower descriptions shortened
sharply over the cohort period. The share of each scoring role's tokens that
the base_fit vocabulary actually covers is recorded in the manifest so the
fusion track can see how much text signal reaches it.

**Why a separate module rather than an extension of evaluate_baseline.py.**
That runner produces the v1 out-of-fold export: it fits on all 86,293 training
loans with random stratified folds, and its schema and behaviour are pinned by
33 tests and consumed through the shipped v1 parquets. This export fits on
base_fit only, selects under expanding-window time-ordered folds, and writes a
different schema under a different role contract. Folding both into one module
would put two incompatible fold schemes and two export schemas behind flags,
where a change meant for one could break the other. The pipeline itself is
genuinely reused rather than copied: ``build_pipeline`` and
``load_baseline_split`` are imported from the v1 modules unchanged.

**Naming.** The config file is named for the model version it produces
(``nlp-tfidf-fusion-v2``), matching the DistilBERT sibling. Note that the older
``configs/nlp_tfidf_v2.toml`` produces ``nlp-tfidf-tuned-v1``: that config's
counter and its model version were never aligned. The model version, not the
config filename, is the identifier recorded in every export and manifest.

``src/nlp/preprocess.py`` and the v1 exports are untouched. The locked test
split is never requested: it is absent from the role mapping by construction
and the shared loader refuses it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import tomllib
from sklearn.metrics import average_precision_score

from src.nlp.baseline_tfidf import load_baseline_split
from src.nlp.evaluate_baseline import build_pipeline
from src.tabular.evaluate import evaluate_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "nlp_tfidf_fusion_v2.toml"
DEFAULT_DATA_DIR = REPO_ROOT / "data"

EXPORT_COLUMNS = ("loan_id", "split", "role", "raw_margin", "raw_prob", "model_version")
REQUIRED_CONFIG_TABLES = ("roles", "expected", "model", "grid", "cv", "comparison", "output")
CLASS_WEIGHT_NONE = "none"


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the v3 configuration, failing loudly on gaps."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if missing := [table for table in REQUIRED_CONFIG_TABLES if table not in config]:
        raise ValueError(f"Config is missing tables: {missing}")
    for key in ("model_version", "model_name", "preprocess_version"):
        if not config.get(key):
            raise ValueError(f"Config is missing {key}")
    if config["model"]["class_weight"] != CLASS_WEIGHT_NONE:
        raise ValueError(
            "The fusion export must be unweighted; [model] class_weight must be 'none'"
        )
    return config


def resolve_class_weight(value: str | None) -> str | None:
    """Map the config's ``"none"`` sentinel to scikit-learn's ``None``.

    TOML has no null, so the unweighted setting is spelled as a string and
    translated here rather than letting the literal "none" reach the estimator,
    where it would raise.
    """
    return None if value in (None, CLASS_WEIGHT_NONE) else value


def load_role_mapping(data_dir: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    """Read the shared role mapping and verify it against the pinned checksum.

    The mapping is a published contract: a changed file means a changed
    experiment, so a checksum mismatch stops the run rather than silently
    rescoring a different population.
    """
    roles = config["roles"]
    path = Path(data_dir).parent / roles["mapping_path"] if not Path(roles["mapping_path"]).is_absolute() else Path(roles["mapping_path"])
    path = REPO_ROOT / roles["mapping_path"]
    if not path.is_file():
        raise FileNotFoundError(f"Shared role mapping not found at {path}")
    actual = sha256_file(path)
    if actual != roles["mapping_sha256"]:
        raise ValueError(
            f"Role mapping checksum mismatch: expected {roles['mapping_sha256']}, got {actual}. "
            "The shared contract changed; re-pin it deliberately before rescoring."
        )
    mapping = pd.read_csv(path, dtype={"loan_id": "string", "split": "string", "role": "string"})
    expected_columns = ["loan_id", "split", "role", "issue_month"]
    if list(mapping.columns) != expected_columns:
        raise ValueError(f"Role mapping must have columns {expected_columns}; got {list(mapping.columns)}")
    if mapping["loan_id"].duplicated().any():
        raise ValueError("Role mapping contains duplicate loan_id values")
    counts = mapping["role"].value_counts().to_dict()
    for role, expected in config["expected"]["role_rows"].items():
        if role == "scoring_total":
            continue
        if counts.get(role) != expected:
            raise ValueError(f"Role {role!r} has {counts.get(role)} rows; config expects {expected}")
    return mapping


def attach_roles(rows: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Join loaded loans to the shared mapping, rejecting any imperfect match.

    A one-to-one validated merge with an indicator: a missing or extra loan is
    an error, never something to fill or drop quietly.
    """
    merged = rows.assign(loan_id=rows["loan_id"].astype("string")).merge(
        mapping[["loan_id", "split", "role"]].rename(columns={"split": "mapping_split"}),
        on="loan_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = int((merged["_merge"] != "both").sum())
        raise ValueError(f"{missing} loaded loans have no row in the shared role mapping")
    if not merged["split"].astype("string").eq(merged["mapping_split"]).all():
        raise ValueError("Split label disagrees with the shared role mapping")
    return merged.drop(columns=["_merge", "mapping_split"])


def load_scored_population(data_dir: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (base_fit rows, scoring rows) with roles attached and verified."""
    mapping = load_role_mapping(data_dir, config)
    frames = [load_baseline_split(data_dir, split) for split in ("train", "validation")]
    rows = attach_roles(pd.concat(frames, ignore_index=True), mapping)
    fit_rows = rows.loc[rows["role"].eq(config["roles"]["fit_role"])].reset_index(drop=True)
    scoring = rows.loc[rows["role"].isin(config["roles"]["scoring_roles"])].reset_index(drop=True)
    expected_total = config["expected"]["role_rows"]["scoring_total"]
    if len(scoring) != expected_total:
        raise ValueError(f"Scoring population is {len(scoring)} rows; expected {expected_total}")
    return fit_rows, scoring


def build_folds(fit_rows: pd.DataFrame, cv_config: dict[str, Any]):
    """Expanding-window folds over base_fit issue months, indexed by loan_id.

    ``time_aware_folds`` is imported here rather than at module scope because it
    lives in ``src/tabular/xgboost_baseline.py``, which imports xgboost at module
    level. Reusing the tabular track's exact fold scheme is worth that coupling -
    both tracks then select settings under identical folds - but a lazy import
    keeps this module usable without xgboost for anyone who only reads the export.
    See the known limitation in the report: the fold builder would be better
    placed in a module with no model-specific dependency.
    """
    from src.tabular.xgboost_baseline import time_aware_folds

    issue_month = pd.Series(
        fit_rows["issue_month"].to_numpy(), index=pd.Index(fit_rows["loan_id"], name="loan_id")
    )
    return time_aware_folds(issue_month, cv_config["n_folds"], cv_config["min_train_share"])


def grid_points(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Every combination of the declared grid, in a stable order."""
    return [
        {"max_features": features, "ngram_range": tuple(ngram), "C": regularisation}
        for features in grid["max_features"]
        for ngram in grid["ngram_range"]
        for regularisation in grid["C"]
    ]


def pipeline_for(params: dict[str, Any], model_config: dict[str, Any], class_weight: str | None):
    """Build the pipeline for one grid point, with the vectoriser as a step.

    Keeping TF-IDF inside the pipeline is what makes fold scores honest: the
    vocabulary and IDF weights are refit on each fold's training months only.
    """
    return build_pipeline(
        max_features=params["max_features"],
        ngram_range=params["ngram_range"],
        C=params["C"],
        sublinear_tf=model_config["sublinear_tf"],
        class_weight=class_weight,
        max_iter=model_config["max_iter"],
        random_state=model_config["random_state"],
    )


def score_folds(
    fit_rows: pd.DataFrame, folds, params: dict[str, Any], model_config: dict[str, Any], class_weight: str | None
) -> list[float]:
    """Average precision on each fold's holdout block."""
    text = pd.Series(fit_rows["text_tfidf"].to_numpy(), index=pd.Index(fit_rows["loan_id"]))
    target = pd.Series(fit_rows["target"].to_numpy(), index=pd.Index(fit_rows["loan_id"]))
    scores = []
    for fold in folds:
        pipeline = pipeline_for(params, model_config, class_weight)
        pipeline.fit(text.loc[fold.train_ids], target.loc[fold.train_ids])
        predicted = pipeline.predict_proba(text.loc[fold.holdout_ids])[:, 1]
        scores.append(float(average_precision_score(target.loc[fold.holdout_ids], predicted)))
    return scores


def run_grid_search(
    fit_rows: pd.DataFrame, folds, config: dict[str, Any], class_weight: str | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select settings on base_fit folds alone; no scoring role is consulted."""
    records = []
    for params in grid_points(config["grid"]):
        fold_scores = score_folds(fit_rows, folds, params, config["model"], class_weight)
        records.append(
            {
                "max_features": params["max_features"],
                "ngram_range": str(params["ngram_range"]),
                "C": params["C"],
                "mean_cv_pr_auc": float(np.mean(fold_scores)),
                "std_cv_pr_auc": float(np.std(fold_scores, ddof=1)),
                "fold_scores": fold_scores,
            }
        )
    results = pd.DataFrame(records).sort_values("mean_cv_pr_auc", ascending=False).reset_index(drop=True)
    winner = results.iloc[0]
    best = {
        "max_features": int(winner["max_features"]),
        "ngram_range": tuple(int(part) for part in winner["ngram_range"].strip("()").split(",")),
        "C": float(winner["C"]),
    }
    return results, best


def vocabulary_coverage(vectorizer, texts: pd.Series) -> dict[str, float]:
    """Share of analysed n-gram occurrences the fitted vocabulary covers.

    Occurrence level rather than a type-level rate: what matters is how much of
    each scored loan's text the model can actually read, given the vocabulary
    was fitted on months that all predate the scored loans.
    """
    analyze = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    totals = {"unigram": 0, "bigram": 0}
    covered = {"unigram": 0, "bigram": 0}
    for text in texts:
        for term in analyze(text):
            kind = "bigram" if " " in term else "unigram"
            totals[kind] += 1
            covered[kind] += term in vocabulary
    total = totals["unigram"] + totals["bigram"]
    result = {
        "unigram_covered_percent": 100 * covered["unigram"] / totals["unigram"] if totals["unigram"] else 0.0,
        "combined_covered_percent": 100 * (covered["unigram"] + covered["bigram"]) / total if total else 0.0,
    }
    if totals["bigram"]:
        result["bigram_covered_percent"] = 100 * covered["bigram"] / totals["bigram"]
    return result


def build_export(
    scoring: pd.DataFrame, margins: np.ndarray, probabilities: np.ndarray, model_version: str
) -> pd.DataFrame:
    """Assemble the agreed six-column export, matching the DistilBERT sibling."""
    frame = pd.DataFrame(
        {
            "loan_id": scoring["loan_id"].astype("string"),
            "split": scoring["split"].astype("string"),
            "role": scoring["role"].astype("string"),
            "raw_margin": np.asarray(margins, dtype=np.float64),
            "raw_prob": np.asarray(probabilities, dtype=np.float64),
            "model_version": pd.Series([model_version] * len(scoring), dtype="string"),
        }
    )
    return frame[list(EXPORT_COLUMNS)].sort_values("loan_id").reset_index(drop=True)


def validate_export(frame: pd.DataFrame, mapping: pd.DataFrame, config: dict[str, Any]) -> None:
    """Reject anything the fusion track would have to clean up after us."""
    if frame.columns.tolist() != list(EXPORT_COLUMNS):
        raise ValueError(f"Export must have columns {list(EXPORT_COLUMNS)}; got {frame.columns.tolist()}")
    if frame["loan_id"].duplicated().any():
        raise ValueError("Export has duplicate loan_id values")

    expected = mapping.loc[mapping["role"].isin(config["roles"]["scoring_roles"])]
    if set(frame["loan_id"]) != set(expected["loan_id"]):
        missing = len(set(expected["loan_id"]) - set(frame["loan_id"]))
        extra = len(set(frame["loan_id"]) - set(expected["loan_id"]))
        raise ValueError(f"Export does not cover the scoring roles exactly: {missing} missing, {extra} extra")
    if "base_fit" in set(frame["role"]):
        raise ValueError("Export must not contain base_fit rows")

    counts = frame["role"].value_counts().to_dict()
    for role in config["roles"]["scoring_roles"]:
        if counts.get(role) != config["expected"]["role_rows"][role]:
            raise ValueError(f"Export role {role!r} has {counts.get(role)} rows, expected {config['expected']['role_rows'][role]}")

    merged = frame.merge(
        mapping[["loan_id", "split", "role"]], on="loan_id", how="left", suffixes=("", "_expected")
    )
    if not merged["split"].eq(merged["split_expected"]).all():
        raise ValueError("Export split labels disagree with the shared role mapping")
    if not merged["role"].eq(merged["role_expected"]).all():
        raise ValueError("Export role labels disagree with the shared role mapping")

    if not np.isfinite(frame["raw_margin"]).all():
        raise ValueError("raw_margin contains non-finite values")
    probabilities = frame["raw_prob"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("raw_prob contains non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("raw_prob values must lie within [0, 1]")
    expected_probabilities = 1.0 / (1.0 + np.exp(-frame["raw_margin"].to_numpy(dtype=np.float64)))
    if not np.allclose(probabilities, expected_probabilities, atol=1e-9):
        raise ValueError("raw_prob is not the logistic transform of raw_margin from the same model")


def write_parquet(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Write Parquet and read it back, following the lexical-feature precedent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    written = pd.read_parquet(path, engine="pyarrow")
    if written.columns.tolist() != list(EXPORT_COLUMNS):
        raise RuntimeError("Written Parquet has unexpected columns")
    if not written["loan_id"].equals(frame["loan_id"]):
        raise RuntimeError("Written Parquet failed its loan_id order verification")
    if not np.allclose(written["raw_prob"], frame["raw_prob"]):
        raise RuntimeError("Written Parquet failed its probability round-trip check")
    return written


def run(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Select on base_fit, fit unweighted, score the three roles, export and record."""
    config = load_config(config_path)
    model_config = config["model"]
    model_version = config["model_version"]
    shipped_weight = resolve_class_weight(model_config["class_weight"])
    comparison_weight = resolve_class_weight(config["comparison"]["class_weight"])

    mapping = load_role_mapping(data_dir, config)
    fit_rows, scoring = load_scored_population(data_dir, config)
    folds = build_folds(fit_rows, config["cv"])

    grid_results, best_params = run_grid_search(fit_rows, folds, config, shipped_weight)

    fit_text, fit_target = fit_rows["text_tfidf"], fit_rows["target"].to_numpy()
    pipeline = pipeline_for(best_params, model_config, shipped_weight)
    pipeline.fit(fit_text, fit_target)

    margins = pipeline.decision_function(scoring["text_tfidf"])
    probabilities = pipeline.predict_proba(scoring["text_tfidf"])[:, 1]
    export = build_export(scoring, margins, probabilities, model_version)
    validate_export(export, mapping, config)
    written = write_parquet(export, REPO_ROOT / config["output"]["scores_path"])

    # Matched comparison: same rows, same folds, same winning settings. The v1
    # balanced figures used all 86,293 train rows and random folds, so they are
    # not a like-for-like reference.
    comparison_pipeline = pipeline_for(best_params, model_config, comparison_weight)
    comparison_pipeline.fit(fit_text, fit_target)
    comparison_folds = score_folds(fit_rows, folds, best_params, model_config, comparison_weight)

    validation = scoring.loc[scoring["role"].eq("validation")]
    validation_index = validation.index.to_numpy()
    y_validation = validation["target"].to_numpy()
    unweighted_metrics = evaluate_predictions(y_validation, probabilities[validation_index])
    balanced_probabilities = comparison_pipeline.predict_proba(validation["text_tfidf"])[:, 1]
    balanced_metrics = evaluate_predictions(y_validation, balanced_probabilities)

    vectorizer = pipeline.named_steps["vectorize"]
    coverage = {
        role: vocabulary_coverage(vectorizer, scoring.loc[scoring["role"].eq(role), "text_tfidf"])
        for role in config["roles"]["scoring_roles"]
    }
    scored_matrix = vectorizer.transform(scoring["text_tfidf"])
    empty_vector_rows = int((np.asarray((scored_matrix != 0).sum(axis=1)).ravel() == 0).sum())

    pipeline_path = REPO_ROOT / config["output"]["pipeline_path"]
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, pipeline_path)

    scores_path = REPO_ROOT / config["output"]["scores_path"]
    manifest = {
        "model_version": model_version,
        "model_name": config["model_name"],
        "preprocess_version": config["preprocess_version"],
        "status": config.get("status"),
        "probability_semantics": "raw_unweighted_model_output_not_post_hoc_calibrated",
        "fusion_development_version": config["roles"]["fusion_development_version"],
        "naming": {
            "authoritative_identifier": "model_version",
            "config_file": "configs/nlp_tfidf_fusion_v2.toml",
            "note": (
                "The config is named for the model version it produces. The older "
                "configs/nlp_tfidf_v2.toml produces nlp-tfidf-tuned-v1; that file's counter "
                "was never aligned with its model version. Exports and manifests are keyed by "
                "model_version, not by config filename."
            ),
            "supersedes_for_fusion": "nlp-tfidf-tuned-v1",
            "v1_exports_unchanged": ["data/nlp/tfidf_oof_train.parquet", "data/nlp/tfidf_oof_validate.parquet"],
        },
        "fitting_boundaries": {
            "fit_role": config["roles"]["fit_role"],
            "fit_rows": int(len(fit_rows)),
            "fit_first_month": str(fit_rows["issue_month"].min()),
            "fit_last_month": str(fit_rows["issue_month"].max()),
            "fitted_on_scoring_roles": False,
            "note": "Vocabulary, IDF weights, hyperparameter selection and coefficients all come from base_fit only.",
        },
        "roles": {
            role: {
                "rows": int(scoring["role"].eq(role).sum()),
                "first_month": str(scoring.loc[scoring["role"].eq(role), "issue_month"].min()),
                "last_month": str(scoring.loc[scoring["role"].eq(role), "issue_month"].max()),
            }
            for role in config["roles"]["scoring_roles"]
        },
        "selection": {
            "scoring": config["cv"]["scoring"],
            "scheme": config["cv"]["scheme"],
            "scheme_source": config["cv"]["scheme_source"],
            "n_folds": config["cv"]["n_folds"],
            "min_train_share": config["cv"]["min_train_share"],
            "grid_size": int(len(grid_results)),
            "selected_on": "base_fit_time_ordered_folds_only",
            "best_params": {**best_params, "ngram_range": list(best_params["ngram_range"])},
            "best_mean_cv_pr_auc": float(grid_results.loc[0, "mean_cv_pr_auc"]),
            "best_std_cv_pr_auc": float(grid_results.loc[0, "std_cv_pr_auc"]),
            "folds": [
                {
                    "fold_id": index,
                    "fit_rows": int(len(fold.train_ids)),
                    "holdout_rows": int(len(fold.holdout_ids)),
                    "holdout_first_month": fold.holdout_first_month,
                    "holdout_last_month": fold.holdout_last_month,
                }
                for index, fold in enumerate(folds)
            ],
            "fixed_model_settings": {**model_config, "class_weight": "None (unweighted)"},
        },
        "comparison_matched_conditions": {
            "note": "Same base_fit rows, same folds, same winning settings; only class_weight differs.",
            "unweighted": {
                "validation": unweighted_metrics,
                "mean_cv_pr_auc": float(grid_results.loc[0, "mean_cv_pr_auc"]),
            },
            "balanced": {
                "validation": balanced_metrics,
                "mean_cv_pr_auc": float(np.mean(comparison_folds)),
            },
        },
        "vocabulary_coverage_of_scoring_roles": coverage,
        "unusable_text": {
            "definition_confirmed_by_leads": False,
            "empty_raw_description_rows": int((scoring["desc"].fillna("").str.strip() == "").sum()),
            "empty_cleaned_payload_rows": int((scoring["text_tfidf"].str.strip() == "").sum()),
            "zero_vector_after_tokenisation_rows": empty_vector_rows,
            "note": (
                "No fallback flag is emitted: the agreed schema has no column for one and the "
                "definition is unconfirmed. Rows with a zero vector still receive a probability, "
                "which is the model intercept rather than a text signal."
            ),
        },
        "output": {
            "path": config["output"]["scores_path"],
            "rows": int(len(written)),
            "columns": list(EXPORT_COLUMNS),
            "sha256": sha256_file(scores_path),
        },
        "pipeline_artifact": {
            "path": config["output"]["pipeline_path"],
            "sha256": sha256_file(pipeline_path),
            "gitignored": True,
            "note": "Convenience for inference. The contract is this manifest: the pipeline rebuilds deterministically from the pinned config, seed and base_fit rows.",
        },
        "inputs": {
            "role_mapping": {
                "path": config["roles"]["mapping_path"],
                "sha256": config["roles"]["mapping_sha256"],
            },
            "config": {
                "path": str(Path(config_path).relative_to(REPO_ROOT)) if Path(config_path).is_absolute() else str(config_path),
                "sha256": sha256_file(config_path),
            },
            "split_manifest_sha256": sha256_file(Path(data_dir) / "split_manifest.csv"),
        },
        "test_rows_used": 0,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    manifest_path = REPO_ROOT / config["output"]["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    grid_results.drop(columns=["fold_scores"]).to_csv(
        manifest_path.parent / "tfidf_scores_v2_grid.csv", index=False
    )
    return {"export": written, "manifest": manifest, "grid_results": grid_results, "pipeline": pipeline}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unweighted TF-IDF fusion scores.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.data_dir, config_path=args.config)
    manifest = result["manifest"]
    print(json.dumps({k: v for k, v in manifest.items() if k != "selection"}, indent=2)[:2600])
    print("\nSelected:", json.dumps(manifest["selection"]["best_params"]))
    print(f"Top grid rows:\n{result['grid_results'].drop(columns=['fold_scores']).head(6).to_string(index=False)}")


if __name__ == "__main__":
    main()

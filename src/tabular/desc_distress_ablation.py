"""Pre-specified description-density XGBoost ablation, gated on shared role approval."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.tabular import evaluate, preprocess, xgboost_baseline as xgb

FEATURE = "desc_distress_keyword_density"
OLD_FEATURE = "financial_distress_keyword_density"
ROOT = preprocess.REPO_ROOT
EXPORT = ROOT / "data/nlp/desc_distress_keyword_density.parquet"
EXPORT_META = ROOT / "reports/nlp/desc_distress_export_metrics.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    existing = result["model"]["numeric"] + result["model"]["categorical"]
    if len(existing) != 9 or FEATURE in existing or OLD_FEATURE in existing:
        raise ValueError("Expected the approved nine-feature baseline without distress columns")
    result["model"]["numeric"].append(FEATURE)
    return result


def join_density(features: pd.DataFrame, split: str, export: pd.DataFrame) -> pd.DataFrame:
    if not features.index.is_unique or features.index.hasnans:
        raise ValueError("Feature IDs must be unique and present")
    if list(export.columns) != ["loan_id", "split", FEATURE]:
        raise ValueError("Unexpected density export schema")
    if export.loan_id.isna().any() or export.loan_id.duplicated().any():
        raise ValueError("Density export has missing or duplicate IDs")
    table = export.copy()
    table["loan_id"] = table.loan_id.astype(str)
    table = table.set_index("loan_id")
    rows = table.reindex(features.index)
    if rows.split.isna().any() or not rows.split.eq(split).all():
        raise ValueError("Density join has missing IDs or split mismatch")
    values = pd.to_numeric(rows[FEATURE], errors="coerce")
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        raise ValueError("Density values must be finite and within [0, 1]")
    if FEATURE in features or OLD_FEATURE in features:
        raise ValueError("Unexpected distress feature in baseline frame")
    result = features.copy()
    result[FEATURE] = values
    return result


def run(output_dir: Path, data_dir: Path, raw_csv: Path) -> None:
    xgb.validate_runtime()
    manifest, _ = evaluate.load_manifest(data_dir)
    metadata = json.loads(EXPORT_META.read_text())
    if digest(EXPORT) != metadata["parquet_sha256"] or metadata["feature_version"] != "nlp-desc-distress-v1":
        raise ValueError("Description export does not match its versioned metadata")
    export = pd.read_parquet(EXPORT)
    expected_ids = manifest.set_index(manifest.loan_id.astype(str))
    # Validate all development rows, including fusion roles, before selecting a subset.
    join_density(pd.DataFrame(index=expected_ids.index[expected_ids.split.eq("train")]), "train", export)
    join_density(pd.DataFrame(index=expected_ids.index[expected_ids.split.eq("validation")]), "validation", export)
    if len(export) != len(expected_ids):
        raise ValueError("Density export has unexpected coverage")
    model_x, model_y, months = xgb.load_model_fit(data_dir, raw_csv)
    val_x, val_y = preprocess.load_tabular_split(data_dir, raw_csv, "validation")
    expected_base = manifest.loc[manifest.split.eq("train") & manifest.issue_month.between("2007-06", "2012-12"), "loan_id"].astype(str)
    expected_val = manifest.loc[manifest.split.eq("validation"), "loan_id"].astype(str)
    if len(expected_base) != 59353 or len(expected_val) != 21784 or set(model_x.index) != set(expected_base) or set(val_x.index) != set(expected_val):
        raise ValueError("Loaded rows disagree with the confirmed fitting and validation boundaries")
    model_lex = join_density(model_x, "train", export)
    val_lex = join_density(val_x, "validation", export)
    config = preprocess.load_feature_config()
    candidate_config = variant_config(config)
    settings = xgb.load_xgboost_config()
    # Trial 1 was selected on model_fit CV in the existing, recorded search.
    params = {**settings["model"], "n_estimators": 200, "max_depth": 2}
    selection = json.loads((ROOT / "reports/tabular/xgboost_selection.json").read_text())
    if selection.get("selected_trial_id") != 1 or selection.get("selected_params") != params:
        raise ValueError("Selected XGBoost settings differ from the pre-specified trial 1")
    folds = xgb.build_folds(months, settings)
    cv_base = xgb.cross_validate(lambda: xgb.build_xgboost_pipeline(config, params), model_x, model_y, folds)
    cv_candidate = xgb.cross_validate(lambda: xgb.build_xgboost_pipeline(candidate_config, params), model_lex, model_y, folds)
    baseline = xgb.build_xgboost_pipeline(config, params).fit(model_x, model_y)
    candidate = xgb.build_xgboost_pipeline(candidate_config, params).fit(model_lex, model_y)
    p_base = baseline.predict_proba(val_x)[:, 1]
    p_candidate = candidate.predict_proba(val_lex)[:, 1]
    rows = []
    for metric in xgb.SUMMARY_METRICS:
        interval = xgb.paired_bootstrap_delta(val_y.to_numpy(), p_base, p_candidate, metric,
            resamples=settings["evaluation"]["bootstrap_resamples"], seed=settings["evaluation"]["seed"])
        fn = xgb.METRIC_FUNCTIONS[metric]
        rows.append({"metric": metric, "baseline": fn(val_y, p_base), "density": fn(val_y, p_candidate), **interval,
                     "base_cv_fold_mean": cv_base.summary[metric]["mean"], "density_cv_fold_mean": cv_candidate.summary[metric]["mean"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "desc_distress_ablation_results.csv", index=False)
    validation_month = manifest.set_index("loan_id").loc[val_x.index, "issue_month"]
    diagnostics = pd.DataFrame({"month": validation_month, "target": val_y,
                                "density_positive": val_lex[FEATURE].gt(0),
                                "baseline": p_base, "density": p_candidate}, index=val_x.index)
    subgroup_rows = []
    for grouping, groups in (("month", diagnostics.groupby("month")),
                             ("keyword_hit", diagnostics.groupby("density_positive"))):
        for name, group in groups:
            item = {"grouping": grouping, "group": str(name), "rows": len(group),
                    "defaults": int(group.target.sum()), "keyword_hit_rate": group.density_positive.mean()}
            for metric, function in xgb.METRIC_FUNCTIONS.items():
                item[f"baseline_{metric}"] = function(group.target, group.baseline)
                item[f"density_{metric}"] = function(group.target, group.density)
            subgroup_rows.append(item)
    pd.DataFrame(subgroup_rows).to_csv(output_dir / "desc_distress_ablation_diagnostics.csv", index=False)
    record = {"split_manifest_sha256": digest(data_dir / "split_manifest.csv"), "density_sha256": digest(EXPORT),
              "density_version": metadata["feature_version"], "feature_set_version": config["feature_set_version"],
              "feature_config_sha256": digest(preprocess.DEFAULT_FEATURE_CONFIG),
              "xgboost_config_sha256": digest(xgb.DEFAULT_XGBOOST_CONFIG),
              "selection_sha256": digest(ROOT / "reports/tabular/xgboost_selection.json"),
              "model_params": params, "folds": xgb.describe_folds(folds), "base_fit_rows": len(model_x),
              "validation_rows": len(val_x), "validation_positives": int(val_y.sum()),
              "baseline_cv_folds": list(cv_base.fold_metrics), "density_cv_folds": list(cv_candidate.fold_metrics),
              "environment": xgb.environment(), "bootstrap": settings["evaluation"],
              "ci_interpretation": "paired row bootstrap, conditional on fitted models; temporal dependence is not accounted for"}
    (output_dir / "desc_distress_ablation_metadata.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads-confirmed", action="store_true", help="Assert Evan and Ariel confirmed the shared development roles")
    parser.add_argument("--data-dir", type=Path, default=preprocess.DEFAULT_DATA_DIR)
    parser.add_argument("--raw-csv", type=Path, default=preprocess.DEFAULT_RAW_CSV)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/tabular")
    args = parser.parse_args()
    if not args.leads_confirmed:
        parser.error("Training requires Evan and Ariel's confirmation of the shared roles")
    run(args.output_dir, args.data_dir, args.raw_csv)


if __name__ == "__main__":
    main()

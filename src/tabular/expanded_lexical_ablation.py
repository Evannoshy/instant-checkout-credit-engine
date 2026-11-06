"""Compare tabular, description density, and lexical XGBoost with LightGBM."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.tabular import development, evaluate, preprocess, xgboost_baseline as xgb
from src.tabular import desc_distress_ablation as desc

EXTRA = ["flesch_reading_ease", "flesch_kincaid_grade", "lexical_diversity_ttr",
         "sentiment_polarity", "sentiment_subjectivity"]
MAPPING_SHA256 = "68599e30673a8099feadd3c7961f70dcc72bf082266d4a891c2e48f54664b20e"
LIGHTGBM_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2,
                   "num_leaves": 4, "min_child_samples": 20, "reg_lambda": 1.0,
                   "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                   "random_state": 42, "n_jobs": 1, "verbosity": -1}


def check_mapping(mapping: pd.DataFrame, manifest: pd.DataFrame) -> None:
    columns = ["loan_id", "split", "role", "issue_month"]
    if list(mapping.columns) != columns or mapping.isna().any().any() or mapping.loan_id.duplicated().any():
        raise ValueError("Invalid shared mapping schema or duplicated or missing IDs")
    expected = manifest[["loan_id", "split", "issue_month"]].copy()
    actual = mapping.merge(expected, on="loan_id", how="outer", validate="one_to_one", indicator=True,
                           suffixes=("_mapping", "_manifest"))
    if not actual._merge.eq("both").all() or not actual.split_mapping.eq(actual.split_manifest).all() or not actual.issue_month_mapping.eq(actual.issue_month_manifest).all():
        raise ValueError("Shared mapping disagrees with development manifest")
    bounds = {"base_fit": ("train", "2007-06", "2012-12", 59353),
              "fusion_fit": ("train", "2013-01", "2013-04", 15587),
              "fusion_calibration": ("train", "2013-05", "2013-07", 11353),
              "validation": ("validation", "2013-08", "2013-12", 21784)}
    if set(mapping.role) != set(bounds):
        raise ValueError("Unexpected roles in shared mapping")
    for role, (split, first, last, count) in bounds.items():
        part = mapping[mapping.role.eq(role)]
        if len(part) != count or not part.split.eq(split).all() or not part.issue_month.between(first, last).all():
            raise ValueError(f"Invalid mapping for {role}")


def lexical_frame(features: pd.DataFrame, target: pd.Series, data_dir: Path) -> pd.DataFrame:
    joined, joined_y = development.join_lexical_features(features, target, data_dir)
    if not joined_y.equals(target) or not np.isfinite(joined[EXTRA].to_numpy(dtype=float)).all():
        raise ValueError("Lexical features or labels are invalid")
    return joined[EXTRA]


def full_config(base: dict) -> dict:
    config = desc.variant_config(base)
    config["model"]["numeric"].extend(EXTRA)
    return config


def light_pipeline(config: dict) -> Pipeline:
    transformer = xgb.build_xgboost_pipeline(config, {}).named_steps["preprocess"]
    return Pipeline([("preprocess", transformer), ("classify", lightgbm.LGBMClassifier(**LIGHTGBM_PARAMS))])


def run(data_dir: Path, raw_csv: Path, output_dir: Path, mapping_path: Path) -> None:
    xgb.validate_runtime()
    if lightgbm.__version__ != "4.7.0":
        raise ValueError("LightGBM 4.7.0 is required for this experiment")
    if desc.digest(mapping_path) != MAPPING_SHA256:
        raise ValueError("Shared role mapping checksum differs from published handoff")
    mapping = pd.read_csv(mapping_path, dtype=str)
    manifest, _ = evaluate.load_manifest(data_dir)
    check_mapping(mapping, manifest)
    metadata = json.loads(desc.EXPORT_META.read_text())
    if metadata["feature_version"] != "nlp-desc-distress-v1" or desc.digest(desc.EXPORT) != metadata["parquet_sha256"]:
        raise ValueError("Description feature export checksum or version differs")
    density = pd.read_parquet(desc.EXPORT)
    model_x, y, months = xgb.load_model_fit(data_dir, raw_csv)
    val_x, val_y = preprocess.load_tabular_split(data_dir, raw_csv, "validation")
    for frame, role in [(model_x, "base_fit"), (val_x, "validation")]:
        if set(frame.index) != set(mapping.loc[mapping.role.eq(role), "loan_id"]):
            raise ValueError(f"Loaded {role} IDs disagree with shared mapping")
    model_d = desc.join_density(model_x, "train", density)
    val_d = desc.join_density(val_x, "validation", density)
    model_all = model_d.join(lexical_frame(model_x, y, data_dir), validate="one_to_one")
    val_all = val_d.join(lexical_frame(val_x, val_y, data_dir), validate="one_to_one")
    if desc.OLD_FEATURE in model_all or desc.OLD_FEATURE in val_all:
        raise ValueError("Combined distress column must not enter the models")
    config = preprocess.load_feature_config()
    settings = xgb.load_xgboost_config()
    selection = json.loads((desc.ROOT / "reports/tabular/xgboost_selection.json").read_text())
    params = {**settings["model"], "n_estimators": 200, "max_depth": 2}
    if selection.get("selected_trial_id") != 1 or selection.get("selected_params") != params:
        raise ValueError("XGBoost parameters do not match the recorded selection")
    density_config = desc.variant_config(config)
    lexical_config = full_config(config)
    folds = xgb.build_folds(months, settings)
    variants = {
        "xgb_tabular": (lambda: xgb.build_xgboost_pipeline(config, params), model_x, val_x),
        "xgb_distress": (lambda: xgb.build_xgboost_pipeline(density_config, params), model_d, val_d),
        "xgb_lexical": (lambda: xgb.build_xgboost_pipeline(lexical_config, params), model_all, val_all),
        "lightgbm_lexical": (lambda: light_pipeline(lexical_config), model_all, val_all),
    }
    predictions, cv_records, summary = {}, {}, []
    for name, (factory, train, valid) in variants.items():
        print(f"Fitting {name}", flush=True)
        cv = xgb.cross_validate(factory, train, y, folds)
        cv_records[name] = list(cv.fold_metrics)
        fitted = factory().fit(train, y)
        probabilities = fitted.predict_proba(valid)[:, 1].astype(float)
        if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
            raise ValueError(f"Invalid probabilities: {name}")
        predictions[name] = probabilities
        for metric, function in xgb.METRIC_FUNCTIONS.items():
            summary.append({"model": name, "metric": metric, "validation": function(val_y, probabilities),
                            "cv_fold_mean": cv.summary[metric]["mean"], "cv_fold_std": cv.summary[metric]["std"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output_dir / "expanded_lexical_metrics.csv", index=False)
    pair_rows = []
    for ref, cand in [("xgb_tabular", "xgb_distress"), ("xgb_tabular", "xgb_lexical"),
                      ("xgb_distress", "xgb_lexical"), ("xgb_lexical", "lightgbm_lexical")]:
        for metric in xgb.SUMMARY_METRICS:
            print(f"Bootstrap {cand} vs {ref}: {metric}", flush=True)
            delta = xgb.paired_bootstrap_delta(val_y.to_numpy(), predictions[ref], predictions[cand], metric,
                resamples=settings["evaluation"]["bootstrap_resamples"], seed=settings["evaluation"]["seed"])
            pair_rows.append({"reference": ref, "candidate": cand, "metric": metric, **delta})
    pd.DataFrame(pair_rows).to_csv(output_dir / "expanded_lexical_deltas.csv", index=False)
    month = manifest.set_index("loan_id").loc[val_x.index, "issue_month"]
    pd.DataFrame({"loan_id": val_x.index, "issue_month": month.to_numpy(), "target": val_y.to_numpy(),
                  "distress_positive": val_d[desc.FEATURE].gt(0).to_numpy(), **predictions}).to_csv(
                      output_dir / "expanded_lexical_validation_predictions.csv", index=False)
    record = {"role_mapping_sha256": desc.digest(mapping_path), "mapping_source_commit": "57b8d564f6e0c0ad90d85012fcc6ddcdc5dc7cb3",
              "split_manifest_sha256": desc.digest(data_dir / "split_manifest.csv"),
              "density_sha256": desc.digest(desc.EXPORT), "density_version": metadata["feature_version"],
              "lexical_sha256": development.load_development_config()["lexical"]["sha256"],
              "lexical_columns": [desc.FEATURE, *EXTRA], "excluded": desc.OLD_FEATURE,
              "base_fit_rows": len(model_x), "validation_rows": len(val_x), "validation_positives": int(val_y.sum()),
              "xgboost_params": params, "lightgbm_params": LIGHTGBM_PARAMS, "lightgbm_version": lightgbm.__version__,
              "folds": xgb.describe_folds(folds), "cv_results": cv_records, "environment": xgb.environment(),
              "bootstrap": settings["evaluation"], "scope": "expanded analysis selected after previous validation review; exploratory"}
    (output_dir / "expanded_lexical_metadata.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=preprocess.DEFAULT_DATA_DIR)
    parser.add_argument("--raw-csv", type=Path, default=preprocess.DEFAULT_RAW_CSV)
    parser.add_argument("--output-dir", type=Path, default=desc.ROOT / "reports/tabular")
    parser.add_argument("--mapping", type=Path, default=desc.ROOT / "data/fusion/fusion_roles_v1.csv")
    args = parser.parse_args()
    run(args.data_dir, args.raw_csv, args.output_dir, args.mapping)


if __name__ == "__main__":
    main()

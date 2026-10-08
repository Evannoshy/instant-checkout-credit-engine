"""Lexical experiment: does logistic regression gain from the six NLP lexical features?

Compares two logistic pipelines that differ only in their inputs:
1. the nine approved tabular_features_v1 features, and
2. the same nine plus the six lexical columns from ``[lexical]`` in
   configs/tabular_development_v1.toml (joined by
   ``development.join_lexical_features``).

Held identical: the model_fit rows (asserted loan_id by loan_id), the
expanding-window folds (src/tabular/xgboost_baseline.py), the approved
logistic settings (configs/tabular_logistic_v1.toml, unchanged), the
metrics and the seed. A metric "improves" only when its paired bootstrap 95%
CI excludes zero in the better direction.

The lexical columns are experimental input, not part of tabular_features_v1;
the nine-feature config is copied, never modified.

Run from the repository root:
    python -m src.tabular.lexical_ablation
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.tabular import development, logistic_baseline, preprocess
from src.tabular.xgboost_baseline import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_XGBOOST_CONFIG,
    HIGHER_IS_BETTER,
    SUMMARY_METRICS,
    Fold,
    build_folds,
    cross_validate,
    load_model_fit,
    load_xgboost_config,
    paired_deltas,
    run_metadata,
    validate_runtime,
)


def lexical_feature_config(
    feature_config: dict[str, Any], lexical_columns: list[str]
) -> dict[str, Any]:
    """A deep copy of the feature config with the lexical columns appended to numeric."""
    variant = copy.deepcopy(feature_config)
    variant["model"]["numeric"] = [*variant["model"]["numeric"], *lexical_columns]
    return variant


def _improves(metric: str, ci_low: float, ci_high: float) -> bool:
    """True only if the whole CI lies on the better side of zero."""
    return ci_low > 0 if HIGHER_IS_BETTER[metric] else ci_high < 0


def compare_lexical(
    tabular_features: pd.DataFrame,
    lexical_features: pd.DataFrame,
    target: pd.Series,
    folds: tuple[Fold, ...],
    feature_config: dict[str, Any],
    lexical_columns: list[str],
    logistic_config: dict[str, Any],
    evaluation: dict[str, Any],
) -> pd.DataFrame:
    """One row per metric: both variants' fold mean/std, the change and whether it is an improvement."""
    if not (
        tabular_features.index.equals(lexical_features.index)
        and tabular_features.index.equals(target.index)
    ):
        raise ValueError("Tabular and lexical frames are not aligned on the same loan_id values")
    lexical_config = lexical_feature_config(feature_config, lexical_columns)
    tabular = cross_validate(
        lambda: logistic_baseline.build_pipeline(feature_config, logistic_config),
        tabular_features, target, folds,
    )
    lexical = cross_validate(
        lambda: logistic_baseline.build_pipeline(lexical_config, logistic_config),
        lexical_features, target, folds,
    )
    deltas = paired_deltas(target, tabular, lexical, SUMMARY_METRICS, evaluation)
    rows = []
    for metric in SUMMARY_METRICS:
        delta = deltas[metric]
        rows.append({
            "metric": metric,
            "higher_is_better": HIGHER_IS_BETTER[metric],
            "tabular_9_mean": tabular.summary[metric]["mean"],
            "tabular_9_std": tabular.summary[metric]["std"],
            "tabular_9_plus_lexical_6_mean": lexical.summary[metric]["mean"],
            "tabular_9_plus_lexical_6_std": lexical.summary[metric]["std"],
            "delta": lexical.summary[metric]["mean"] - tabular.summary[metric]["mean"],
            "delta_pooled": delta["delta"],
            "delta_ci_low": delta["ci_low"],
            "delta_ci_high": delta["ci_high"],
            "improves": _improves(metric, delta["ci_low"], delta["ci_high"]),
        })
    return pd.DataFrame(rows)


def run_lexical_ablation(
    data_dir: str | Path,
    raw_csv_path: str | Path,
    output_dir: str | Path,
    *,
    xgboost_config_path: str | Path = DEFAULT_XGBOOST_CONFIG,
) -> pd.DataFrame:
    """Load and join model_fit, run the experiment, write lexical_ablation_results.csv."""
    validate_runtime()
    xgboost_config = load_xgboost_config(xgboost_config_path)
    logistic_config = logistic_baseline.load_logistic_config()
    feature_config = preprocess.load_feature_config()
    lexical_settings = development.load_development_config()["lexical"]
    features, target, issue_month = load_model_fit(data_dir, raw_csv_path)
    lexical_features, lexical_target = development.join_lexical_features(features, target, data_dir)
    if not lexical_target.equals(target):
        raise ValueError("The lexical join changed the target")
    folds = build_folds(issue_month, xgboost_config)

    results = compare_lexical(
        features, lexical_features, target, folds, feature_config,
        lexical_settings["columns"], logistic_config, xgboost_config["evaluation"],
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "lexical_ablation_results.csv", index=False)
    metadata = {
        **run_metadata(data_dir, xgboost_config, folds),
        "model": "logistic_regression",
        "logistic_params": logistic_config["model"],
        "aligned_loan_ids": len(features),
        "lexical_columns": lexical_settings["columns"],
        "lexical_sha256": lexical_settings["sha256"],
    }
    (output_dir / "lexical_ablation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic regression: 9 tabular vs 9 + 6 lexical.")
    parser.add_argument("--data-dir", default=preprocess.DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--raw-csv", default=preprocess.DEFAULT_RAW_CSV, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    args = parser.parse_args()
    results = run_lexical_ablation(args.data_dir, args.raw_csv, args.output_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

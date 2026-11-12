"""Late-fusion stacker (fusion-stacker-v1): fit, calibrate and grade each run variant.

For each variant in ``[run] variants`` of ``configs/fusion_stacker_v1.toml``:

1. Fit an unweighted (``class_weight=None``) L2 logistic regression on the
   15,587 ``fusion_fit`` loans. Its inputs are the variant's raw base-model
   probabilities only (``inputs.score_columns``).
2. Score ``fusion_calibration`` and ``validation`` with it.
3. Fit a Platt calibrator on the 11,353 ``fusion_calibration`` stacker outputs
   only (``calibrate.fit_platt``), and apply it to validation.
4. Grade raw and calibrated outputs on the same 21,784 validation loans,
   against the calibrated tabular reference, with 1,000 paired bootstrap
   resamples (``xgboost_baseline.paired_bootstrap_delta``).

Validation fits nothing; the locked test split is never read.

The comparison does not choose a winner: Evan and Ariel have not yet confirmed
the primary metric, probability-quality safeguards or decision rule. The
config's +0.005 margin is reported as a proposed target only.

Each fitted variant is saved as a JSON artifact (coefficients, intercept,
Platt slope and intercept, source versions) so ``src/fusion/predict.py`` can
rebuild it without a pickle.

Run from the repository root (no raw CSV needed):
    python -m src.fusion.stacker

Known limitations:
- With one input (``tabular_only``), a logistic regression on the raw
  probability is a monotone transform, so ROC-AUC and PR-AUC cannot differ from
  the reference. Only probability quality (Brier, log loss) can change.
- The paired bootstrap resamples loans independently; validation loans are
  clustered in time, so the intervals are probably somewhat narrow.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from src.fusion import inputs
from src.tabular import calibrate, provenance
from src.tabular.preprocess import DEFAULT_DATA_DIR, REPO_ROOT
from src.tabular.xgboost_baseline import paired_bootstrap_delta

DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "fusion"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "data" / "fusion" / "stacker_v1"
METRICS = calibrate.METRICS
REFERENCE = "reference"
BASELINE_VARIANT = "tabular_only"
# (candidate, baseline) variants compared when both run: does DistilBERT add over TF-IDF?
INCREMENTAL_PAIRS = (("tabular_tfidf_distilbert", "tabular_tfidf"),)


@dataclass(frozen=True)
class FittedVariant:
    """One variant's stacker and calibrator, with its predictions on the held-out roles."""

    name: str
    tracks: list[str]
    stacker: LogisticRegression
    platt: calibrate.Calibrator
    fusion_calibration_raw: np.ndarray
    validation_raw: np.ndarray
    validation_platt: np.ndarray


def fit_stacker(fusion_fit: pd.DataFrame, tracks: list[str], settings: dict[str, Any]) -> LogisticRegression:
    """Unweighted L2 logistic regression on the variant's score columns, fitted on fusion_fit only."""
    model = LogisticRegression(C=settings["C"], class_weight=None, max_iter=settings["max_iter"])
    return model.fit(fusion_fit[inputs.score_columns(tracks)].to_numpy(), fusion_fit["target"].to_numpy())


def stacker_probability(model: LogisticRegression, table: pd.DataFrame, tracks: list[str]) -> np.ndarray:
    return model.predict_proba(table[inputs.score_columns(tracks)].to_numpy())[:, 1]


def fit_variant(
    name: str, tracks: list[str], tables: dict[str, pd.DataFrame], config: dict[str, Any]
) -> FittedVariant:
    """Stacker on fusion_fit, Platt on fusion_calibration, both applied to validation."""
    stacker = fit_stacker(tables["fusion_fit"], tracks, config["stacker"])
    calibration_raw = stacker_probability(stacker, tables["fusion_calibration"], tracks)
    platt = calibrate.fit_platt(
        calibration_raw, tables["fusion_calibration"]["target"].to_numpy(),
        config["calibration"]["probability_clip"],
    )
    validation_raw = stacker_probability(stacker, tables["validation"], tracks)
    return FittedVariant(
        name=name,
        tracks=tracks,
        stacker=stacker,
        platt=platt,
        fusion_calibration_raw=calibration_raw,
        validation_raw=validation_raw,
        validation_platt=np.asarray(platt.transform(validation_raw), dtype=np.float64),
    )


def validation_versions(reference: np.ndarray, fitted: list[FittedVariant]) -> dict[str, np.ndarray]:
    """Every probability graded on validation, keyed ``reference`` or ``<variant>_raw|_platt``."""
    versions = {REFERENCE: reference}
    for variant in fitted:
        versions[f"{variant.name}_raw"] = variant.validation_raw
        versions[f"{variant.name}_platt"] = variant.validation_platt
    return versions


def comparison_pairs(versions: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    """(candidate, baseline) pairs: every stacker output vs the reference, and each text variant's
    calibrated output vs the calibrated tabular-only stacker (the "does text help" question), and
    each INCREMENTAL_PAIRS candidate vs its smaller variant, calibrated."""
    pairs = [(name, REFERENCE) for name in versions if name != REFERENCE]
    baseline = f"{BASELINE_VARIANT}_platt"
    if baseline in versions:
        pairs += [(name, baseline) for name in versions if name.endswith("_platt") and name != baseline]
    incremental = [(f"{candidate}_platt", f"{smaller}_platt") for candidate, smaller in INCREMENTAL_PAIRS]
    return pairs + [pair for pair in incremental if set(pair) <= set(versions)]


def compare(
    y: np.ndarray, versions: dict[str, np.ndarray], evaluation: dict[str, Any]
) -> dict[str, dict[str, dict[str, float]]]:
    """``{"<candidate>_minus_<baseline>": {metric: {"delta", "ci_low", "ci_high"}}}`` on validation."""
    return {
        f"{candidate}_minus_{baseline}": {
            metric: paired_bootstrap_delta(
                y, versions[baseline], versions[candidate], metric,
                resamples=evaluation["bootstrap_resamples"], seed=evaluation["seed"],
            )
            for metric in METRICS
        }
        for candidate, baseline in comparison_pairs(versions)
    }


def variant_artifact(variant: FittedVariant, config: dict[str, Any]) -> dict[str, Any]:
    """Everything ``predict.py`` needs to rebuild one variant, as plain JSON."""
    return {
        "stacker_version": config["stacker_version"],
        "variant": variant.name,
        "tracks": variant.tracks,
        "input_columns": inputs.score_columns(variant.tracks),
        "input_space": config["stacker"]["input_space"],
        "stacker": {
            "coefficients": variant.stacker.coef_[0].tolist(),
            "intercept": float(variant.stacker.intercept_[0]),
            "C": config["stacker"]["C"],
            "class_weight": None,
            "fit_role": "fusion_fit",
        },
        "calibrator": {
            "method": "platt",
            "fit_role": "fusion_calibration",
            "probability_clip": config["calibration"]["probability_clip"],
            **variant.platt.details,
        },
        "source_versions": {track: config["sources"][track]["model_version"] for track in variant.tracks},
    }


def prediction_frames(
    tables: dict[str, pd.DataFrame], reference: np.ndarray, fitted: list[FittedVariant],
    stacker_version: str,
) -> dict[str, pd.DataFrame]:
    """Versioned prediction tables for fusion_calibration (raw stacker) and validation (all versions)."""
    calibration = tables["fusion_calibration"][["loan_id"]].assign(role="fusion_calibration")
    validation = tables["validation"][["loan_id"]].assign(role="validation", p_reference=reference)
    for variant in fitted:
        calibration[f"p_{variant.name}_raw"] = variant.fusion_calibration_raw
        validation[f"p_{variant.name}_raw"] = variant.validation_raw
        validation[f"p_{variant.name}_platt"] = variant.validation_platt
    return {
        "fusion_calibration": calibration.assign(stacker_version=stacker_version),
        "validation": validation.assign(stacker_version=stacker_version),
    }


def build_metrics(
    config: dict[str, Any], tables: dict[str, pd.DataFrame], versions: dict[str, np.ndarray],
    comparisons: dict[str, Any], fitted: list[FittedVariant], sources: list[Path],
) -> dict[str, Any]:
    y = tables["validation"]["target"].to_numpy()
    return {
        "stacker_version": config["stacker_version"],
        "fusion_development_version": config["fusion_development_version"],
        "status": config["status"],
        "variants_run": [variant.name for variant in fitted],
        "rows": {role: len(frame) for role, frame in tables.items()},
        "test_rows_read": 0,
        "validation_metrics": {name: calibrate.metric_row(y, p) for name, p in versions.items()},
        "comparisons": comparisons,
        "decision": {
            "status": config["evaluation"]["decision_rule_status"],
            "proposed_margin": config["evaluation"]["proposed_margin"],
            "note": "Proposed practical target only, not a significance test and not an approved rule.",
        },
        "artifacts": {variant.name: variant_artifact(variant, config) for variant in fitted},
        "inputs_sha256": provenance.file_checksums(sources),
        "commands": ["python -m src.fusion.stacker", "python -m pytest src/fusion -q"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "platform": platform.system(),
        },
    }


def results_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """One row per graded probability version on validation."""
    return pd.DataFrame([{"version": name, **row} for name, row in metrics["validation_metrics"].items()])


def source_paths(config: dict[str, Any], tracks: set[str], config_path: Path) -> list[Path]:
    """Every input file the run read, for the checksum record."""
    paths = [REPO_ROOT / config["role_mapping"]["path"], REPO_ROOT / config["reference"]["path"]]
    for track in sorted(tracks):
        source = config["sources"][track]
        keys = ("fusion_path", "validation_path") if "fusion_path" in source else ("path",)
        paths += [REPO_ROOT / source[key] for key in keys]
    return [*paths, config_path]


def write_outputs(
    metrics: dict[str, Any], frames: dict[str, pd.DataFrame], y: np.ndarray,
    versions: dict[str, np.ndarray], config: dict[str, Any], report_dir: Path, artifact_dir: Path,
) -> None:
    """Predictions and JSON artifacts under ``artifact_dir``; results, plot and metrics under ``report_dir``."""
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for role, frame in frames.items():
        path = artifact_dir / f"predictions_{role}.parquet"
        frame.to_parquet(path, index=False)
        outputs.append(path)
    for name, artifact in metrics["artifacts"].items():
        path = artifact_dir / f"{name}.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8", newline="\n")
        outputs.append(path)
    results_path = report_dir / "stacker_v1_results.csv"
    results_table(metrics).to_csv(results_path, index=False, lineterminator="\n")
    plot_path = report_dir / "stacker_v1_calibration.png"
    calibrate.plot_calibration(
        "Late fusion stacker v1", y, versions, config["calibration"]["plot_bins"],
        config["calibration"]["plot_strategy"], plot_path,
    )
    metrics["outputs_sha256"] = provenance.file_checksums([*outputs, results_path, plot_path])
    metrics_path = report_dir / "stacker_v1_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_stacker(
    config_path: str | Path = inputs.DEFAULT_STACKER_CONFIG,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Fit and grade every run variant, then write predictions, artifacts, metrics and the plot.

    Nothing is written unless every input check and fit passes.
    """
    config = inputs.load_stacker_config(config_path)
    variants = {name: config["variants"][name] for name in config["run"]["variants"]}
    tracks = sorted({track for variant_tracks in variants.values() for track in variant_tracks})
    tables = inputs.build_role_tables(config, tracks, data_dir=data_dir)
    reference_frame = inputs.load_reference(config, tables["validation"]["loan_id"])
    if "tabular" in tracks:
        inputs.check_reference_matches_tabular(reference_frame, tables["validation"])
    reference = reference_frame["p_reference"].to_numpy()

    fitted = [fit_variant(name, variant_tracks, tables, config) for name, variant_tracks in variants.items()]
    versions = validation_versions(reference, fitted)
    y = tables["validation"]["target"].to_numpy()
    comparisons = compare(y, versions, config["evaluation"])
    metrics = build_metrics(
        config, tables, versions, comparisons, fitted, source_paths(config, set(tracks), Path(config_path)),
    )
    frames = prediction_frames(tables, reference, fitted, config["stacker_version"])
    write_outputs(metrics, frames, y, versions, config, Path(report_dir), Path(artifact_dir))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and grade the late-fusion stacker variants.")
    parser.add_argument("--config", default=inputs.DEFAULT_STACKER_CONFIG, type=Path)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path)
    args = parser.parse_args()
    metrics = run_stacker(args.config, data_dir=args.data_dir)
    print(results_table(metrics)[["version", *METRICS]].to_string(index=False))


if __name__ == "__main__":
    main()

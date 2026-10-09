"""Reproduce training-only EDA for the approved lexical feature table.

The analysis joins ``text_lexical_features.parquet`` to the frozen manifest,
loads original text for the training split only, and writes aggregate evidence
to ``reports/nlp``.  Validation and test outcomes are deliberately excluded.

Run from the repository root with::

    python -m src.nlp.lexical_eda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pointbiserialr

from src.nlp.features_lexical import (
    DISTRESS_KEYWORDS,
    FEATURE_COLUMNS,
    tokenize_words,
)
from src.nlp.preprocess import audit_source_files, clean_text, load_original_split

CORRELATION_COLUMNS = (
    "feature",
    "point_biserial_r",
    "absolute_r",
    "p_value",
    "non_default_mean",
    "default_mean",
    "default_minus_non_default",
)

FEATURE_LABELS = {
    "flesch_reading_ease": "Reading ease",
    "flesch_kincaid_grade": "Reading grade",
    "lexical_diversity_ttr": "Lexical diversity",
    "sentiment_polarity": "Sentiment polarity",
    "sentiment_subjectivity": "Sentiment subjectivity",
    "financial_distress_keyword_density": "Distress keyword density",
}
OUTCOME_COLORS = {0: "#4C78A8", 1: "#E45756"}


def _validate_binary_target(rows: pd.DataFrame) -> None:
    if "target" not in rows:
        raise ValueError("Rows are missing target")
    if rows["target"].isna().any() or not rows["target"].isin([0, 1]).all():
        raise ValueError("target must be present and binary")
    if rows["target"].nunique() != 2:
        raise ValueError("Both target outcomes are required for correlation analysis")


def join_training_features(
    lexical_features: pd.DataFrame, training_rows: pd.DataFrame
) -> pd.DataFrame:
    """Validate and join the lexical table to frozen training labels/text."""
    feature_required = {"loan_id", "split", *FEATURE_COLUMNS}
    if missing := feature_required - set(lexical_features.columns):
        raise ValueError(f"Lexical feature table is missing columns: {sorted(missing)}")
    if lexical_features["loan_id"].isna().any() or lexical_features["loan_id"].duplicated().any():
        raise ValueError("Lexical feature loan_id values must be present and unique")
    if lexical_features[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Lexical feature table contains missing feature values")

    training_features = lexical_features.loc[
        lexical_features["split"].eq("train"), ["loan_id", *FEATURE_COLUMNS]
    ].copy()
    required_rows = {"loan_id", "split", "target", "title", "purpose", "desc", "text_payload"}
    if missing := required_rows - set(training_rows.columns):
        raise ValueError(f"Training rows are missing columns: {sorted(missing)}")
    if not training_rows["split"].eq("train").all():
        raise ValueError("Training rows must contain only the training split")
    if training_rows["loan_id"].isna().any() or training_rows["loan_id"].duplicated().any():
        raise ValueError("Training row loan_id values must be present and unique")
    _validate_binary_target(training_rows)

    feature_ids = set(training_features["loan_id"].astype(str))
    row_ids = set(training_rows["loan_id"].astype(str))
    if feature_ids != row_ids:
        raise ValueError("Lexical feature IDs do not exactly match frozen training IDs")

    joined = training_rows.merge(
        training_features,
        on="loan_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Joined training data contains missing lexical features")
    return joined


def compute_point_biserial_correlations(rows: pd.DataFrame) -> pd.DataFrame:
    """Rank lexical features by absolute point-biserial correlation."""
    _validate_binary_target(rows)
    if missing := set(FEATURE_COLUMNS) - set(rows.columns):
        raise ValueError(f"Rows are missing lexical features: {sorted(missing)}")

    records: list[dict[str, float | str]] = []
    non_default = rows.loc[rows["target"].eq(0)]
    default = rows.loc[rows["target"].eq(1)]
    for feature in FEATURE_COLUMNS:
        values = rows[feature].astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{feature} contains a non-finite value")
        if values.nunique() < 2:
            raise ValueError(f"{feature} is constant; point-biserial correlation is undefined")
        result = pointbiserialr(rows["target"].astype(int), values)
        r = float(result.statistic)
        records.append(
            {
                "feature": feature,
                "point_biserial_r": r,
                "absolute_r": abs(r),
                "p_value": float(result.pvalue),
                "non_default_mean": float(non_default[feature].mean()),
                "default_mean": float(default[feature].mean()),
                "default_minus_non_default": float(
                    default[feature].mean() - non_default[feature].mean()
                ),
            }
        )
    return pd.DataFrame(records, columns=CORRELATION_COLUMNS).sort_values(
        ["absolute_r", "feature"], ascending=[False, True], ignore_index=True
    )


def compute_outcome_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Return long-form distribution summaries for each feature and outcome."""
    _validate_binary_target(rows)
    if missing := set(FEATURE_COLUMNS) - set(rows.columns):
        raise ValueError(f"Rows are missing lexical features: {sorted(missing)}")

    records: list[dict[str, float | int | str]] = []
    labels = {0: "non_default", 1: "default"}
    for outcome, group in rows.groupby("target", sort=True):
        for feature in FEATURE_COLUMNS:
            values = group[feature].astype(float)
            records.append(
                {
                    "target": int(outcome),
                    "outcome": labels[int(outcome)],
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame.from_records(records)


def _word_and_distress_counts(text: object) -> tuple[int, int]:
    words = tokenize_words(clean_text(text))
    return len(words), sum(word in DISTRESS_KEYWORDS for word in words)


def add_keyword_source_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Attribute distress-keyword hits to purpose, title and description."""
    required = {"title", "purpose", "desc", "text_payload", "financial_distress_keyword_density"}
    if missing := required - set(rows.columns):
        raise ValueError(f"Rows are missing keyword-analysis columns: {sorted(missing)}")

    result = rows.copy()
    for source in ("title", "purpose", "desc"):
        counts = result[source].map(_word_and_distress_counts)
        result[f"{source}_word_count"] = counts.str[0].astype(int)
        result[f"{source}_distress_count"] = counts.str[1].astype(int)

    payload_counts = result["text_payload"].map(_word_and_distress_counts)
    result["payload_word_count"] = payload_counts.str[0].astype(int)
    result["payload_distress_count"] = payload_counts.str[1].astype(int)
    result["desc_distress_keyword_density"] = np.divide(
        result["desc_distress_count"],
        result["desc_word_count"],
        out=np.zeros(len(result), dtype=float),
        where=result["desc_word_count"].to_numpy() != 0,
    )
    recomputed_density = result["payload_distress_count"] / result["payload_word_count"]
    if not np.allclose(
        recomputed_density,
        result["financial_distress_keyword_density"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Recomputed payload distress density disagrees with the feature table")

    result["payload_has_distress"] = result["payload_distress_count"].gt(0)
    result["desc_has_distress"] = result["desc_distress_count"].gt(0)
    result["purpose_label_only_hit"] = result["purpose_distress_count"].gt(0) & result[
        "desc_distress_count"
    ].eq(0)
    return result


def compute_purpose_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reading, distress and default measures by loan purpose."""
    required = {
        "purpose",
        "target",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "financial_distress_keyword_density",
        "desc_distress_keyword_density",
        "payload_distress_count",
        "purpose_distress_count",
        "title_distress_count",
        "desc_distress_count",
        "payload_has_distress",
        "desc_has_distress",
        "purpose_label_only_hit",
    }
    if missing := required - set(rows.columns):
        raise ValueError(f"Rows are missing purpose-analysis columns: {sorted(missing)}")
    _validate_binary_target(rows)
    if rows["purpose"].isna().any() or rows["purpose"].astype(str).str.strip().eq("").any():
        raise ValueError("purpose must be present for every training row")

    grouped = rows.groupby("purpose", observed=True, sort=True)
    summary = grouped.agg(
        n_loans=("loan_id", "size"),
        defaults=("target", "sum"),
        empirical_default_rate=("target", "mean"),
        mean_reading_ease=("flesch_reading_ease", "mean"),
        mean_reading_grade=("flesch_kincaid_grade", "mean"),
        mean_distress_keyword_density=("financial_distress_keyword_density", "mean"),
        mean_desc_distress_keyword_density=("desc_distress_keyword_density", "mean"),
        payload_distress_mention_rate=("payload_has_distress", "mean"),
        desc_distress_mention_rate=("desc_has_distress", "mean"),
        purpose_label_only_hit_rate=("purpose_label_only_hit", "mean"),
        payload_distress_hits=("payload_distress_count", "sum"),
        purpose_label_distress_hits=("purpose_distress_count", "sum"),
        title_distress_hits=("title_distress_count", "sum"),
        description_distress_hits=("desc_distress_count", "sum"),
    ).reset_index()
    summary["purpose_label_share_of_payload_hits"] = np.divide(
        summary["purpose_label_distress_hits"],
        summary["payload_distress_hits"],
        out=np.zeros(len(summary), dtype=float),
        where=summary["payload_distress_hits"].to_numpy() != 0,
    )
    return summary.sort_values("purpose", ignore_index=True)


def build_metrics(
    rows: pd.DataFrame,
    correlations: pd.DataFrame,
    purpose_summary: pd.DataFrame,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Create compact machine-readable headline findings."""
    total_payload_hits = int(rows["payload_distress_count"].sum())
    total_purpose_hits = int(rows["purpose_distress_count"].sum())
    purpose_categories_with_label_hits = purpose_summary.loc[
        purpose_summary["purpose_label_distress_hits"].gt(0), "purpose"
    ].astype(str).tolist()
    medical = purpose_summary.loc[purpose_summary["purpose"].eq("medical")]
    medical_record = medical.iloc[0].to_dict() if len(medical) == 1 else None

    return {
        "analysis_version": "nlp-lexical-eda-v1",
        "analysis_split": "train",
        "training_rows": int(len(rows)),
        "defaults": int(rows["target"].sum()),
        "empirical_default_rate": float(rows["target"].mean()),
        "purpose_categories": int(rows["purpose"].nunique()),
        "strongest_feature": {
            "feature": str(correlations.iloc[0]["feature"]),
            "point_biserial_r": float(correlations.iloc[0]["point_biserial_r"]),
            "absolute_r": float(correlations.iloc[0]["absolute_r"]),
        },
        "keyword_artifact": {
            "payload_distress_mention_rate": float(rows["payload_has_distress"].mean()),
            "description_distress_mention_rate": float(rows["desc_has_distress"].mean()),
            "purpose_label_only_hit_rate": float(rows["purpose_label_only_hit"].mean()),
            "purpose_label_share_of_all_payload_hits": (
                float(total_purpose_hits / total_payload_hits) if total_payload_hits else 0.0
            ),
            "purpose_categories_with_distress_label_hits": purpose_categories_with_label_hits,
            "medical_purpose": medical_record,
        },
        "provenance": provenance,
    }


def _format_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _finish_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_correlation_figure(correlations: pd.DataFrame) -> plt.Figure:
    """Build the signed point-biserial correlation figure."""
    ordered = correlations.sort_values("point_biserial_r", ascending=True)
    colors = ["#E45756" if value > 0 else "#4C78A8" for value in ordered["point_biserial_r"]]
    labels = [FEATURE_LABELS[name] for name in ordered["feature"]]

    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.barh(labels, ordered["point_biserial_r"], color=colors)
    axis.axvline(0, color="#444444", linewidth=0.9)
    axis.set_xlabel("Point-biserial correlation with default")
    axis.set_title(
        "No lexical feature strongly separates the two outcomes",
        loc="left",
        weight="bold",
        y=1.10,
    )
    axis.text(
        0,
        1.03,
        "Positive values are higher among defaulted borrowers",
        transform=axis.transAxes,
        color="#555555",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, ordered["point_biserial_r"], strict=True):
        offset = 0.0006 if value >= 0 else -0.0006
        axis.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    bound = max(0.025, float(ordered["absolute_r"].max()) * 1.3)
    axis.set_xlim(-bound, bound)
    return figure


def plot_correlations(correlations: pd.DataFrame, output_path: str | Path) -> None:
    """Write the signed point-biserial correlation figure."""
    _finish_figure(build_correlation_figure(correlations), Path(output_path))


def build_outcome_means_figure(outcome_summary: pd.DataFrame) -> plt.Figure:
    """Build outcome means on each feature's natural scale."""
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    figure.suptitle("Defaulted borrowers differ only slightly on lexical averages", weight="bold")
    for axis, feature in zip(axes.flat, FEATURE_COLUMNS, strict=True):
        subset = outcome_summary.loc[outcome_summary["feature"].eq(feature)].sort_values("target")
        values = subset["mean"].to_numpy(copy=True)
        suffix = ""
        if feature == "financial_distress_keyword_density":
            values *= 100
            suffix = "% of words"
        axis.bar(
            ["Non-default", "Default"],
            values,
            color=[OUTCOME_COLORS[0], OUTCOME_COLORS[1]],
            width=0.65,
        )
        axis.set_title(FEATURE_LABELS[feature], fontsize=10, weight="bold")
        axis.set_ylabel(suffix)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_outcome_means(outcome_summary: pd.DataFrame, output_path: str | Path) -> None:
    """Write outcome means on each feature's natural scale."""
    _finish_figure(build_outcome_means_figure(outcome_summary), Path(output_path))


def build_purpose_profile_figure(purpose_summary: pd.DataFrame) -> plt.Figure:
    """Build the purpose comparison figure."""
    ordered = purpose_summary.sort_values("empirical_default_rate", ascending=True)
    y = np.arange(len(ordered))
    labels = ordered["purpose"].str.replace("_", " ")
    figure, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    figure.suptitle("Loan purposes differ more in default rate than in writing style", weight="bold")

    axes[0].barh(y, ordered["empirical_default_rate"] * 100, color="#E45756")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Default rate (%)")
    axes[0].set_title("Observed outcome", fontsize=10, weight="bold")

    axes[1].scatter(ordered["mean_reading_grade"], y, color="#4C78A8", s=48)
    axes[1].set_xlabel("Mean Flesch-Kincaid grade")
    axes[1].set_title("Reading grade", fontsize=10, weight="bold")

    axes[2].barh(y, ordered["mean_distress_keyword_density"] * 100, color="#F2CF5B")
    axes[2].set_xlabel("Mean distress density (% of words)")
    axes[2].set_title("Full text payload", fontsize=10, weight="bold")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[2].tick_params(axis="y", left=False, labelleft=False)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_purpose_profile(purpose_summary: pd.DataFrame, output_path: str | Path) -> None:
    """Write the purpose comparison figure."""
    _finish_figure(build_purpose_profile_figure(purpose_summary), Path(output_path))


def build_keyword_sources_figure(purpose_summary: pd.DataFrame) -> plt.Figure:
    """Build a figure attributing distress hits to their source fields."""
    ordered = purpose_summary.sort_values("payload_distress_hits", ascending=True).copy()
    total = ordered["payload_distress_hits"].replace(0, np.nan)
    purpose_share = (ordered["purpose_label_distress_hits"] / total).fillna(0)
    title_share = (ordered["title_distress_hits"] / total).fillna(0)
    desc_share = (ordered["description_distress_hits"] / total).fillna(0)
    y = np.arange(len(ordered))
    labels = ordered["purpose"].str.replace("_", " ")

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(y, purpose_share * 100, color="#E45756", label="Purpose label")
    axis.barh(y, title_share * 100, left=purpose_share * 100, color="#F2CF5B", label="Title")
    axis.barh(
        y,
        desc_share * 100,
        left=(purpose_share + title_share) * 100,
        color="#4C78A8",
        label="Description",
    )
    axis.set_yticks(y, labels)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of distress-keyword hits (%)")
    axis.set_title(
        "The medical purpose label supplies many of its keyword hits",
        loc="left",
        weight="bold",
        y=1.10,
    )
    axis.text(
        0,
        1.03,
        "Each bar splits all keyword hits in that purpose into their source field",
        transform=axis.transAxes,
        color="#555555",
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        ncol=3,
    )
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return figure


def plot_keyword_sources(purpose_summary: pd.DataFrame, output_path: str | Path) -> None:
    """Write a figure attributing distress hits to their source fields."""
    _finish_figure(build_keyword_sources_figure(purpose_summary), Path(output_path))


def create_plots(
    correlations: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    purpose_summary: pd.DataFrame,
    figure_dir: str | Path,
) -> None:
    """Write the four figures used by the report and consolidation notebook."""
    figure_dir = Path(figure_dir)
    plot_correlations(correlations, figure_dir / "lexical_correlations.png")
    plot_outcome_means(outcome_summary, figure_dir / "lexical_outcome_means.png")
    plot_purpose_profile(purpose_summary, figure_dir / "lexical_purpose_profile.png")
    plot_keyword_sources(purpose_summary, figure_dir / "lexical_keyword_sources.png")


def build_markdown_report(
    metrics: dict[str, Any], correlations: pd.DataFrame, purpose_summary: pd.DataFrame
) -> str:
    """Render the aggregate evidence as a concise review document."""
    correlation_rows = "\n".join(
        "| {feature} | {r} | {non_default} | {default} | {difference} |".format(
            feature=row.feature,
            r=_format_float(row.point_biserial_r),
            non_default=_format_float(row.non_default_mean),
            default=_format_float(row.default_mean),
            difference=_format_float(row.default_minus_non_default),
        )
        for row in correlations.itertuples(index=False)
    )
    purpose_rows = "\n".join(
        "| {purpose} | {n:,} | {ease} | {grade} | {density} | {rate} | {desc_rate} | {label_only} |".format(
            purpose=row.purpose,
            n=int(row.n_loans),
            ease=_format_float(row.mean_reading_ease, 2),
            grade=_format_float(row.mean_reading_grade, 2),
            density=_format_float(row.mean_distress_keyword_density, 5),
            rate=_format_float(row.empirical_default_rate),
            desc_rate=_format_float(row.desc_distress_mention_rate),
            label_only=_format_float(row.purpose_label_only_hit_rate),
        )
        for row in purpose_summary.itertuples(index=False)
    )
    artifact = metrics["keyword_artifact"]
    strongest = metrics["strongest_feature"]
    correlation_index = correlations.set_index("feature")
    grade_r = float(correlation_index.loc["flesch_kincaid_grade", "point_biserial_r"])
    distress_r = float(
        correlation_index.loc[
            "financial_distress_keyword_density", "point_biserial_r"
        ]
    )
    purpose_index = purpose_summary.set_index("purpose")
    small_business = purpose_index.loc["small_business"]
    debt_consolidation = purpose_index.loc["debt_consolidation"]
    highest_default = purpose_summary.loc[purpose_summary["empirical_default_rate"].idxmax()]
    medical = artifact.get("medical_purpose")
    if medical is None:
        medical_sentence = "No `medical` purpose row was present."
    else:
        medical_sentence = (
            "For `medical`, {label_only:.1%} of rows have a purpose-label hit with no "
            "distress keyword in `desc`; only {desc_rate:.1%} of descriptions contain a "
            "distress keyword."
        ).format(
            label_only=float(medical["purpose_label_only_hit_rate"]),
            desc_rate=float(medical["desc_distress_mention_rate"]),
        )

    return f"""# NLP Lexical Feature Correlation and EDA

**Analysis version:** `nlp-lexical-eda-v1`<br>
**Dataset version:** `{metrics['provenance'].get('dataset_version')}`<br>
**Split:** frozen training split only (`{metrics['training_rows']:,}` rows; `{metrics['defaults']:,}` defaults)<br>
**Test/validation outcomes used:** no

## Headline finding

The honest answer is that none of these features separates the two borrower outcomes well.
`{strongest['feature']}` ranks first, but its point-biserial correlation is only
`r = {strongest['point_biserial_r']:.4f}`. Reading grade (`r = {grade_r:.4f}`) is only slightly
stronger than distress-keyword density (`r = {distress_r:.4f}`). These are small shifts in group
averages, not useful standalone rules for predicting default.

![Point-biserial correlations](../reports/nlp/figures/lexical_correlations.png)

## Point-biserial correlations

Positive `r` means the feature is higher among defaulted borrowers. The full-precision table,
including p-values, is `reports/nlp/lexical_feature_correlations.csv`.

| Feature | r | Non-default mean | Default mean | Difference |
|---|---:|---:|---:|---:|
{correlation_rows}

Detailed count, standard deviation, quartile and range statistics for both outcomes are in
`reports/nlp/lexical_feature_outcome_summary.csv`.

![Outcome means by lexical feature](../reports/nlp/figures/lexical_outcome_means.png)

## Purpose-level aggregation

| Purpose | Loans | Mean reading ease | Mean reading grade | Mean distress density | Default rate | Desc distress mention rate | Purpose-label-only hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
{purpose_rows}

The contrast between `small_business` and `debt_consolidation` is a useful example.
`small_business` has a {float(small_business['empirical_default_rate']):.1%} empirical default
rate and a mean grade of {float(small_business['mean_reading_grade']):.2f}. For
`debt_consolidation`, the corresponding figures are
{float(debt_consolidation['empirical_default_rate']):.1%} and
{float(debt_consolidation['mean_reading_grade']):.2f}. Writing complexity is similar, while the
observed default rates are far apart. The purpose field is likely carrying differences in the
loans and borrowers that this simple table does not control for.

`{highest_default['purpose']}` has the highest purpose-level default rate at
{float(highest_default['empirical_default_rate']):.1%}. Treat that as a description of this
training sample, not as an estimate of what the purpose itself causes.

![Purpose-level profile](../reports/nlp/figures/lexical_purpose_profile.png)

## Is distress density a category-label artifact?

Yes, in one important case. Across training rows,
{artifact['purpose_label_share_of_all_payload_hits']:.1%} of all distress-keyword hits in the
full payload come directly from the `purpose` field. The only purpose label that contains one of
the configured distress words is {', '.join(f'`{x}`' for x in artifact['purpose_categories_with_distress_label_hits'])}.
{medical_sentence}

The full-payload density therefore mixes two ideas: what the borrower wrote and which category
the loan belongs to. For questions about borrower prose, use the description-only density and
mention-rate columns in `reports/nlp/lexical_purpose_summary.csv`. Title hits remain part of the
full payload, but the source chart keeps them separate from both purpose and description.

![Sources of distress-keyword hits](../reports/nlp/figures/lexical_keyword_sources.png)

## Method and limitations

- Point-biserial correlations are computed against the binary target on training rows only.
- Reading grade is `flesch_kincaid_grade`; distress density uses the fixed keyword set in
  `src/nlp/features_lexical.py`.
- Source-field keyword counts are recomputed and checked against the tracked full-payload density.
- P-values are included in the CSV. With more than 86,000 rows, a small p-value can accompany a
  correlation that has little practical value, so the effect size matters more here.
- LendingClub is an evidence-lane proxy, not a representative Singapore BNPL population.
- No multiple-testing correction, causal claim, or model-selection decision is made here.

## Reproduction

From the repository root, with the approved `data/raw/loan.csv` present:

```text
python -m src.nlp.lexical_eda
```
"""


def run_analysis(
    data_dir: str | Path,
    feature_path: str | Path,
    report_dir: str | Path,
    document_path: str | Path,
    *,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    """Run the complete training-only analysis and write aggregate artifacts."""
    data_dir = Path(data_dir)
    feature_path = Path(feature_path)
    report_dir = Path(report_dir)
    document_path = Path(document_path)

    provenance = audit_source_files(data_dir)
    lexical_features = pd.read_parquet(feature_path, engine="pyarrow")
    training_rows = load_original_split(data_dir, "train", chunksize=chunksize)
    joined = join_training_features(lexical_features, training_rows)
    enriched = add_keyword_source_metrics(joined)

    correlations = compute_point_biserial_correlations(enriched)
    outcome_summary = compute_outcome_summary(enriched)
    purpose_summary = compute_purpose_summary(enriched)
    if len(purpose_summary) != 14:
        raise ValueError(f"Expected 14 loan-purpose categories; found {len(purpose_summary)}")
    metrics = build_metrics(enriched, correlations, purpose_summary, provenance)

    report_dir.mkdir(parents=True, exist_ok=True)
    document_path.parent.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(report_dir / "lexical_feature_correlations.csv", index=False)
    outcome_summary.to_csv(report_dir / "lexical_feature_outcome_summary.csv", index=False)
    purpose_summary.to_csv(report_dir / "lexical_purpose_summary.csv", index=False)
    create_plots(correlations, outcome_summary, purpose_summary, report_dir / "figures")
    with (report_dir / "lexical_eda_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    document_path.write_text(
        build_markdown_report(metrics, correlations, purpose_summary), encoding="utf-8"
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument(
        "--features",
        type=Path,
        default=root / "data" / "nlp" / "text_lexical_features.parquet",
    )
    parser.add_argument("--report-dir", type=Path, default=root / "reports" / "nlp")
    parser.add_argument(
        "--document",
        type=Path,
        default=root / "docs" / "nlp-lexical-eda-report.md",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = run_analysis(
        args.data_dir,
        args.features,
        args.report_dir,
        args.document,
        chunksize=args.chunksize,
    )
    strongest = metrics["strongest_feature"]
    print(
        f"Analyzed {metrics['training_rows']:,} training rows across "
        f"{metrics['purpose_categories']} purposes; strongest feature: "
        f"{strongest['feature']} (r={strongest['point_biserial_r']:.4f})"
    )


if __name__ == "__main__":
    main()

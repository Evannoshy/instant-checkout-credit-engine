"""Draw figures from the recorded four-model validation predictions and intervals."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/tabular"
PLOTS = OUT / "figures"
LABELS = {"xgb_tabular": "XGB: tabular", "xgb_distress": "XGB: + distress", "xgb_lexical": "XGB: + lexical", "lightgbm_lexical": "LightGBM: + lexical"}
COLORS = ["#315d8c", "#9c6045", "#13817d", "#7f56a1"]


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(OUT / "expanded_lexical_validation_predictions.csv")
    deltas = pd.read_csv(OUT / "expanded_lexical_deltas.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, label), color in zip(LABELS.items(), COLORS):
        x, y, _ = roc_curve(data.target, data[name]); axes[0].plot(x, y, label=label, color=color, lw=1.7)
        precision, recall, _ = precision_recall_curve(data.target, data[name]); axes[1].plot(recall, precision, label=label, color=color, lw=1.5)
    axes[0].plot([0, 1], [0, 1], color="#888888", ls="--", lw=1)
    axes[1].axhline(data.target.mean(), color="#888888", ls="--", lw=1, label="Default prevalence")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves", ylim=(0, 1))
    axes[0].legend(fontsize=8, loc="lower right"); axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Validation discrimination: 21,784 loans", fontsize=14, fontweight="bold"); fig.tight_layout(); fig.savefig(PLOTS / "expanded_discrimination.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, label), color in zip(LABELS.items(), COLORS):
        fraction, mean = calibration_curve(data.target, data[name], n_bins=10, strategy="quantile")
        ax.plot(mean, fraction, marker="o", markersize=3, color=color, lw=1.5, label=label)
    ax.plot([0, 0.6], [0, 0.6], "--", color="#888888", lw=1, label="Ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Validation reliability by prediction decile")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(PLOTS / "expanded_calibration.png", bbox_inches="tight"); plt.close(fig)
    order = [("xgb_tabular", "xgb_distress", "+ distress vs tabular"), ("xgb_tabular", "xgb_lexical", "+ lexical vs tabular"), ("xgb_distress", "xgb_lexical", "+ lexical vs distress"), ("xgb_lexical", "lightgbm_lexical", "LightGBM vs XGB, lexical")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, metric in zip(axes.flat, ["pr_auc", "roc_auc", "brier_score", "log_loss"]):
        sub = deltas[deltas.metric.eq(metric)]
        for i, (ref, candidate, label) in enumerate(order):
            row = sub[(sub.reference == ref) & (sub.candidate == candidate)].iloc[0]
            ax.errorbar(row.delta, i, xerr=[[row.delta-row.ci_low], [row.ci_high-row.delta]], fmt="o", capsize=3, color="#315d8c", markersize=4)
        ax.axvline(0, ls="--", color="#666666", lw=1)
        ax.set_yticks(range(4), [x[2] for x in order], fontsize=8); ax.invert_yaxis(); ax.set_title(metric.replace("_", " ").upper()); ax.set_xlabel("Candidate minus reference (95% paired CI)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Validation metric differences and uncertainty", fontsize=14, fontweight="bold"); fig.tight_layout(); fig.savefig(PLOTS / "expanded_paired_differences.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()

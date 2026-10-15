"""Create reproducible 300-dpi PNG and vector PDF research figures.

Input artifacts must come from experiments already evaluated; this script does
not fit models or change metrics. Missing optional artifacts are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = {"droughtcast": "#0072B2", "persistence": "#E69F00",
           "seasonal_climatology": "#009E73", "ridge": "#CC79A7",
           "hist_gradient_boosting": "#D55E00"}
SCALES = ["spei_3", "spei_6", "spei_12"]


def save(fig, folder: Path, stem: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def metric_figures(summary_path: Path, out: Path):
    table = pd.read_csv(summary_path)
    required = {"method", "scale", "lead", "rmse_mean", "rmse_std"}
    if not required.issubset(table):
        raise ValueError(f"Summary table missing fields: {sorted(required - set(table))}")
    table = table[table.method.isin(PALETTE)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, scale in zip(axes, SCALES):
        subset = table[table.scale == scale]
        leads = [f"T+{i}" for i in range(1, 5)]
        for method in PALETTE:
            rows = subset[subset.method == method].set_index("lead").reindex(leads)
            if rows.rmse_mean.notna().any():
                ax.errorbar(range(1, 5), rows.rmse_mean, yerr=rows.rmse_std,
                            marker="o", capsize=3, color=PALETTE[method], label=method.replace("_", " "))
        ax.set(title=scale.upper(), xlabel="Forecast lead", xticks=range(1, 5),
               xticklabels=leads, ylabel="RMSE")
        ax.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Test RMSE by SPEI accumulation scale and forecast lead\n"
                 "NASA POWER/NOAA derived targets; test period 2017–2024; mean ± SD across available seeds")
    save(fig, out, "test_rmse_by_scale_lead")


def interpretation_figures(path: Path, out: Path):
    if not path.exists():
        print(f"Skip interpretation figures: {path} not found")
        return
    info = json.loads(path.read_text(encoding="utf-8"))
    attention = np.asarray(info["attention"])
    if attention.ndim == 3 and attention.shape[-1] == 1:
        attention = attention[..., 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(attention.mean(axis=0)[None, :], cmap="cividis", vmin=0,
                xticklabels=[f"T-{attention.shape[1]-i}" for i in range(attention.shape[1])],
                yticklabels=["Shared attention"], ax=ax, cbar_kws={"label": "Mean attention weight"})
    ax.set_title("Mean temporal attention across sampled held-out windows")
    ax.set_xlabel("Input month")
    fig.tight_layout()
    save(fig, out, "attention_summary")
    names = info["feature_names"]
    scales = info["integrated_gradients"]
    matrix = np.stack([np.abs(np.asarray(scales[s])).mean(axis=(0, 1)) for s in SCALES])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(matrix, cmap="cividis", xticklabels=names, yticklabels=[s.upper() for s in SCALES],
                ax=ax, cbar_kws={"label": "Mean absolute Integrated Gradients"})
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Input attribution by SPEI scale (model association, not causal effect)")
    fig.tight_layout()
    save(fig, out, "integrated_gradients_by_scale")


def probabilistic_figures(paths: list[Path], out: Path):
    rows = []
    for path in paths:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for scale, leads in report.get("probabilistic", {}).get("scales", {}).items():
            for lead, values in leads.items():
                rows.append({"scale": scale, "lead": lead,
                             "before": values["before"]["coverage"],
                             "after": values["after"]["coverage"]})
    if not rows:
        print("Skip coverage figure: no probabilistic reports found")
        return
    frame = pd.DataFrame(rows).groupby(["scale", "lead"], as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=frame.melt(["scale", "lead"], var_name="stage", value_name="coverage"),
                 x="lead", y="coverage", hue="scale", style="stage", markers=True, dashes=True, ax=ax)
    ax.axhline(.8, color="#D55E00", linestyle="--", label="Nominal 80%")
    ax.set_ylim(0, 1)
    ax.set_title("Held-out interval coverage before and after validation CQR")
    ax.set_ylabel("Empirical coverage")
    fig.tight_layout()
    save(fig, out, "interval_coverage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--interpretability", type=Path, default=Path("results/interpretability.json"))
    parser.add_argument("--probabilistic", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    if args.summary.exists():
        metric_figures(args.summary, args.out)
    else:
        print(f"Skip metric figures: {args.summary} not found")
    interpretation_figures(args.interpretability, args.out)
    probabilistic_figures(args.probabilistic, args.out)
    print(f"Figures are in {args.out}")


if __name__ == "__main__":
    main()

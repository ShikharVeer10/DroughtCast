"""Build paper figures only from checked DroughtCast configs and result files.

Run `python make_paper_figures.py --check` to audit the ten figure inputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import yaml

from paper.figure_style import OKABE_ITO, apply_style

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "figures"
DATA = OUT / "data"


def _csv_has(path: Path, columns: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        return columns.issubset(set(pd.read_csv(path, nrows=0).columns))
    except Exception:
        return False


def missing_inputs() -> dict[str, list[str]]:
    """Return absent or schema-inadequate source files by figure."""
    missing: dict[str, list[str]] = {}
    config = ROOT / "configs" / "base_diagram.yaml"
    if not config.is_file():
        missing["F1"] = [str(config.relative_to(ROOT))]
    required = {
        "F2": [("results/site_skill_t4.csv", {"site", "scale", "latitude", "longitude", "skill_vs_persistence"})],
        "F3": [("results/per_seed.csv", {"seed", "method", "scale", "lead", "skill_vs_persistence"})],
        "F4": [("results/case_study_predictions.csv", {"site", "date", "scale", "lead", "observed", "forecast", "lower", "upper"})],
        "F5": [("results/reliability_and_coverage.csv", {"model", "scale", "lead", "bin_mean_probability", "observed_frequency", "nominal_coverage", "empirical_coverage"})],
        "F6": [("results/drought_event_bootstrap.csv", {"scale", "lead", "threshold", "metric", "estimate", "lower", "upper"})],
        "F7": [("results/ablation_forest.csv", {"variant", "delta_rmse", "lower", "upper"})],
        "F8": [("results/attention_occlusion.csv", {"scale", "lead", "season", "input_step", "attention", "occlusion_delta_validation_mse"})],
        "F9": [
            ("results/speibase_validation_points.csv", {"site", "date", "scale", "computed_spei", "speibase_spei"}),
            ("results/speibase_validation.csv", {"site", "scale", "correlation", "rmse", "bias"}),
        ],
        "F10": [("results/leave_region_out_skill.csv", {"heldout_region", "scale", "lead", "skill_vs_persistence"})],
    }
    if not config.is_file():
        missing["F2"] = missing.get("F2", []) + ["configs/config.yaml (site coordinates)"]
    if not importlib.util.find_spec("cartopy"):
        try:
            import geopandas as gpd
            gpd.datasets.get_path("naturalearth_lowres")
        except Exception:
            missing.setdefault("F2", []).append(
                "Natural Earth coastlines (Cartopy is unavailable and no local Natural Earth dataset is installed)"
            )
    for figure, sources in required.items():
        bad = [name for name, columns in sources if not _csv_has(ROOT / name, columns)]
        if figure == "F3" and not bad:
            frame = pd.read_csv(ROOT / sources[0][0])
            group_counts = frame.groupby(["method", "scale", "lead"]).seed.nunique()
            if group_counts.empty or (group_counts < 5).any():
                bad.append("results/per_seed.csv (requires five seeds for every method, scale, and forecast lead)")
        if figure == "F7" and not bad:
            frame = pd.read_csv(ROOT / sources[0][0])
            if "base_diagram" not in set(frame.variant):
                bad.append("results/ablation_forest.csv (missing base_diagram reference row)")
        if figure == "F10" and not bad:
            frame = pd.read_csv(ROOT / sources[0][0])
            if frame.heldout_region.nunique() < 2:
                bad.append("results/leave_region_out_skill.csv (requires multiple held-out regions)")
        if bad:
            missing[figure] = bad
    run_manifest = ROOT / "results" / "run_manifest.json"
    expected_alignment = (
        "ONI/DMI/NAO/PDO are shifted one month: the feature at month m uses the value published for m-1; "
        "centered ONI at m-1 contains no month m or later"
    )
    try:
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        current = manifest.get("protocol_version") == 1 and manifest.get("dataset_schema_version") == 2 \
            and manifest.get("index_alignment") == expected_alignment
        if current and "F3" not in missing:
            seed_frame = pd.read_csv(ROOT / "results/per_seed.csv")
            if not set(manifest.get("seeds", [])).issubset(set(seed_frame.seed.unique())):
                missing.setdefault("F3", []).append(
                    "results/per_seed.csv (seed rows do not match results/run_manifest.json)"
                )
    except (OSError, ValueError):
        current = False
    if not current:
        try:
            evaluation = json.loads((ROOT / "results/evaluation_manifest.json").read_text(encoding="utf-8"))
            current_evaluation = evaluation.get("protocol_version") == 1 \
                and evaluation.get("dataset_schema_version") == 2 \
                and evaluation.get("index_alignment") == expected_alignment
        except (OSError, ValueError):
            current_evaluation = False
        if not current_evaluation:
            missing.setdefault("F2", []).append(
                "results/evaluation_manifest.json (current evaluation from dataset schema v2 is required)"
            )
        for figure in (f"F{i}" for i in range(3, 11)):
            missing.setdefault(figure, []).append(
                "results/run_manifest.json (current evaluation from dataset schema v2 is required)"
            )
    return missing


def _save(fig, name: str, data: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    data.to_csv(DATA / f"{name}.csv", index=False)
    plt.close(fig)


def figure_f1():
    config_path = ROOT / "configs" / "base_diagram.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Required architecture config is missing: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seq_len = int(cfg["data"]["seq_len"])
    forecast = int(cfg["data"]["forecast_steps"])
    model = cfg["model"]
    if (seq_len, int(model["hidden_dim_1"]), int(model["hidden_dim_2"]), forecast,
        len(cfg["data"]["features"])) != (6, 64, 32, 4, 12):
        raise ValueError("F1 requires base_diagram.yaml to specify 6 steps, 64-to-32 LSTM, and four forecast leads")
    if model.get("attention") != "shared" or model.get("heads") != "direct" or cfg["training"].get("loss") != "mse":
        raise ValueError("F1 requires shared attention, direct point heads, and plain MSE in base_diagram.yaml")
    phases = [
        ("Phase 1: Inputs", f"6 monthly steps\n{len(cfg['data']['features'])} climate, index, and\nobserved SPEI features"),
        ("Phase 2: Encoder", "LSTM(64) -> Dropout(0.2)\n-> LSTM(32)\n[B, 6, 32]"),
        ("Phase 3: Attention", "Additive attention\nover 6 steps\n[B, 6, 1]"),
        ("Phase 4: Context", "One shared context\nc in R^32"),
        ("Phase 5: Heads", "SPEI-3 / SPEI-6 / SPEI-12\naccumulation-scale heads\nT+1 through T+4"),
        ("Phase 6: Output", "Point output\n[B, 3, 4]\nplain MSE"),
    ]
    rows = []
    for i, (phase, description) in enumerate(phases):
        rows.append({"phase": i + 1, "component": phase.split(": ", 1)[1],
                     "description": description.replace("\n", "; "), "order": i})
    fig, ax = plt.subplots(figsize=(12.6, 2.7))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = [OKABE_ITO["sky_blue"], OKABE_ITO["blue"], OKABE_ITO["bluish_green"],
              OKABE_ITO["orange"], OKABE_ITO["reddish_purple"], OKABE_ITO["vermillion"]]
    for i, (title, description) in enumerate(phases):
        x = i + 0.08
        rect = plt.Rectangle((x, 0.20), 0.82, 0.62, facecolor=colors[i],
                             edgecolor=OKABE_ITO["black"], linewidth=0.7, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x + 0.41, 0.70, title, ha="center", va="center", fontsize=8.2, weight="bold")
        ax.text(x + 0.41, 0.45, description, ha="center", va="center", fontsize=7.1, linespacing=1.25)
        if i < 5:
            ax.annotate("", xy=(x + 1.00, 0.51), xytext=(x + 0.84, 0.51),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": OKABE_ITO["black"]})
    data = pd.DataFrame(rows)
    _save(fig, "F1_architecture", data)


def _load_csv(name: str) -> pd.DataFrame:
    path = ROOT / "results" / name
    if not path.is_file():
        raise FileNotFoundError(f"Required paper figure input is missing: {path}")
    return pd.read_csv(path)


def figure_f2():
    data = _load_csv("site_skill_t4.csv")
    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    coords = pd.DataFrame([{"site": name, "latitude": spec["lat"], "longitude": spec["lon"]}
                           for name, spec in cfg["sites"].items()])
    data = data.merge(coords, on="site", how="left", validate="one_to_one")
    if data[["latitude", "longitude", "skill_vs_persistence"]].isna().any().any():
        raise ValueError("F2 requires configured coordinates and finite T+4 skill for every site")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    scales = ["spei_3", "spei_6", "spei_12"]
    try:
        import cartopy.crs as ccrs
        for ax in axes:
            ax.remove()
        axes = [fig.add_subplot(1, 3, i + 1, projection=ccrs.PlateCarree()) for i in range(3)]
        for ax in axes:
            ax.coastlines(resolution="110m", linewidth=0.5)
            ax.set_global()
        transform = ccrs.PlateCarree()
    except ImportError:
        for ax in axes:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 85)
            ax.set_xlabel("Longitude (degrees east)")
            ax.grid(alpha=0.2, linewidth=0.4)
        axes[0].set_ylabel("Latitude (degrees north)")
        transform = None
        try:
            import geopandas as gpd
            natural_earth = gpd.datasets.get_path("naturalearth_lowres")
            for ax in axes:
                gpd.read_file(natural_earth).boundary.plot(ax=ax, color="#777777", linewidth=0.35)
        except Exception as exc:
            raise FileNotFoundError("F2 requires Cartopy coastlines or local Natural Earth coastline data") from exc
    points = None
    for ax, scale in zip(axes, scales):
        subset = data[data.scale == scale]
        skill_map = LinearSegmentedColormap.from_list(
            "okabe_ito_skill", [OKABE_ITO["blue"], OKABE_ITO["yellow"], OKABE_ITO["vermillion"]])
        scatter_kw = {"c": subset.skill_vs_persistence, "cmap": skill_map, "s": 45,
                      "edgecolors": "black", "linewidths": 0.4}
        if transform is not None:
            scatter_kw["transform"] = transform
        points = ax.scatter(subset.longitude, subset.latitude, **scatter_kw)
        for row in subset.itertuples():
            ax.annotate(row.site.replace("_", " "), (row.longitude, row.latitude), xytext=(3, 2),
                        textcoords="offset points", fontsize=5.5,
                        **({"transform": transform._as_mpl_transform(ax)} if transform is not None else {}))
        ax.text(0.03, 0.97, scale.replace("_", "-"), transform=ax.transAxes,
                ha="left", va="top", fontsize=8, weight="bold")
    fig.colorbar(points, ax=axes, label="Skill vs persistence at T+4 (unitless)")
    _save(fig, "F2_site_skill", data)


def figure_f3():
    source = _load_csv("per_seed.csv")
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    scales = ["spei_3", "spei_6", "spei_12"]
    methods = sorted(source.method.unique())
    colors = [OKABE_ITO[k] for k in ("blue", "orange", "bluish_green", "vermillion", "reddish_purple", "sky_blue", "black")]
    for ax, scale in zip(axes, scales):
        for mi, method in enumerate(methods):
            sub = source[(source.scale == scale) & (source.method == method)]
            means, lows, highs = [], [], []
            for lead in range(1, 5):
                vals = sub.loc[sub.lead == f"T+{lead}", "skill_vs_persistence"].dropna().to_numpy()
                if len(vals) < 5:
                    raise ValueError(f"F3 needs five seeds for {method}/{scale}/T+{lead}; got {len(vals)}")
                mean = vals.mean()
                se = vals.std(ddof=1) / np.sqrt(len(vals))
                from scipy.stats import t
                half = float(t.ppf(0.975, len(vals) - 1) * se)
                means.append(mean); lows.append(mean - half); highs.append(mean + half)
                rows.append({"method": method, "scale": scale, "lead": f"T+{lead}", "n_seeds": len(vals),
                             "skill_mean": mean, "ci95_lower": mean - half, "ci95_upper": mean + half})
            x = np.arange(1, 5)
            ax.plot(x, means, marker="o", linewidth=1.1, label=method,
                    color=colors[mi % len(colors)])
            ax.fill_between(x, lows, highs, alpha=0.14, color=colors[mi % len(colors)])
        ax.axhline(0, color="#444444", lw=0.7, ls="--")
        ax.text(0.03, 0.97, scale.replace("_", "-"), transform=ax.transAxes,
                ha="left", va="top", fontsize=8, weight="bold")
        ax.set_xticks(range(1, 5), [f"T+{i}" for i in range(1, 5)])
        ax.set_xlabel("Forecast horizon (months)")
    axes[0].set_ylabel("Skill vs persistence (unitless)")
    axes[-1].legend(frameon=False, loc="best")
    _save(fig, "F3_skill_by_lead", pd.DataFrame(rows))


def figure_f4():
    data = _load_csv("case_study_predictions.csv")
    episodes = list(data.episode.dropna().unique())
    scales = list(data.scale.dropna().unique())
    leads = sorted(data.lead.unique())
    fig, axes = plt.subplots(len(episodes), len(scales), figsize=(3.2 * len(scales), 2.4 * len(episodes)),
                             squeeze=False, sharex="col")
    for i, episode in enumerate(episodes):
        for j, scale in enumerate(scales):
            ax = axes[i, j]
            subset = data[(data.episode == episode) & (data.scale == scale)]
            observed = subset.sort_values("date").drop_duplicates("date")
            ax.plot(pd.to_datetime(observed.date), observed.observed, color=OKABE_ITO["black"],
                    lw=1.0, label="Observed")
            for k, lead in enumerate(leads):
                line = subset[subset.lead == lead].sort_values("date")
                color = OKABE_ITO[["blue", "orange", "bluish_green", "vermillion"][k % 4]]
                ax.plot(pd.to_datetime(line.date), line.forecast, color=color, lw=0.9, label=f"T+{lead}")
                ax.fill_between(pd.to_datetime(line.date), line.lower, line.upper, color=color, alpha=0.12)
            ax.axhline(-1, color="#777777", lw=0.5, ls="--")
            ax.set_ylabel("SPEI (unitless)")
            ax.set_xlabel("Target month")
            ax.text(0.03, 0.97, f"{episode} | {scale.replace('_', '-')}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7)
            ax.legend(frameon=False, fontsize=6, ncol=2)
    _save(fig, "F4_case_studies", data)


def figure_f5():
    data = _load_csv("reliability_and_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for i, (scale, sub) in enumerate(data.groupby("scale", sort=True)):
        for model, model_data in sub.groupby("model", sort=True):
            rel = model_data.dropna(subset=["bin_mean_probability", "observed_frequency"])
            axes[0].plot(rel.bin_mean_probability, rel.observed_frequency, marker="o", ms=3,
                         color=OKABE_ITO[["blue", "orange", "bluish_green", "vermillion"][i % 4]],
                         label=f"{scale.replace('_', '-')} {model}")
        cov = sub.drop_duplicates(["model", "lead"])
        for model, model_data in cov.groupby("model", sort=True):
            axes[1].plot(model_data.lead, model_data.empirical_coverage, marker="o", ms=3,
                         label=f"{scale.replace('_', '-')} {model}")
    axes[0].plot([0, 1], [0, 1], color="#555555", ls="--", lw=0.7)
    axes[0].set(xlabel="Forecast event probability (unitless)", ylabel="Observed event frequency (unitless)")
    axes[1].set(xlabel="Forecast horizon (months)", ylabel="Empirical interval coverage (unitless)")
    axes[1].axhline(float(data.nominal_coverage.iloc[0]), color="#555555", ls="--", lw=0.7)
    for ax in axes:
        ax.legend(frameon=False, fontsize=6)
    _save(fig, "F5_reliability_coverage", data)


def figure_f6():
    data = _load_csv("drought_event_bootstrap.csv")
    metrics = ["POD", "FAR", "CSI"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, metric in zip(axes, metrics):
        sub = data[data.metric.str.upper() == metric]
        for scale, scale_data in sub.groupby("scale", sort=True):
            scale_data = scale_data.sort_values(["lead", "threshold"])
            x = np.arange(len(scale_data))
            yerr = np.vstack([scale_data.estimate - scale_data.lower,
                              scale_data.upper - scale_data.estimate])
            ax.errorbar(x, scale_data.estimate, yerr=yerr, marker="o", ms=2.5, lw=0.8,
                        capsize=2, label=scale.replace("_", "-"))
        ax.set_ylabel(f"{metric} (unitless)")
        ax.set_xlabel("Threshold and forecast lead")
        tick_data = sub.drop_duplicates(["threshold", "lead"]).sort_values(["lead", "threshold"])
        labels = [f"{row.threshold:g},T{str(row.lead).replace('T+', '')}"
                  for row in tick_data.itertuples()]
        ax.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=6)
    axes[-1].legend(frameon=False)
    _save(fig, "F6_drought_events", data)


def figure_f7():
    data = _load_csv("ablation_forest.csv")
    data = data[data.variant != "base_diagram"].reset_index(drop=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.4, max(2.8, 0.28 * len(data))))
    ax.errorbar(data.delta_rmse, y,
                xerr=np.vstack([data.delta_rmse - data.lower, data.upper - data.delta_rmse]),
                fmt="o", color=OKABE_ITO["blue"], ecolor="#555555", capsize=2)
    ax.axvline(0, color=OKABE_ITO["black"], lw=0.7, ls="--")
    ax.set_yticks(y, data.variant)
    ax.set_xlabel("RMSE difference from base_diagram (SPEI units)")
    _save(fig, "F7_ablation_forest", data)


def figure_f8():
    data = _load_csv("attention_occlusion.csv")
    data = data.sort_values(["scale", "lead", "season", "input_step"])
    groups = data[["scale", "lead", "season"]].drop_duplicates().itertuples(index=False, name=None)
    groups = list(groups)
    matrix = []
    labels = []
    for scale, lead, season in groups:
        row = data[(data.scale == scale) & (data.lead == lead) & (data.season == season)].sort_values("input_step")
        matrix.append(row.attention.to_numpy())
        labels.append(f"{scale.replace('_', '-')} T+{lead} {season}")
    fig, axes = plt.subplots(1, 2, figsize=(9, max(3.2, 0.22 * len(matrix))))
    im = axes[0].imshow(np.asarray(matrix), aspect="auto", cmap="viridis")
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=6)
    axes[0].set_xlabel("Input month step (months before forecast origin)")
    axes[0].set_ylabel("Scale, forecast horizon, season")
    fig.colorbar(im, ax=axes[0], fraction=0.045)
    occ = data.groupby("input_step", as_index=False).occlusion_delta_validation_mse.mean()
    axes[1].bar(occ.input_step, occ.occlusion_delta_validation_mse, color=OKABE_ITO["orange"])
    axes[1].set_xlabel("Occluded input step")
    axes[1].set_ylabel("Change in validation MSE (SPEI squared)")
    _save(fig, "F8_attention_occlusion", data)


def figure_f9():
    data = _load_csv("speibase_validation_points.csv")
    metrics = _load_csv("speibase_validation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    for ax, scale in zip(axes, [3, 6, 12]):
        sub = data[data.scale == scale]
        if sub.empty:
            raise ValueError(f"F9 input has no validation pairs for SPEI-{scale}")
        ax.scatter(sub.speibase_spei, sub.computed_spei, s=5, alpha=0.25,
                   color=OKABE_ITO["blue"], edgecolors="none")
        lo = min(sub.speibase_spei.min(), sub.computed_spei.min())
        hi = max(sub.speibase_spei.max(), sub.computed_spei.max())
        ax.plot([lo, hi], [lo, hi], color=OKABE_ITO["vermillion"], ls="--", lw=0.8)
        ax.text(0.04, 0.96, f"SPEI-{scale}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, weight="bold")
        metric = metrics[metrics.scale == scale]
        if metric.empty:
            raise ValueError(f"F9 validation summary is missing SPEI-{scale}")
        ax.text(0.04, 0.87,
                f"r={metric.correlation.mean():.2f}\nRMSE={metric.rmse.mean():.2f}\nBias={metric.bias.mean():.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
        ax.set_xlabel("SPEIbase SPEI (unitless)")
    axes[0].set_ylabel("Computed SPEI (unitless)")
    data = data.merge(metrics, on=["site", "scale"], how="left", validate="many_to_one")
    _save(fig, "F9_speibase_validation", data)


def figure_f10():
    data = _load_csv("leave_region_out_skill.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    regions = list(data.heldout_region.unique())
    for ax, scale in zip(axes, ["spei_3", "spei_6", "spei_12"]):
        sub = data[data.scale == scale]
        for lead, lead_data in sub.groupby("lead", sort=True):
            lead_data = lead_data.set_index("heldout_region").reindex(regions).reset_index()
            ax.plot(lead_data.heldout_region, lead_data.skill_vs_persistence, marker="o",
                    label=lead, color=OKABE_ITO[["blue", "orange", "bluish_green", "vermillion"][int(str(lead).replace("T+", "")) - 1]])
        ax.text(0.03, 0.97, scale.replace("_", "-"), transform=ax.transAxes,
                ha="left", va="top", fontsize=8, weight="bold")
        ax.set_xticks(np.arange(len(regions)), regions, rotation=45, ha="right")
        ax.set_xlabel("Held-out region")
    axes[0].set_ylabel("Skill vs persistence (unitless)")
    axes[-1].legend(frameon=False)
    _save(fig, "F10_leave_region_out", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="List missing or insufficient inputs for F1–F10")
    parser.add_argument("--figure", choices=[f"F{i}" for i in range(1, 11)],
                        help="Generate one figure (default: generate available figures, then report blockers)")
    args = parser.parse_args()
    apply_style()
    missing = missing_inputs()
    if args.check:
        print("F1: config-driven architecture diagram" if "F1" not in missing else f"F1: missing {missing['F1']}")
        for i in range(2, 11):
            name = f"F{i}"
            print(f"{name}: missing " + "; ".join(missing.get(name, [])) if name in missing else f"{name}: ready")
        return 1 if missing else 0
    plotters = {"F1": figure_f1, "F2": figure_f2, "F3": figure_f3, "F4": figure_f4,
                "F5": figure_f5, "F6": figure_f6, "F7": figure_f7, "F8": figure_f8,
                "F9": figure_f9, "F10": figure_f10}
    requested = [args.figure] if args.figure else [f"F{i}" for i in range(1, 11)]
    errors = []
    for name in requested:
        if name in missing:
            errors.append(f"{name}: missing required input(s): " + "; ".join(missing[name]))
            continue
        try:
            plotters[name]()
            print(f"Wrote {name} PDF, 300-dpi PNG, and plotted data under {OUT.relative_to(ROOT)}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise FileNotFoundError("Paper figure generation blocked:\n" + "\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sqlite3
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path as MplPath
from shapely import wkb
from shapely.geometry import Point, box

from paper.figure_style import (
    OKABE_ITO,
    SINGLE_COLUMN_IN,
    apply_style,
    audit_figure,
    save_figure,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "figures"
DATA = OUT / "data"
PROVENANCE = OUT / "provenance"
QUALITY_RESULTS = []


def _safe_print(value):
    """Keep diagnostics printable in legacy Windows terminals."""
    print(str(value).encode("ascii", "backslashreplace").decode("ascii"))


RESULT_COMMANDS = {
    "G4": "python train.py --config configs/base_diagram.yaml --seed SEED --checkpoint results/SEED/model.pt",
    "G5,G6,G18": "python evaluate.py --config configs/base_diagram.yaml --checkpoint CHECKPOINT --output results/metrics_seed_SEED.json",
    "G7": "python -m data.spatial_inference --config configs/spatial_config.yaml (requires downloaded real spatial inputs)",
    "G8": "python evaluate.py --config configs/base_diagram.yaml --checkpoint CHECKPOINT --output results/metrics_seed_SEED.json",
    "G10,T3": "python -m experiments.summarize_runs SEED... (requires current five-seed evaluation reports)",
    "G12,G13": "python experiments/export_interpretability.py --config configs/base_diagram.yaml --checkpoint CHECKPOINT --output results/interpretability_validation.json",
    "G14,T4": "python -m experiments.run_ablations --seeds 11 22 33 44 55",
    "G16,T5": "python experiments/run_region_holdouts.py --seeds 42 101 2024 2025 2026",
    "G15": "python -m data.build_dataset --speibase-dir data/raw/spatial/speibase",
    "G11": "python -m experiments.summarize_probability_calibration --input-dir results/probability_calibration",
}

FIGURE_REQUIREMENTS = {
    "G2": ["results/site_skill_t4.csv", "configs/config.yaml"],
    "G3": ["data/dataset_metadata.json", "configs/base_diagram.yaml"],
    "G4": ["results/*/history.json"],
    "G5": ["results/test_predictions_seed_*.csv"],
    "G6": ["results/per_seed.csv", "results/run_manifest.json"],
    "G7": ["results/spatial_pipeline/spatial_evidence.nc"],
    "G8": ["results/site_skill_t4.csv", "configs/config.yaml"],
    "G9": ["results/case_study_predictions.csv"],
    "G10": ["results/drought_event_bootstrap.csv"],
    "G11": [
        "results/probability_calibration/predictions_seed_*.csv",
        "results/probability_calibration/summary.csv",
    ],
    "G12": ["results/attention_occlusion_validation.csv"],
    "G13": ["results/feature_importance_validation.csv"],
    "G14": ["results/ablation_forest.csv"],
    "G15": ["results/speibase_validation_points.csv"],
    "G16": ["results/leave_region_out_skill.csv"],
    "G17": ["results/test_predictions_seed_*.csv"],
    "G18": ["results/test_predictions_seed_*.csv"],
}

P_COMMANDS = {
    "P2": "fill paper/prisma_counts.yaml with your real review counts",
    "P5": "python experiments/run_all.py --seeds 42 101 2024 2025 2026 (training now saves per-seed histories)",
    "P6": "python evaluate.py --config configs/base_diagram.yaml --checkpoint CHECKPOINT --seed SEED --output results/seeds/metrics_seed_SEED.json --skip-classical",
    "P10": "python evaluate.py --config configs/base_diagram.yaml --checkpoint CHECKPOINT --seed SEED --output results/seeds/metrics_seed_SEED.json --skip-classical",
    "P7,P8": "python experiments/run_all.py --seeds 42 101 2024 2025 2026",
    "P9": "python data/spatial_download.py --download-missing; python -m data.spatial_preprocess --speibase-dir data/raw/spatial/speibase --terraclimate-dir data/raw/spatial/terraclimate --output data/spatial_harmonized.nc; python experiments/run_spatial_pipeline.py --harmonized-nc data/spatial_harmonized.nc --checkpoint results/seeds/model_seed_42.pt (current runner hardcodes T+1; multi-lead export must be implemented before P9 can run)",
    "P11": "python experiments/run_ablations.py --variants quantile_cqr --seeds 42 --force; then python make_paper_figures.py --only P11 --strict",
    "P12": "python experiments/run_ablations.py --variants quantile_cqr --seeds 11 22 33 44 55 --force --probability-calibration; then python make_paper_figures.py --only P12 --strict",
    "P13": "python -m experiments.export_interpretability --config configs/base_diagram.yaml --checkpoint results/seeds/model_seed_42.pt --output results/interpretability_validation.json; then python make_paper_figures.py --only P13 --strict",
    "P14": "python experiments/run_ablations.py --seeds 11 22 33 44 55 --force; python experiments/run_region_holdouts.py --seeds 42 101 2024 2025 2026 --force; then python make_paper_figures.py --only P14 --strict",
    "P15": "python -m data.build_dataset --speibase data/raw/spatial/speibase (requires official SPEIbase files)",
}

P_REQUIRED = {
    "P1": ["data/dataset_metadata.json", "configs/base_diagram.yaml", "data/loader.py"],
    "P2": ["paper/prisma_counts.yaml"],
    "P3": [
        "results/spatial_pipeline/spatial_evidence.nc",
        "data/raw/spatial/gadm/IND.gpkg",
    ],
    "P4": ["configs/base_diagram.yaml"],
    "P5": ["results/seeds/history_seed_*.json"],
    "P6": ["results/test_predictions_seed_*.csv"],
    "P7": ["results/run_manifest.json", "results/summary.csv"],
    "P8": ["results/run_manifest.json", "results/per_seed.csv"],
    "P9": ["results/spatial_pipeline/spatial_evidence.nc"],
    "P10": ["results/site_skill_t4.csv", "configs/config.yaml"],
    "P11": ["results/ablations/quantile_cqr/quantile_test_predictions_seed_*.csv"],
    "P12": [
        "results/probability_calibration/predictions_seed_*.csv",
        "results/probability_calibration/summary.csv",
        "results/ablations/quantile_cqr/metrics_seed_*.json",
    ],
    "P13": [
        "results/interpretability_validation.json",
        "results/attention_occlusion_validation.csv",
        "results/feature_importance_validation.csv",
    ],
    "P14": [
        "results/ablation_forest.csv",
        "results/leave_region_out_skill.csv",
        "results/test_predictions_seed_*.csv",
    ],
    "P15": ["results/speibase_validation_points.csv"],
}

P2_MISSING_MESSAGE = "fill paper/prisma_counts.yaml with your real review counts"


def _csv_has(path: Path, columns: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        return columns.issubset(set(pd.read_csv(path, nrows=0).columns))
    except (OSError, ValueError):
        return False


def missing_inputs() -> dict[str, list[str]]:
    """Return absent or schema-inadequate source files by figure."""
    missing: dict[str, list[str]] = {}
    config = ROOT / "configs" / "base_diagram.yaml"
    if not config.is_file():
        missing["F1"] = [str(config.relative_to(ROOT))]
    for figure, sources in FIGURE_REQUIREMENTS.items():
        absent = []
        for source in sources:
            if "*" in source:
                parent = ROOT / Path(source).parent
                pattern = Path(source).name
                if not parent.is_dir() or not list(parent.glob(pattern)):
                    absent.append(
                        f"{source} (produce with {RESULT_COMMANDS.get(figure, 'the corresponding evaluation/export command')})"
                    )
            elif not (ROOT / source).is_file():
                absent.append(
                    f"{source} (produce with {RESULT_COMMANDS.get(figure, 'the corresponding evaluation/export command')})"
                )
        if absent:
            missing[figure] = absent
    for figure, sources in P_REQUIRED.items():
        absent = []
        for source in sources:
            if "*" in source:
                parent, pattern = ROOT / Path(source).parent, Path(source).name
                if not parent.is_dir() or not list(parent.glob(pattern)):
                    command = next(
                        (
                            cmd
                            for key, cmd in P_COMMANDS.items()
                            if figure in key.split(",")
                        ),
                        "run the corresponding project pipeline",
                    )
                    absent.append(f"{source} (run {command})")
            elif not (ROOT / source).is_file():
                if figure == "P2":
                    absent.append(P2_MISSING_MESSAGE)
                else:
                    command = next(
                        (
                            cmd
                            for key, cmd in P_COMMANDS.items()
                            if figure in key.split(",")
                        ),
                        "run the corresponding project pipeline",
                    )
                    absent.append(f"{source} (run {command})")
        if figure == "P2" and not absent:
            try:
                prisma = yaml.safe_load(
                    (ROOT / "paper/prisma_counts.yaml").read_text(encoding="utf-8")
                )
                if not isinstance(prisma, dict) or any(
                    value is None for value in prisma.values()
                ):
                    absent.append(P2_MISSING_MESSAGE)
            except (OSError, ValueError, yaml.YAMLError):
                absent.append(P2_MISSING_MESSAGE)
        if figure == "P9" and not absent:
            import xarray as xr

            with xr.open_dataset(ROOT / sources[0]) as spatial:
                lead = str(spatial.attrs.get("target_lead", ""))
                if lead == "T+1":
                    absent.append(
                        f"{sources[0]} contains only T+1 forecasts, but P9 requires T+1, T+3, and T+4 "
                        f"(run {P_COMMANDS['P9']})"
                    )
        if figure == "P12" and not absent:
            data_path = ROOT / "data/monthly_sites.csv"
            current_hash = (
                hashlib.sha256(data_path.read_bytes()).hexdigest()
                if data_path.is_file()
                else None
            )
            for seed in (11, 22, 33, 44, 55):
                checkpoint = (
                    ROOT / "results/ablations/quantile_cqr" / f"model_seed_{seed}.pt"
                )
                report_path = (
                    ROOT
                    / "results/probability_calibration"
                    / f"calibration_seed_{seed}.json"
                )
                if not checkpoint.is_file() or not report_path.is_file():
                    absent.append(
                        f"missing checkpoint or calibration report for seed {seed} (run {P_COMMANDS['P12']})"
                    )
                    continue
                state = __import__("torch").load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                trained_hash = state.get("artifact_metadata", {}).get("dataset_sha256")
                if current_hash and trained_hash != current_hash:
                    absent.append(
                        f"{checkpoint.relative_to(ROOT)} has a stale dataset hash; do not plot these probabilities. "
                        f"Regenerate with: {P_COMMANDS['P12']}"
                    )
                    break
        if figure == "P10" and not absent:
            skill_path = ROOT / "results/site_skill_t4.csv"
            manifest_path = ROOT / "results/evaluation_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                skill = pd.read_csv(skill_path)
                expected_alignment = (
                    "ONI/DMI/NAO/PDO are shifted one month: the feature at month m uses the value published for m-1; "
                    "centered ONI at m-1 contains no month m or later"
                )
                current = (
                    manifest.get("protocol_version") == 1
                    and manifest.get("dataset_schema_version") == 2
                    and manifest.get("index_alignment") == expected_alignment
                )
                if (
                    not current
                    or "seed" not in skill
                    or set(skill.seed.unique()) != {manifest.get("seed")}
                ):
                    absent.append(
                        f"{skill_path.relative_to(ROOT)} or evaluation manifest is stale or lacks seed provenance "
                        f"(rerun {P_COMMANDS['P10']})"
                    )
                else:
                    checkpoint_path = (
                        ROOT
                        / "results/seeds"
                        / f"model_seed_{int(manifest['seed'])}.pt"
                    )
                    data_path = ROOT / "data/monthly_sites.csv"
                    checkpoint = __import__("torch").load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )
                    current_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
                    if (
                        checkpoint.get("artifact_metadata", {}).get("dataset_sha256")
                        != current_hash
                    ):
                        absent.append(
                            f"{checkpoint_path.relative_to(ROOT)} is trained on stale data (run {P_COMMANDS['P10']})"
                        )
            except (OSError, ValueError, KeyError):
                absent.append(
                    f"results/evaluation_manifest.json is missing or invalid (run {P_COMMANDS['P10']})"
                )
        if figure == "P13" and not absent:
            try:
                provenance = json.loads(
                    (ROOT / "results/interpretability_validation.json").read_text(
                        encoding="utf-8"
                    )
                )
                checkpoint_path = ROOT / provenance["checkpoint"]
                config_path = ROOT / provenance["config"]
                dataset_hash = hashlib.sha256(
                    (ROOT / "data/monthly_sites.csv").read_bytes()
                ).hexdigest()
                if (
                    provenance.get("split") != "validation only"
                    or provenance.get("dataset_sha256") != dataset_hash
                    or provenance.get("checkpoint_sha256")
                    != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                    or provenance.get("config_sha256")
                    != hashlib.sha256(config_path.read_bytes()).hexdigest()
                ):
                    absent.append(
                        f"results/interpretability_validation.json is stale or not validation-only (run {P_COMMANDS['P13']})"
                    )
            except (OSError, ValueError, KeyError):
                absent.append(
                    f"results/interpretability_validation.json is invalid (run {P_COMMANDS['P13']})"
                )
        if figure == "P14" and not absent:
            forest_path = ROOT / "results/ablation_forest.csv"
            regions_path = ROOT / "results/leave_region_out_skill.csv"
            try:
                forest = pd.read_csv(forest_path)
                regions = pd.read_csv(regions_path)
                if not {"variant", "delta_rmse", "lower", "upper", "n_seeds"}.issubset(
                    forest.columns
                ):
                    absent.append(
                        f"{forest_path.relative_to(ROOT)} lacks the required forest schema (run {P_COMMANDS['P14']})"
                    )
                elif (
                    "base_diagram" not in set(forest.variant)
                    or (forest.n_seeds < 5).any()
                ):
                    absent.append(
                        f"{forest_path.relative_to(ROOT)} lacks five current seeds and a base_diagram row (run {P_COMMANDS['P14']})"
                    )
                if not {
                    "heldout_region",
                    "seed",
                    "scale",
                    "lead",
                    "skill_vs_persistence",
                }.issubset(regions.columns):
                    absent.append(
                        f"{regions_path.relative_to(ROOT)} lacks the required held-region schema (run {P_COMMANDS['P14']})"
                    )
                elif (
                    regions.empty
                    or (
                        regions.groupby(
                            ["heldout_region", "scale", "lead"]
                        ).seed.nunique()
                        < 5
                    ).any()
                ):
                    absent.append(
                        f"{regions_path.relative_to(ROOT)} lacks five seeds per region/scale/lead (run {P_COMMANDS['P14']})"
                    )
            except (OSError, ValueError, KeyError) as exc:
                absent.append(
                    f"P14 result summaries could not be read ({exc}); run {P_COMMANDS['P14']}"
                )
        if absent:
            missing[figure] = absent
    required = {
        "F2": [
            (
                "results/site_skill_t4.csv",
                {"site", "scale", "latitude", "longitude", "skill_vs_persistence"},
            )
        ],
        "F3": [
            (
                "results/per_seed.csv",
                {"seed", "method", "scale", "lead", "skill_vs_persistence"},
            )
        ],
        "F4": [
            (
                "results/case_study_predictions.csv",
                {
                    "site",
                    "date",
                    "scale",
                    "lead",
                    "observed",
                    "forecast",
                    "lower",
                    "upper",
                },
            )
        ],
        "F5": [
            (
                "results/reliability_and_coverage.csv",
                {
                    "model",
                    "scale",
                    "lead",
                    "bin_mean_probability",
                    "observed_frequency",
                    "nominal_coverage",
                    "empirical_coverage",
                },
            )
        ],
        "F6": [
            (
                "results/drought_event_bootstrap.csv",
                {"scale", "lead", "threshold", "metric", "estimate", "lower", "upper"},
            )
        ],
        "F7": [
            ("results/ablation_forest.csv", {"variant", "delta_rmse", "lower", "upper"})
        ],
        "F8": [
            (
                "results/attention_occlusion.csv",
                {
                    "scale",
                    "lead",
                    "season",
                    "input_step",
                    "attention",
                    "occlusion_delta_validation_mse",
                },
            )
        ],
        "F9": [
            (
                "results/speibase_validation_points.csv",
                {"site", "date", "scale", "computed_spei", "speibase_spei"},
            ),
            (
                "results/speibase_validation.csv",
                {"site", "scale", "correlation", "rmse", "bias"},
            ),
        ],
        "F10": [
            (
                "results/leave_region_out_skill.csv",
                {"heldout_region", "scale", "lead", "skill_vs_persistence"},
            )
        ],
    }
    if not config.is_file():
        missing["F2"] = missing.get("F2", []) + [
            "configs/config.yaml (site coordinates)"
        ]
    if not importlib.util.find_spec("cartopy"):
        try:
            import geopandas as gpd

            gpd.datasets.get_path("naturalearth_lowres")
        except (ImportError, AttributeError, FileNotFoundError, OSError):
            missing.setdefault("F2", []).append(
                "Natural Earth coastlines (Cartopy is unavailable and no local Natural Earth dataset is installed)"
            )
    for figure, sources in required.items():
        bad = [name for name, columns in sources if not _csv_has(ROOT / name, columns)]
        if figure == "F3" and not bad:
            frame = pd.read_csv(ROOT / sources[0][0])
            group_counts = frame.groupby(["method", "scale", "lead"]).seed.nunique()
            if group_counts.empty or (group_counts < 5).any():
                bad.append(
                    "results/per_seed.csv (requires five seeds for every method, scale, and forecast lead)"
                )
        if figure == "F7" and not bad:
            frame = pd.read_csv(ROOT / sources[0][0])
            if "base_diagram" not in set(frame.variant):
                bad.append(
                    "results/ablation_forest.csv (missing base_diagram reference row)"
                )
        if figure == "F10" and not bad:
            frame = pd.read_csv(ROOT / sources[0][0])
            if frame.heldout_region.nunique() < 2:
                bad.append(
                    "results/leave_region_out_skill.csv (requires multiple held-out regions)"
                )
        if bad:
            missing[figure] = bad
    run_manifest = ROOT / "results" / "run_manifest.json"
    expected_alignment = (
        "ONI/DMI/NAO/PDO are shifted one month: the feature at month m uses the value published for m-1; "
        "centered ONI at m-1 contains no month m or later"
    )
    try:
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        current = (
            manifest.get("protocol_version") == 1
            and manifest.get("dataset_schema_version") == 2
            and manifest.get("index_alignment") == expected_alignment
        )
        if current and "F3" not in missing:
            seed_frame = pd.read_csv(ROOT / "results/per_seed.csv")
            if not set(manifest.get("seeds", [])).issubset(
                set(seed_frame.seed.unique())
            ):
                missing.setdefault("F3", []).append(
                    "results/per_seed.csv (seed rows do not match results/run_manifest.json)"
                )
    except (OSError, ValueError):
        current = False
    if not current:
        try:
            evaluation = json.loads(
                (ROOT / "results/evaluation_manifest.json").read_text(encoding="utf-8")
            )
            current_evaluation = (
                evaluation.get("protocol_version") == 1
                and evaluation.get("dataset_schema_version") == 2
                and evaluation.get("index_alignment") == expected_alignment
            )
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


def _save(fig, name: str, data: pd.DataFrame, *, kind="line"):
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    width = "single" if abs(fig.get_figwidth() - SINGLE_COLUMN_IN) <= 0.02 else "double"
    dpi = 300 if kind in {"map", "heatmap"} else 600
    outputs = save_figure(fig, OUT / name, width=width, dpi=dpi, kind=kind)
    expected_aspect = (
        1 / np.cos(np.deg2rad(float(data.attrs["map_mean_lat"])))
        if kind == "map" and data.attrs.get("map_mean_lat") is not None
        else None
    )
    quality = audit_figure(
        fig,
        outputs["pdf"],
        outputs["png"],
        width=width,
        dpi=dpi,
        expected_geographic_aspect=expected_aspect,
    )
    QUALITY_RESULTS.append({"figure": name, **quality})
    _safe_print(
        f"QA {name}: {'PASS' if quality['ok'] else 'FAIL'}"
        + ("; " + "; ".join(quality["errors"]) if quality["errors"] else "")
    )
    data.to_csv(DATA / f"{name}.csv", index=False)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / data.attrs.get("config_path", "configs/base_diagram.yaml")
    config_hash = (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_path.is_file()
        else None
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    try:
        dataset_meta = json.loads(
            (ROOT / "data/dataset_metadata.json").read_text(encoding="utf-8")
        )
        schema = dataset_meta.get("schema_version")
    except (OSError, ValueError):
        schema = None
    (PROVENANCE / f"{name}.json").write_text(
        json.dumps(
            {
                "git_commit": commit,
                "config_sha256": config_hash,
                "config_path": str(config_path.relative_to(ROOT))
                if config_path.is_relative_to(ROOT)
                else str(config_path),
                "seeds": data.attrs.get(
                    "seeds",
                    sorted(data.seed.dropna().unique().tolist())
                    if "seed" in data
                    else [],
                ),
                "dataset_schema_version": schema,
                "test_period": data.attrs.get("test_period"),
                "source_rows": len(data),
                "source_data_file": f"data/{name}.csv",
                "figure_quality": quality,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plt.close(fig)


def figure_f1():
    config_path = ROOT / "configs" / "base_diagram.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Required architecture config is missing: {config_path}"
        )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seq_len = int(cfg["data"]["seq_len"])
    forecast = int(cfg["data"]["forecast_steps"])
    model = cfg["model"]
    if (
        seq_len,
        int(model["hidden_dim_1"]),
        int(model["hidden_dim_2"]),
        forecast,
        len(cfg["data"]["features"]),
    ) != (6, 64, 32, 4, 12):
        raise ValueError(
            "F1 requires base_diagram.yaml to specify 6 steps, 64-to-32 LSTM, and four forecast leads"
        )
    if (
        model.get("attention") != "shared"
        or model.get("heads") != "direct"
        or cfg["training"].get("loss") != "mse"
    ):
        raise ValueError(
            "F1 requires shared attention, direct point heads, and plain MSE in base_diagram.yaml"
        )
    phases = [
        (
            "Phase 1: Inputs",
            f"6 monthly steps\n{len(cfg['data']['features'])} climate and index\nfeatures + observed\nSPEI state",
        ),
        ("Phase 2: Encoder", "LSTM 64\nDropout 0.2\nLSTM 32\n[B, 6, 32]"),
        ("Phase 3: Attention", "Additive attention\nover 6 input steps\n[B, 6, 1]"),
        ("Phase 4: Context", "One shared context\nc in R^32"),
        (
            "Phase 5: Heads",
            "Three direct heads\nSPEI-3, SPEI-6,\nSPEI-12 scales\nT+1..T+4",
        ),
        ("Phase 6: Output", "Point output\n[B, 3, 4]\nplain MSE"),
    ]
    rows = []
    for i, (phase, description) in enumerate(phases):
        rows.append(
            {
                "phase": i + 1,
                "component": phase.split(": ", 1)[1],
                "description": description.replace("\n", "; "),
                "order": i,
            }
        )
    fig, ax = plt.subplots(figsize=(7.16, 2.7), constrained_layout=True)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = [
        OKABE_ITO["sky_blue"],
        OKABE_ITO["blue"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["orange"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["vermillion"],
    ]
    for i, (title, description) in enumerate(phases):
        x = i + 0.08
        rect = plt.Rectangle(
            (x, 0.20),
            0.82,
            0.62,
            facecolor=colors[i],
            edgecolor=OKABE_ITO["black"],
            linewidth=0.7,
            alpha=0.92,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.41, 0.70, title, ha="center", va="center", fontsize=8.2, weight="bold"
        )
        ax.text(
            x + 0.41,
            0.45,
            description,
            ha="center",
            va="center",
            fontsize=7.1,
            linespacing=1.25,
        )
        if i < 5:
            ax.annotate(
                "",
                xy=(x + 0.98, 0.51),
                xytext=(x + 0.84, 0.51),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 1.0,
                    "color": OKABE_ITO["black"],
                },
            )
    data = pd.DataFrame(rows)
    _save(fig, "P4_architecture", data)


def _load_csv(name: str) -> pd.DataFrame:
    path = ROOT / "results" / name
    if not path.is_file():
        raise FileNotFoundError(f"Required paper figure input is missing: {path}")
    return pd.read_csv(path)


def figure_f2():
    data = _load_csv("site_skill_t4.csv")
    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    coords = pd.DataFrame(
        [
            {"site": name, "latitude": spec["lat"], "longitude": spec["lon"]}
            for name, spec in cfg["sites"].items()
        ]
    )
    data = data.merge(coords, on="site", how="left", validate="one_to_one")
    if data[["latitude", "longitude", "skill_vs_persistence"]].isna().any().any():
        raise ValueError(
            "F2 requires configured coordinates and finite T+4 skill for every site"
        )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.6), constrained_layout=True)
    scales = ["spei_3", "spei_6", "spei_12"]
    try:
        import cartopy.crs as ccrs

        for ax in axes:
            ax.remove()
        axes = [
            fig.add_subplot(1, 3, i + 1, projection=ccrs.PlateCarree())
            for i in range(3)
        ]
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
                gpd.read_file(natural_earth).boundary.plot(
                    ax=ax, color="#777777", linewidth=0.35
                )
        except Exception as exc:
            raise FileNotFoundError(
                "F2 requires Cartopy coastlines or local Natural Earth coastline data"
            ) from exc
    points = None
    for ax, scale in zip(axes, scales):
        subset = data[data.scale == scale]
        skill_map = LinearSegmentedColormap.from_list(
            "okabe_ito_skill",
            [OKABE_ITO["blue"], OKABE_ITO["yellow"], OKABE_ITO["vermillion"]],
        )
        scatter_kw = {
            "c": subset.skill_vs_persistence,
            "cmap": skill_map,
            "s": 45,
            "edgecolors": "black",
            "linewidths": 0.4,
        }
        if transform is not None:
            scatter_kw["transform"] = transform
        points = ax.scatter(subset.longitude, subset.latitude, **scatter_kw)
        for row in subset.itertuples():
            ax.annotate(
                row.site.replace("_", " "),
                (row.longitude, row.latitude),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                **(
                    {"transform": transform._as_mpl_transform(ax)}
                    if transform is not None
                    else {}
                ),
            )
        ax.text(
            0.03,
            0.97,
            scale.replace("_", "-"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            weight="bold",
        )
    fig.colorbar(points, ax=axes, label="Skill vs persistence at T+4 (unitless)")
    _save(fig, "F2_site_skill", data)


def figure_f3():
    source = _load_csv("per_seed.csv")
    rows = []
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 3.2), sharey=True, constrained_layout=True
    )
    scales = ["spei_3", "spei_6", "spei_12"]
    methods = sorted(source.method.unique())
    colors = [
        OKABE_ITO[k]
        for k in (
            "blue",
            "orange",
            "bluish_green",
            "vermillion",
            "reddish_purple",
            "sky_blue",
            "black",
        )
    ]
    for ax, scale in zip(axes, scales):
        for mi, method in enumerate(methods):
            sub = source[(source.scale == scale) & (source.method == method)]
            means, lows, highs = [], [], []
            for lead in range(1, 5):
                vals = (
                    sub.loc[sub.lead == f"T+{lead}", "skill_vs_persistence"]
                    .dropna()
                    .to_numpy()
                )
                if len(vals) < 5:
                    raise ValueError(
                        f"F3 needs five seeds for {method}/{scale}/T+{lead}; got {len(vals)}"
                    )
                mean = vals.mean()
                se = vals.std(ddof=1) / np.sqrt(len(vals))
                from scipy.stats import t

                half = float(t.ppf(0.975, len(vals) - 1) * se)
                means.append(mean)
                lows.append(mean - half)
                highs.append(mean + half)
                rows.append(
                    {
                        "method": method,
                        "scale": scale,
                        "lead": f"T+{lead}",
                        "n_seeds": len(vals),
                        "skill_mean": mean,
                        "ci95_lower": mean - half,
                        "ci95_upper": mean + half,
                    }
                )
            x = np.arange(1, 5)
            ax.plot(
                x,
                means,
                marker="o",
                linewidth=1.1,
                label=method,
                color=colors[mi % len(colors)],
            )
            ax.fill_between(x, lows, highs, alpha=0.14, color=colors[mi % len(colors)])
        ax.axhline(0, color="#444444", lw=0.7, ls="--")
        ax.text(
            0.03,
            0.97,
            scale.replace("_", "-"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            weight="bold",
        )
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
    fig, axes = plt.subplots(
        len(episodes),
        len(scales),
        figsize=(7.16, 2.4 * len(episodes)),
        squeeze=False,
        sharex="col",
        constrained_layout=True,
    )
    for i, episode in enumerate(episodes):
        for j, scale in enumerate(scales):
            ax = axes[i, j]
            subset = data[(data.episode == episode) & (data.scale == scale)]
            observed = subset.sort_values("date").drop_duplicates("date")
            ax.plot(
                pd.to_datetime(observed.date),
                observed.observed,
                color=OKABE_ITO["black"],
                lw=1.0,
                label="Observed",
            )
            for k, lead in enumerate(leads):
                line = subset[subset.lead == lead].sort_values("date")
                color = OKABE_ITO[
                    ["blue", "orange", "bluish_green", "vermillion"][k % 4]
                ]
                ax.plot(
                    pd.to_datetime(line.date),
                    line.forecast,
                    color=color,
                    lw=0.9,
                    label=f"T+{lead}",
                )
                ax.fill_between(
                    pd.to_datetime(line.date),
                    line.lower,
                    line.upper,
                    color=color,
                    alpha=0.12,
                )
            ax.axhline(-1, color="#777777", lw=0.5, ls="--")
            ax.set_ylabel("SPEI (unitless)")
            ax.set_xlabel("Target month")
            ax.text(
                0.03,
                0.97,
                f"{episode} | {scale.replace('_', '-')}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            ax.legend(frameon=False, fontsize=6, ncol=2)
    _save(fig, "F4_case_studies", data)


def figure_f5():
    data = _load_csv("reliability_and_coverage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4), constrained_layout=True)
    for i, (scale, sub) in enumerate(data.groupby("scale", sort=True)):
        for model, model_data in sub.groupby("model", sort=True):
            rel = model_data.dropna(
                subset=["bin_mean_probability", "observed_frequency"]
            )
            axes[0].plot(
                rel.bin_mean_probability,
                rel.observed_frequency,
                marker="o",
                ms=3,
                color=OKABE_ITO[
                    ["blue", "orange", "bluish_green", "vermillion"][i % 4]
                ],
                label=f"{scale.replace('_', '-')} {model}",
            )
        cov = sub.drop_duplicates(["model", "lead"])
        for model, model_data in cov.groupby("model", sort=True):
            axes[1].plot(
                model_data.lead,
                model_data.empirical_coverage,
                marker="o",
                ms=3,
                label=f"{scale.replace('_', '-')} {model}",
            )
    axes[0].plot([0, 1], [0, 1], color="#555555", ls="--", lw=0.7)
    axes[0].set(
        xlabel="Forecast event probability (unitless)",
        ylabel="Observed event frequency (unitless)",
    )
    axes[1].set(
        xlabel="Forecast horizon (months)",
        ylabel="Empirical interval coverage (unitless)",
    )
    axes[1].axhline(
        float(data.nominal_coverage.iloc[0]), color="#555555", ls="--", lw=0.7
    )
    for ax in axes:
        ax.legend(frameon=False, fontsize=6)
    _save(fig, "F5_reliability_coverage", data)


def figure_f6():
    data = _load_csv("drought_event_bootstrap.csv")
    metrics = ["POD", "FAR", "CSI"]
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 3.3), sharey=True, constrained_layout=True
    )
    for ax, metric in zip(axes, metrics):
        sub = data[data.metric.str.upper() == metric]
        for scale, scale_data in sub.groupby("scale", sort=True):
            scale_data = scale_data.sort_values(["lead", "threshold"])
            x = np.arange(len(scale_data))
            yerr = np.vstack(
                [
                    scale_data.estimate - scale_data.lower,
                    scale_data.upper - scale_data.estimate,
                ]
            )
            ax.errorbar(
                x,
                scale_data.estimate,
                yerr=yerr,
                marker="o",
                ms=2.5,
                lw=0.8,
                capsize=2,
                label=scale.replace("_", "-"),
            )
        ax.set_ylabel(f"{metric} (unitless)")
        ax.set_xlabel("Threshold and forecast lead")
        tick_data = sub.drop_duplicates(["threshold", "lead"]).sort_values(
            ["lead", "threshold"]
        )
        labels = [
            f"{row.threshold:g},T{str(row.lead).replace('T+', '')}"
            for row in tick_data.itertuples()
        ]
        ax.set_xticks(
            np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=6
        )
    axes[-1].legend(frameon=False)
    _save(fig, "F6_drought_events", data)


def figure_f7():
    data = _load_csv("ablation_forest.csv")
    data = data[data.variant != "base_diagram"].reset_index(drop=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(
        figsize=(7.16, max(2.8, 0.28 * len(data))), constrained_layout=True
    )
    ax.errorbar(
        data.delta_rmse,
        y,
        xerr=np.vstack([data.delta_rmse - data.lower, data.upper - data.delta_rmse]),
        fmt="o",
        color=OKABE_ITO["blue"],
        ecolor="#555555",
        capsize=2,
    )
    ax.axvline(0, color=OKABE_ITO["black"], lw=0.7, ls="--")
    ax.set_yticks(y, data.variant)
    ax.set_xlabel("RMSE difference from base_diagram (SPEI units)")
    _save(fig, "F7_ablation_forest", data)


def figure_f8():
    data = _load_csv("attention_occlusion.csv")
    data = data.sort_values(["scale", "lead", "season", "input_step"])
    groups = (
        data[["scale", "lead", "season"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    groups = list(groups)
    matrix = []
    labels = []
    for scale, lead, season in groups:
        row = data[
            (data.scale == scale) & (data.lead == lead) & (data.season == season)
        ].sort_values("input_step")
        matrix.append(row.attention.to_numpy())
        labels.append(f"{scale.replace('_', '-')} T+{lead} {season}")
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, max(3.2, 0.22 * len(matrix))), constrained_layout=True
    )
    im = axes[0].imshow(np.asarray(matrix), aspect="auto", cmap="viridis")
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=6)
    axes[0].set_xlabel("Input month step (months before forecast origin)")
    axes[0].set_ylabel("Scale, forecast horizon, season")
    fig.colorbar(im, ax=axes[0], fraction=0.045)
    occ = data.groupby(
        "input_step", as_index=False
    ).occlusion_delta_validation_mse.mean()
    axes[1].bar(
        occ.input_step, occ.occlusion_delta_validation_mse, color=OKABE_ITO["orange"]
    )
    axes[1].set_xlabel("Occluded input step")
    axes[1].set_ylabel("Change in validation MSE (SPEI squared)")
    _save(fig, "F8_attention_occlusion", data)


def figure_f9():
    data = _load_csv("speibase_validation_points.csv")
    metrics = _load_csv("speibase_validation.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 3), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, scale in zip(axes, [3, 6, 12]):
        sub = data[data.scale == scale]
        if sub.empty:
            raise ValueError(f"F9 input has no validation pairs for SPEI-{scale}")
        ax.scatter(
            sub.speibase_spei,
            sub.computed_spei,
            s=5,
            alpha=0.25,
            color=OKABE_ITO["blue"],
            edgecolors="none",
        )
        lo = min(sub.speibase_spei.min(), sub.computed_spei.min())
        hi = max(sub.speibase_spei.max(), sub.computed_spei.max())
        ax.plot([lo, hi], [lo, hi], color=OKABE_ITO["vermillion"], ls="--", lw=0.8)
        ax.text(
            0.04,
            0.96,
            f"SPEI-{scale}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            weight="bold",
        )
        metric = metrics[metrics.scale == scale]
        if metric.empty:
            raise ValueError(f"F9 validation summary is missing SPEI-{scale}")
        ax.text(
            0.04,
            0.87,
            f"r={metric.correlation.mean():.2f}\nRMSE={metric.rmse.mean():.2f}\nBias={metric.bias.mean():.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
        )
        ax.set_xlabel("SPEIbase SPEI (unitless)")
    axes[0].set_ylabel("Computed SPEI (unitless)")
    data = data.merge(metrics, on=["site", "scale"], how="left", validate="many_to_one")
    _save(fig, "F9_speibase_validation", data)


def figure_f10():
    data = _load_csv("leave_region_out_skill.csv")
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 3), sharey=True, constrained_layout=True
    )
    regions = list(data.heldout_region.unique())
    for ax, scale in zip(axes, ["spei_3", "spei_6", "spei_12"]):
        sub = data[data.scale == scale]
        for lead, lead_data in sub.groupby("lead", sort=True):
            lead_data = (
                lead_data.set_index("heldout_region").reindex(regions).reset_index()
            )
            ax.plot(
                lead_data.heldout_region,
                lead_data.skill_vs_persistence,
                marker="o",
                label=lead,
                color=OKABE_ITO[
                    ["blue", "orange", "bluish_green", "vermillion"][
                        int(str(lead).replace("T+", "")) - 1
                    ]
                ],
            )
        ax.text(
            0.03,
            0.97,
            scale.replace("_", "-"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            weight="bold",
        )
        ax.set_xticks(np.arange(len(regions)), regions, rotation=45, ha="right")
        ax.set_xlabel("Held-out region")
    axes[0].set_ylabel("Skill vs persistence (unitless)")
    axes[-1].legend(frameon=False)
    _save(fig, "F10_leave_region_out", data)


def figure_g8():
    """Per-site T+4 skill map from the real chronological test evaluation."""
    skills = _load_csv("site_skill_t4.csv")
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    coords = pd.DataFrame(
        [
            {
                "site": site,
                "config_latitude": spec["lat"],
                "config_longitude": spec["lon"],
                "region": spec["region"],
            }
            for site, spec in config["sites"].items()
        ]
    )
    source = skills.merge(coords, on="site", how="left", validate="many_to_one")
    source["latitude"] = source.config_latitude
    source["longitude"] = source.config_longitude
    if source[["latitude", "longitude"]].isna().any().any():
        raise ValueError(
            "G8 site coordinates are missing for one or more test skill rows"
        )
    skill_columns = ["spei_3", "spei_6", "spei_12"]
    data = source.pivot(
        index=["site", "latitude", "longitude", "region"],
        columns="scale",
        values="skill_vs_persistence",
    ).reset_index()
    data.columns.name = None
    data = data.rename(columns={scale: f"skill_{scale}" for scale in skill_columns})
    if "seed" in source:
        seeds = source.seed.dropna().unique()
        if len(seeds) != 1:
            raise ValueError(
                "P10 site-skill map requires one clearly identified evaluation seed"
            )
        data["seed"] = int(seeds[0])
    count_table = source.pivot(index="site", columns="scale", values="n")
    for scale in skill_columns:
        if f"skill_{scale}" not in data:
            raise ValueError(f"G8 is missing T+4 skill for accumulation scale {scale}")
    data["mean_t4_skill_across_scales"] = data[
        [f"skill_{scale}" for scale in skill_columns]
    ].mean(axis=1)
    data["mean_test_case_count_across_scales"] = data.site.map(
        count_table[skill_columns].mean(axis=1)
    )
    skill_cmap = LinearSegmentedColormap.from_list(
        "okabe_ito_skill",
        [OKABE_ITO["vermillion"], OKABE_ITO["yellow"], OKABE_ITO["blue"]],
    )
    fig, ax = plt.subplots(figsize=(7.16, 2.9), constrained_layout=True)
    marker_size = 35 + 35 * data.mean_t4_skill_across_scales.abs().clip(0, 1)
    scatter = ax.scatter(
        data.longitude,
        data.latitude,
        c=data.mean_t4_skill_across_scales,
        s=marker_size,
        cmap=skill_cmap,
        vmin=-1,
        vmax=1,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    offsets = {
        "jaipur": (7, 7),
        "nagpur": (7, -7),
        "guntur": (7, 7),
        "phoenix": (7, 7),
        "seville": (7, 7),
        "addis_ababa": (7, 7),
        "pretoria": (7, -8),
        "murray_darling": (7, 7),
    }
    for row in data.itertuples():
        dx, dy = offsets.get(row.site, (6, 6))
        ax.annotate(
            row.site.replace("_", " "),
            (row.longitude, row.latitude),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="center",
        )
    ax.text(
        0.015,
        0.97,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlim(-125, 155)
    ax.set_ylim(-42, 46)
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.set_aspect(1 / np.cos(np.deg2rad(float(data.latitude.mean()))), adjustable="box")
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.grid(True, lw=0.4, linestyle=":", color="#666666", alpha=0.55)
    fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.035,
        pad=0.025,
        label="Mean T+4 skill vs persistence across SPEI scales (unitless)",
    )
    prediction_path = next(
        iter(sorted((ROOT / "results").glob("test_predictions_seed_*.csv"))), None
    )
    if prediction_path:
        predictions = pd.read_csv(prediction_path, usecols=["target_month"])
        data.attrs["test_period"] = [
            str(predictions.target_month.min()),
            str(predictions.target_month.max()),
        ]
    else:
        data.attrs["test_period"] = None
    data.attrs["map_mean_lat"] = float(data.latitude.mean())
    _save(fig, "P10_site_skill", data, kind="map")


def figure_g11_probability_calibration():
    """Test reliability and per-seed Brier metrics from the five event-calibration runs."""
    folder = ROOT / "results" / "probability_calibration"
    paths = sorted(folder.glob("predictions_seed_*.csv"))
    if len(paths) < 5:
        raise FileNotFoundError(
            f"G11 needs five prediction_seed CSVs under {folder}. Run "
            "experiments/evaluate_event_calibration.py for seeds 11, 22, 33, 44, and 55."
        )
    frames = [
        pd.read_csv(path).assign(seed=int(path.stem.rsplit("_", 1)[1]))
        for path in paths
    ]
    predictions = pd.concat(frames, ignore_index=True)
    required = {
        "site",
        "date",
        "lead",
        "scale",
        "calibrated_probability",
        "observed_event",
        "seed",
    }
    if required.difference(predictions.columns):
        raise ValueError(
            f"G11 predictions are missing columns: {sorted(required.difference(predictions.columns))}"
        )
    coverage = predictions.groupby(["scale", "lead"]).seed.nunique()
    if (coverage < 5).any():
        raise ValueError("G11 requires five seeds for every scale and lead")
    labels = predictions.groupby(
        ["site", "date", "lead", "scale"]
    ).observed_event.nunique()
    if (labels > 1).any():
        raise ValueError("Observed event differs by seed for the same test case")
    cases = predictions.groupby(["site", "date", "lead", "scale"], as_index=False).agg(
        probability=("calibrated_probability", "mean"),
        observed_event=("observed_event", "first"),
    )
    summary = pd.read_csv(folder / "summary.csv")
    needed_summary = {
        "scale",
        "lead",
        "seeds",
        "raw_brier_mean",
        "raw_brier_std",
        "calibrated_brier_mean",
        "calibrated_brier_std",
    }
    if needed_summary.difference(summary.columns) or (summary.seeds < 5).any():
        raise ValueError(
            "G11 summary must contain raw/calibrated Brier mean/std for five seeds"
        )

    scales = ["spei_3", "spei_6", "spei_12"]
    leads = [1, 2, 3, 4]
    colors = [OKABE_ITO[k] for k in ("blue", "orange", "bluish_green", "vermillion")]
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 7.25), constrained_layout=True)
    rows = []
    z = 1.959963984540054
    for scale_idx, scale in enumerate(scales):
        reliability_ax, brier_ax = axes[scale_idx]
        scale_cases = cases[cases.scale == scale]
        for lead_idx, lead in enumerate(leads):
            subset = scale_cases[scale_cases.lead.astype(int) == lead].copy()
            if subset.empty:
                raise ValueError(f"No test prediction rows for {scale}, T+{lead}")
            bin_index = (subset.probability.clip(0, 1) * 10).astype(int).clip(upper=9)
            lead_label_used = False
            for probability_bin in range(10):
                block = subset[bin_index == probability_bin]
                if block.empty:
                    continue
                n = len(block)
                observed = float(block.observed_event.mean())
                predicted = float(block.probability.mean())
                denom = 1 + z * z / n
                center = (observed + z * z / (2 * n)) / denom
                half = (
                    z
                    * math.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n))
                    / denom
                )
                lower, upper = max(0.0, center - half), min(1.0, center + half)
                reliability_ax.errorbar(
                    predicted,
                    observed,
                    yerr=[[observed - lower], [upper - observed]],
                    fmt="o",
                    ms=2.8,
                    capsize=1.5,
                    lw=0.55,
                    color=colors[lead_idx],
                    label=f"T+{lead}" if not lead_label_used else None,
                )
                lead_label_used = True
                rows.append(
                    {
                        "panel": "reliability",
                        "scale": scale,
                        "lead": lead,
                        "probability_bin": probability_bin,
                        "n_test_cases": n,
                        "mean_predicted_probability": predicted,
                        "observed_event_frequency": observed,
                        "wilson95_lower": lower,
                        "wilson95_upper": upper,
                        "event_definition": "observed SPEI <= -1",
                    }
                )
            reliability_ax.plot(
                [0, 1], [0, 1], color=OKABE_ITO["black"], lw=0.6, ls="--"
            )
            reliability_ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Mean predicted drought probability" if scale_idx == 2 else "",
            )
            if lead_idx == 0:
                reliability_ax.set_ylabel("Observed drought frequency")
            row = summary[
                (summary.scale == scale) & (summary.lead.astype(str) == f"T+{lead}")
            ]
            if row.empty:
                raise ValueError(f"Summary is missing {scale}, T+{lead}")
            metric = row.iloc[0]
            brier_ax.bar(
                lead_idx - 0.17,
                metric.raw_brier_mean,
                width=0.32,
                color=OKABE_ITO["sky_blue"],
                edgecolor="black",
                linewidth=0.35,
                yerr=metric.raw_brier_std,
                capsize=1.7,
                label="Raw" if lead_idx == 0 else None,
            )
            brier_ax.bar(
                lead_idx + 0.17,
                metric.calibrated_brier_mean,
                width=0.32,
                color=OKABE_ITO["blue"],
                edgecolor="black",
                linewidth=0.35,
                yerr=metric.calibrated_brier_std,
                capsize=1.7,
                label="Validation calibrated" if lead_idx == 0 else None,
            )
            for label, mean, std in (
                ("raw", metric.raw_brier_mean, metric.raw_brier_std),
                (
                    "validation_calibrated",
                    metric.calibrated_brier_mean,
                    metric.calibrated_brier_std,
                ),
            ):
                rows.append(
                    {
                        "panel": "brier",
                        "scale": scale,
                        "lead": lead,
                        "method": label,
                        "brier_mean_five_seeds": mean,
                        "brier_std_five_seeds": std,
                        "n_seeds": int(metric.seeds),
                        "event_definition": "observed SPEI <= -1",
                    }
                )
        reliability_ax.text(
            0.02,
            0.97,
            f"({chr(97 + scale_idx)}) {scale.replace('_', '-')} reliability",
            transform=reliability_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            weight="bold",
        )
        reliability_handles, reliability_labels = (
            reliability_ax.get_legend_handles_labels()
        )
        brier_ax.set_xticks(range(4), [f"T+{n}" for n in leads])
        brier_ax.set_xlabel("Forecast lead" if scale_idx == 2 else "")
        brier_ax.set_ylabel("Brier score (lower is better)")
        brier_ax.text(
            0.02,
            0.97,
            f"({chr(100 + scale_idx)}) {scale.replace('_', '-')} Brier",
            transform=brier_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            weight="bold",
        )
    brier_handles, brier_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        reliability_handles + brier_handles,
        reliability_labels + brier_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=6,
    )
    fig.subplots_adjust(
        top=0.96, bottom=0.09, left=0.10, right=0.98, hspace=0.42, wspace=0.50
    )
    data = pd.DataFrame(rows)
    data.attrs["test_period"] = [
        str(predictions.date.min()),
        str(predictions.date.max()),
    ]
    _save(fig, "P12_probability_reliability", data)
    figure_p12_interval_coverage()


def figure_p12_interval_coverage():
    """Validation and test coverage by lead from current five-seed CQR reports."""
    folder = ROOT / "results/ablations/quantile_cqr"
    reports = sorted(folder.glob("metrics_seed_*.json"))
    if len(reports) < 5:
        raise FileNotFoundError(
            f"P12 coverage requires five current quantile reports in {folder}"
        )
    records = []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("dataset_protocol", {}).get("schema_version") != 2:
            raise ValueError(
                f"P12 report is stale or lacks schema-v2 provenance: {path}"
            )
        for scale, leads in report.get("probabilistic", {}).get("scales", {}).items():
            for lead, metrics in leads.items():
                for split in ("validation", "test"):
                    metric = metrics.get(f"{split}_after")
                    if metric is None:
                        continue
                    records.append(
                        {
                            "seed": int(report["seed"]),
                            "scale": scale,
                            "lead": lead,
                            "split": split,
                            "coverage": metric["coverage"],
                            "interval_width": metric["width"],
                            "n": metric["n"],
                        }
                    )
    raw = pd.DataFrame(records)
    if raw.empty or raw.seed.nunique() < 5 or set(raw.split) != {"validation", "test"}:
        raise ValueError(
            "P12 coverage needs validation and test CQR coverage for at least five current seeds"
        )
    summary = raw.groupby(["scale", "lead", "split"], as_index=False).agg(
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        interval_width_mean=("interval_width", "mean"),
        n_seeds=("seed", "nunique"),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 2.65), sharey=True, constrained_layout=True
    )
    colors = {"validation": OKABE_ITO["orange"], "test": OKABE_ITO["blue"]}
    markers = {"validation": "o", "test": "s"}
    leads = ["T+1", "T+2", "T+3", "T+4"]
    for index, (ax, scale) in enumerate(zip(axes, ["spei_3", "spei_6", "spei_12"])):
        subset = summary[summary.scale == scale]
        for split in ("validation", "test"):
            rows = subset[subset.split == split].set_index("lead").reindex(leads)
            if rows.coverage_mean.isna().any():
                raise ValueError(f"P12 interval coverage missing {scale} {split} lead")
            ax.errorbar(
                np.arange(1, 5),
                rows.coverage_mean,
                yerr=rows.coverage_std.fillna(0),
                marker=markers[split],
                color=colors[split],
                linestyle="-" if split == "test" else "--",
                linewidth=0.9,
                markersize=3.2,
                capsize=2,
                label=split.title(),
            )
            for lead, row in zip(leads, rows.itertuples()):
                records.append(
                    {
                        "record_type": "coverage_summary",
                        "scale": scale,
                        "lead": lead,
                        "split": split,
                        "coverage_mean_five_seeds": row.coverage_mean,
                        "coverage_std_five_seeds": row.coverage_std,
                        "mean_interval_width_spei": row.interval_width_mean,
                        "n_seeds": row.n_seeds,
                    }
                )
        ax.axhline(0.8, color=OKABE_ITO["black"], linewidth=0.8, linestyle=":")
        ax.set_xticks(range(1, 5), leads)
        ax.set_xlabel("Forecast lead" if index == 1 else "")
        ax.text(
            0.03,
            0.97,
            f"({chr(97 + index)}) {scale.replace('_', '-')} CQR coverage",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            weight="bold",
        )
        ax.grid(axis="y", linewidth=0.4, alpha=0.45)
    axes[0].set_ylabel("Empirical interval coverage")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="lower right")
    coverage_data = pd.concat(
        [raw.assign(record_type="seed_metric"), pd.DataFrame(records)],
        ignore_index=True,
        sort=False,
    )
    coverage_data.attrs["seeds"] = sorted(raw.seed.unique().tolist())
    coverage_data.attrs["config_path"] = "results/ablations/quantile_cqr/config.yaml"
    _save(fig, "P12_validation_test_coverage", coverage_data)


def _read_gpkg_layer(path: Path, table: str, name_column: str, extent):
    """Read GeoPackage WKB directly when GDAL/Fiona is not installed."""
    geometries = []
    spatial_window = box(*extent)
    with sqlite3.connect(path) as connection:
        for blob, name in connection.execute(
            f"SELECT geom, {name_column} FROM {table}"
        ):
            if not blob or blob[:2] != b"GP":
                continue
            envelope_code = (blob[3] >> 1) & 0b111
            envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_code)
            if envelope_bytes is None:
                raise ValueError(
                    f"Unsupported GeoPackage envelope type {envelope_code} in {path}"
                )
            geometry = wkb.loads(blob[8 + envelope_bytes :])
            if geometry.intersects(spatial_window):
                geometries.append((geometry.intersection(spatial_window), str(name)))
    return geometries


def _plot_boundary(ax, geometry, *, color, linewidth, zorder):
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        x, y = geometry.xy
        ax.plot(x, y, color=color, linewidth=linewidth, zorder=zorder)
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            _plot_boundary(ax, part, color=color, linewidth=linewidth, zorder=zorder)


def _boundary_paths(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield list(geometry.coords)
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _boundary_paths(part)


def _geometry_clip_patch(geometry, transform):
    """Convert an India GADM polygon (including holes) to a Matplotlib clip path."""
    from matplotlib.patches import PathPatch

    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    vertices, codes = [], []
    for polygon in polygons:
        if polygon.geom_type != "Polygon":
            continue
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = np.asarray(ring.coords, dtype=float)
            vertices.extend(coordinates.tolist())
            codes.extend(
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(coordinates) - 2)
                + [MplPath.CLOSEPOLY]
            )
    path = MplPath(np.asarray(vertices), np.asarray(codes, dtype=np.uint8))
    return PathPatch(path, transform=transform)


def figure_p3_elevation_map():
    """Elevation and configured sites over real GADM India administrative outlines."""
    import xarray as xr

    evidence = ROOT / "results/spatial_pipeline/spatial_evidence.nc"
    gpkg = ROOT / "data/raw/spatial/gadm/IND.gpkg"
    if not evidence.is_file():
        raise FileNotFoundError(
            f"P3 spatial elevation evidence is missing: {evidence}. "
            "Run python experiments/run_spatial_pipeline.py after preparing spatial inputs."
        )
    if not gpkg.is_file():
        raise FileNotFoundError(
            f"P3 GADM boundaries are missing: {gpkg}. "
            "Run python data/spatial_download.py --download-missing"
        )
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    with xr.open_dataset(evidence) as dataset:
        if (
            "elevation" not in dataset
            or "lat" not in dataset.coords
            or "lon" not in dataset.coords
        ):
            raise ValueError(
                f"P3 needs elevation, lat, and lon variables in {evidence}"
            )
        lat = np.asarray(dataset.lat.values, dtype=float)
        lon = np.asarray(dataset.lon.values, dtype=float)
        elevation = np.asarray(dataset.elevation.squeeze().values, dtype=float)
        if elevation.shape != (len(lat), len(lon)):
            raise ValueError(
                f"P3 elevation array shape {elevation.shape} does not match coordinates"
            )
        grid_shape = elevation.shape
    extent = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    state_geometries = _read_gpkg_layer(gpkg, "ADM_ADM_1", "NAME_1", extent)
    if not state_geometries:
        raise ValueError(
            "P3 found no India state boundaries intersecting the real elevation grid"
        )
    display_states = [
        (geometry.simplify(0.03, preserve_topology=True), name)
        for geometry, name in state_geometries
    ]
    india_union = state_geometries[0][0]
    for state_geometry, _ in state_geometries[1:]:
        india_union = india_union.union(state_geometry)
    mask = np.array([[india_union.covers(Point(x, y)) for x in lon] for y in lat])
    mask &= np.isfinite(elevation)
    if not mask.any():
        raise ValueError(
            "P3 has no finite elevation cells inside the GADM India boundary"
        )

    def cell_edges(values):
        middle = (values[1:] + values[:-1]) / 2
        return np.concatenate(
            (
                [values[0] - (middle[0] - values[0])],
                middle,
                [values[-1] + (values[-1] - middle[-1])],
            )
        )

    lon_edges, lat_edges = cell_edges(lon), cell_edges(lat)
    # Discrete quantile bins are computed from the real elevation output, never typed from a reference figure.
    breaks = np.unique(np.quantile(elevation[mask], np.linspace(0, 1, 6)))
    if len(breaks) < 3:
        raise ValueError(
            "P3 elevation grid has too few distinct values for a discrete elevation legend"
        )
    palette = plt.get_cmap("terrain")(np.linspace(0.12, 0.88, len(breaks) - 1))
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(breaks, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(7.16, 4.5), constrained_layout=True)
    image = ax.pcolormesh(
        lon_edges,
        lat_edges,
        np.ma.masked_where(~mask, elevation),
        shading="flat",
        cmap=cmap,
        norm=norm,
        rasterized=True,
        edgecolors="none",
        antialiased=False,
    )
    image.set_clip_path(_geometry_clip_patch(india_union, ax.transData))
    for state_geometry, _ in display_states:
        _plot_boundary(
            ax, state_geometry.boundary, color="#303030", linewidth=0.65, zorder=3
        )
    site_rows = []
    for site, spec in config["sites"].items():
        x, y = float(spec["lon"]), float(spec["lat"])
        if (
            extent[0] <= x <= extent[2]
            and extent[1] <= y <= extent[3]
            and india_union.covers(Point(x, y))
        ):
            site_rows.append(
                {
                    "record_type": "site",
                    "site": site,
                    "latitude": y,
                    "longitude": x,
                    "region": spec.get("region"),
                    "elevation_m": np.nan,
                }
            )
    if site_rows:
        sites = pd.DataFrame(site_rows)
        ax.scatter(
            sites.longitude,
            sites.latitude,
            s=20,
            marker="o",
            color=OKABE_ITO["vermillion"],
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        for index, row in sites.iterrows():
            dx, dy = (3, 3) if index % 2 == 0 else (3, -8)
            ax.annotate(
                row.site.replace("_", " "),
                (row.longitude, row.latitude),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.5,
                color="#111111",
                zorder=6,
            )
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(lat)))), adjustable="box")
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_xticks(np.arange(np.ceil(extent[0] / 5) * 5, extent[2] + 0.1, 5))
    ax.set_yticks(np.arange(np.ceil(extent[1] / 5) * 5, extent[3] + 0.1, 5))
    ax.grid(True, linewidth=0.45, linestyle=":", color="#555555", alpha=0.65)
    ax.annotate(
        "N",
        xy=(0.95, 0.91),
        xytext=(0.95, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "black"},
    )
    mean_lat = float(np.mean(lat))
    scale_km = 300
    scale_deg = scale_km / (111.32 * np.cos(np.deg2rad(mean_lat)))
    x0 = extent[0] + 0.05 * (extent[2] - extent[0])
    y0 = extent[1] + 0.055 * (extent[3] - extent[1])
    ax.plot([x0, x0 + scale_deg], [y0, y0], color="black", linewidth=1.3, zorder=7)
    ax.plot([x0, x0], [y0 - 0.12, y0 + 0.12], color="black", linewidth=0.8, zorder=7)
    ax.plot(
        [x0 + scale_deg, x0 + scale_deg],
        [y0 - 0.12, y0 + 0.12],
        color="black",
        linewidth=0.8,
        zorder=7,
    )
    ax.text(
        x0 + scale_deg / 2,
        y0 + 0.22,
        f"{scale_km} km",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )
    fig.colorbar(
        image,
        ax=ax,
        boundaries=breaks,
        ticks=(breaks[:-1] + breaks[1:]) / 2,
        label="Elevation (m)",
        fraction=0.046,
        pad=0.03,
    )
    fig.text(
        0.08,
        0.012,
        f"Elevation from spatial_evidence.nc ({grid_shape[0]} × {grid_shape[1]} grid cells); administrative outlines: GADM 4.1",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#222222",
    )
    rows = [
        {
            "record_type": "elevation_cell",
            "site": "",
            "administrative_area": "",
            "latitude": float(lat[i]),
            "longitude": float(lon[j]),
            "region": "",
            "elevation_m": float(elevation[i, j]),
        }
        for i, j in zip(*np.where(mask))
    ]
    rows.extend({**row, "administrative_area": ""} for row in site_rows)
    for state_geometry, state_name in display_states:
        boundary = state_geometry.boundary.intersection(box(*extent))
        for line_index, line in enumerate(_boundary_paths(boundary)):
            for vertex_index, (x_coord, y_coord) in enumerate(line):
                rows.append(
                    {
                        "record_type": "gadm_state_boundary_vertex",
                        "site": "",
                        "administrative_area": state_name,
                        "line_id": line_index,
                        "vertex_id": vertex_index,
                        "latitude": float(y_coord),
                        "longitude": float(x_coord),
                        "region": "",
                        "elevation_m": np.nan,
                    }
                )
    data = pd.DataFrame(rows)
    data.attrs["test_period"] = None
    data.attrs["map_mean_lat"] = float(np.mean(lat))
    if not (ROOT / "data/raw/spatial/dem/copernicus_dem_glo30.tif").is_file():
        _safe_print(
            "P3 note: plotting only the coarse elevation field already present in spatial_evidence.nc; "
            "native DEM absent. To refresh native spatial sources, run: python data/spatial_download.py --download-missing"
        )
    _save(fig, "P3_study_area_elevation", data, kind="map")


def figure_p1_pipeline():
    """Real-data flowchart and timeline using current dataset/config split dates."""
    from data.loader import split_for_date

    metadata_path = ROOT / "data/dataset_metadata.json"
    config_path = ROOT / "configs/base_diagram.yaml"
    if not metadata_path.is_file() or not config_path.is_file():
        missing = metadata_path if not metadata_path.is_file() else config_path
        raise FileNotFoundError(
            f"P1 source is missing: {missing}; build/update the dataset and config first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2:
        raise ValueError("P1 requires current dataset metadata schema version 2")
    period = metadata["period"]
    begin = f"{period[0]}-01-01"
    end = f"{period[1]}-12-01"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    steps = int(cfg["data"]["seq_len"])
    rows = [
        {
            "kind": "pipeline",
            "name": "Data",
            "detail": "NASA POWER climate\nprecipitation + temperature\nNOAA monthly indices\nONI, DMI, NAO, PDO",
        },
        {
            "kind": "pipeline",
            "name": "PET + SPEI",
            "detail": "Hargreaves-Samani PET\ntrain-period calibration\nSPEI accumulation scales\n3, 6, and 12 months",
        },
        {
            "kind": "pipeline",
            "name": "Windowing",
            "detail": f"{steps} monthly inputs\nobservations through m-1\nindex at month m uses data\nthrough month m-1 only",
        },
        {
            "kind": "pipeline",
            "name": "Forecaster",
            "detail": "LSTM + additive attention\none shared context vector\nthree scale heads\nleads T+1 to T+4",
        },
        {
            "kind": "pipeline",
            "name": "Evaluation",
            "detail": "Chronological test only\nall forecast leads reported\ncompared with persistence\nno train/validation results",
        },
    ]
    month_dates = pd.date_range(begin, end, freq="MS")
    month_splits = [split_for_date(date) for date in month_dates]
    split_rows = []
    group_start = 0
    for index in range(1, len(month_dates) + 1):
        if (
            index == len(month_dates)
            or month_splits[index] != month_splits[group_start]
        ):
            split_name = month_splits[group_start]
            display_name = {
                "train": "Train",
                "val": "Validation",
                "test": "Test",
                None: "Gap",
            }[split_name]
            color = {
                "Train": OKABE_ITO["blue"],
                "Validation": OKABE_ITO["orange"],
                "Test": OKABE_ITO["bluish_green"],
                "Gap": "#BDBDBD",
            }[display_name]
            split_rows.append(
                (
                    display_name,
                    str(month_dates[group_start].date()),
                    str(month_dates[index - 1].date()),
                    color,
                )
            )
            group_start = index
    rows.extend(
        {"kind": "split", "name": name, "start": start, "end": finish}
        for name, start, finish, _ in split_rows
    )
    fig, ax = plt.subplots(figsize=(7.16, 3.9), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x_positions = [0.025, 0.225, 0.425, 0.625, 0.825]
    colors = ["#EAF2F8", "#E8F3EA", "#FFF5DC", "#F4ECF7", "#E9EEF6"]
    for i, (row, x, fill) in enumerate(zip(rows[:5], x_positions, colors)):
        w, y, h = 0.16, 0.57, 0.34
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                facecolor=fill,
                edgecolor="#34495E",
                linewidth=0.8,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.80,
            row["name"],
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=8,
        )
        ax.text(
            x + w / 2,
            y + h * 0.43,
            row["detail"],
            ha="center",
            va="center",
            fontsize=6.5,
            linespacing=1.12,
        )
        if i < 4:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.01, y + h / 2),
                xytext=(x + w + 0.01, y + h / 2),
                arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#34495E"},
            )
    ax.text(
        0.025,
        0.46,
        "Chronological split (monthly targets)",
        ha="left",
        va="center",
        fontsize=7.5,
        weight="bold",
    )
    all_start, all_end = pd.Timestamp(begin), pd.Timestamp(end)
    total_months = (
        (all_end.year - all_start.year) * 12 + all_end.month - all_start.month + 1
    )
    left, width, y, height = 0.04, 0.92, 0.22, 0.16
    for label, start, finish, color in split_rows:
        start_date, end_date = pd.Timestamp(start), pd.Timestamp(finish)
        start_index = (
            (start_date.year - all_start.year) * 12 + start_date.month - all_start.month
        )
        count = (
            (end_date.year - start_date.year) * 12
            + end_date.month
            - start_date.month
            + 1
        )
        x = left + width * start_index / total_months
        w = width * count / total_months
        ax.add_patch(
            plt.Rectangle(
                (x, y), w, height, facecolor=color, edgecolor="white", linewidth=0.8
            )
        )
        if label == "Gap":
            ax.text(x + w / 2, y - 0.035, "gap", ha="center", va="top", fontsize=6.5)
        else:
            ax.text(
                x + w / 2,
                y + height / 2,
                f"{label}\n{start[:4]}-{finish[:4]}",
                ha="center",
                va="center",
                fontsize=6.5,
                weight="bold",
            )
    ax.text(left, 0.15, begin[:4], ha="left", va="center", fontsize=6.5)
    ax.text(left + width, 0.15, end[:4], ha="right", va="center", fontsize=6.5)
    gap_years = [
        start[:4] if start[:4] == finish[:4] else f"{start[:4]}-{finish[:4]}"
        for label, start, finish, _ in split_rows
        if label == "Gap"
    ]
    gap_summary = ", ".join(gap_years) if gap_years else "none"
    ax.text(
        0.04,
        0.055,
        f"Gap periods: {gap_summary}. Chronological split preserves separation; no random split.",
        ha="left",
        va="center",
        fontsize=6.5,
    )
    data = pd.DataFrame(rows)
    test_period = [
        (start, finish) for label, start, finish, _ in split_rows if label == "Test"
    ]
    data.attrs["test_period"] = (
        [test_period[0][0], test_period[-1][1]] if test_period else None
    )
    _save(fig, "P1_data_model_pipeline", data)


def figure_p6_predicted_actual():
    """Dense held-out test predictions with persistence comparator and metrics."""
    paths = sorted((ROOT / "results").glob("test_predictions_seed_*.csv"))
    if not paths:
        raise FileNotFoundError(
            "P6 needs results/test_predictions_seed_<seed>.csv. Run "
            "python evaluate.py --config configs/base_diagram.yaml --checkpoint CHECKPOINT "
            "--seed SEED --output results/seeds/metrics_seed_SEED.json --skip-classical."
        )
    frames = [pd.read_csv(path) for path in paths]
    predictions = pd.concat(frames, ignore_index=True)
    required = {
        "split",
        "seed",
        "site",
        "target_month",
        "scale",
        "lead",
        "observed",
        "prediction_droughtcast",
        "prediction_persistence",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"P6 predictions are missing columns: {sorted(missing)}")
    if set(predictions.split) != {"test"}:
        raise ValueError("P6 accepts chronological test predictions only")
    if predictions.duplicated(["seed", "site", "target_month", "scale", "lead"]).any():
        raise ValueError(
            "P6 prediction rows contain duplicate seed/site/month/scale/lead keys"
        )
    for seed in sorted(predictions.seed.unique()):
        report_path = ROOT / "results" / "seeds" / f"metrics_seed_{int(seed)}.json"
        checkpoint = ROOT / "results" / "seeds" / f"model_seed_{int(seed)}.pt"
        if not report_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"P6 needs current report/checkpoint for seed {seed}: {report_path}; {checkpoint}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        protocol = report.get("dataset_protocol", {})
        if protocol.get("schema_version") != 2:
            raise ValueError(
                f"P6 seed {seed} report is not dataset schema v2; rerun evaluation"
            )
        state = __import__("torch").load(
            checkpoint, map_location="cpu", weights_only=False
        )
        data_path = ROOT / "data/monthly_sites.csv"
        current_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if state.get("artifact_metadata", {}).get("dataset_sha256") != current_hash:
            raise ValueError(
                f"P6 seed {seed} checkpoint was trained on a different dataset; retrain and reevaluate"
            )
    scales = ["spei_3", "spei_6", "spei_12"]
    selected_leads = ["T+1", "T+4"]
    values = predictions[
        predictions.scale.isin(scales) & predictions.lead.isin(selected_leads)
    ]
    all_values = np.concatenate(
        [
            values.observed.to_numpy(),
            values.prediction_droughtcast.to_numpy(),
            values.prediction_persistence.to_numpy(),
        ]
    )
    low, high = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    pad = max((high - low) * 0.025, 0.05)
    limits = (low - pad, high + pad)
    fig, axes = plt.subplots(
        2, 3, figsize=(7.16, 4.65), sharex=True, sharey=True, constrained_layout=True
    )
    hexbin = None
    plotted = []
    for row, lead in enumerate(selected_leads):
        for col, scale in enumerate(scales):
            ax = axes[row, col]
            subset = values[(values.scale == scale) & (values.lead == lead)]
            if subset.empty:
                raise ValueError(f"P6 no test rows for {scale}, {lead}")
            observed = subset.observed.to_numpy(dtype=float)
            model = subset.prediction_droughtcast.to_numpy(dtype=float)
            persistence = subset.prediction_persistence.to_numpy(dtype=float)
            hexbin = ax.hexbin(
                observed,
                model,
                gridsize=35,
                bins="log",
                mincnt=1,
                cmap="Blues",
                linewidths=0.0,
                rasterized=True,
            )
            hist, xedges, yedges = np.histogram2d(
                observed,
                persistence,
                bins=32,
                range=[[limits[0], limits[1]], [limits[0], limits[1]]],
            )
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            nonzero = hist[hist > 0]
            if nonzero.size >= 3:
                levels = np.unique(np.quantile(nonzero, [0.60, 0.82, 0.95]))
                if levels.size:
                    ax.contour(
                        xcenters,
                        ycenters,
                        hist.T,
                        levels=levels,
                        colors=[OKABE_ITO["vermillion"]],
                        linewidths=0.8,
                    )
            ax.plot(
                limits, limits, color=OKABE_ITO["black"], linestyle="--", linewidth=0.8
            )
            model_mse = float(np.mean((model - observed) ** 2))
            pers_mse = float(np.mean((persistence - observed) ** 2))
            model_rmse, persistence_rmse = math.sqrt(model_mse), math.sqrt(pers_mse)
            model_r2 = float(1 - model_mse / np.mean((observed - observed.mean()) ** 2))
            pers_r2 = float(1 - pers_mse / np.mean((observed - observed.mean()) ** 2))
            skill = float(1 - model_mse / pers_mse) if pers_mse > 0 else float("nan")
            metrics = (
                f"Model: R²={model_r2:.2f}, RMSE={model_rmse:.2f}\n"
                f"Persistence: R²={pers_r2:.2f}, RMSE={persistence_rmse:.2f}\n"
                f"Model skill vs persistence={skill:.2f}"
            )
            ax.text(
                0.04,
                0.96,
                metrics,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#777777",
                    "linewidth": 0.4,
                    "alpha": 0.92,
                    "pad": 2,
                },
            )
            panel = chr(97 + row * 3 + col)
            ax.text(
                0.03,
                0.04,
                f"({panel}) {scale.upper().replace('_', '-')} {lead}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=7,
                weight="bold",
            )
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal", adjustable="box")
            tick_start = math.ceil(limits[0] / 2) * 2
            tick_end = math.floor(limits[1] / 2) * 2
            ticks = np.arange(tick_start, tick_end + 0.1, 2)
            if ticks.size < 2:
                ticks = np.linspace(limits[0], limits[1], 3)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            plotted.append(subset.assign(panel=f"({panel})"))
    for ax in axes[-1]:
        ax.set_xlabel("Observed SPEI (unitless)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Forecast SPEI (unitless)")
    colorbar = fig.colorbar(hexbin, ax=axes, location="right", shrink=0.88, pad=0.025)
    maximum_count = max(
        float(collection.get_array().max())
        for collection in fig.axes[0].collections
        if hasattr(collection, "get_array") and collection.get_array().size
    )
    colorbar.set_ticks([tick for tick in (1, 10, 100, 1000) if tick <= maximum_count])
    colorbar.set_label("Model test cases per hexagon (log count)", fontsize=7)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=OKABE_ITO["black"],
                linestyle="--",
                linewidth=0.8,
                label="1:1 line",
            ),
            Line2D(
                [0],
                [0],
                color=OKABE_ITO["vermillion"],
                linewidth=0.8,
                label="Persistence density contours",
            ),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    data = pd.concat(plotted, ignore_index=True)
    data.attrs["test_period"] = [
        str(predictions.target_month.min()),
        str(predictions.target_month.max()),
    ]
    _save(fig, "P6_test_predicted_vs_observed", data)


def figure_p11_case_studies():
    """Plot three objectively selected SPEI-6 drought episodes using held-out quantile forecasts."""
    folder = ROOT / "results/ablations/quantile_cqr"
    paths = sorted(folder.glob("quantile_test_predictions_seed_*.csv"))
    if not paths:
        raise FileNotFoundError(
            "P11 needs results/quantile_test_predictions_seed_<seed>.csv. Run "
            + P_COMMANDS["P11"]
        )
    frames = []
    current_hash = hashlib.sha256(
        (ROOT / "data/monthly_sites.csv").read_bytes()
    ).hexdigest()
    for path in paths:
        frame = pd.read_csv(path)
        seed = int(path.stem.rsplit("_", 1)[1])
        report_path = folder / f"metrics_seed_{seed}.json"
        checkpoint_path = folder / f"model_seed_{seed}.pt"
        if not report_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"P11 needs current quantile report and checkpoint for seed {seed}; run {P_COMMANDS['P11']}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        state = __import__("torch").load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            report.get("dataset_protocol", {}).get("schema_version") != 2
            or state.get("artifact_metadata", {}).get("dataset_sha256") != current_hash
        ):
            raise ValueError(
                f"P11 seed {seed} quantile output is stale; rerun {P_COMMANDS['P11']}"
            )
        required = {
            "split",
            "seed",
            "site",
            "target_month",
            "scale",
            "lead",
            "observed",
            "q10",
            "q50",
            "q90",
            "cqr_lower",
            "cqr_upper",
        }
        if required.difference(frame.columns):
            raise ValueError(
                f"P11 {path.name} lacks required columns: {sorted(required.difference(frame.columns))}"
            )
        if set(frame.split) != {"test"} or set(frame.seed) != {seed}:
            raise ValueError(
                f"P11 accepts only chronological test rows with matching seed in {path.name}"
            )
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    keys = ["seed", "site", "target_month", "scale", "lead"]
    if raw.duplicated(keys).any():
        raise ValueError(
            "P11 quantile predictions contain duplicate seed/site/month/scale/lead rows"
        )
    obs = (
        raw[raw.scale == "spei_6"]
        .groupby(["site", "target_month"], as_index=False)
        .agg(
            observed_min=("observed", "min"),
            observed_max=("observed", "max"),
            observed=("observed", "first"),
        )
    )
    if (obs.observed_min - obs.observed_max).abs().max() > 1e-6:
        raise ValueError("P11 test observations differ across seeds or forecast leads")
    obs["target_month"] = pd.to_datetime(obs.target_month)
    episodes = []
    for site, site_obs in obs.sort_values("target_month").groupby("site"):
        drought = site_obs[site_obs.observed <= -1.0].copy().sort_values("target_month")
        if drought.empty:
            continue
        month_index = drought.target_month.dt.year * 12 + drought.target_month.dt.month
        group_id = month_index.diff().ne(1).cumsum()
        for _, event in drought.groupby(group_id):
            if len(event) >= 2:
                episodes.append(
                    {
                        "site": site,
                        "start": event.target_month.min(),
                        "end": event.target_month.max(),
                        "minimum_spei6": event.observed.min(),
                        "duration_months": len(event),
                    }
                )
    episodes = sorted(
        episodes, key=lambda event: (event["minimum_spei6"], -event["duration_months"])
    )
    selected = []
    for event in episodes:
        if any(
            event["site"] == chosen["site"]
            and not (event["end"] < chosen["start"] or event["start"] > chosen["end"])
            for chosen in selected
        ):
            continue
        selected.append(event)
        if len(selected) == 3:
            break
    if len(selected) < 2:
        raise ValueError(
            f"P11 found only {len(selected)} multi-month test-period SPEI-6 drought episodes at SPEI <= -1"
        )
    raw["target_month"] = pd.to_datetime(raw.target_month)
    summary = (
        raw[raw.scale == "spei_6"]
        .groupby(["site", "target_month", "lead"], as_index=False)
        .agg(
            observed=("observed", "first"),
            forecast_q50=("q50", "mean"),
            lower=("cqr_lower", "mean"),
            upper=("cqr_upper", "mean"),
        )
    )
    colors = [
        OKABE_ITO[key] for key in ("blue", "orange", "bluish_green", "vermillion")
    ]
    styles = ["-", "--", "-.", ":"]
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(7.16, 5.0),
        sharex=False,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    plotted = []
    for panel_index, (ax, event) in enumerate(zip(axes, selected)):
        start = event["start"] - pd.DateOffset(months=1)
        end = event["end"] + pd.DateOffset(months=1)
        actual = obs[
            (obs.site == event["site"]) & obs.target_month.between(start, end)
        ].sort_values("target_month")
        ax.plot(
            actual.target_month,
            actual.observed,
            color=OKABE_ITO["black"],
            linewidth=1.1,
            marker="o",
            markersize=3.0,
            label="Observed SPEI-6",
        )
        panel_rows = []
        for lead_index, lead in enumerate(("T+1", "T+2", "T+3", "T+4")):
            forecast = summary[
                (summary.site == event["site"])
                & (summary.lead == lead)
                & summary.target_month.between(start, end)
            ].sort_values("target_month")
            if forecast.empty:
                raise ValueError(
                    f"P11 episode has no {lead} forecasts for site {event['site']}"
                )
            ax.fill_between(
                forecast.target_month,
                forecast.lower,
                forecast.upper,
                color=colors[lead_index],
                alpha=0.09,
                linewidth=0,
            )
            ax.plot(
                forecast.target_month,
                forecast.forecast_q50,
                color=colors[lead_index],
                linestyle=styles[lead_index],
                linewidth=0.9,
                marker="s",
                markersize=3.0,
                label=f"{lead} forecast median",
            )
            panel_rows.append(
                forecast.assign(
                    episode_site=event["site"],
                    episode_start=event["start"].strftime("%Y-%m"),
                    episode_end=event["end"].strftime("%Y-%m"),
                    episode_minimum_spei6=event["minimum_spei6"],
                    panel=f"({chr(97 + panel_index)})",
                )
            )
        for threshold in (-1.0, -1.5, -2.0):
            ax.axhline(
                threshold, color="#777777", linestyle=(0, (2, 2)), linewidth=0.65
            )
        label = (
            f"({chr(97 + panel_index)}) {event['site'].replace('_', ' ')}; "
            f"{event['start']:%Y-%m} to {event['end']:%Y-%m}"
        )
        ax.text(
            0.015,
            0.97,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        ax.set_ylabel("SPEI-6 (unitless)")
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)
        plotted.extend(panel_rows)
    axes[-1].set_xlabel("Target month")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=5,
        frameon=False,
        fontsize=6.5,
    )
    data = pd.concat(plotted, ignore_index=True)
    data.attrs["test_period"] = [
        str(raw.target_month.min()),
        str(raw.target_month.max()),
    ]
    data.attrs["seeds"] = sorted(raw.seed.unique().tolist())
    data.attrs["config_path"] = "results/ablations/quantile_cqr/config.yaml"
    _save(fig, "P11_case_studies", data)


def figure_p13_interpretability():
    """Render validation-only attention, lag occlusion, and deterministic permutation diagnostics."""
    attention_path = ROOT / "results/attention_occlusion_validation.csv"
    feature_path = ROOT / "results/feature_importance_validation.csv"
    report_path = ROOT / "results/interpretability_validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("split") != "validation only":
        raise ValueError(
            "P13 only accepts diagnostics explicitly computed on validation windows"
        )
    attention = pd.read_csv(attention_path)
    features = pd.read_csv(feature_path)
    required_attention = {
        "seed",
        "split",
        "scale",
        "lead",
        "season",
        "input_step",
        "attention",
        "occlusion_delta_validation_mse",
    }
    required_features = {
        "seed",
        "split",
        "feature",
        "feature_group",
        "input_step",
        "scale",
        "lead",
        "delta_validation_mse",
    }
    if required_attention.difference(attention.columns) or required_features.difference(
        features.columns
    ):
        raise ValueError(
            "P13 validation CSV schemas do not contain required attention/feature diagnostics"
        )
    if set(attention.split) != {"validation"} or set(features.split) != {"validation"}:
        raise ValueError("P13 rejects any diagnostic rows that are not validation-only")
    if set(attention.seed) != set(features.seed):
        raise ValueError(
            "P13 attention and feature diagnostics have different seed provenance"
        )
    scales = ["spei_3", "spei_6", "spei_12"]
    seasons = ["DJF", "MAM", "JJA", "SON"]
    lags = [f"t-{lag}" for lag in range(6, 0, -1)]
    attention_summary = attention.groupby(
        ["scale", "season", "input_step"], as_index=False
    ).agg(attention_mean=("attention", "mean"))
    attention_rows, attention_labels = [], []
    for scale in scales:
        for season in seasons:
            row = attention_summary[
                (attention_summary.scale == scale)
                & (attention_summary.season == season)
            ].set_index("input_step")
            if row.empty:
                continue
            attention_rows.append(row.reindex(lags).attention_mean.to_numpy())
            attention_labels.append(f"{scale.upper().replace('_', '-')} / {season}")
    if not attention_rows:
        raise ValueError("P13 contains no seasonal attention rows")
    attention_matrix = np.vstack(attention_rows)

    occlusion_summary = attention.groupby(["scale", "input_step"], as_index=False).agg(
        mean_delta_mse=("occlusion_delta_validation_mse", "mean"),
        sd_across_leads_seasons=("occlusion_delta_validation_mse", "std"),
    )
    feature_summary = features.groupby(
        ["feature", "feature_group", "input_step"], as_index=False
    ).agg(mean_delta_mse=("delta_validation_mse", "mean"))
    feature_order = list(dict.fromkeys(features.feature.tolist()))
    feature_labels = {
        "PRECTOTCORR": "Precipitation",
        "T2M": "Mean temperature",
        "T2MDEW": "Dew-point temperature",
        "WS10M": "Wind speed",
        "ALLSKY_SFC_SW_DWN": "Shortwave radiation",
        "oni": "ONI",
        "dmi": "DMI",
        "nao": "NAO",
        "pdo": "PDO",
        "spei_3_obs": "SPEI-3 observed",
        "spei_6_obs": "SPEI-6 observed",
        "spei_12_obs": "SPEI-12 observed",
    }
    feature_matrix = np.vstack(
        [
            feature_summary[feature_summary.feature == feature]
            .set_index("input_step")
            .reindex(lags)
            .mean_delta_mse.to_numpy()
            for feature in feature_order
        ]
    )
    magnitude = float(np.nanmax(np.abs(feature_matrix)))
    if not np.isfinite(magnitude) or magnitude == 0:
        magnitude = 1.0

    fig = plt.figure(figsize=(7.16, 5.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax_attention = fig.add_subplot(grid[0, :])
    ax_occlusion = fig.add_subplot(grid[1, 0])
    ax_features = fig.add_subplot(grid[1, 1])
    att_image = ax_attention.imshow(
        attention_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=max(0.2, float(np.nanmax(attention_matrix))),
    )
    ax_attention.set_yticks(range(len(attention_labels)), attention_labels)
    ax_attention.set_xticks(range(len(lags)), lags)
    ax_attention.set_xlabel("Input lag (months before target)")
    ax_attention.set_ylabel("Accumulation scale and local season")
    ax_attention.text(
        -0.065,
        1.02,
        "(a)",
        transform=ax_attention.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.colorbar(
        att_image,
        ax=ax_attention,
        fraction=0.018,
        pad=0.012,
        label="Mean attention weight",
    )

    palette = {
        "spei_3": OKABE_ITO["blue"],
        "spei_6": OKABE_ITO["orange"],
        "spei_12": OKABE_ITO["bluish_green"],
    }
    for scale in scales:
        rows = (
            occlusion_summary[occlusion_summary.scale == scale]
            .set_index("input_step")
            .reindex(lags)
        )
        if rows.mean_delta_mse.isna().any():
            raise ValueError(f"P13 occlusion rows are incomplete for {scale}")
        ax_occlusion.errorbar(
            range(len(lags)),
            rows.mean_delta_mse,
            yerr=rows.sd_across_leads_seasons.fillna(0),
            marker="o",
            markersize=3.0,
            linewidth=0.9,
            color=palette[scale],
            label=scale.upper().replace("_", "-"),
            capsize=1.5,
        )
    ax_occlusion.axhline(0, color=OKABE_ITO["black"], linestyle=":", linewidth=0.8)
    ax_occlusion.set_xticks(range(len(lags)), lags)
    ax_occlusion.set_xlabel("Input lag (months before target)")
    ax_occlusion.set_ylabel("Occlusion change in validation MSE (SPEI²)")
    ax_occlusion.text(
        -0.13,
        1.03,
        "(b)",
        transform=ax_occlusion.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax_occlusion.legend(frameon=False, fontsize=6.5, ncol=3, loc="best")
    ax_occlusion.grid(axis="y", linewidth=0.35, alpha=0.4)

    feature_image = ax_features.imshow(
        feature_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-magnitude,
        vmax=magnitude,
    )
    ax_features.set_yticks(
        range(len(feature_order)),
        [feature_labels.get(name, name) for name in feature_order],
    )
    ax_features.set_xticks(range(len(lags)), lags)
    ax_features.set_xlabel("Input lag (months before target)")
    ax_features.text(
        -0.13,
        1.03,
        "(c)",
        transform=ax_features.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax_features.tick_params(axis="both", labelsize=6.5)
    group_colors = {
        "local climate": OKABE_ITO["orange"],
        "ocean-teleconnection": OKABE_ITO["blue"],
        "SPEI state": OKABE_ITO["bluish_green"],
    }
    for tick, feature in zip(ax_features.get_yticklabels(), feature_order):
        group = features.loc[features.feature == feature, "feature_group"].iloc[0]
        tick.set_color(group_colors.get(group, OKABE_ITO["black"]))
    fig.colorbar(
        feature_image,
        ax=ax_features,
        fraction=0.045,
        pad=0.025,
        label="Permutation change in validation MSE (SPEI²)",
    )

    plotted_attention = attention_summary.assign(panel="attention", split="validation")
    plotted_occlusion = occlusion_summary.assign(panel="occlusion", split="validation")
    plotted_features = feature_summary.assign(
        panel="feature_permutation", split="validation"
    )
    data = pd.concat(
        [plotted_attention, plotted_occlusion, plotted_features],
        ignore_index=True,
        sort=False,
    )
    data.attrs["seeds"] = sorted(attention.seed.unique().tolist())
    data.attrs["config_path"] = "configs/base_diagram.yaml"
    data.attrs["test_period"] = None
    _save(fig, "P13_validation_diagnostics", data)


def figure_p14_ablation_forest():
    """Render five-seed ablation deltas and held-region test skill intervals."""
    forest_path = ROOT / "results/ablation_forest.csv"
    regions_path = ROOT / "results/leave_region_out_skill.csv"
    forest = pd.read_csv(forest_path)
    regions = pd.read_csv(regions_path)
    required_forest = {"variant", "delta_rmse", "lower", "upper", "n_seeds"}
    required_regions = {
        "heldout_region",
        "seed",
        "scale",
        "lead",
        "skill_vs_persistence",
    }
    if required_forest.difference(forest.columns):
        raise ValueError(
            f"P14 ablation forest CSV lacks columns {sorted(required_forest.difference(forest.columns))}"
        )
    if required_regions.difference(regions.columns):
        raise ValueError(
            f"P14 leave-region CSV lacks columns {sorted(required_regions.difference(regions.columns))}"
        )
    if "base_diagram" not in set(forest.variant) or (forest.n_seeds < 5).any():
        raise ValueError(
            "P14 ablation forest requires base_diagram and at least five seeds per variant"
        )
    group_sizes = regions.groupby(["heldout_region", "scale", "lead"]).seed.nunique()
    if group_sizes.empty or (group_sizes < 5).any():
        raise ValueError(
            "P14 leave-region results require at least five current seeds per region, scale, and lead"
        )
    from scipy.stats import t as student_t

    forest = forest.sort_values("delta_rmse", ascending=True).reset_index(drop=True)
    forest_data = forest.copy()
    forest_data.attrs["seeds"] = sorted(
        {
            seed
            for value in forest.get("seeds", pd.Series(dtype=str)).dropna()
            for seed in str(value).split(",")
            if seed
        },
        key=int,
    )
    test_predictions = ROOT / "results/test_predictions_seed_42.csv"
    if not test_predictions.is_file():
        raise FileNotFoundError(
            f"P14 provenance requires chronological test predictions: {test_predictions}"
        )
    base_period = pd.read_csv(test_predictions, usecols=["target_month"])
    test_period = [
        str(base_period.target_month.min()),
        str(base_period.target_month.max()),
    ]
    forest_data.attrs["test_period"] = test_period
    forest_data.attrs["config_path"] = "configs/base_diagram.yaml"
    fig, ax = plt.subplots(
        figsize=(7.16, max(2.4, 0.30 * len(forest) + 0.8)), constrained_layout=True
    )
    y = np.arange(len(forest))
    colors = [
        OKABE_ITO["black"] if v == "base_diagram" else OKABE_ITO["blue"]
        for v in forest.variant
    ]
    for index, row in forest.iterrows():
        ax.errorbar(
            row.delta_rmse,
            y[index],
            xerr=[[row.delta_rmse - row.lower], [row.upper - row.delta_rmse]],
            fmt="o",
            markersize=4,
            capsize=2,
            linewidth=0.9,
            color=colors[index],
            markeredgecolor=OKABE_ITO["black"],
        )
    ax.axvline(0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.8)
    ax.set_yticks(
        y,
        [
            "Base diagram" if x == "base_diagram" else x.replace("_", " ")
            for x in forest.variant
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Change in test RMSE vs base diagram (SPEI units)")
    ax.set_ylabel("Ablation variant")
    ax.grid(axis="x", linewidth=0.4, alpha=0.45)
    ax.text(
        -0.08,
        1.02,
        "(a)",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )
    _save(fig, "P14_ablation_forest", forest_data)

    scales = ["spei_3", "spei_6", "spei_12"]
    leads = ["T+1", "T+2", "T+3", "T+4"]
    colors = [OKABE_ITO[k] for k in ("blue", "orange", "bluish_green", "vermillion")]
    markers = ["o", "s", "^", "D"]
    rows = []
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 3.0), sharey=True, constrained_layout=True
    )
    for scale_index, (ax, scale) in enumerate(zip(axes, scales)):
        subset = regions[regions.scale == scale]
        if subset.empty:
            raise ValueError(f"P14 leave-region results have no rows for {scale}")
        held_regions = sorted(subset.heldout_region.unique())
        x = np.arange(len(held_regions))
        offsets = np.linspace(-0.27, 0.27, len(leads))
        for lead_index, lead in enumerate(leads):
            means, lows, highs = [], [], []
            for region in held_regions:
                values = subset[
                    (subset.heldout_region == region) & (subset.lead == lead)
                ]
                vals = values.skill_vs_persistence.dropna().to_numpy(dtype=float)
                if len(vals) < 5:
                    raise ValueError(
                        f"P14 needs five seeds for {region}/{scale}/{lead}; got {len(vals)}"
                    )
                mean = float(np.mean(vals))
                half = float(
                    student_t.ppf(0.975, len(vals) - 1)
                    * np.std(vals, ddof=1)
                    / np.sqrt(len(vals))
                )
                means.append(mean)
                lows.append(mean - half)
                highs.append(mean + half)
                rows.append(
                    {
                        "heldout_region": region,
                        "scale": scale,
                        "lead": lead,
                        "skill_mean": mean,
                        "ci95_lower": mean - half,
                        "ci95_upper": mean + half,
                        "n_seeds": len(vals),
                        "seeds": ",".join(map(str, sorted(values.seed.unique()))),
                    }
                )
            ax.errorbar(
                x + offsets[lead_index],
                means,
                yerr=[
                    np.asarray(means) - np.asarray(lows),
                    np.asarray(highs) - np.asarray(means),
                ],
                color=colors[lead_index],
                marker=markers[lead_index],
                markersize=3.0,
                linewidth=0.85,
                capsize=1.6,
                label=lead,
            )
        ax.axhline(0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.8)
        ax.set_xticks(
            x,
            [name.replace("_", " ") for name in held_regions],
            rotation=35,
            ha="right",
        )
        ax.set_xlabel("Held-out region" if scale_index == 1 else "")
        ax.grid(axis="y", linewidth=0.4, alpha=0.45)
        ax.text(
            0.03,
            0.97,
            f"({chr(98 + scale_index)}) {scale.replace('_', '-')}",
            transform=ax.transAxes,
            fontsize=7,
            fontweight="bold",
            va="top",
        )
    axes[0].set_ylabel("Skill vs persistence (unitless)")
    axes[-1].legend(
        frameon=False, fontsize=6.5, title="Forecast lead", title_fontsize=6.5
    )
    region_data = pd.DataFrame(rows)
    region_data.attrs["seeds"] = sorted(regions.seed.unique().tolist())
    region_data.attrs["test_period"] = test_period
    region_data.attrs["config_path"] = "configs/base_diagram.yaml"
    _save(fig, "P14_leave_region_out", region_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="List missing or insufficient inputs per figure",
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all supported figures"
    )
    parser.add_argument("--only", help="Comma-separated figure IDs, e.g. P3,P6,P10")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero when any requested figure is blocked",
    )
    parser.add_argument(
        "--latex", action="store_true", help="Generate LaTeX figure/table snippets"
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Run fixed-size PDF/PNG and font QA on generated figures",
    )
    args = parser.parse_args()
    apply_style()
    missing = missing_inputs()
    if args.latex:
        write_latex()
        return 0
    if args.check:
        for figure in [f"P{i}" for i in range(1, 16)]:
            entries = missing.get(figure, [])
            if figure in {"P2", "P5", "P7", "P8", "P9", "P12", "P15"} and not entries:
                entries = [
                    "renderer not available or input provenance has not passed validation"
                ]
            print(
                f"{figure}: blocked — " + "; ".join(entries)
                if entries
                else f"{figure}: inputs ready"
            )
        return (
            1 if any(name in missing for name in (f"P{i}" for i in range(1, 16))) else 0
        )

    renderers = {
        "P1": figure_p1_pipeline,
        "P3": figure_p3_elevation_map,
        "P4": figure_f1,
        "P6": figure_p6_predicted_actual,
        "P10": figure_g8,
        "P11": figure_p11_case_studies,
        "P13": figure_p13_interpretability,
        "P12": figure_p12_interval_coverage,
        "P14": figure_p14_ablation_forest,
    }
    if args.only:
        requested = [
            item.strip().upper() for item in args.only.split(",") if item.strip()
        ]
    else:
        requested = [f"P{i}" for i in range(1, 16)] if args.all else list(renderers)
    known_ids = {f"P{i}" for i in range(1, 16)}
    unknown = sorted(set(requested) - known_ids)
    if unknown:
        parser.error(f"unknown figure id(s): {', '.join(unknown)}")
    errors = []
    for name in requested:
        if name not in renderers:
            details = missing.get(name, []) or [
                "renderer not implemented from validated project outputs"
            ]
            _safe_print(f"{name}: blocked - " + "; ".join(details))
            errors.append(f"{name}: blocked")
            continue
        if name in missing:
            _safe_print(f"{name}: blocked - " + "; ".join(missing[name]))
            errors.append(f"{name}: blocked")
            continue
        try:
            renderers[name]()
            _safe_print(f"{name}: generated PDF, PNG, plotted CSV, and provenance JSON")
        except Exception as exc:  # noqa: BLE001 - record a blocked figure and continue reporting others
            _safe_print(f"{name}: blocked - {exc}")
            errors.append(f"{name}: {exc}")
    if args.qa:
        for result in QUALITY_RESULTS:
            print(
                f"QA {result['figure']}: {'PASS' if result['ok'] else 'FAIL'} "
                f"({result['pdf_width_in']} in PDF; {result['png_width_px']} px PNG)"
            )
            if not result["ok"]:
                errors.append(f"QA failed for {result['figure']}")
    generated = [item["figure"] for item in QUALITY_RESULTS if item["ok"]]
    if generated:
        _write_contact_sheet(generated)
    if args.all:
        _write_readme(missing, generated)
    if errors:
        print("Figures blocked or failed:\n" + "\n".join(errors))
        return 1 if args.strict or args.all or args.only else 0
    return 0


def write_latex():
    latex = ROOT / "paper" / "latex"
    latex.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by python make_paper_figures.py --latex"]
    captions = {
        "P1_data_model_pipeline": "Data processing and chronological split. The shaded timeline marks train, validation, and held-out test periods; gaps separate fitting and evaluation windows.",
        "P3_study_area_elevation": "Study area and configured sites over the available elevation evidence grid; the grid resolution and boundary source are recorded in the plotted data.",
        "P4_architecture": "DroughtCast architecture. SPEI-3, SPEI-6, and SPEI-12 denote accumulation scales; each output head forecasts leads T+1 through T+4.",
        "P6_test_predicted_vs_observed": "Predicted versus observed values on the chronological test period. Density shows case counts; persistence contours provide the baseline comparison.",
        "P10_site_skill": "Site-level mean T+4 skill relative to persistence across SPEI accumulation scales during the chronological test period.",
        "P11_case_studies": "Observed SPEI-6 and CQR forecast intervals for objectively selected multi-month drought episodes during the chronological test period.",
        "P13_validation_diagnostics": "Validation-only attention and deterministic perturbation diagnostics. Attention weights are descriptive diagnostics, not causal explanations; color-coded feature labels identify local climate, ocean teleconnection, and SPEI state inputs.",
        "P12_validation_test_coverage": "Validation and test interval coverage by accumulation scale and forecast lead; error bars show seed-to-seed standard deviation.",
        "P14_ablation_forest": "Ablation effects are mean per-seed test RMSE differences from the base diagram across SPEI accumulation scales and T+1..T+4; bars show two-sided 95% t intervals across seeds.",
        "P14_leave_region_out": "Skill relative to persistence on chronological test windows from each held-out region; points and error bars show five-seed means and 95% t intervals for each SPEI accumulation scale and forecast lead.",
    }
    for figure, caption in captions.items():
        if not (OUT / f"{figure}.pdf").is_file():
            continue
        provenance_path = PROVENANCE / f"{figure}.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else {}
        )
        seeds = provenance.get("seeds", [])
        seed_text = (
            ", ".join(str(seed) for seed in seeds) if seeds else "not applicable"
        )
        period = provenance.get("test_period")
        if figure in {
            "P6_test_predicted_vs_observed",
            "P10_site_skill",
            "P11_case_studies",
        }:
            caption += f" Chronological test period: {period}; seed(s): {seed_text}."
        if figure == "P6_test_predicted_vs_observed":
            caption += " Hexbin color indicates test-case density; orange contours show persistence density."
        if figure == "P11_case_studies":
            caption += " Shading is the validation-calibrated CQR interval; episodes are selected by observed SPEI-6 <= -1."
        if figure == "P10_site_skill":
            caption += (
                " Marker color encodes mean T+4 skill across SPEI accumulation scales."
            )
        if figure == "P13_validation_diagnostics":
            caption += f" Validation windows only; seed(s): {seed_text}. Attention is a diagnostic, not an explanation."
        if figure.startswith("P14_"):
            caption += f" Seeds: {seed_text}."
        width = (
            r"\textwidth"
            if figure
            in {
                "P1_data_model_pipeline",
                "P4_architecture",
                "P6_test_predicted_vs_observed",
            }
            else r"\columnwidth"
        )
        lines += [
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width={width}]{{figures/{figure}.pdf}}",
            rf"\caption{{{caption}}}",
            rf"\label{{fig:{figure.lower()}}}",
            r"\end{figure}",
            "",
        ]
    (latex / "figures.tex").write_text("\n".join(lines), encoding="utf-8")
    (latex / "tables.tex").write_text(
        "% No validated multi-seed result table is available yet. Regenerate current five-seed evaluation results first.\n",
        encoding="utf-8",
    )


def _write_contact_sheet(names):
    from PIL import Image, ImageDraw

    thumbs = []
    for name in names:
        path = OUT / f"{name}.png"
        if not path.is_file():
            continue
        with Image.open(path) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((720, 520))
            canvas = Image.new("RGB", (740, 560), "white")
            canvas.paste(thumb, ((740 - thumb.width) // 2, 24))
            ImageDraw.Draw(canvas).text((12, 6), name, fill="black")
            thumbs.append(canvas)
    if not thumbs:
        return
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 740, rows * 560), "#dddddd")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 740, (index // columns) * 560))
    sheet.save(OUT / "contact_sheet.png", dpi=(150, 150))


def _write_readme(missing, generated):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    captions = {
        "P1": (
            "Data processing and chronological train/validation/test timeline.",
            "data/dataset_metadata.json; configs/base_diagram.yaml; data/loader.py",
            "P1_data_model_pipeline",
        ),
        "P2": (
            "PRISMA literature flow using only user-supplied systematic-review counts.",
            "paper/prisma_counts.yaml",
            None,
        ),
        "P3": (
            "Study area, sites, GADM outlines, and elevation values from the available spatial evidence grid.",
            "results/spatial_pipeline/spatial_evidence.nc; data/raw/spatial/gadm/IND.gpkg; configs/config.yaml",
            "P3_study_area_elevation",
        ),
        "P4": (
            "Six-phase model architecture from the base diagram config.",
            "configs/base_diagram.yaml",
            "P4_architecture",
        ),
        "P5": (
            "Training and validation losses with selected checkpoint epoch.",
            "results/seeds/history_seed_*.json",
            None,
        ),
        "P6": (
            "Test-period predicted versus observed density with persistence diagnostics.",
            "results/test_predictions_seed_*.csv; results/seeds/metrics_seed_*.json; results/seeds/model_seed_*.pt",
            "P6_test_predicted_vs_observed",
        ),
        "P7/P8": (
            "Multi-seed comparison table and skill by lead.",
            "results/run_manifest.json; results/summary.csv; results/per_seed.csv",
            None,
        ),
        "P9": (
            "Test-period gridded observations, forecasts, errors, interval width, and event probability.",
            "results/spatial_pipeline/spatial_evidence.nc",
            None,
        ),
        "P10": (
            "Site-level T+4 skill relative to persistence.",
            "results/site_skill_t4.csv; configs/config.yaml",
            "P10_site_skill",
        ),
        "P11": (
            "Observed SPEI-6 and quantile forecasts for the three most severe non-overlapping multi-month test drought episodes (threshold SPEI <= -1).",
            "results/quantile_test_predictions_seed_*.csv; results/ablations/quantile_cqr/metrics_seed_*.json; results/ablations/quantile_cqr/model_seed_*.pt",
            "P11_case_studies",
        ),
        "P12": (
            "Quantile reliability and validation/test coverage.",
            "results/probability_calibration/*; results/ablations/quantile_cqr/metrics_seed_*.json",
            "P12_validation_test_coverage",
        ),
        "P13": (
            "Validation-only attention by scale, season, and lag, plus lag occlusion and feature permutation diagnostics. Attention is descriptive, not an explanation.",
            "results/interpretability_validation.json; results/attention_occlusion_validation.csv; results/feature_importance_validation.csv",
            "P13_validation_diagnostics",
        ),
        "P14": (
            "Ablation and leave-region-out skill intervals.",
            "results/ablation_forest.csv; results/leave_region_out_skill.csv",
            "P14_ablation_forest; P14_leave_region_out",
        ),
        "P15": (
            "Computed SPEI against official SPEIbase.",
            "results/speibase_validation_points.csv",
            None,
        ),
    }
    lines = [
        "# Paper figure assets",
        "",
        f"Generation commit: `{commit}`.",
        "",
        "Figures use real project result files only. Blocked items are not drawn.",
        "",
        "| ID | Caption draft | Source files | Regeneration command | State |",
        "|---|---|---|---|---|",
    ]
    commands = {
        "P1": "python make_paper_figures.py --only P1 --strict",
        "P2": "fill paper/prisma_counts.yaml with real review counts; python make_paper_figures.py --only P2 --strict",
        "P3": "python make_paper_figures.py --only P3 --strict",
        "P4": "python make_paper_figures.py --only P4 --strict",
        "P5": "python experiments/run_all.py --seeds 42 101 2024 2025 2026; python make_paper_figures.py --only P5 --strict",
        "P6": "python evaluate.py --config configs/base_diagram.yaml --checkpoint results/seeds/model_seed_42.pt --seed 42 --skip-classical --output results/seeds/metrics_seed_42.json; python make_paper_figures.py --only P6 --strict",
        "P7/P8": "python experiments/run_all.py --seeds 42 101 2024 2025 2026; python make_paper_figures.py --only P7,P8 --strict",
        "P9": P_COMMANDS["P9"],
        "P10": "python evaluate.py --config configs/base_diagram.yaml --checkpoint results/seeds/model_seed_42.pt --seed 42 --skip-classical --output results/seeds/metrics_seed_42.json; python make_paper_figures.py --only P10 --strict",
        "P11": P_COMMANDS["P11"],
        "P12": P_COMMANDS["P12"],
        "P13": P_COMMANDS["P13"],
        "P14": P_COMMANDS["P14"],
        "P15": "python -m data.build_dataset --speibase data/raw/spatial/speibase; python make_paper_figures.py --only P15 --strict",
    }
    for identifier, (caption, sources, stem) in captions.items():
        command = commands[identifier]
        state = (
            "Generated"
            if stem and stem in generated
            else "Blocked: " + "; ".join(missing.get(identifier.split("/")[0], []))
        )
        if state == "Blocked: ":
            state = "Blocked: renderer or validated inputs unavailable"
        lines.append(
            f"| {identifier} | {caption} | `{sources}` | `{command}` | {state} |"
        )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

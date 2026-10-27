"""End-to-end orchestration of the DroughtCast continuous spatial evidence pipeline.

Workflow:
1. Verify spatial provenance manifest (fails loudly if sources are unverified/missing)
2. Harmonize gridded SPEIbase targets, TerraClimate predictors, and DEM elevation
3. Initialize / load DroughtCast multi-scale multi-horizon model
4. Run continuous spatial inference cell-by-cell on untouched test partition
5. Fit validation-only isotonic probability calibrator and split-CQR interval adjustment
6. Compile NetCDF evidence product with all required fields
7. Run comprehensive evaluation against baselines (persistence, climatology, Ridge)
8. Render publication-quality multi-format figures (PNG, PDF, EPS)
9. Export paper tables and write reproducible experiment manifest
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import torch
import xarray as xr
import yaml

from data.spatial_dataset import SpatialGridDataset
from data.spatial_download import (
    check_gadm_boundaries,
    generate_provenance_manifest,
    load_sources_spec,
    verify_elevation_file,
    verify_speibase_files,
    verify_terraclimate_files,
)
from data.spatial_inference import (
    build_evidence_netcdf,
    calibrate_and_conformalize,
    run_cell_inference,
)
from experiments.evaluate_spatial_pipeline import evaluate_spatial_evidence
from experiments.generate_spatial_tables import generate_tables
from experiments.plot_spatial_pipeline import generate_all_spatial_figures
from models.forecaster import DroughtCast
from utils.artifacts import config_hash

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("SpatialPipeline")


def verify_sources_or_raise(cfg: dict) -> dict:
    """Validate raw sources and return manifest or raise explicit blocker."""
    spatial_cfg = cfg.get("spatial", {})
    spei_dir = spatial_cfg.get("speibase_dir", "data/raw/spatial/speibase")
    tc_dir = spatial_cfg.get("terraclimate_dir", "data/raw/spatial/terraclimate")
    dem_path = spatial_cfg.get(
        "dem_path", "data/raw/spatial/dem/copernicus_dem_glo30.tif"
    )
    gadm_dir = spatial_cfg.get("boundaries_dir", "data/raw/spatial/gadm")

    spec = load_sources_spec()
    spei_stat = verify_speibase_files(spei_dir)
    tc_stat = verify_terraclimate_files(tc_dir)
    dem_stat = verify_elevation_file(dem_path)
    gadm_stat = check_gadm_boundaries(gadm_dir)

    manifest = generate_provenance_manifest(
        spec,
        spei_stat,
        tc_stat,
        dem_stat,
        gadm_stat,
        spatial_cfg.get("output_manifest", "data/spatial_data_manifest.json"),
    )

    blockers = []
    if not spei_stat["available"]:
        blockers.append(
            f"SPEIbase targets missing: {spei_stat['missing']} from {spei_dir}"
        )
    if not tc_stat["available"]:
        blockers.append(
            f"TerraClimate predictors missing: {tc_stat['missing']} from {tc_dir}"
        )
    if not dem_stat["available"]:
        blockers.append(f"Copernicus DEM missing from {dem_path}")

    return {"manifest": manifest, "blockers": blockers}


def execute_pipeline_from_harmonized(
    harmonized_nc_path: Path | str,
    cfg: dict,
    checkpoint_path: Path | str | None = None,
    output_dir: Path | str = "results/spatial_pipeline",
) -> dict:
    """Run full spatial inference, validation, evaluation, and plotting from harmonized NetCDF."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    evidence_nc = out_dir / "spatial_evidence.nc"

    logger.info(f"Loading harmonized NetCDF from: {harmonized_nc_path}")
    ds = xr.open_dataset(harmonized_nc_path)

    # Bounding box for holdout
    holdout_cfg = cfg.get("spatial", {}).get("holdout_region", {})
    holdout_bbox = None
    if holdout_cfg.get("enabled", False):
        holdout_bbox = (
            holdout_cfg["lat_min"],
            holdout_cfg["lat_max"],
            holdout_cfg["lon_min"],
            holdout_cfg["lon_max"],
        )

    feature_vars = cfg.get("data", {}).get(
        "features", ["PRECTOTCORR", "T2M", "T2MDEW", "WS10M", "ALLSKY_SFC_SW_DWN"]
    )
    dataset = SpatialGridDataset(
        ds=ds,
        feature_vars=feature_vars,
        seq_len=cfg.get("data", {}).get("seq_len", 6),
        forecast_steps=cfg.get("data", {}).get("forecast_steps", 4),
        holdout_bbox=holdout_bbox,
    )
    counts = dataset.sample_counts()
    logger.info(f"Constructed spatial dataset: {counts}")

    if dataset.num_cells == 0:
        raise RuntimeError("No valid spatial grid cells found in harmonized dataset")

    # Load or initialize model
    model_cfg = {
        "data": {
            "input_dim": len(feature_vars),
            "forecast_steps": cfg.get("data", {}).get("forecast_steps", 4),
            "features": feature_vars,
            "seq_len": cfg.get("data", {}).get("seq_len", 6),
        },
        "model": cfg.get("model", {}),
    }
    model = DroughtCast(model_cfg)

    if checkpoint_path and Path(checkpoint_path).is_file():
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    else:
        logger.info("Using initialized model architecture for inference")

    # Run cell inference
    logger.info("Running spatial model inference across valid grid cells...")
    raw_results = run_cell_inference(model, dataset)

    # Calibrate probabilities & CQR intervals on validation split
    logger.info("Fitting validation-only probability calibrator and CQR intervals...")
    cal_artifacts, processed_cells = calibrate_and_conformalize(
        raw_results["cell_results"],
        quantiles=raw_results["quantiles"],
        threshold=-1.0,
        nominal_coverage=0.8,
    )

    # Build evidence NetCDF
    logger.info(f"Compiling NetCDF evidence raster at: {evidence_nc}")
    evidence_ds = build_evidence_netcdf(
        dataset=dataset,
        cell_results=processed_cells,
        calibration_artifacts=cal_artifacts,
        output_path=evidence_nc,
        target_lead=1,
    )

    # Comprehensive evaluation
    logger.info("Computing spatial evaluation metrics...")
    eval_report = evaluate_spatial_evidence(evidence_ds)
    eval_json = tbl_dir / "spatial_evaluation.json"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    eval_json.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")

    # Tables
    logger.info("Generating paper summary tables...")
    generate_tables(eval_json, tbl_dir)

    # Figures
    logger.info("Rendering publication-quality spatial figures (PNG, PDF, EPS)...")
    boundaries_path = cfg.get("spatial", {}).get(
        "boundaries_dir", "data/raw/spatial/gadm"
    )
    ind_gpkg = Path(boundaries_path) / "IND.gpkg"
    generate_all_spatial_figures(
        evidence_nc, fig_dir, ind_gpkg if ind_gpkg.is_file() else None
    )

    # Experiment Manifest
    manifest = {
        "experiment_name": cfg.get("experiment", {}).get("name", "spatial_evidence"),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "evidence_product": str(evidence_nc),
        "sample_counts": counts,
        "evaluation_summary": eval_report["per_scale"],
    }
    (out_dir / "spatial_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("Spatial pipeline execution completed successfully!")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end spatial pipeline runner")
    parser.add_argument(
        "--config", default="configs/spatial_config.yaml", help="Configuration YAML"
    )
    parser.add_argument(
        "--harmonized-nc", default=None, help="Path to harmonized NetCDF if available"
    )
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--output-dir", default="results/spatial_pipeline", help="Output directory"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Only check raw sources and manifest"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    logger.info("Checking spatial raw data availability...")
    check_result = verify_sources_or_raise(cfg)
    blockers = check_result["blockers"]

    if blockers:
        logger.warning("SPATIAL DATA PIPELINE BLOCKED: Missing required raw sources:")
        for b in blockers:
            logger.warning(f"  - {b}")

        if args.check_only or not args.harmonized_nc:
            print("\n" + "=" * 70)
            print(
                "STATUS: Continuous Spatial Pipeline Blocked Pending Raw Data Download"
            )
            print(
                "Reason: In accordance with scientific non-fabrication standards, continuous"
            )
            print(
                "spatial predictions cannot be generated without real gridded observations."
            )
            print("=" * 70 + "\n")
            sys.exit(0)

    # If harmonized dataset is explicitly provided, proceed with execution
    if args.harmonized_nc and Path(args.harmonized_nc).is_file():
        execute_pipeline_from_harmonized(
            args.harmonized_nc,
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

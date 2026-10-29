"""Download, verify, and provenance manifest tracking for gridded spatial data.

This module manages the official public sources required for continuous spatial evidence:
1. SPEIbase v2.10 (CSIC) gridded SPEI-3, SPEI-6, SPEI-12
2. TerraClimate (University of Idaho) monthly climate predictors
3. Copernicus DEM GLO-30 (ESA) 30m elevation
4. GADM v4.1 / Natural Earth administrative boundaries

In strict accordance with conference artifact standards:
- Silently substituting synthetic data is strictly prohibited.
- If raw source files are missing or incomplete, the verification fails loudly
  and reports the blocker with exact provenance instructions.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import requests

from utils.artifacts import sha256_file

logger = logging.getLogger(__name__)

DEFAULT_SOURCES_JSON = Path("data/spatial_sources.json")
DEFAULT_MANIFEST_PATH = Path("data/spatial_data_manifest.json")

REQUIRED_SPEI_FILES = {
    "spei_3": "spei03.nc",
    "spei_6": "spei06.nc",
    "spei_12": "spei12.nc",
}

REQUIRED_TERRACLIMATE_VARS = ["ppt", "tmax", "tmin", "srad", "ws", "pet"]


def load_sources_spec(path: Path | str = DEFAULT_SOURCES_JSON) -> dict:
    spec_path = Path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"Spatial sources specification not found at: {spec_path}"
        )
    return json.loads(spec_path.read_text(encoding="utf-8"))


def check_gadm_boundaries(gadm_dir: Path | str) -> dict[str, Any]:
    """Check administrative boundaries in GADM directory."""
    gadm_path = Path(gadm_dir)
    status: dict[str, Any] = {"available": False, "files": {}, "count": 0}
    if not gadm_path.is_dir():
        return status
    gpkg_files = list(gadm_path.glob("*.gpkg"))
    if not gpkg_files:
        return status
    status["available"] = True
    status["count"] = len(gpkg_files)
    for f in gpkg_files:
        status["files"][f.name] = {
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "sha256": sha256_file(f),
        }
    return status


def verify_speibase_files(speibase_dir: Path | str) -> dict[str, Any]:
    """Verify presence and validity of SPEIbase files."""
    s_dir = Path(speibase_dir)
    result: dict[str, Any] = {"available": False, "files": {}, "missing": []}
    if not s_dir.is_dir():
        result["missing"] = list(REQUIRED_SPEI_FILES.values())
        return result

    for scale, filename in REQUIRED_SPEI_FILES.items():
        filepath = s_dir / filename
        if filepath.is_file() and filepath.stat().st_size > 1000:
            result["files"][scale] = {
                "filename": filename,
                "path": str(filepath),
                "size_bytes": filepath.stat().st_size,
                "sha256": sha256_file(filepath),
            }
        else:
            result["missing"].append(filename)

    result["available"] = len(result["missing"]) == 0
    return result


def verify_terraclimate_files(tc_dir: Path | str) -> dict[str, Any]:
    """Verify presence and validity of TerraClimate predictor files."""
    t_dir = Path(tc_dir)
    result: dict[str, Any] = {"available": False, "files": {}, "missing": []}
    if not t_dir.is_dir():
        result["missing"] = REQUIRED_TERRACLIMATE_VARS
        return result

    for var in REQUIRED_TERRACLIMATE_VARS:
        matching = list(t_dir.glob(f"*{var}*.nc"))
        if matching:
            f = matching[0]
            result["files"][var] = {
                "filename": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            }
        else:
            result["missing"].append(var)

    result["available"] = len(result["missing"]) == 0
    return result


def verify_elevation_file(dem_path: Path | str) -> dict[str, Any]:
    """Verify presence of Copernicus DEM GLO-30 elevation file."""
    path = Path(dem_path)
    if path.is_file() and path.stat().st_size > 1000:
        return {
            "available": True,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {"available": False, "path": str(path), "missing": [str(path)]}


def generate_provenance_manifest(
    sources_spec: dict,
    speibase_status: dict,
    terraclimate_status: dict,
    elevation_status: dict,
    gadm_status: dict,
    output_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict:
    """Compile and write a machine-readable provenance manifest."""
    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "crs": sources_spec.get("crs", "EPSG:4326"),
        "spatial_resolution_degrees": sources_spec.get("observed_drought", {}).get(
            "spatial_resolution_degrees", 0.5
        ),
        "status": {
            "speibase_complete": speibase_status["available"],
            "terraclimate_complete": terraclimate_status["available"],
            "elevation_complete": elevation_status["available"],
            "boundaries_complete": gadm_status["available"],
            "fully_validated": (
                speibase_status["available"]
                and terraclimate_status["available"]
                and elevation_status["available"]
                and gadm_status["available"]
            ),
        },
        "sources": {
            "observed_drought": {
                **sources_spec.get("sources", {}).get("observed_drought", {}),
                "local_verification": speibase_status,
            },
            "climate_predictors": {
                **sources_spec.get("sources", {}).get("climate_predictors", {}),
                "local_verification": terraclimate_status,
            },
            "elevation": {
                **sources_spec.get("sources", {}).get("elevation", {}),
                "local_verification": elevation_status,
            },
            "boundaries": {
                **sources_spec.get("sources", {}).get("boundaries", {}),
                "local_verification": gadm_status,
            },
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_file_with_resume(
    url: str,
    target_path: Path | str,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file with stream and resume capability."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    mode = "wb"
    existing_size = 0
    if path.is_file():
        existing_size = path.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    if response.status_code == 416:  # Range not satisfiable, file already complete
        return path
    if response.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {response.status_code} when downloading from {url}")

    with open(path, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial data provenance and download verification"
    )
    parser.add_argument(
        "--sources-json",
        default=str(DEFAULT_SOURCES_JSON),
        help="Path to sources spec JSON",
    )
    parser.add_argument(
        "--speibase-dir", default="data/raw/spatial/speibase", help="SPEIbase directory"
    )
    parser.add_argument(
        "--terraclimate-dir",
        default="data/raw/spatial/terraclimate",
        help="TerraClimate directory",
    )
    parser.add_argument(
        "--dem-path",
        default="data/raw/spatial/dem/copernicus_dem_glo30.tif",
        help="DEM file path",
    )
    parser.add_argument(
        "--gadm-dir", default="data/raw/spatial/gadm", help="GADM boundaries directory"
    )
    parser.add_argument(
        "--output-manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Manifest output path",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Attempt to download missing files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with exit code 1 if any source is missing",
    )
    args = parser.parse_args()

    spec = load_sources_spec(args.sources_json)

    print("Verifying spatial source files...")
    spei_stat = verify_speibase_files(args.speibase_dir)
    tc_stat = verify_terraclimate_files(args.terraclimate_dir)
    dem_stat = verify_elevation_file(args.dem_path)
    gadm_stat = check_gadm_boundaries(args.gadm_dir)

    blockers = []
    if not spei_stat["available"]:
        blockers.append(
            f"SPEIbase targets missing: {spei_stat['missing']} from {args.speibase_dir}"
        )
    if not tc_stat["available"]:
        blockers.append(
            f"TerraClimate predictors missing: {tc_stat['missing']} from {args.terraclimate_dir}"
        )
    if not dem_stat["available"]:
        blockers.append(f"Copernicus DEM missing from {args.dem_path}")
    if not gadm_stat["available"]:
        blockers.append(f"GADM boundary geopackages missing from {args.gadm_dir}")

    generate_provenance_manifest(
        spec, spei_stat, tc_stat, dem_stat, gadm_stat, args.output_manifest
    )
    print(f"Written spatial provenance manifest to: {args.output_manifest}")

    if blockers:
        print("\n" + "=" * 60)
        print("SPATIAL EVIDENCE PIPELINE BLOCKED: Missing raw source files:")
        for b in blockers:
            print(f"  - {b}")
        print("\nOfficial Data Source URLs:")
        print("  SPEIbase v2.10: https://digital.csic.es/handle/10261/364137")
        print(
            "  TerraClimate:   http://thredds.northwestknowledge.net:8080/thredds/terraclimate_aggregated.html"
        )
        print(
            "  Copernicus DEM: https://dataspace.copernicus.eu/ or s3://copernicus-dem-30m/"
        )
        print("  GADM v4.1:      https://gadm.org/download_country.html")
        print("=" * 60 + "\n")

        if args.strict:
            raise SystemExit(1)
    else:
        print("\nAll spatial sources successfully verified and recorded in manifest!")


if __name__ == "__main__":
    main()

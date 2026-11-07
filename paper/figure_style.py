"""Reusable IEEE conference figure styling and export checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}


def apply_style() -> None:
    """Set legible, vector-friendly defaults for research figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, path: str | Path, *, width: str = "single", dpi: int = 600, kind: str = "plot") -> dict[str, str]:
    """Save a figure as vector PDF and raster PNG at a fixed column width."""
    if width not in {"single", "double"}:
        raise ValueError("width must be 'single' or 'double'")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    width_in = SINGLE_COLUMN_IN if width == "single" else DOUBLE_COLUMN_IN
    current = fig.get_size_inches()
    fig.set_size_inches(width_in, current[1] * width_in / current[0], forward=True)
    pdf_path = target.with_suffix(".pdf")
    png_path = target.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    return {"pdf": str(pdf_path), "png": str(png_path), "kind": kind}


def audit_figure(
    fig,
    pdf_path: str | Path,
    png_path: str | Path,
    *,
    width: str = "single",
    dpi: int = 600,
    expected_geographic_aspect: float | None = None,
) -> dict:
    """Check exported file existence and figure canvas dimensions."""
    errors = []
    expected_width = SINGLE_COLUMN_IN if width == "single" else DOUBLE_COLUMN_IN
    actual_width = float(fig.get_size_inches()[0])
    if abs(actual_width - expected_width) > 0.08:
        errors.append(f"canvas width {actual_width:.3f}in != {expected_width:.3f}in")
    for file_path in (Path(pdf_path), Path(png_path)):
        if not file_path.is_file() or file_path.stat().st_size == 0:
            errors.append(f"missing or empty export: {file_path}")
    if dpi <= 0:
        errors.append("dpi must be positive")
    if expected_geographic_aspect is not None and expected_geographic_aspect <= 0:
        errors.append("expected geographic aspect must be positive")
    return {"ok": not errors, "errors": errors, "width_in": actual_width, "dpi": dpi}

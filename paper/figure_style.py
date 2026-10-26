"""Shared IEEE-sized Matplotlib style and export checks for paper figures."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

SINGLE_COLUMN_IN = 3.5
DOUBLE_COLUMN_IN = 7.16
BODY_PT = 8
OKABE_ITO = {
    "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7", "black": "#000000",
}


def apply_style() -> None:
    """Set consistent serif typography and embedded TrueType PDF fonts."""
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.prop_cycle": mpl.cycler(color=list(OKABE_ITO.values())),
    })


def new_figure(width: str = "single", height_in: float = 2.4, **kwargs):
    width_in = SINGLE_COLUMN_IN if width == "single" else DOUBLE_COLUMN_IN if width == "double" else None
    if width_in is None:
        raise ValueError("width must be 'single' or 'double'")
    return plt.subplots(figsize=(width_in, height_in), constrained_layout=True, **kwargs)


def add_panel_labels(axes, labels=None) -> None:
    axes = [axes] if not isinstance(axes, (list, tuple)) else list(axes)
    labels = labels or [f"({chr(ord('a') + index)})" for index in range(len(axes))]
    for ax, label in zip(axes, labels):
        ax.text(-0.12, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, fontweight="bold", clip_on=False)


def save_figure(fig, path, width: str = "single", dpi: int = 600, kind: str = "line") -> dict[str, str]:
    """Write fixed-canvas vector PDF and high-resolution opaque PNG."""
    expected = SINGLE_COLUMN_IN if width == "single" else DOUBLE_COLUMN_IN if width == "double" else None
    if expected is None:
        raise ValueError("width must be 'single' or 'double'")
    if abs(fig.get_figwidth() - expected) > 0.02:
        raise ValueError(f"Figure width {fig.get_figwidth():.3f} in does not match requested {expected:.2f} in")
    base = Path(path)
    if base.suffix.lower() in {".pdf", ".png"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = base.with_suffix(".pdf"), base.with_suffix(".png")
    fig.savefig(pdf, format="pdf", facecolor="white", transparent=False, pad_inches=0.02)
    fig.savefig(png, format="png", dpi=dpi, facecolor="white", transparent=False, pad_inches=0.02)
    return {"pdf": str(pdf), "png": str(png)}


def audit_figure(fig, pdf_path, png_path, width: str = "single", dpi: int = 600,
                 expected_geographic_aspect: float | None = None) -> dict:
    expected = SINGLE_COLUMN_IN if width == "single" else DOUBLE_COLUMN_IN
    errors: list[str] = []
    if abs(fig.get_figwidth() - expected) > 0.02:
        errors.append(f"canvas width is {fig.get_figwidth():.3f} in; expected {expected:.2f} in")
    pdf_bytes = Path(pdf_path).read_bytes()
    media = re.search(rb"/MediaBox\s*\[\s*[-\d.]+\s+[-\d.]+\s+([\d.]+)\s+([\d.]+)\s*\]", pdf_bytes)
    if not media:
        errors.append("could not read PDF page dimensions")
    elif abs(float(media.group(1)) / 72.0 - expected) > 0.02:
        errors.append("PDF page width does not match requested IEEE column width")
    with Image.open(png_path) as image:
        if abs(image.width - round(expected * dpi)) > 2:
            errors.append(f"PNG width is {image.width}px; expected {round(expected * dpi)}px")
        if ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white")).getbbox() is None:
            errors.append("PNG appears blank")
    return {"ok": not errors, "errors": errors}

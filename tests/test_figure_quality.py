from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper.figure_style import (
    DOUBLE_COLUMN_IN,
    SINGLE_COLUMN_IN,
    apply_style,
    audit_figure,
    save_figure,
)


@pytest.mark.parametrize("width,width_in", [("single", SINGLE_COLUMN_IN), ("double", DOUBLE_COLUMN_IN)])
def test_fixed_ieee_canvas_exports_pdf_png_and_quality(tmp_path, width, width_in):
    apply_style()
    fig, ax = plt.subplots(figsize=(width_in, 2.2), constrained_layout=True)
    ax.plot([0, 1, 2], [0, 1, 0], color="#0072B2", linewidth=1.0)
    ax.set_xlabel("Forecast lead (months)")
    ax.set_ylabel("SPEI (unitless)")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.5, 1])
    output = save_figure(fig, tmp_path / "quality", width=width, dpi=300)
    report = audit_figure(fig, output["pdf"], output["png"], width=width, dpi=300)
    assert report["ok"], report["errors"]
    assert Path(output["pdf"]).is_file()
    assert Path(output["png"]).is_file()
    plt.close(fig)


def test_generated_figure_files_are_paired_with_data_and_provenance():
    root = Path(__file__).parents[1]
    out = root / "paper" / "figures"
    provenance = out / "provenance"
    for pdf in out.glob("P*.pdf"):
        stem = pdf.stem
        assert (out / f"{stem}.png").is_file(), stem
        assert (out / "data" / f"{stem}.csv").is_file(), stem
        assert (provenance / f"{stem}.json").is_file(), stem

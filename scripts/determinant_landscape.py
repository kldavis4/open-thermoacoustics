#!/usr/bin/env python3
"""Determinant-landscape diagnostic for traveling-wave complex-frequency onset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openthermoacoustics.validation.traveling_wave_engine import (
    compute_determinant_landscape,
    compute_trunk_transfer_matrix,
    tuned_traveling_wave_engine_candidate_config,
)


def main() -> None:
    """Generate determinant landscape heatmap and transfer-matrix sensitivity table."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    t_hot = 600.0
    f_real_center = 120.0
    f_real_values = np.linspace(f_real_center - 30.0, f_real_center + 30.0, 61)
    f_imag_values = np.linspace(-10.0, 10.0, 41)

    landscape = compute_determinant_landscape(
        cfg,
        t_hot=t_hot,
        f_real_values=f_real_values,
        f_imag_values=f_imag_values,
    )

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    m = ax.pcolormesh(
        landscape["f_real_values"],
        landscape["f_imag_values"],
        np.log10(landscape["det_magnitude"] + 1e-30),
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(m, ax=ax, label="log10(|det(M)|)")
    ax.axhline(0.0, color="white", linestyle="--", alpha=0.6)
    ax.set_xlabel("f_real (Hz)")
    ax.set_ylabel("f_imag (Hz)")
    ax.set_title(f"Determinant Landscape at T_hot={t_hot:.0f} K")
    fig.tight_layout()
    fig.savefig(output_dir / "det_landscape_600K.png", dpi=150)
    plt.close(fig)

    lines = []
    lines.append(
        f"Landscape minimum: f_real={landscape['min_f_real_hz']:.3f} Hz, "
        f"f_imag={landscape['min_f_imag_hz']:.3f} Hz, "
        f"|det|={landscape['min_det_magnitude']:.3e}"
    )
    lines.append("")
    lines.append(
        f"{'f_imag':>8s}  {'T[0,0]':>24s}  {'T[0,1]':>24s}  {'T[1,0]':>24s}  {'T[1,1]':>24s}"
    )
    for f_imag in [0.0, 0.1, 1.0, 5.0, 10.0]:
        omega = 2.0 * np.pi * (f_real_center + 1j * f_imag)
        tmat = compute_trunk_transfer_matrix(cfg, t_hot=t_hot, omega=omega)
        lines.append(
            f"{f_imag:8.1f}  {str(tmat[0, 0]):>24s}  {str(tmat[0, 1]):>24s}  "
            f"{str(tmat[1, 0]):>24s}  {str(tmat[1, 1]):>24s}"
        )

    (output_dir / "transfer_matrix_sensitivity.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'det_landscape_600K.png'}")
    print(f"Wrote {output_dir / 'transfer_matrix_sensitivity.txt'}")
    print(lines[0])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Loop-only self-oscillation diagnostic based on det(T_branch - I)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openthermoacoustics.validation.traveling_wave_engine import (
    _loop_det_value,
    detect_onset_from_complex_frequency,
    find_onset_ratio_proxy,
    scan_loop_eigenvalues,
    solve_loop_self_oscillation,
    sweep_loop_self_oscillation,
    tuned_traveling_wave_engine_candidate_config,
)


def _write_scan_data(path: Path, rows: list[dict]) -> None:
    lines = [
        "f_real_hz,lambda1_real,lambda1_imag,lambda1_mag,"
        "lambda2_real,lambda2_imag,lambda2_mag,closest_abs_error"
    ]
    for row in rows:
        ev0 = complex(row["eigvals"][0])
        ev1 = complex(row["eigvals"][1])
        lines.append(
            f"{row['f_real_hz']:.6f},"
            f"{ev0.real:.9e},{ev0.imag:.9e},{abs(ev0):.9e},"
            f"{ev1.real:.9e},{ev1.imag:.9e},{abs(ev1):.9e},"
            f"{row['closest_abs_error']:.9e}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_loop_det_landscape(cfg, output_dir: Path, *, t_hot: float, f0: float) -> tuple[float, float, float]:
    f_real_values = np.linspace(f0 - 30.0, f0 + 30.0, 121)
    f_imag_values = np.linspace(-30.0, 30.0, 121)
    det_mag = np.zeros((len(f_imag_values), len(f_real_values)), dtype=float)
    for i, fi in enumerate(f_imag_values):
        for j, fr in enumerate(f_real_values):
            d, _ = _loop_det_value(cfg, t_hot=t_hot, f_real_hz=float(fr), f_imag_hz=float(fi))
            det_mag[i, j] = float(abs(d))
    idx = np.unravel_index(int(np.argmin(det_mag)), det_mag.shape)
    min_fr = float(f_real_values[idx[1]])
    min_fi = float(f_imag_values[idx[0]])
    min_det = float(det_mag[idx])

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        f_real_values,
        f_imag_values,
        np.log10(det_mag + 1e-30),
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="log10(|det(T_branch-I)|)")
    ax.axhline(0.0, color="white", linestyle="--", alpha=0.6)
    ax.scatter([min_fr], [min_fi], color="red", s=25)
    ax.set_xlabel("f_real (Hz)")
    ax.set_ylabel("f_imag (Hz)")
    ax.set_title(f"Loop Determinant Landscape (T_hot={t_hot:.0f} K)")
    fig.tight_layout()
    fig.savefig(output_dir / "loop_det_landscape_600K.png", dpi=150)
    plt.close(fig)
    return min_fr, min_fi, min_det


def main() -> None:
    cfg = replace(tuned_traveling_wave_engine_candidate_config(), n_points_per_segment=24)
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    f_real_values = np.linspace(20.0, 300.0, 141)
    scan = scan_loop_eigenvalues(cfg, t_hot=600.0, f_real_values=f_real_values)
    _write_scan_data(output_dir / "loop_eigenvalue_scan_data.txt", scan)

    # Plot |lambda - 1| for both eigenvalues.
    err0 = []
    err1 = []
    for row in scan:
        ev0 = complex(row["eigvals"][0])
        ev1 = complex(row["eigvals"][1])
        err0.append(abs(ev0 - (1.0 + 0.0j)))
        err1.append(abs(ev1 - (1.0 + 0.0j)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(f_real_values, err0, label="|lambda1-1|")
    ax.plot(f_real_values, err1, label="|lambda2-1|")
    ax.set_xlabel("f_real (Hz)")
    ax.set_ylabel("|lambda - 1|")
    ax.set_title("Loop Eigenvalue Proximity to Unity (T_hot=600 K)")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "loop_eigenvalue_scan_600K.png", dpi=150)
    plt.close(fig)

    best_scan = min(scan, key=lambda r: float(r["closest_abs_error"]))
    seed_fr = float(best_scan["f_real_hz"])
    seed_fi = 0.0

    # Refine using loop-only complex solve and local determinant landscape.
    offaxis = solve_loop_self_oscillation(
        cfg,
        t_hot=600.0,
        f_real_guess=seed_fr,
        f_imag_guess=seed_fi,
        f_real_span_hz=60.0,
        f_imag_span_hz=30.0,
    )
    min_fr, min_fi, min_det = _plot_loop_det_landscape(
        cfg,
        output_dir,
        t_hot=600.0,
        f0=float(offaxis["frequency_real"]),
    )

    # Retry from loop-det minimum if needed.
    offaxis = solve_loop_self_oscillation(
        cfg,
        t_hot=600.0,
        f_real_guess=min_fr,
        f_imag_guess=min_fi,
        f_real_span_hz=60.0,
        f_imag_span_hz=30.0,
    )

    sweep = sweep_loop_self_oscillation(
        cfg,
        t_hot_values=np.arange(300.0, 901.0, 25.0),
        f_real_guess=float(offaxis["frequency_real"]),
        f_imag_guess=float(offaxis["frequency_imag"]),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([p["t_hot"] for p in sweep], [p["frequency_imag"] for p in sweep], marker="o", ms=3)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("T_hot (K)")
    ax.set_ylabel("f_imag (Hz)")
    ax.set_title("Loop Self-Oscillation f_imag vs Temperature")
    fig.tight_layout()
    fig.savefig(output_dir / "tw_loop_fimag_vs_temp.png", dpi=150)
    plt.close(fig)

    onset = detect_onset_from_complex_frequency(sweep, max_residual_norm=1e-3)
    proxy_onset, _ = find_onset_ratio_proxy(
        cfg,
        frequency_hz=120.0,
        t_hot_min=300.0,
        t_hot_max=900.0,
        coarse_step=100.0,
        fine_step=20.0,
    )
    print(f"Scan best seed: f_real={seed_fr:.3f} Hz")
    print(
        "Off-axis @600K: "
        f"f={offaxis['frequency_real']:.6f}+j{offaxis['frequency_imag']:.6f} Hz, "
        f"res={offaxis['residual_norm']:.3e}"
    )
    print(
        "Loop-det local minimum @600K: "
        f"f={min_fr:.3f}+j{min_fi:.3f} Hz, |det|={min_det:.3e}"
    )
    print(f"Loop onset ratio: {onset}")
    print(f"Gain-proxy onset ratio: {proxy_onset}")


if __name__ == "__main__":
    main()

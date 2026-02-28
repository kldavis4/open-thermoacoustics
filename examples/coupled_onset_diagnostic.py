#!/usr/bin/env python3
"""Coupled onset diagnostic: trunk resonance + loop gain + coupled determinant."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-config")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openthermoacoustics.validation.traveling_wave_engine import (
    build_boundary_matrix,
    compute_branch_transfer_matrix,
    compute_trunk_transfer_matrix,
    solve_traveling_wave_engine_determinant_complex_frequency,
    tuned_traveling_wave_engine_candidate_config,
)


def _find_local_minima(values: np.ndarray) -> list[int]:
    idx: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i - 1] > values[i] and values[i + 1] > values[i]:
            idx.append(i)
    return idx


def _select_resonances(freqs: np.ndarray, t11_abs: np.ndarray, max_count: int = 4) -> list[int]:
    minima = _find_local_minima(t11_abs)
    if not minima:
        return [int(np.argmin(t11_abs))]
    # Prefer deepest minima while keeping spacing ~15 Hz.
    ranked = sorted(minima, key=lambda i: float(t11_abs[i]))
    selected: list[int] = []
    for i in ranked:
        if all(abs(float(freqs[i] - freqs[j])) > 15.0 for j in selected):
            selected.append(i)
        if len(selected) >= max_count:
            break
    return sorted(selected, key=lambda i: float(freqs[i]))


def _coupled_det(t_trunk: np.ndarray, t_branch: np.ndarray) -> complex:
    m = build_boundary_matrix(t_trunk, t_branch)
    return complex(np.linalg.det(m))


def _loop_det(t_branch: np.ndarray) -> complex:
    return complex(
        (t_branch[0, 0] - 1.0) * (t_branch[1, 1] - 1.0) - t_branch[0, 1] * t_branch[1, 0]
    )


def main() -> None:
    cfg = replace(tuned_traveling_wave_engine_candidate_config(), n_points_per_segment=8)
    output = Path("examples/output")
    output.mkdir(parents=True, exist_ok=True)

    t_hot_loop = 600.0
    t_hot_trunk = 300.0

    # A. Trunk resonance scan.
    f_scan = np.linspace(20.0, 400.0, 801)
    t11_abs = np.zeros_like(f_scan)
    for i, f in enumerate(f_scan):
        t_trunk = compute_trunk_transfer_matrix(
            cfg,
            t_hot=t_hot_trunk,
            omega=2.0 * np.pi * f,
        )
        t11_abs[i] = abs(t_trunk[1, 1])
        if i % 200 == 0:
            print(f"[scan] trunk {i}/{len(f_scan)}")

    reson_idx = _select_resonances(f_scan, t11_abs)
    reson_freqs = [float(f_scan[i]) for i in reson_idx]

    scan_lines = ["f_hz,abs_T11"]
    scan_lines.extend(f"{float(f):.6f},{float(v):.9e}" for f, v in zip(f_scan, t11_abs))
    (output / "trunk_resonance_scan.txt").write_text("\n".join(scan_lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(f_scan, t11_abs, linewidth=1.2)
    for f in reson_freqs:
        ax.axvline(f, color="red", linestyle="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|T_trunk[1,1]|")
    ax.set_title("Trunk Resonance Scan")
    fig.tight_layout()
    fig.savefig(output / "trunk_resonances.png", dpi=150)
    plt.close(fig)

    # B. Loop gain at trunk resonances.
    gain_lines = ["f_trunk_hz,lambda_dom_real,lambda_dom_imag,abs_lambda_dom,abs_lambda_dom_minus_1"]
    for f in reson_freqs:
        t_branch = compute_branch_transfer_matrix(
            cfg,
            t_hot=t_hot_loop,
            omega=2.0 * np.pi * f,
        )
        eigvals = np.linalg.eigvals(t_branch)
        lam_dom = max((complex(e) for e in eigvals), key=lambda z: abs(z))
        gain_lines.append(
            f"{f:.6f},{lam_dom.real:.9e},{lam_dom.imag:.9e},{abs(lam_dom):.9e},{abs(lam_dom-1):.9e}"
        )
    (output / "loop_gain_at_trunk_resonances.txt").write_text(
        "\n".join(gain_lines) + "\n",
        encoding="utf-8",
    )

    # C/D. Coupled determinant local maps + seeded solves.
    coupled_lines = ["f_seed_hz,f_solved_real_hz,f_solved_imag_hz,residual_norm,det_mag_at_solution"]
    for f0 in reson_freqs:
        f_real = np.linspace(f0 - 10.0, f0 + 10.0, 81)
        f_imag = np.linspace(-5.0, 5.0, 41)
        det_abs = np.zeros((len(f_imag), len(f_real)))
        for i, fi in enumerate(f_imag):
            for j, fr in enumerate(f_real):
                omega = 2.0 * np.pi * (fr + 1j * fi)
                t_trunk = compute_trunk_transfer_matrix(cfg, t_hot=t_hot_trunk, omega=omega)
                t_branch = compute_branch_transfer_matrix(cfg, t_hot=t_hot_loop, omega=omega)
                det_abs[i, j] = abs(_coupled_det(t_trunk, t_branch))
            if i % 10 == 0:
                print(f"[landscape {f0:.1f} Hz] row {i}/{len(f_imag)}")

        fig, ax = plt.subplots(figsize=(10, 6))
        mesh = ax.pcolormesh(f_real, f_imag, np.log10(det_abs + 1e-30), shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="log10(|det(M)|)")
        ax.axhline(0.0, color="white", linestyle="--", alpha=0.6)
        ax.axvline(f0, color="red", linestyle="--", alpha=0.6)
        ax.set_xlabel("f_real (Hz)")
        ax.set_ylabel("f_imag (Hz)")
        ax.set_title(f"Coupled det(M) near trunk resonance {f0:.2f} Hz")
        fig.tight_layout()
        fig.savefig(output / f"det_near_trunk_resonance_{int(round(f0))}.png", dpi=150)
        plt.close(fig)

        solved = solve_traveling_wave_engine_determinant_complex_frequency(
            cfg,
            t_hot=t_hot_loop,
            f_real_guess=f0,
            f_imag_guess=0.0,
            f_real_span_hz=10.0,
            f_imag_span_hz=5.0,
        )
        coupled_lines.append(
            f"{f0:.6f},{float(solved['frequency_real']):.9e},{float(solved['frequency_imag']):.9e},"
            f"{float(solved['residual_norm']):.9e},{float(solved['det_magnitude']):.9e}"
        )

    (output / "coupled_eigenfreq_at_trunk_resonances.txt").write_text(
        "\n".join(coupled_lines) + "\n",
        encoding="utf-8",
    )

    # E. Check 120 Hz explicitly.
    omega_120 = 2.0 * np.pi * 120.0
    t_trunk_120 = compute_trunk_transfer_matrix(cfg, t_hot=t_hot_trunk, omega=omega_120)
    t_branch_120 = compute_branch_transfer_matrix(cfg, t_hot=t_hot_loop, omega=omega_120)
    det_coupled_120 = _coupled_det(t_trunk_120, t_branch_120)
    det_loop_120 = _loop_det(t_branch_120)
    nearest = min(reson_freqs, key=lambda f: abs(f - 120.0))
    check_lines = [
        f"nearest_trunk_resonance_hz={nearest:.6f}",
        f"distance_from_120_hz={abs(nearest-120.0):.6f}",
        f"abs_Ttrunk11_at_120={abs(t_trunk_120[1,1]):.9e}",
        f"abs_det_loop_at_120={abs(det_loop_120):.9e}",
        f"abs_det_coupled_at_120={abs(det_coupled_120):.9e}",
    ]
    (output / "check_120Hz.txt").write_text("\n".join(check_lines) + "\n", encoding="utf-8")

    print(f"Resonance candidates: {', '.join(f'{f:.2f}' for f in reson_freqs)} Hz")
    print(f"Nearest trunk resonance to 120 Hz: {nearest:.2f} Hz")
    print(f"|T11(120 Hz)|={abs(t_trunk_120[1,1]):.3e}")
    print(f"|det(T_branch-I)|(120 Hz)={abs(det_loop_120):.3e}")
    print(f"|det(M)|(120 Hz)={abs(det_coupled_120):.3e}")


if __name__ == "__main__":
    main()

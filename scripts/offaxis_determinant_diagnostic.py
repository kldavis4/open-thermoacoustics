#!/usr/bin/env python3
"""Off-axis determinant diagnostic for traveling-wave complex-frequency onset."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openthermoacoustics import gas
from openthermoacoustics.solver.distributed_loop import integrate_segment_chain
from openthermoacoustics.validation.traveling_wave_engine import (
    build_boundary_matrix,
    build_traveling_wave_paths,
    detect_onset_from_complex_frequency,
    find_best_frequency_by_residual,
    find_onset_ratio_proxy,
    solve_traveling_wave_engine_determinant_complex_frequency,
    sweep_traveling_wave_frequency,
    tuned_traveling_wave_engine_candidate_config,
)


def _transfer_matrix_fast(
    segs,
    helium,
    cfg,
    *,
    omega: complex,
    p_scale: float = 1_000.0,
    u_scale: float = 1e-3,
) -> np.ndarray:
    a = integrate_segment_chain(
        segs,
        p1_start=complex(p_scale),
        u1_start=0.0 + 0.0j,
        t_m_start=cfg.t_m_start,
        omega=omega,
        gas=helium,
        n_points_per_segment=cfg.n_points_per_segment,
    )
    b = integrate_segment_chain(
        segs,
        p1_start=0.0 + 0.0j,
        u1_start=complex(u_scale),
        t_m_start=cfg.t_m_start,
        omega=omega,
        gas=helium,
        n_points_per_segment=cfg.n_points_per_segment,
    )
    return np.array(
        [
            [a.p1_end / p_scale, b.p1_end / u_scale],
            [a.U1_end / p_scale, b.U1_end / u_scale],
        ],
        dtype=complex,
    )


def _det_eval_fast(cfg, trunk_segs, branch_segs, helium, *, t_hot: float, f_real: float, f_imag: float):
    omega = 2.0 * np.pi * (float(f_real) + 1j * float(f_imag))
    t_trunk = _transfer_matrix_fast(trunk_segs, helium, cfg, omega=omega)
    t_branch = _transfer_matrix_fast(branch_segs, helium, cfg, omega=omega)
    m = build_boundary_matrix(t_trunk, t_branch)
    for i in range(3):
        norm = np.linalg.norm(m[i, :])
        if norm > 0.0:
            m[i, :] /= norm
    return complex(np.linalg.det(m))


def _save_landscape(
    cfg,
    *,
    t_hot: float,
    out_path: Path,
    f_real_values: np.ndarray,
    f_imag_values: np.ndarray,
) -> dict[str, float]:
    helium = gas.Helium(mean_pressure=cfg.mean_pressure)
    trunk_segs, branch_segs = build_traveling_wave_paths(cfg, t_hot=t_hot)
    det_mag = np.zeros((len(f_imag_values), len(f_real_values)), dtype=float)
    for i, f_imag in enumerate(f_imag_values):
        for j, f_real in enumerate(f_real_values):
            det_mag[i, j] = abs(
                _det_eval_fast(
                    cfg,
                    trunk_segs,
                    branch_segs,
                    helium,
                    t_hot=t_hot,
                    f_real=float(f_real),
                    f_imag=float(f_imag),
                )
            )
    idx = np.unravel_index(int(np.argmin(det_mag)), det_mag.shape)
    landscape = {
        "f_real_values": np.asarray(f_real_values, dtype=float),
        "f_imag_values": np.asarray(f_imag_values, dtype=float),
        "det_magnitude": det_mag,
        "min_f_real_hz": float(f_real_values[idx[1]]),
        "min_f_imag_hz": float(f_imag_values[idx[0]]),
        "min_det_magnitude": float(det_mag[idx]),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        landscape["f_real_values"],
        landscape["f_imag_values"],
        np.log10(landscape["det_magnitude"] + 1e-30),
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="log10(|det(M)|)")
    ax.axhline(0.0, color="white", linestyle="--", alpha=0.6)
    ax.set_xlabel("f_real (Hz)")
    ax.set_ylabel("f_imag (Hz)")
    ax.set_title(f"Determinant Landscape (T_hot={t_hot:.0f} K)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "t_hot": float(t_hot),
        "min_f_real_hz": float(landscape["min_f_real_hz"]),
        "min_f_imag_hz": float(landscape["min_f_imag_hz"]),
        "min_det_magnitude": float(landscape["min_det_magnitude"]),
    }


def _real_freq_det_table(cfg, t_hot_values: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for t_hot in t_hot_values:
        helium = gas.Helium(mean_pressure=cfg.mean_pressure)
        trunk_segs, branch_segs = build_traveling_wave_paths(cfg, t_hot=float(t_hot))
        freq_grid = np.linspace(60.0, 180.0, 13)
        sweep = sweep_traveling_wave_frequency(cfg, frequencies_hz=freq_grid, t_hot=float(t_hot))
        best = find_best_frequency_by_residual(sweep)
        f_real = float(best["frequency_hz"])
        det_eval = _det_eval_fast(
            cfg,
            trunk_segs,
            branch_segs,
            helium,
            t_hot=float(t_hot),
            f_real=f_real,
            f_imag=0.0,
        )
        rows.append(
            {
                "t_hot": float(t_hot),
                "f_real_hz": f_real,
                "loop_residual": float(best["result"].residual_norm),
                "det_magnitude": float(abs(det_eval)),
            }
        )
    return rows


def _offaxis_continuation(cfg, t_hot_values: np.ndarray, f_real0: float, f_imag0: float):
    points = []
    f_real = float(f_real0)
    f_imag = float(f_imag0)
    for t_hot in t_hot_values:
        point = solve_traveling_wave_engine_determinant_complex_frequency(
            cfg,
            t_hot=float(t_hot),
            f_real_guess=f_real,
            f_imag_guess=f_imag,
            f_real_span_hz=60.0,
            f_imag_span_hz=30.0,
        )
        points.append(point)
        f_real = float(point["frequency_real"])
        f_imag = float(point["frequency_imag"])
    return points


def main() -> None:
    cfg_base = tuned_traveling_wave_engine_candidate_config()
    # Use fewer integration points to keep wide-grid runtime tractable.
    cfg = replace(cfg_base, n_points_per_segment=24)

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    f_real_values = np.linspace(50.0, 200.0, 151)
    f_imag_values = np.linspace(-30.0, 30.0, 121)

    mins = []
    mins.append(
        _save_landscape(
            cfg,
            t_hot=300.0,
            out_path=output_dir / "det_landscape_300K_wide.png",
            f_real_values=f_real_values,
            f_imag_values=f_imag_values,
        )
    )
    mins.append(
        _save_landscape(
            cfg,
            t_hot=400.0,
            out_path=output_dir / "det_landscape_400K_wide.png",
            f_real_values=f_real_values,
            f_imag_values=f_imag_values,
        )
    )
    mins.append(
        _save_landscape(
            cfg,
            t_hot=600.0,
            out_path=output_dir / "det_landscape_600K_wide.png",
            f_real_values=f_real_values,
            f_imag_values=f_imag_values,
        )
    )

    real_rows = _real_freq_det_table(cfg, np.arange(300.0, 901.0, 100.0))
    real_lines = ["t_hot[K],f_real_hz,loop_residual,det_magnitude"]
    for row in real_rows:
        real_lines.append(
            f"{row['t_hot']:.1f},{row['f_real_hz']:.6f},{row['loop_residual']:.6e},"
            f"{row['det_magnitude']:.6e}"
        )
    (output_dir / "det_at_real_freq_vs_temp.txt").write_text("\n".join(real_lines) + "\n")

    # Seed off-axis solve from the 600 K landscape minimum.
    min_600 = mins[-1]
    offaxis_600 = solve_traveling_wave_engine_determinant_complex_frequency(
        cfg,
        t_hot=600.0,
        f_real_guess=min_600["min_f_real_hz"],
        f_imag_guess=min_600["min_f_imag_hz"],
        f_real_span_hz=60.0,
        f_imag_span_hz=30.0,
    )
    offaxis_lines = [
        f"seed_f_real={min_600['min_f_real_hz']:.6f}",
        f"seed_f_imag={min_600['min_f_imag_hz']:.6f}",
        f"solution_f_real={offaxis_600['frequency_real']:.6f}",
        f"solution_f_imag={offaxis_600['frequency_imag']:.6f}",
        f"residual_norm={offaxis_600['residual_norm']:.6e}",
        f"det_magnitude={offaxis_600['det_magnitude']:.6e}",
    ]
    (output_dir / "offaxis_zero_600K.txt").write_text("\n".join(offaxis_lines) + "\n")

    # Sweep from off-axis seed around known point.
    t_hot_values = np.arange(300.0, 901.0, 25.0)
    sweep = _offaxis_continuation(
        cfg,
        t_hot_values=t_hot_values,
        f_real0=float(offaxis_600["frequency_real"]),
        f_imag0=float(offaxis_600["frequency_imag"]),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [float(p["t_hot"]) for p in sweep],
        [float(p["frequency_imag"]) for p in sweep],
        marker="o",
        linewidth=1.5,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("T_hot (K)")
    ax.set_ylabel("f_imag (Hz)")
    ax.set_title("Determinant Complex-Frequency Sweep (Off-Axis Seed)")
    fig.tight_layout()
    fig.savefig(output_dir / "tw_det_fimag_vs_temp_offaxis.png", dpi=150)
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

    print("Landscape minima:")
    for row in mins:
        print(
            f"  T_hot={row['t_hot']:.0f} K -> "
            f"f_real={row['min_f_real_hz']:.3f} Hz, "
            f"f_imag={row['min_f_imag_hz']:.3f} Hz, "
            f"|det|={row['min_det_magnitude']:.3e}"
        )
    print(f"Off-axis solve @600K: f={offaxis_600['frequency_real']:.6f}+j{offaxis_600['frequency_imag']:.6f} Hz")
    print(f"Determinant onset ratio (off-axis sweep): {onset}")
    print(f"Gain-proxy onset ratio: {proxy_onset}")


if __name__ == "__main__":
    main()

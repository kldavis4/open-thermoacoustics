#!/usr/bin/env python3
"""Loop-eigenvalue unity diagnostics for traveling-wave onset."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openthermoacoustics.validation.traveling_wave_engine import (
    detect_onset_from_complex_frequency,
    find_onset_ratio_proxy,
    scan_loop_eigenvalues_multi_temp,
    solve_loop_lambda_unity,
    sweep_loop_lambda_unity,
    tuned_traveling_wave_engine_candidate_config,
)


def main() -> None:
    cfg = replace(tuned_traveling_wave_engine_candidate_config(), n_points_per_segment=16)
    out = Path("examples/output")
    out.mkdir(parents=True, exist_ok=True)

    f_real_values = np.concatenate(
        [np.linspace(1.0, 20.0, 100), np.linspace(20.0, 300.0, 100)]
    )
    t_hot_values = np.array([300.0, 350.0, 400.0, 500.0, 600.0, 700.0, 800.0])
    scans = scan_loop_eigenvalues_multi_temp(
        cfg,
        t_hot_values=t_hot_values,
        f_real_values=f_real_values,
    )

    # Extended scan table at 600 K.
    rows_600 = scans[600.0]
    lines = [
        "f_real_hz,lambda1_real,lambda1_imag,lambda1_mag,"
        "lambda2_real,lambda2_imag,lambda2_mag,closest_abs_error"
    ]
    for row in rows_600:
        ev0 = complex(row["eigvals"][0])
        ev1 = complex(row["eigvals"][1])
        lines.append(
            f"{row['f_real_hz']:.6f},"
            f"{ev0.real:.9e},{ev0.imag:.9e},{abs(ev0):.9e},"
            f"{ev1.real:.9e},{ev1.imag:.9e},{abs(ev1):.9e},"
            f"{row['closest_abs_error']:.9e}"
        )
    (out / "loop_eigenvalue_scan_extended.txt").write_text("\n".join(lines) + "\n")

    # Plot |lambda| for tracked branch closest to unity.
    fig, ax = plt.subplots(figsize=(10, 6))
    for t_hot in t_hot_values:
        rows = scans[float(t_hot)]
        mag = [float(row["closest_mag"]) for row in rows]
        ax.plot(f_real_values, mag, label=f"T_hot={int(t_hot)} K")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("f_real (Hz)")
    ax.set_ylabel("|tracked lambda|")
    ax.set_title("Loop Eigenvalue Magnitude vs Frequency")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loop_eigenvalue_vs_freq_multi_temp.png", dpi=150)
    plt.close(fig)

    # Seed from 600 K minimum distance to unity.
    best_600 = min(rows_600, key=lambda row: float(row["closest_abs_error"]))
    seed_fr = float(best_600["f_real_hz"])
    lam = complex(best_600["closest_to_one"])
    seed_fi = float(seed_fr * np.log(max(abs(lam), 1e-12)) / (2.0 * np.pi))

    root_600 = solve_loop_lambda_unity(
        cfg,
        t_hot=600.0,
        f_real_guess=seed_fr,
        f_imag_guess=seed_fi,
        f_real_span_hz=100.0,
        f_imag_span_hz=40.0,
    )
    root_text = "\n".join(
        [
            f"seed_f_real={seed_fr:.6f}",
            f"seed_f_imag={seed_fi:.6f}",
            f"tracked_lambda_seed={lam.real:.9e}+j{lam.imag:.9e}",
            f"solution_f_real={root_600['frequency_real']:.6f}",
            f"solution_f_imag={root_600['frequency_imag']:.6f}",
            f"tracked_lambda_solution={root_600['tracked_lambda'].real:.9e}+j{root_600['tracked_lambda'].imag:.9e}",
            f"residual_norm={root_600['residual_norm']:.9e}",
        ]
    )
    (out / "loop_lambda1_root_600K.txt").write_text(root_text + "\n")

    sweep = sweep_loop_lambda_unity(
        cfg,
        t_hot_values=np.arange(300.0, 901.0, 25.0),
        f_real_guess=float(root_600["frequency_real"]),
        f_imag_guess=float(root_600["frequency_imag"]),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([p["t_hot"] for p in sweep], [p["frequency_imag"] for p in sweep], marker="o", ms=3)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("T_hot (K)")
    ax.set_ylabel("f_imag (Hz)")
    ax.set_title("Loop λ=1 Complex-Frequency Sweep")
    fig.tight_layout()
    fig.savefig(out / "tw_loop_lambda1_fimag_vs_temp.png", dpi=150)
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
    print(f"seed @600K: f_real={seed_fr:.3f}, f_imag={seed_fi:.6f}, |lambda-1|={best_600['closest_abs_error']:.3e}")
    print(
        "root @600K: "
        f"f={root_600['frequency_real']:.6f}+j{root_600['frequency_imag']:.6f}, "
        f"res={root_600['residual_norm']:.3e}"
    )
    print(f"onset_lambda_unity={onset}")
    print(f"onset_gain_proxy={proxy_onset}")


if __name__ == "__main__":
    main()

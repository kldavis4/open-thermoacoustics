#!/usr/bin/env python3
"""Optimize standing-wave engine onset via complex-frequency sweeps."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openthermoacoustics import gas
from openthermoacoustics.validation.standing_wave_engine import (
    default_standing_wave_engine_config,
    detect_onset_from_complex_frequency,
    solve_standing_wave_engine_complex_frequency,
    solve_standing_wave_engine_complex_frequency_with_profiles,
    sweep_standing_wave_engine_complex_frequency,
)


def _solve_point(cfg, t_hot: float, fr_guess: float | None, fi_guess: float | None):
    kwargs: dict[str, float] = {}
    if fr_guess is not None:
        kwargs["frequency_real_guess"] = fr_guess
    if fi_guess is not None:
        kwargs["frequency_imag_guess"] = fi_guess
    try:
        pt = solve_standing_wave_engine_complex_frequency(cfg, t_hot=t_hot, **kwargs)
        return pt
    except Exception:
        return None


def _interp_zero_crossing(temps: np.ndarray, f_imag: np.ndarray) -> float | None:
    for i in range(len(temps) - 1):
        if f_imag[i] > 0.0 and f_imag[i + 1] <= 0.0:
            t0, t1 = temps[i], temps[i + 1]
            y0, y1 = f_imag[i], f_imag[i + 1]
            if abs(y1 - y0) < 1e-15:
                return float(t1)
            return float(t0 + (t1 - t0) * (y0 / (y0 - y1)))
    return None


def _segment_power_deltas(cfg, x: np.ndarray, p1: np.ndarray, u1: np.ndarray) -> dict[str, float]:
    w = 0.5 * np.real(p1 * np.conj(u1))
    x1 = cfg.left_duct_length
    x2 = x1 + cfg.cold_hx_length
    x3 = x2 + cfg.stack_length
    x4 = x3 + cfg.hot_hx_length
    x5 = cfg.total_length

    def idx(x_target: float) -> int:
        return int(np.argmin(np.abs(x - x_target)))

    i0, i1, i2, i3, i4, i5 = idx(0.0), idx(x1), idx(x2), idx(x3), idx(x4), idx(x5)
    return {
        "Left Duct": float(w[i1] - w[i0]),
        "Cold HX": float(w[i2] - w[i1]),
        "Stack": float(w[i3] - w[i2]),
        "Hot HX": float(w[i4] - w[i3]),
        "Right Duct": float(w[i5] - w[i4]),
    }


def main() -> None:
    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = default_standing_wave_engine_config()
    target_porosity = 0.889

    print("=== Phase 1: Stack Position Sweep (y0=0.4 mm, total length fixed) ===")
    left_values = np.arange(0.05, 0.56, 0.05)
    fi700: list[float] = []
    fi500: list[float] = []
    fr_guess, fi_guess = None, None
    for left in left_values:
        cfg = replace(base, left_duct_length=float(left), right_duct_length=float(0.60 - left))
        p700 = _solve_point(cfg, 700.0, fr_guess, fi_guess)
        if p700 is None:
            fi700.append(np.nan)
            fi500.append(np.nan)
            continue
        fr_guess = float(p700["frequency_real"])
        fi_guess = float(p700["frequency_imag"])
        p500 = _solve_point(cfg, 500.0, fr_guess, fi_guess)
        fi700.append(float(p700["frequency_imag"]))
        fi500.append(float(p500["frequency_imag"]) if p500 is not None else np.nan)
        print(
            f"left={left:.2f} right={0.60-left:.2f} "
            f"f_imag(700K)={fi700[-1]:+.4f} f_imag(500K)={fi500[-1]:+.4f}"
        )

    fi700_arr = np.array(fi700)
    best_idx = int(np.nanargmin(fi700_arr))
    best_left = float(left_values[best_idx])
    best_right = float(0.60 - best_left)
    best_fraction = best_left / 0.60
    print(f"Best position by f_imag@700K: left={best_left:.3f} m right={best_right:.3f} m")

    plt.figure(figsize=(8, 4.8))
    plt.plot(left_values, fi700, "o-", label="T_hot=700 K")
    plt.plot(left_values, fi500, "s--", label="T_hot=500 K")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Left Duct Length (m)")
    plt.ylabel("f_imag (Hz)")
    plt.title("Complex-Frequency Growth Rate vs Stack Position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "onset_vs_position.png", dpi=160)
    plt.close()

    print("\n=== Phase 2: Plate Half-Gap Sweep at Best Position ===")
    y0_values = np.array([0.10e-3, 0.15e-3, 0.20e-3, 0.25e-3, 0.30e-3, 0.40e-3, 0.50e-3, 0.60e-3])
    fi_y0: list[float] = []
    y0_over_dk: list[float] = []
    fr_guess, fi_guess = None, None
    helium = gas.Helium(mean_pressure=base.mean_pressure)
    for y0 in y0_values:
        spacing = 2.0 * y0
        thickness = spacing * (1.0 - target_porosity) / target_porosity
        cfg = replace(
            base,
            left_duct_length=best_left,
            right_duct_length=best_right,
            plate_spacing=spacing,
            plate_thickness=thickness,
        )
        point = _solve_point(cfg, 600.0, fr_guess, fi_guess)
        if point is None:
            fi_y0.append(np.nan)
            y0_over_dk.append(np.nan)
            continue
        fr_guess = float(point["frequency_real"])
        fi_guess = float(point["frequency_imag"])
        fi_val = float(point["frequency_imag"])
        fi_y0.append(fi_val)

        t_mid = 450.0
        omega = 2.0 * np.pi * abs(float(point["frequency_real"]))
        rho = helium.density(t_mid)
        kappa = helium.thermal_conductivity(t_mid)
        cp = helium.specific_heat_cp(t_mid)
        delta_kappa = np.sqrt(2.0 * kappa / (rho * cp * omega))
        y0_over_dk.append(float(y0 / delta_kappa))
        print(
            f"y0={1e3*y0:.3f} mm spacing={1e3*spacing:.3f} mm "
            f"f_imag(600K)={fi_val:+.4f} y0/delta_kappa={y0_over_dk[-1]:.3f}"
        )

    fi_y0_arr = np.array(fi_y0)
    best_y0_idx = int(np.nanargmin(fi_y0_arr))
    best_y0 = float(y0_values[best_y0_idx])
    best_spacing = 2.0 * best_y0
    best_thickness = best_spacing * (1.0 - target_porosity) / target_porosity
    print(f"Best y0 by f_imag@600K: y0={best_y0*1e3:.3f} mm")

    plt.figure(figsize=(8, 4.8))
    plt.plot(1e3 * y0_values, fi_y0, "o-")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Plate Half-Gap y0 (mm)")
    plt.ylabel("f_imag (Hz) at 600 K")
    plt.title("Complex-Frequency Growth Rate vs Stack Half-Gap")
    plt.tight_layout()
    plt.savefig(out_dir / "onset_vs_y0.png", dpi=160)
    plt.close()

    print("\n=== Phase 3: Total Duct Length Sweep (Best Position Fraction + y0) ===")
    total_duct_values = np.array([0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20])
    fi_length: list[float] = []
    fr_guess, fi_guess = None, None
    best_cfg = None
    best_length_score = None
    for total_duct in total_duct_values:
        left = float(total_duct * best_fraction)
        right = float(total_duct * (1.0 - best_fraction))
        cfg = replace(
            base,
            left_duct_length=left,
            right_duct_length=right,
            plate_spacing=best_spacing,
            plate_thickness=best_thickness,
        )
        point = _solve_point(cfg, 600.0, fr_guess, fi_guess)
        if point is None:
            fi_length.append(np.nan)
            continue
        fr_guess = float(point["frequency_real"])
        fi_guess = float(point["frequency_imag"])
        fi_val = float(point["frequency_imag"])
        fi_length.append(fi_val)
        print(f"total_duct={total_duct:.2f} left/right={left:.3f}/{right:.3f} f_imag(600K)={fi_val:+.4f}")
        if best_length_score is None or fi_val < best_length_score:
            best_length_score = fi_val
            best_cfg = cfg

    if best_cfg is None:
        raise RuntimeError("Failed to identify best configuration from length sweep.")

    plt.figure(figsize=(8, 4.8))
    plt.plot(total_duct_values, fi_length, "o-")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Total Duct Length L_left + L_right (m)")
    plt.ylabel("f_imag (Hz) at 600 K")
    plt.title("Complex-Frequency Growth Rate vs Duct Length")
    plt.tight_layout()
    plt.savefig(out_dir / "onset_vs_length.png", dpi=160)
    plt.close()

    print("\n=== Phase 4: Fine Onset Sweep for Best Configuration ===")
    t_values = np.arange(300.0, 801.0, 10.0)
    fine = sweep_standing_wave_engine_complex_frequency(best_cfg, t_hot_values=t_values)
    f_imag = np.array([float(p["frequency_imag"]) for p in fine])
    onset_ratio = detect_onset_from_complex_frequency(fine)
    onset_t = _interp_zero_crossing(t_values, f_imag)

    print("Best configuration:")
    print(f"  left_duct_length = {best_cfg.left_duct_length:.10f} m")
    print(f"  right_duct_length = {best_cfg.right_duct_length:.10f} m")
    print(f"  plate_spacing = {best_cfg.plate_spacing:.10f} m")
    print(f"  plate_thickness = {best_cfg.plate_thickness:.10f} m")
    print(f"  total_length = {best_cfg.total_length:.6f} m")
    print(f"  porosity = {best_cfg.porosity:.6f}")
    print(f"  onset ratio (interpolated) = {onset_ratio}")
    print(f"  onset T_hot [K] = {onset_t}")

    plt.figure(figsize=(8, 4.8))
    plt.plot(t_values, f_imag, "o-")
    plt.axhline(0.0, color="black", linewidth=1.0)
    if onset_t is not None:
        plt.axvline(onset_t, color="tab:red", linestyle="--", linewidth=1.2, label=f"Onset ~ {onset_t:.1f} K")
        plt.legend()
    plt.xlabel("T_hot (K)")
    plt.ylabel("f_imag (Hz)")
    plt.title("Best Configuration: Complex-Frequency Onset Sweep")
    plt.tight_layout()
    plt.savefig(out_dir / "onset_fine_sweep.png", dpi=160)
    plt.close()

    # Profiles + power-budget waterfall above onset.
    t_profile = 650.0 if onset_t is None else float(min(800.0, onset_t + 80.0))
    sol = solve_standing_wave_engine_complex_frequency_with_profiles(best_cfg, t_hot=t_profile)
    x = np.asarray(sol["profiles"]["x"])
    p1 = np.asarray(sol["profiles"]["p1"])
    u1 = np.asarray(sol["profiles"]["U1"])
    w = 0.5 * np.real(p1 * np.conj(u1))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(x, np.abs(p1), label="|p1|")
    axes[0].set_ylabel("|p1| (Pa)")
    axes[0].grid(alpha=0.3)
    ax2 = axes[0].twinx()
    ax2.plot(x, np.abs(u1), color="tab:red", label="|U1|")
    ax2.set_ylabel("|U1| (m^3/s)")
    axes[0].set_title(f"Optimized Benchmark Profiles at T_hot={t_profile:.1f} K")
    axes[1].plot(x, w, color="tab:green")
    axes[1].set_ylabel("W_dot (W)")
    axes[1].set_xlabel("x (m)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "optimized_profiles.png", dpi=160)
    plt.close()

    segment_deltas = _segment_power_deltas(best_cfg, x, p1, u1)
    labels = list(segment_deltas.keys())
    values = [segment_deltas[k] for k in labels]
    colors = ["tab:green" if v >= 0.0 else "tab:red" for v in values]
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.ylabel("Delta Acoustic Power (W)")
    plt.title(f"Optimized Configuration Power Budget at T_hot={t_profile:.1f} K")
    plt.tight_layout()
    plt.savefig(out_dir / "optimized_power_budget.png", dpi=160)
    plt.close()

    print("\nSaved plots:")
    print("  examples/output/onset_vs_position.png")
    print("  examples/output/onset_vs_y0.png")
    print("  examples/output/onset_vs_length.png")
    print("  examples/output/onset_fine_sweep.png")
    print("  examples/output/optimized_profiles.png")
    print("  examples/output/optimized_power_budget.png")


if __name__ == "__main__":
    main()

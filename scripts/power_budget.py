#!/usr/bin/env python3
"""Power-budget analysis for standing-wave engine validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openthermoacoustics import gas
from openthermoacoustics.geometry.parallel_plate import ParallelPlate
from openthermoacoustics.validation.standing_wave_engine import (
    default_standing_wave_engine_config,
    solve_standing_wave_engine_complex_frequency_with_profiles,
)


def idx_at(x: np.ndarray, x_target: float) -> int:
    """Return index nearest the requested coordinate."""
    return int(np.argmin(np.abs(x - x_target)))


def main() -> None:
    """Run power budget for shifted layout at T_hot=600 K."""
    base = default_standing_wave_engine_config()
    cfg = replace(base, left_duct_length=0.10, right_duct_length=0.50)
    t_hot = 600.0
    t_cold = cfg.t_cold

    solution = solve_standing_wave_engine_complex_frequency_with_profiles(cfg, t_hot=t_hot)
    profiles = solution["profiles"]
    x = np.asarray(profiles["x"])
    p1 = np.asarray(profiles["p1"])
    u1 = np.asarray(profiles["U1"])
    w_dot = 0.5 * np.real(p1 * np.conj(u1))

    x0 = 0.0
    x1 = cfg.left_duct_length
    x2 = x1 + cfg.cold_hx_length
    x3 = x2 + cfg.stack_length
    x4 = x3 + cfg.hot_hx_length
    x5 = cfg.total_length
    x_mid = 0.5 * (x2 + x3)

    points = [
        ("Left duct start", x0),
        ("Left duct end", x1),
        ("Cold HX start", x1),
        ("Cold HX end", x2),
        ("Stack start", x2),
        ("Stack mid", x_mid),
        ("Stack end", x3),
        ("Hot HX start", x3),
        ("Hot HX end", x4),
        ("Right duct start", x4),
        ("Right duct end", x5),
    ]

    print("=== Standing-Wave Engine Power Budget (Shifted 0.10/0.50, T_hot=600 K) ===")
    print(f"Complex-frequency solution: f = {solution['frequency_real']:.3f} + j{solution['frequency_imag']:.3f} Hz")
    print(f"Residual norm: {solution['residual_norm']:.3e}")
    print()
    print(f"{'Position [m]':>12}  {'Segment':<16}  {'W_dot [W]':>14}")
    for label, xpos in points:
        i = idx_at(x, xpos)
        print(f"{x[i]:12.3f}  {label:<16}  {w_dot[i]:14.6e}")

    i0 = idx_at(x, x0)
    i1 = idx_at(x, x1)
    i2 = idx_at(x, x2)
    i3 = idx_at(x, x3)
    i4 = idx_at(x, x4)
    i5 = idx_at(x, x5)

    delta_left = float(w_dot[i1] - w_dot[i0])
    delta_cold = float(w_dot[i2] - w_dot[i1])
    delta_stack = float(w_dot[i3] - w_dot[i2])
    delta_hot = float(w_dot[i4] - w_dot[i3])
    delta_right = float(w_dot[i5] - w_dot[i4])

    segment_deltas = {
        "Left Duct": delta_left,
        "Cold HX": delta_cold,
        "Stack": delta_stack,
        "Hot HX": delta_hot,
        "Right Duct": delta_right,
    }
    total_dissipation = sum(abs(v) for v in segment_deltas.values() if v < 0.0 and v != delta_stack)
    stack_gain = delta_stack
    net = stack_gain - total_dissipation
    ratio = stack_gain / total_dissipation if total_dissipation > 0 else np.nan

    print()
    print("Segment ΔW")
    for name, delta_w in segment_deltas.items():
        print(f"  {name:<10}: {delta_w:+.6e} W")
    print()
    print(f"Total dissipation (excluding stack): {total_dissipation:.6e} W")
    print(f"Stack gain:                          {stack_gain:.6e} W")
    print(f"Net (gain - dissipation):            {net:.6e} W")
    print(f"Gain/Dissipation ratio:              {ratio:.6f}")

    # Penetration-depth regime check at stack-mid temperature.
    he = gas.Helium(mean_pressure=cfg.mean_pressure)
    t_mid = 0.5 * (t_cold + t_hot)
    f_real = float(solution["frequency_real"])
    omega = 2.0 * np.pi * f_real
    rho = he.density(t_mid)
    mu = he.viscosity(t_mid)
    kappa = he.thermal_conductivity(t_mid)
    cp = he.specific_heat_cp(t_mid)
    delta_nu = np.sqrt(2.0 * mu / (rho * omega))
    delta_kappa = np.sqrt(2.0 * kappa / (rho * cp * omega))
    y0 = cfg.hydraulic_radius
    geom = ParallelPlate()
    f_nu = complex(geom.f_nu(omega, delta_nu, y0))
    f_kappa = complex(geom.f_kappa(omega, delta_kappa, y0))

    print()
    print("Penetration-depth ratios at T_mid=450 K")
    print(f"  y0/delta_kappa = {y0/delta_kappa:.3f}")
    print(f"  y0/delta_nu    = {y0/delta_nu:.3f}")
    print(f"  |f_kappa|      = {abs(f_kappa):.6f}")
    print(f"  |f_nu|         = {abs(f_nu):.6f}")
    print(f"  |f_kappa-f_nu| = {abs(f_kappa - f_nu):.6f}")

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(segment_deltas.keys())
    values = [segment_deltas[key] for key in labels]
    colors = ["tab:red" if val < 0 else "tab:green" for val in values]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.ylabel("Delta Acoustic Power (W)")
    plt.title("Standing-Wave Engine Power Budget at T_hot=600 K")
    plt.tight_layout()
    plt.savefig(output_dir / "power_budget_600K.png", dpi=150)
    plt.close()
    print("\nSaved: examples/output/power_budget_600K.png")


if __name__ == "__main__":
    main()

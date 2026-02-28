#!/usr/bin/env python3
"""Standing-wave thermoacoustic refrigerator benchmark example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openthermoacoustics import viz
from openthermoacoustics.validation.standing_wave_refrigerator import (
    compute_refrigerator_cop,
    default_standing_wave_refrigerator_config,
    solve_standing_wave_refrigerator,
    sweep_cold_temperature,
    sweep_drive_ratio,
)


def main() -> None:
    cfg = default_standing_wave_refrigerator_config()
    print("Standing-Wave Refrigerator Benchmark")
    print("-" * 40)
    print(f"Mean pressure: {cfg.mean_pressure/1e6:.2f} MPa")
    print(f"T_hot / T_cold: {cfg.t_hot:.1f} / {cfg.t_cold:.1f} K")
    print(f"Drive ratio: {cfg.drive_ratio:.3f} (|p1|/p_m)")

    baseline = solve_standing_wave_refrigerator(cfg)
    cop_info = compute_refrigerator_cop(baseline)
    print("\nBaseline operating point:")
    print(f"  Frequency:      {baseline['frequency_hz']:.2f} Hz")
    print(f"  Cooling power:  {baseline['cooling_power']:.3e} W")
    print(f"  W_stack_in:     {baseline['acoustic_input_power']:.3e} W")
    print(f"  W_system_in:    {baseline['acoustic_input_power_system']:.3e} W")
    print(f"  COP_stack:      {cop_info['cop']:.4f}")
    print(f"  COP_system:     {cop_info['cop_system']:.4f}")
    print(f"  COP_Carnot:     {cop_info['cop_carnot']:.4f}")
    print(f"  COPR_stack:     {cop_info['cop_relative']:.4f}")

    drive_ratios = np.linspace(0.01, 0.05, 9)
    drive_sweep = sweep_drive_ratio(cfg, drive_ratios)
    print("\nDrive-ratio sweep:")
    for row in drive_sweep:
        c = row["config"]
        print(
            f"  drive={c.drive_ratio:.3f}  Q_cold={row['cooling_power']:.3e} W  "
            f"W_stack={row['acoustic_input_power']:.3e} W  COP_stack={row['cop']:.4f}"
        )

    t_cold_values = np.linspace(250.0, 290.0, 9)
    cold_sweep = sweep_cold_temperature(cfg, t_cold_values)
    print("\nCold-temperature sweep:")
    for row in cold_sweep:
        c = row["config"]
        dt = c.t_hot - c.t_cold
        print(
            f"  T_cold={c.t_cold:.1f} K  ΔT={dt:.1f} K  "
            f"Q_cold={row['cooling_power']:.3e} W  COP={row['cop']:.4f}  "
            f"COP/Carnot={row['cop_relative']:.4f}"
        )

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz.plot_frequency_sweep(
        frequencies=[float(row["config"].drive_ratio) for row in drive_sweep],
        values=[float(row["cop"]) for row in drive_sweep],
        xlabel="Drive ratio |p1|/p_m",
        ylabel="COP",
        title="Standing-Wave Refrigerator COP vs Drive Ratio",
        save_path=output_dir / "sw_refrigerator_cop_vs_drive.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(row["config"].t_hot - row["config"].t_cold) for row in cold_sweep],
        values=[float(row["cooling_power"]) for row in cold_sweep],
        xlabel="Temperature lift ΔT = T_hot - T_cold (K)",
        ylabel="Cooling power (W)",
        title="Standing-Wave Refrigerator Cooling Power vs ΔT",
        save_path=output_dir / "sw_refrigerator_cooling_power_vs_delta_T.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(row["config"].t_hot - row["config"].t_cold) for row in cold_sweep],
        values=[float(row["cop"]) for row in cold_sweep],
        xlabel="Temperature lift ΔT = T_hot - T_cold (K)",
        ylabel="COP",
        title="Standing-Wave Refrigerator COP vs ΔT",
        save_path=output_dir / "sw_refrigerator_cop_vs_delta_T.png",
        show=False,
    )

    print("\nSaved figures:")
    print("  examples/output/sw_refrigerator_cop_vs_drive.png")
    print("  examples/output/sw_refrigerator_cooling_power_vs_delta_T.png")
    print("  examples/output/sw_refrigerator_cop_vs_delta_T.png")


if __name__ == "__main__":
    main()

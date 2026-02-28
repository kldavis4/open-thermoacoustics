#!/usr/bin/env python3
"""Tijani et al. standing-wave refrigerator validation run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openthermoacoustics import viz
from openthermoacoustics.validation.standing_wave_refrigerator import (
    compute_refrigerator_performance_short_stack,
    solve_standing_wave_refrigerator,
    sweep_cold_temperature,
    sweep_drive_ratio,
    tijani_refrigerator_config,
)


def main() -> None:
    cfg = tijani_refrigerator_config()
    point = solve_standing_wave_refrigerator(cfg)
    perf_short = compute_refrigerator_performance_short_stack(point)
    q_h2 = float(point["cooling_power_h2"])
    q_short = float(point["cooling_power_short_stack"])
    q_proxy = float(point["cooling_power_proxy"])
    w_stack = float(point["w_stack_absorbed"])
    cop_h2 = q_h2 / w_stack if w_stack > 0.0 else 0.0
    cop_proxy = q_proxy / w_stack if w_stack > 0.0 else 0.0

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "Tijani Refrigerator Diagnostic",
        "=============================",
        f"frequency_hz={point['frequency_hz']:.6f}",
        f"drive_ratio={cfg.drive_ratio:.6f}",
        f"t_hot={cfg.t_hot:.6f}",
        f"t_cold={cfg.t_cold:.6f}",
        "",
        "--- stack performance ---",
        f"W_stack_hot={point['w_stack_hot']:.6f} W",
        f"W_stack_cold={point['w_stack_cold']:.6f} W",
        f"W_stack_absorbed={point['w_stack_absorbed']:.6f} W",
        "method_comparison:",
        f"  Q_cold_short_stack={q_short:.6f} W",
        f"  Q_cold_h2_boundary={q_h2:.6f} W",
        f"  Q_cold_acoustic_proxy={q_proxy:.6f} W",
        f"  COP_short_stack={perf_short['cop']:.6f}",
        f"  COP_h2_boundary={cop_h2:.6f}",
        f"  COP_acoustic_proxy={cop_proxy:.6f}",
        f"Q_cold(selected)={point['cooling_power']:.6f} W",
        f"COP_stack(selected)={point['cop']:.6f}",
        f"COP_system={point['cop_system']:.6f}",
        f"COP_Carnot={point['cop_carnot']:.6f}",
        f"COPR_stack={point['cop_relative']:.6f}",
        "",
        "--- H2 components at stack boundaries ---",
        "hot_side:",
    ]
    for key, value in point["stack_h2_components_hot"].items():
        lines.append(f"  {key}={value:.6f} W")
    lines.append("cold_side:")
    for key, value in point["stack_h2_components_cold"].items():
        lines.append(f"  {key}={value:.6f} W")
    lines += [
        "",
        "--- Acoustic power at each segment ---",
    ]
    for row in point["power_profile"]:
        lines.append(
            f"  {row['name']}: W_in={row['w_in']:.6f} W, W_out={row['w_out']:.6f} W, "
            f"delta_W={row['delta_w']:.6f} W"
        )

    (output_dir / "tijani_diagnostic.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    drive_sweep = sweep_drive_ratio(cfg, np.array([0.005, 0.01, 0.014, 0.02, 0.03, 0.04, 0.05]))
    temp_sweep = sweep_cold_temperature(cfg, np.linspace(220.0, 290.0, 15))

    viz.plot_frequency_sweep(
        frequencies=[float(row["config"].drive_ratio) for row in drive_sweep],
        values=[float(row["cooling_power"]) for row in drive_sweep],
        xlabel="Drive ratio |p1|/p_m",
        ylabel="Q_cold (W)",
        title="Tijani Approximation: Cooling Power vs Drive",
        save_path=output_dir / "tijani_Qcold_vs_drive.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(row["config"].drive_ratio) for row in drive_sweep],
        values=[
            float(row["cooling_power"]) / (float(row["config"].drive_ratio) ** 2)
            for row in drive_sweep
        ],
        xlabel="Drive ratio |p1|/p_m",
        ylabel="Q_cold / D^2 (W)",
        title="Tijani Approximation: D^2 Scaling Check",
        save_path=output_dir / "tijani_Qcold_scaling.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(row["config"].drive_ratio) for row in drive_sweep],
        values=[float(row["cop"]) for row in drive_sweep],
        xlabel="Drive ratio |p1|/p_m",
        ylabel="COP_stack",
        title="Tijani Approximation: COP vs Drive",
        save_path=output_dir / "tijani_cop_vs_drive.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(row["config"].t_hot - row["config"].t_cold) for row in temp_sweep],
        values=[float(row["cop"]) for row in temp_sweep],
        xlabel="Temperature lift ΔT = T_hot - T_cold (K)",
        ylabel="COP_stack",
        title="Tijani Approximation: COP vs ΔT",
        save_path=output_dir / "tijani_cop_vs_deltaT.png",
        show=False,
    )

    print("\n".join(lines[:18]))
    print("...")
    print("Saved:")
    print("  examples/output/tijani_diagnostic.txt")
    print("  examples/output/tijani_Qcold_vs_drive.png")
    print("  examples/output/tijani_Qcold_scaling.png")
    print("  examples/output/tijani_cop_vs_drive.png")
    print("  examples/output/tijani_cop_vs_deltaT.png")


if __name__ == "__main__":
    main()

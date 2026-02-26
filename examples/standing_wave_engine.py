#!/usr/bin/env python3
"""Standing-wave thermoacoustic engine validation example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openthermoacoustics import gas, viz
from openthermoacoustics.validation.standing_wave_engine import (
    default_standing_wave_engine_config,
    sweep_standing_wave_engine,
)


def main() -> None:
    """Solve and visualize the canonical standing-wave validation case."""
    config = default_standing_wave_engine_config()
    helium = gas.Helium(mean_pressure=config.mean_pressure)
    f_simple = helium.sound_speed(config.t_cold) / (2.0 * config.total_length)

    print("Standing-Wave Engine Validation")
    print("-" * 40)
    print(f"Gas: Helium at {config.mean_pressure/1e6:.1f} MPa")
    print(f"Total length: {config.total_length:.3f} m")
    print(f"Simple half-wave estimate: {f_simple:.2f} Hz")
    print(f"Porosity: {config.porosity:.3f}")
    print(f"Hydraulic radius: {config.hydraulic_radius*1e3:.3f} mm")

    t_hot_values = np.arange(300.0, 801.0, 50.0)
    sweep = sweep_standing_wave_engine(config, t_hot_values=t_hot_values)

    print("\nSweep summary")
    print("T_hot [K]  f_res [Hz]  residual    dW_stack [W]")
    for point in sweep:
        print(
            f"{point.t_hot:8.1f}  {point.result.frequency:10.2f}  "
            f"{point.result.residual_norm:9.3e}  {point.stack_power_change:11.5f}"
        )

    onset_ratio = None
    for point in sweep:
        if point.stack_power_change > 0.0:
            onset_ratio = point.t_hot / config.t_cold
            break

    if onset_ratio is None:
        print("\nOnset not detected in 300-800 K sweep with current linear model.")
    else:
        print(f"\nEstimated onset temperature ratio: {onset_ratio:.3f}")

    result_600 = min(sweep, key=lambda p: abs(p.t_hot - 600.0)).result
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz.plot_profiles(
        result_600,
        units="m",
        save_path=output_dir / "standing_wave_engine_profiles_600K.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[p.t_hot for p in sweep],
        values=[p.result.frequency for p in sweep],
        xlabel="Hot-side Temperature (K)",
        ylabel="Resonant Frequency (Hz)",
        title="Standing-Wave Engine Frequency Shift",
        save_path=output_dir / "standing_wave_engine_frequency_shift.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[p.t_hot for p in sweep],
        values=[p.stack_power_change for p in sweep],
        xlabel="Hot-side Temperature (K)",
        ylabel="Delta Acoustic Power Across Stack (W)",
        title="Stack Acoustic Power Change vs Temperature",
        save_path=output_dir / "standing_wave_engine_stack_power_change.png",
        show=False,
    )

    print("\nSaved figures:")
    print("  examples/output/standing_wave_engine_profiles_600K.png")
    print("  examples/output/standing_wave_engine_frequency_shift.png")
    print("  examples/output/standing_wave_engine_stack_power_change.png")


if __name__ == "__main__":
    main()

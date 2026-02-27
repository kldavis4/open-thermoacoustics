#!/usr/bin/env python3
"""Standing-wave thermoacoustic engine validation example."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openthermoacoustics import gas, viz
from openthermoacoustics.validation.standing_wave_engine import (
    detect_onset_from_complex_frequency,
    geometry_sensitive_reference_config,
    optimized_standing_wave_engine_config,
    shifted_negative_control_config,
    solve_standing_wave_engine_complex_frequency_with_profiles,
    sweep_standing_wave_engine_complex_frequency,
    symmetric_negative_control_config,
)


def main() -> None:
    """Run the standing-wave benchmark set with complex-frequency onset metric."""
    base = symmetric_negative_control_config()
    helium = gas.Helium(mean_pressure=base.mean_pressure)
    f_simple = helium.sound_speed(base.t_cold) / (2.0 * base.total_length)
    t_hot_values = np.arange(300.0, 801.0, 25.0)

    print("Standing-Wave Engine Benchmark")
    print("-" * 40)
    print(f"Gas: Helium at {base.mean_pressure/1e6:.1f} MPa")
    print(f"Baseline length: {base.total_length:.3f} m")
    print(f"Half-wave estimate (baseline): {f_simple:.2f} Hz")

    cases = [
        ("symmetric_030_030", symmetric_negative_control_config()),
        ("shifted_010_050", shifted_negative_control_config()),
        ("reference_045_015", geometry_sensitive_reference_config()),
        ("optimized_100_020", optimized_standing_wave_engine_config()),
    ]

    sweeps: dict[str, list[dict[str, float | bool | str]]] = {}
    for label, cfg in cases:
        sweep = sweep_standing_wave_engine_complex_frequency(
            cfg,
            t_hot_values=t_hot_values,
        )
        sweeps[label] = sweep
        onset_ratio = detect_onset_from_complex_frequency(sweep)
        print(f"\nCase: {label}")
        print(f"  left/right ducts: {cfg.left_duct_length:.2f}/{cfg.right_duct_length:.2f} m")
        print(f"  plate spacing: {cfg.plate_spacing*1e3:.3f} mm")
        if onset_ratio is None:
            print("  onset: none <= 800 K")
        else:
            print(f"  onset ratio: {onset_ratio:.3f} (T_hot ~ {onset_ratio * cfg.t_cold:.1f} K)")

    optimized = optimized_standing_wave_engine_config()
    optimized_sweep = sweeps["optimized_100_020"]
    idx_650 = int(np.argmin(np.abs(np.array([float(p["t_hot"]) for p in optimized_sweep]) - 650.0)))
    point_650 = optimized_sweep[idx_650]
    profile_solution = solve_standing_wave_engine_complex_frequency_with_profiles(
        optimized,
        t_hot=float(point_650["t_hot"]),
        frequency_real_guess=float(point_650["frequency_real"]),
        frequency_imag_guess=float(point_650["frequency_imag"]),
    )
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_result = SimpleNamespace(
        x_profile=np.asarray(profile_solution["profiles"]["x"]),
        p1_profile=np.asarray(profile_solution["profiles"]["p1"]),
        U1_profile=np.asarray(profile_solution["profiles"]["U1"]),
        acoustic_power=np.asarray(profile_solution["profiles"]["acoustic_power"]),
        T_m_profile=np.asarray(profile_solution["profiles"]["T_m"]),
        frequency=float(point_650["frequency_real"]),
        converged=bool(point_650["converged"]),
    )
    viz.plot_profiles(
        profile_result,
        units="m",
        save_path=output_dir / "standing_wave_engine_profiles_optimized.png",
        show=False,
    )
    for label, _ in cases:
        sweep = sweeps[label]
        viz.plot_frequency_sweep(
            frequencies=[float(p["t_hot"]) for p in sweep],
            values=[float(p["frequency_imag"]) for p in sweep],
            xlabel="Hot-side Temperature (K)",
            ylabel="Imaginary Frequency (Hz)",
            title=f"Complex-Frequency Growth/Damping ({label})",
            save_path=output_dir / f"standing_wave_engine_fimag_{label}.png",
            show=False,
        )

    print("\nSaved figures:")
    print("  examples/output/standing_wave_engine_profiles_optimized.png")
    print("  examples/output/standing_wave_engine_fimag_symmetric_030_030.png")
    print("  examples/output/standing_wave_engine_fimag_shifted_010_050.png")
    print("  examples/output/standing_wave_engine_fimag_reference_045_015.png")
    print("  examples/output/standing_wave_engine_fimag_optimized_100_020.png")


if __name__ == "__main__":
    main()

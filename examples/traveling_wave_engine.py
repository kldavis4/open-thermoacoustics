#!/usr/bin/env python3
"""Distributed-loop traveling-wave engine example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openthermoacoustics import viz
from openthermoacoustics.validation.traveling_wave_engine import (
    compute_efficiency_estimate,
    compute_energy_balance_growth_rate,
    compute_net_acoustic_power,
    compute_regenerator_phase_profile,
    compute_stored_energy,
    default_traveling_wave_engine_config,
    detect_onset_from_complex_frequency,
    detect_onset_from_gain_proxy,
    estimate_loop_frequency_range,
    find_best_frequency_by_residual,
    find_onset_ratio_proxy,
    solve_traveling_wave_engine_complex_frequency,
    solve_traveling_wave_engine_determinant_complex_frequency,
    solve_traveling_wave_engine_fixed_frequency,
    sweep_efficiency_estimate,
    sweep_energy_balance_growth_rate,
    sweep_traveling_wave_complex_frequency,
    sweep_traveling_wave_determinant_complex_frequency,
    sweep_traveling_wave_frequency,
    sweep_traveling_wave_temperature,
    tuned_traveling_wave_engine_candidate_config,
)


def main() -> None:
    """Run fixed-frequency and sweep analyses for traveling-wave loop topology."""
    cfg = default_traveling_wave_engine_config()
    print("Traveling-Wave Loop Validation")
    print("-" * 40)
    print(f"Mean pressure: {cfg.mean_pressure/1e6:.1f} MPa")
    print(f"T_cold/T_hot: {cfg.t_cold:.1f}/{cfg.t_hot:.1f} K")
    print(f"Resonator length: {cfg.resonator_length:.3f} m")
    print(f"Feedback length: {cfg.feedback_length:.3f} m")
    freq_est = estimate_loop_frequency_range(cfg)
    print(
        "Expected loop-loaded frequency band: "
        f"{freq_est['f_expected_low_hz']:.1f}-{freq_est['f_expected_high_hz']:.1f} Hz"
    )

    point_100 = solve_traveling_wave_engine_fixed_frequency(cfg, frequency_hz=100.0)
    result_100 = point_100["result"]
    print("\nFixed frequency (100 Hz):")
    print(f"  Converged: {result_100.converged}")
    print(f"  Residual:  {result_100.residual_norm:.3e}")
    print(f"  Zb:        {result_100.Zb_real:.3e} + j{result_100.Zb_imag:.3e}")
    print(f"  Regenerator phase(p-U): {point_100['phase_regenerator_mid_deg']:.2f} deg")

    freqs = np.arange(50.0, 251.0, 25.0)
    sweep = sweep_traveling_wave_frequency(cfg, frequencies_hz=freqs)
    best = find_best_frequency_by_residual(sweep)
    print("\nFrequency sweep:")
    for p in sweep:
        r = p["result"]
        print(
            f"  f={p['frequency_hz']:6.1f} Hz  residual={r.residual_norm:9.3e}  "
            f"phase={p['phase_regenerator_mid_deg']:7.2f} deg"
        )
    print(
        f"\nBest by residual: f={best['frequency_hz']:.1f} Hz, "
        f"residual={best['result'].residual_norm:.3e}"
    )

    t_hot_values = np.arange(300.0, 751.0, 25.0)
    temp_sweep = sweep_traveling_wave_temperature(
        cfg,
        frequency_hz=float(best["frequency_hz"]),
        t_hot_values=t_hot_values,
    )
    onset_ratio = detect_onset_from_gain_proxy(temp_sweep, t_cold=cfg.t_cold)
    print("\nTemperature sweep at best frequency (gain proxy):")
    for p in temp_sweep:
        print(
            f"  T_hot={p['t_hot']:6.1f} K  net_gain_proxy={p['net_gain_proxy']:+.3e}  "
            f"regen_dW={p['regenerator_power_delta']:+.3e}"
        )
    if onset_ratio is None:
        print("  No proxy onset crossing in scanned range.")
    else:
        print(
            f"  Proxy onset ratio ~ {onset_ratio:.3f} "
            f"(T_hot ~ {onset_ratio * cfg.t_cold:.1f} K)"
        )

    tuned = tuned_traveling_wave_engine_candidate_config()
    tuned_onset, tuned_sweep = find_onset_ratio_proxy(
        tuned,
        frequency_hz=120.0,
        t_hot_min=300.0,
        t_hot_max=900.0,
        coarse_step=100.0,
        fine_step=20.0,
    )
    print("\nTuned candidate check:")
    if tuned_onset is None:
        print("  No proxy onset crossing in scanned range.")
    else:
        print(
            f"  Proxy onset ratio ~ {tuned_onset:.3f} "
            f"(T_hot ~ {tuned_onset * tuned.t_cold:.1f} K)"
        )
    tuned_point = solve_traveling_wave_engine_fixed_frequency(
        tuned,
        frequency_hz=120.0,
        t_hot=600.0,
    )
    phase_profile = compute_regenerator_phase_profile(tuned_point)
    eff = compute_efficiency_estimate(tuned_point, t_cold=tuned.t_cold, t_hot=600.0)
    print(
        "  Regenerator phase [in/mid/out]: "
        f"{phase_profile['inlet_deg']:.1f}/{phase_profile['mid_deg']:.1f}/"
        f"{phase_profile['outlet_deg']:.1f} deg"
    )
    print(
        "  Efficiency estimate at 600 K: "
        f"eta_thermal~{eff['eta_thermal_est']:.3e}, "
        f"eta_Carnot={eff['eta_carnot']:.3f}"
    )

    eff_rows = sweep_efficiency_estimate(
        tuned,
        frequency_hz=120.0,
        t_hot_values=np.arange(350.0, 801.0, 50.0),
    )
    energy_sweep = sweep_energy_balance_growth_rate(
        tuned,
        t_hot_values=np.arange(300.0, 901.0, 25.0),
        frequency_guess_hz=120.0,
    )
    w_net_600 = compute_net_acoustic_power(tuned, t_hot=600.0, solve_result=tuned_point)
    e_600 = compute_stored_energy(tuned, t_hot=600.0, solve_result=tuned_point)
    fi_600 = compute_energy_balance_growth_rate(tuned, t_hot=600.0, solve_result=tuned_point)
    print(
        "  Energy balance @600 K: "
        f"W_net={w_net_600:+.3e} W, E={e_600:.3e} J, f_imag~{fi_600:+.3f} Hz"
    )
    complex_sweep = sweep_traveling_wave_complex_frequency(
        tuned,
        t_hot_values=np.arange(350.0, 801.0, 50.0),
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    onset_complex = detect_onset_from_complex_frequency(complex_sweep)
    if onset_complex is None:
        print("  Complex-frequency onset crossing not found in scanned range.")
    else:
        print(
            f"  Complex-frequency onset ratio ~ {onset_complex:.3f} "
            f"(T_hot ~ {onset_complex * tuned.t_cold:.1f} K)"
        )
    point_complex = solve_traveling_wave_engine_complex_frequency(
        tuned,
        t_hot=600.0,
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    print(
        "  Complex-frequency @600 K: "
        f"f={point_complex['frequency_real']:.2f}+j{point_complex['frequency_imag']:.2f} Hz, "
        f"residual={point_complex['residual_norm']:.3e}"
    )
    det_sweep = sweep_traveling_wave_determinant_complex_frequency(
        tuned,
        t_hot_values=np.arange(350.0, 801.0, 50.0),
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    onset_det = detect_onset_from_complex_frequency(det_sweep)
    if onset_det is None:
        print("  Determinant onset crossing not found in scanned range.")
    else:
        print(
            f"  Determinant onset ratio ~ {onset_det:.3f} "
            f"(T_hot ~ {onset_det * tuned.t_cold:.1f} K)"
        )
    point_det = solve_traveling_wave_engine_determinant_complex_frequency(
        tuned,
        t_hot=600.0,
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    print(
        "  Determinant complex-frequency @600 K: "
        f"f={point_det['frequency_real']:.2f}+j{point_det['frequency_imag']:.2f} Hz, "
        f"residual={point_det['residual_norm']:.3e}"
    )

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz.plot_frequency_sweep(
        frequencies=[float(p["frequency_hz"]) for p in sweep],
        values=[float(p["result"].residual_norm) for p in sweep],
        xlabel="Frequency (Hz)",
        ylabel="Loop Residual Norm",
        title="Traveling-Wave Loop Residual vs Frequency",
        save_path=output_dir / "traveling_wave_residual_vs_frequency.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(p["frequency_hz"]) for p in sweep],
        values=[float(p["phase_regenerator_mid_deg"]) for p in sweep],
        xlabel="Frequency (Hz)",
        ylabel="Phase(p1)-Phase(U1) in Regenerator (deg)",
        title="Regenerator Phase vs Frequency",
        save_path=output_dir / "traveling_wave_regenerator_phase_vs_frequency.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(p["t_hot"]) for p in tuned_sweep],
        values=[float(p["net_gain_proxy"]) for p in tuned_sweep],
        xlabel="T_hot (K)",
        ylabel="Net Gain Proxy (W)",
        title="Traveling-Wave Gain Proxy vs Hot Temperature",
        save_path=output_dir / "tw_onset_sweep.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(r["t_hot"]) for r in energy_sweep],
        values=[float(r["w_net"]) for r in energy_sweep],
        xlabel="T_hot (K)",
        ylabel="Net Acoustic Power (W)",
        title="Traveling-Wave Net Acoustic Power vs Hot Temperature",
        save_path=output_dir / "tw_Wnet_vs_temp.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(r["t_hot"]) for r in energy_sweep],
        values=[float(r["frequency_imag_energy_balance"]) for r in energy_sweep],
        xlabel="T_hot (K)",
        ylabel="Energy-Balance Imaginary Frequency (Hz)",
        title="Traveling-Wave Growth Rate from Energy Balance",
        save_path=output_dir / "tw_energy_balance_fimag_vs_temp.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(r["t_hot"]) for r in eff_rows],
        values=[float(r["eta_thermal_est"]) for r in eff_rows],
        xlabel="T_hot (K)",
        ylabel="Estimated Thermal Efficiency",
        title="Traveling-Wave Efficiency Estimate vs Hot Temperature",
        save_path=output_dir / "tw_efficiency_vs_temp.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(p["t_hot"]) for p in complex_sweep],
        values=[float(p["frequency_imag"]) for p in complex_sweep],
        xlabel="T_hot (K)",
        ylabel="Imaginary Frequency (Hz)",
        title="Traveling-Wave Complex-Frequency Growth Rate",
        save_path=output_dir / "tw_complex_frequency_imag_vs_temp.png",
        show=False,
    )
    viz.plot_frequency_sweep(
        frequencies=[float(p["t_hot"]) for p in det_sweep],
        values=[float(p["frequency_imag"]) for p in det_sweep],
        xlabel="T_hot (K)",
        ylabel="Imaginary Frequency (Hz)",
        title="Traveling-Wave Determinant Growth Rate",
        save_path=output_dir / "tw_det_fimag_vs_temp.png",
        show=False,
    )
    print("\nSaved figures:")
    print("  examples/output/traveling_wave_residual_vs_frequency.png")
    print("  examples/output/traveling_wave_regenerator_phase_vs_frequency.png")
    print("  examples/output/tw_onset_sweep.png")
    print("  examples/output/tw_Wnet_vs_temp.png")
    print("  examples/output/tw_energy_balance_fimag_vs_temp.png")
    print("  examples/output/tw_efficiency_vs_temp.png")
    print("  examples/output/tw_complex_frequency_imag_vs_temp.png")
    print("  examples/output/tw_det_fimag_vs_temp.png")


if __name__ == "__main__":
    main()

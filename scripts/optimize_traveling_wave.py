#!/usr/bin/env python3
"""Coarse design sweep for distributed-loop traveling-wave engine tuning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openthermoacoustics.validation.traveling_wave_engine import (
    default_traveling_wave_engine_config,
    detect_onset_from_gain_proxy,
    find_best_frequency_by_residual,
    sweep_traveling_wave_frequency,
    sweep_traveling_wave_temperature,
)


def _phase_distance_to_traveling(phase_deg: float) -> float:
    """Distance to 0 deg phase with wrap at +/-180."""
    wrapped = (phase_deg + 180.0) % 360.0 - 180.0
    return abs(wrapped)


def _score(net_gain_proxy: float, phase_deg: float) -> float:
    """Simple multi-objective scalar score (higher is better)."""
    return net_gain_proxy - 1e-3 * _phase_distance_to_traveling(phase_deg)


def main() -> None:
    base = default_traveling_wave_engine_config()
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: broad geometry/frequency tuning at fixed hot temperature.
    feedback_radius_values = [0.022, 0.028, 0.035, 0.042]
    resonator_length_values = [0.60, 0.80, 1.00, 1.20]
    regenerator_rh_values = [0.02e-3, 0.04e-3, 0.08e-3]
    pressure_values = [3.0e6, 4.0e6]
    freq_grid = np.arange(40.0, 241.0, 20.0)

    stage1: list[dict[str, float | dict]] = []
    print("=== Stage 1: Frequency/Phase Screening ===")
    for pressure in pressure_values:
        for feedback_radius in feedback_radius_values:
            for resonator_length in resonator_length_values:
                for regen_rh in regenerator_rh_values:
                    cfg = replace(
                        base,
                        mean_pressure=pressure,
                        feedback_radius=feedback_radius,
                        resonator_length=resonator_length,
                        regenerator_hydraulic_radius=regen_rh,
                    )
                    sweep = sweep_traveling_wave_frequency(cfg, frequencies_hz=freq_grid, t_hot=600.0)
                    best = find_best_frequency_by_residual(sweep)
                    phase = float(best["phase_regenerator_mid_deg"])
                    residual = float(best["result"].residual_norm)
                    best_freq = float(best["frequency_hz"])
                    stage1.append(
                        {
                            "mean_pressure": pressure,
                            "feedback_radius": feedback_radius,
                            "resonator_length": resonator_length,
                            "regenerator_hydraulic_radius": regen_rh,
                            "best_frequency_hz": best_freq,
                            "phase_deg": phase,
                            "phase_dist": _phase_distance_to_traveling(phase),
                            "residual_norm": residual,
                            "best_point": best,
                        }
                    )
                    print(
                        f"P={pressure / 1e6:.1f}MPa fb_r={feedback_radius:.3f}m "
                        f"Lr={resonator_length:.2f}m rh={regen_rh * 1e3:.3f}mm -> "
                        f"f*={best_freq:.1f}Hz phase={phase:+.1f}deg res={residual:.2e}"
                    )

    stage1_sorted = sorted(stage1, key=lambda row: (row["phase_dist"], row["residual_norm"]))
    top_candidates = stage1_sorted[:10]

    # Stage 2: temperature/onset proxy for top candidates.
    print("\n=== Stage 2: Temperature Sweep on Top Candidates ===")
    t_hot_values = np.arange(300.0, 851.0, 50.0)
    stage2: list[dict[str, float | None]] = []
    for row in top_candidates:
        cfg = replace(
            base,
            mean_pressure=float(row["mean_pressure"]),
            feedback_radius=float(row["feedback_radius"]),
            resonator_length=float(row["resonator_length"]),
            regenerator_hydraulic_radius=float(row["regenerator_hydraulic_radius"]),
        )
        best_freq = float(row["best_frequency_hz"])
        temp_sweep = sweep_traveling_wave_temperature(
            cfg,
            frequency_hz=best_freq,
            t_hot_values=t_hot_values,
        )
        onset_ratio = detect_onset_from_gain_proxy(temp_sweep, t_cold=cfg.t_cold)
        final_gain = float(temp_sweep[-1]["net_gain_proxy"])
        phase = float(row["phase_deg"])
        score = _score(final_gain, phase)
        summary = {
            "mean_pressure": float(row["mean_pressure"]),
            "feedback_radius": float(row["feedback_radius"]),
            "resonator_length": float(row["resonator_length"]),
            "regenerator_hydraulic_radius": float(row["regenerator_hydraulic_radius"]),
            "best_frequency_hz": best_freq,
            "phase_deg": phase,
            "final_gain_proxy": final_gain,
            "onset_ratio_proxy": onset_ratio,
            "score": score,
        }
        stage2.append(summary)
        print(
            "P={:.1f}MPa fb_r={:.3f}m Lr={:.2f}m rh={:.3f}mm "
            "f*={:.1f}Hz phase={:+.1f}deg gain850={:+.3e} onset_proxy={}".format(
                summary["mean_pressure"] / 1e6,
                summary["feedback_radius"],
                summary["resonator_length"],
                summary["regenerator_hydraulic_radius"] * 1e3,
                summary["best_frequency_hz"],
                summary["phase_deg"],
                summary["final_gain_proxy"],
                "None" if summary["onset_ratio_proxy"] is None else f"{summary['onset_ratio_proxy']:.3f}",
            )
        )

    stage2_sorted = sorted(stage2, key=lambda row: row["score"], reverse=True)
    best = stage2_sorted[0]
    print("\n=== Recommended Next Candidate ===")
    print(best)

    # Stage 3: focused second pass around best Stage-2 candidate with extra knobs.
    print("\n=== Stage 3: Focused Pass (feedback/tbt/pressure refinement) ===")
    base2 = replace(
        base,
        mean_pressure=float(best["mean_pressure"]),
        feedback_radius=float(best["feedback_radius"]),
        resonator_length=float(best["resonator_length"]),
        regenerator_hydraulic_radius=float(best["regenerator_hydraulic_radius"]),
    )
    pressure_values_2 = [4.0e6, 5.0e6, 6.0e6]
    feedback_radius_values_2 = sorted(
        max(0.015, base2.feedback_radius + dr) for dr in (-0.004, 0.0, 0.004)
    )
    feedback_length_values_2 = [0.30, 0.50, 0.70]
    tbt_length_values_2 = [0.15, 0.25, 0.35]
    regen_rh_values_2 = [
        base2.regenerator_hydraulic_radius,
        1.5 * base2.regenerator_hydraulic_radius,
    ]
    freq_grid_2 = np.arange(30.0, 191.0, 20.0)
    stage3: list[dict[str, float | dict]] = []

    for pressure in pressure_values_2:
        for feedback_radius in feedback_radius_values_2:
            for feedback_length in feedback_length_values_2:
                for tbt_length in tbt_length_values_2:
                    for regen_rh in regen_rh_values_2:
                        cfg = replace(
                            base2,
                            mean_pressure=pressure,
                            feedback_radius=feedback_radius,
                            feedback_length=feedback_length,
                            tbt_length=tbt_length,
                            regenerator_hydraulic_radius=regen_rh,
                        )
                        sweep = sweep_traveling_wave_frequency(
                            cfg,
                            frequencies_hz=freq_grid_2,
                            t_hot=600.0,
                        )
                        best_point = find_best_frequency_by_residual(sweep)
                        phase = float(best_point["phase_regenerator_mid_deg"])
                        residual = float(best_point["result"].residual_norm)
                        stage3.append(
                            {
                                "mean_pressure": pressure,
                                "feedback_radius": feedback_radius,
                                "feedback_length": feedback_length,
                                "tbt_length": tbt_length,
                                "regenerator_hydraulic_radius": regen_rh,
                                "best_frequency_hz": float(best_point["frequency_hz"]),
                                "phase_deg": phase,
                                "phase_dist": _phase_distance_to_traveling(phase),
                                "residual_norm": residual,
                            }
                        )

    stage3_sorted = sorted(stage3, key=lambda row: (row["phase_dist"], row["residual_norm"]))
    top3_stage3 = stage3_sorted[:8]
    stage3_eval: list[dict[str, float | None]] = []
    for row in top3_stage3:
        cfg = replace(
            base2,
            mean_pressure=float(row["mean_pressure"]),
            feedback_radius=float(row["feedback_radius"]),
            feedback_length=float(row["feedback_length"]),
            tbt_length=float(row["tbt_length"]),
            regenerator_hydraulic_radius=float(row["regenerator_hydraulic_radius"]),
        )
        temp_sweep = sweep_traveling_wave_temperature(
            cfg,
            frequency_hz=float(row["best_frequency_hz"]),
            t_hot_values=np.arange(300.0, 951.0, 50.0),
        )
        onset_ratio = detect_onset_from_gain_proxy(temp_sweep, t_cold=cfg.t_cold)
        final_gain = float(temp_sweep[-1]["net_gain_proxy"])
        stage3_eval.append(
            {
                "mean_pressure": float(row["mean_pressure"]),
                "feedback_radius": float(row["feedback_radius"]),
                "feedback_length": float(row["feedback_length"]),
                "tbt_length": float(row["tbt_length"]),
                "regenerator_hydraulic_radius": float(row["regenerator_hydraulic_radius"]),
                "best_frequency_hz": float(row["best_frequency_hz"]),
                "phase_deg": float(row["phase_deg"]),
                "phase_dist": float(row["phase_dist"]),
                "final_gain_proxy": final_gain,
                "onset_ratio_proxy": onset_ratio,
                "score": _score(final_gain, float(row["phase_deg"])),
            }
        )
    stage3_eval_sorted = sorted(stage3_eval, key=lambda row: row["score"], reverse=True)
    best_stage3 = stage3_eval_sorted[0]
    print("Top Stage-3 candidate:")
    print(best_stage3)

    # Plot summaries.
    x = np.arange(len(stage2_sorted))
    phase_dist = [_phase_distance_to_traveling(float(row["phase_deg"])) for row in stage2_sorted]
    gains = [float(row["final_gain_proxy"]) for row in stage2_sorted]

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, phase_dist, "o-", label="|phase-0| [deg]")
    plt.plot(x, np.array(gains) * 1e3, "s-", label="gain@850K [mW]")
    plt.xlabel("Candidate rank (best score first)")
    plt.ylabel("Metric value")
    plt.title("Traveling-Wave Tuning Candidates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "traveling_wave_candidate_ranking.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    fb = [float(row["feedback_radius"]) * 1e3 for row in stage1]
    pdist = [float(row["phase_dist"]) for row in stage1]
    plt.scatter(fb, pdist, c=[float(row["resonator_length"]) for row in stage1], cmap="viridis")
    plt.colorbar(label="Resonator length [m]")
    plt.xlabel("Feedback radius [mm]")
    plt.ylabel("|phase-0| [deg] at best frequency")
    plt.title("Stage-1 Phase Screening")
    plt.tight_layout()
    plt.savefig(output_dir / "traveling_wave_phase_screening.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    x3 = np.arange(len(stage3_eval_sorted))
    phase3 = [float(row["phase_dist"]) for row in stage3_eval_sorted]
    gain3 = [float(row["final_gain_proxy"]) for row in stage3_eval_sorted]
    plt.plot(x3, phase3, "o-", label="|phase-0| [deg]")
    plt.plot(x3, np.array(gain3) * 1e3, "s-", label="gain@900K [mW]")
    plt.xlabel("Stage-3 candidate rank")
    plt.ylabel("Metric value")
    plt.title("Traveling-Wave Focused Pass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "traveling_wave_focused_pass.png", dpi=160)
    plt.close()

    print("\nSaved plots:")
    print("  examples/output/traveling_wave_candidate_ranking.png")
    print("  examples/output/traveling_wave_phase_screening.png")
    print("  examples/output/traveling_wave_focused_pass.png")


if __name__ == "__main__":
    main()

"""Validation tests for distributed-loop traveling-wave engine plumbing."""

from __future__ import annotations

import numpy as np

from openthermoacoustics.validation.traveling_wave_engine import (
    compute_efficiency_estimate,
    compute_loop_power_profile,
    compute_regenerator_phase_profile,
    default_traveling_wave_engine_config,
    detect_onset_from_gain_proxy,
    estimate_loop_frequency_range,
    find_best_frequency_by_residual,
    find_onset_ratio_proxy,
    solve_traveling_wave_engine_fixed_frequency,
    sweep_efficiency_estimate,
    sweep_traveling_wave_frequency,
    sweep_traveling_wave_temperature,
    tuned_traveling_wave_engine_candidate_config,
)


def test_fixed_frequency_loop_solve_converges() -> None:
    """Fixed-frequency distributed TBRANCH solve should converge robustly."""
    cfg = default_traveling_wave_engine_config()
    point = solve_traveling_wave_engine_fixed_frequency(cfg, frequency_hz=100.0)
    result = point["result"]

    assert result.converged
    assert result.residual_norm < 1e-6
    assert np.isfinite(result.Zb_real)
    assert np.isfinite(result.Zb_imag)
    assert point["profiles"]["trunk"] is not None
    assert point["profiles"]["branch"] is not None


def test_frequency_sweep_selects_low_residual_region() -> None:
    """Frequency sweep should provide a best point with finite residual/phase."""
    cfg = default_traveling_wave_engine_config()
    freqs = np.arange(50.0, 251.0, 25.0)
    sweep = sweep_traveling_wave_frequency(cfg, frequencies_hz=freqs)
    best = find_best_frequency_by_residual(sweep)

    assert len(sweep) == len(freqs)
    assert float(best["result"].residual_norm) < 1e-6
    phase = best["phase_regenerator_mid_deg"]
    assert phase is not None
    assert np.isfinite(phase)


def test_temperature_sweep_returns_finite_gain_proxy() -> None:
    """Temperature sweep at fixed frequency should return finite gain proxies."""
    cfg = default_traveling_wave_engine_config()
    t_hot_values = np.arange(300.0, 651.0, 50.0)
    sweep = sweep_traveling_wave_temperature(
        cfg,
        frequency_hz=100.0,
        t_hot_values=t_hot_values,
    )
    assert len(sweep) == len(t_hot_values)
    gain = np.array([float(p["net_gain_proxy"]) for p in sweep])
    assert np.all(np.isfinite(gain))

    onset_ratio = detect_onset_from_gain_proxy(sweep, t_cold=cfg.t_cold)
    if onset_ratio is not None:
        assert onset_ratio > 1.0


def test_tuned_candidate_shows_proxy_onset_below_1p5() -> None:
    """Tuned candidate should cross gain-proxy onset in low temperature ratio range."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    temp_sweep = sweep_traveling_wave_temperature(
        cfg,
        frequency_hz=120.0,
        t_hot_values=np.arange(300.0, 901.0, 100.0),
    )
    onset_ratio = detect_onset_from_gain_proxy(temp_sweep, t_cold=cfg.t_cold)
    assert onset_ratio is not None
    assert 1.0 < onset_ratio < 1.5
    assert float(temp_sweep[-1]["net_gain_proxy"]) > 0.0


def test_onset_ratio_lower_than_standing_wave_benchmark() -> None:
    """Traveling-wave proxy onset should substantially beat standing-wave onset."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    onset_ratio, _ = find_onset_ratio_proxy(
        cfg,
        frequency_hz=120.0,
        t_hot_max=900.0,
        coarse_step=100.0,
        fine_step=20.0,
    )
    assert onset_ratio is not None
    standing_wave_onset_ratio = 2.01
    assert onset_ratio < 0.8 * standing_wave_onset_ratio


def test_regenerator_phase_improves_vs_baseline() -> None:
    """Tuned candidate should reduce phase lag magnitude vs baseline default."""
    baseline = default_traveling_wave_engine_config()
    tuned = tuned_traveling_wave_engine_candidate_config()
    point_baseline = solve_traveling_wave_engine_fixed_frequency(
        baseline, frequency_hz=100.0, t_hot=600.0
    )
    point_tuned = solve_traveling_wave_engine_fixed_frequency(
        tuned, frequency_hz=120.0, t_hot=600.0
    )
    ph_base = abs(compute_regenerator_phase_profile(point_baseline)["mid_deg"])
    ph_tuned = abs(compute_regenerator_phase_profile(point_tuned)["mid_deg"])
    assert ph_tuned < ph_base
    assert ph_tuned < 120.0


def test_efficiency_estimate_obeys_carnot_bound() -> None:
    """Estimated thermal efficiency should satisfy second-law bound."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_traveling_wave_engine_fixed_frequency(cfg, frequency_hz=120.0, t_hot=600.0)
    eff = compute_efficiency_estimate(point, t_cold=cfg.t_cold, t_hot=600.0)
    assert 0.0 <= eff["eta_thermal_est"] <= eff["eta_carnot"]
    assert 0.0 <= eff["eta_relative"] <= 1.0


def test_power_flow_is_directionally_consistent_in_each_path() -> None:
    """Branch and trunk power signs should each be internally consistent."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_traveling_wave_engine_fixed_frequency(cfg, frequency_hz=120.0, t_hot=600.0)
    power_profile = compute_loop_power_profile(point)

    def nonzero_signs(rows: list[dict[str, float | str]]) -> set[int]:
        signs: set[int] = set()
        for row in rows:
            for key in ("w_in", "w_out"):
                val = float(row[key])
                if abs(val) > 1e-10:
                    signs.add(int(np.sign(val)))
        return signs

    branch_signs = nonzero_signs(power_profile["branch"])
    trunk_signs = nonzero_signs(power_profile["trunk"])
    assert len(branch_signs) <= 1
    assert len(trunk_signs) <= 1


def test_frequency_estimate_range_is_order_of_magnitude_correct() -> None:
    """Quarter-wave estimate and loaded range should be ordered and positive."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    freq = estimate_loop_frequency_range(cfg)
    assert freq["f_quarter_hz"] > 0.0
    assert freq["f_expected_low_hz"] < freq["f_expected_high_hz"] < freq["f_quarter_hz"]


def test_efficiency_sweep_rows_are_finite() -> None:
    """Efficiency sweep should return finite, bounded metrics."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    rows = sweep_efficiency_estimate(
        cfg,
        frequency_hz=120.0,
        t_hot_values=np.arange(350.0, 651.0, 100.0),
    )
    assert len(rows) > 0
    for row in rows:
        assert np.isfinite(row["eta_carnot"])
        assert np.isfinite(row["eta_thermal_est"])
        assert np.isfinite(row["eta_relative"])
        assert 0.0 <= row["eta_thermal_est"] <= row["eta_carnot"]
        assert 0.0 <= row["eta_relative"] <= 1.0

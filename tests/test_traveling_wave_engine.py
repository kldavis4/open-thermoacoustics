"""Validation tests for distributed-loop traveling-wave engine plumbing."""

from __future__ import annotations

import numpy as np

from openthermoacoustics.validation.traveling_wave_engine import (
    default_traveling_wave_engine_config,
    detect_onset_from_gain_proxy,
    find_best_frequency_by_residual,
    solve_traveling_wave_engine_fixed_frequency,
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

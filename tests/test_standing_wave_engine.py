"""Validation tests for standing-wave thermoacoustic engine onset behavior."""

from __future__ import annotations

import numpy as np
import pytest

from openthermoacoustics import gas
from openthermoacoustics.validation.standing_wave_engine import (
    EngineSweepPoint,
    StandingWaveEngineConfig,
    detect_onset_from_complex_frequency,
    geometry_sensitive_reference_config,
    optimized_standing_wave_engine_config,
    shifted_negative_control_config,
    solve_standing_wave_engine,
    solve_standing_wave_engine_complex_frequency_with_profiles,
    sweep_standing_wave_engine,
    sweep_standing_wave_engine_complex_frequency,
    symmetric_negative_control_config,
)


@pytest.fixture(scope="module")
def symmetric_config() -> StandingWaveEngineConfig:
    """Baseline symmetric layout (negative control for onset <= 800 K)."""
    return symmetric_negative_control_config()


@pytest.fixture(scope="module")
def shifted_config() -> StandingWaveEngineConfig:
    """Stack shifted toward left closed end while keeping total length fixed."""
    return shifted_negative_control_config()


@pytest.fixture(scope="module")
def optimized_config() -> StandingWaveEngineConfig:
    """Benchmark candidate selected by onset optimization sweep."""
    return optimized_standing_wave_engine_config()


@pytest.fixture(scope="module")
def reference_config() -> StandingWaveEngineConfig:
    """Geometry-sensitive reference with higher onset than optimized benchmark."""
    return geometry_sensitive_reference_config()


@pytest.fixture(scope="module")
def symmetric_real_sweep(
    symmetric_config: StandingWaveEngineConfig,
) -> list[EngineSweepPoint]:
    t_hot_values = np.arange(300.0, 801.0, 100.0)
    return sweep_standing_wave_engine(symmetric_config, t_hot_values=t_hot_values)


@pytest.fixture(scope="module")
def symmetric_complex_sweep(
    symmetric_config: StandingWaveEngineConfig,
) -> list[dict[str, float | bool | str]]:
    t_hot_values = np.arange(300.0, 801.0, 100.0)
    return sweep_standing_wave_engine_complex_frequency(
        symmetric_config, t_hot_values=t_hot_values
    )


@pytest.fixture(scope="module")
def shifted_complex_sweep(
    shifted_config: StandingWaveEngineConfig,
) -> list[dict[str, float | bool | str]]:
    t_hot_values = np.arange(300.0, 801.0, 100.0)
    return sweep_standing_wave_engine_complex_frequency(shifted_config, t_hot_values=t_hot_values)


@pytest.fixture(scope="module")
def optimized_complex_sweep(
    optimized_config: StandingWaveEngineConfig,
) -> list[dict[str, float | bool | str]]:
    t_hot_values = np.arange(300.0, 801.0, 25.0)
    return sweep_standing_wave_engine_complex_frequency(optimized_config, t_hot_values=t_hot_values)


@pytest.fixture(scope="module")
def reference_complex_sweep(
    reference_config: StandingWaveEngineConfig,
) -> list[dict[str, float | bool | str]]:
    t_hot_values = np.arange(300.0, 1001.0, 25.0)
    return sweep_standing_wave_engine_complex_frequency(reference_config, t_hot_values=t_hot_values)


def _point_at_temperature(sweep_points: list[EngineSweepPoint], t_hot: float) -> EngineSweepPoint:
    return min(sweep_points, key=lambda point: abs(point.t_hot - t_hot))


def test_isothermal_resonant_frequency_within_expected_range(
    symmetric_config: StandingWaveEngineConfig,
    symmetric_real_sweep: list[EngineSweepPoint],
) -> None:
    """Isothermal resonance should be close to half-wave estimate."""
    iso = _point_at_temperature(symmetric_real_sweep, 300.0)
    helium = gas.Helium(mean_pressure=symmetric_config.mean_pressure)
    expected = helium.sound_speed(symmetric_config.t_cold) / (2.0 * symmetric_config.total_length)
    relative_error = abs(iso.result.frequency - expected) / expected

    assert iso.result.frequency > 0.0
    assert 300.0 < iso.result.frequency < 1200.0
    assert relative_error < 0.15


def test_complex_frequency_primary_metric_is_well_converged(
    symmetric_config: StandingWaveEngineConfig,
) -> None:
    """Complex-frequency solve should satisfy closed-end boundary robustly."""
    point = solve_standing_wave_engine(
        symmetric_config, t_hot=600.0, frequency_guess=700.0, p1_phase_guess=0.0
    )
    # Real-frequency solve is intentionally approximate in this formulation.
    assert point.result.residual_norm < 1e-3


def test_symmetric_layout_has_no_onset_below_800k(
    symmetric_complex_sweep: list[dict[str, float | bool | str]],
) -> None:
    """Negative control: symmetric stack placement should not onset below 800 K."""
    onset_ratio = detect_onset_from_complex_frequency(symmetric_complex_sweep)
    assert onset_ratio is None


def test_optimized_layout_onset_target_range(
    optimized_complex_sweep: list[dict[str, float | bool | str]],
) -> None:
    """Optimized benchmark should onset in a practical standing-wave range."""
    onset_ratio = detect_onset_from_complex_frequency(optimized_complex_sweep)
    assert onset_ratio is not None
    assert 1.3 <= onset_ratio <= 2.5


def test_optimized_layout_onset_regression_temperature(
    optimized_complex_sweep: list[dict[str, float | bool | str]],
) -> None:
    """Regression guard: optimized onset should stay near the validated ~604 K."""
    onset_ratio = detect_onset_from_complex_frequency(optimized_complex_sweep)
    assert onset_ratio is not None
    onset_hot = onset_ratio * 300.0
    assert 560.0 <= onset_hot <= 650.0


def test_shifted_layout_has_no_onset_below_800k(
    shifted_complex_sweep: list[dict[str, float | bool | str]],
) -> None:
    """Secondary negative control: 0.10/0.50 layout stays damped <= 800 K."""
    onset_ratio = detect_onset_from_complex_frequency(shifted_complex_sweep)
    assert onset_ratio is None


def test_geometry_sensitive_reference_onset_near_890k(
    reference_complex_sweep: list[dict[str, float | bool | str]],
) -> None:
    """0.45/0.15 reference should onset around 890 K for current stack spacing."""
    onset_ratio = detect_onset_from_complex_frequency(reference_complex_sweep)
    assert onset_ratio is not None
    onset_hot = onset_ratio * 300.0
    assert 840.0 <= onset_hot <= 940.0


def test_optimized_layout_is_above_onset_with_margin(
    optimized_config: StandingWaveEngineConfig,
    optimized_complex_sweep: list[dict[str, float | bool | str]],
) -> None:
    """50 K above interpolated onset should have negative f_imag."""
    onset_ratio = detect_onset_from_complex_frequency(optimized_complex_sweep)
    assert onset_ratio is not None
    t_check = min(800.0, onset_ratio * optimized_config.t_cold + 50.0)
    point = solve_standing_wave_engine(
        optimized_config,
        t_hot=t_check,
        frequency_guess=float(optimized_complex_sweep[0]["frequency_real"]),
        p1_phase_guess=0.0,
    )
    assert point.result.residual_norm < 1e-2
    cf_point = min(
        optimized_complex_sweep,
        key=lambda p: abs(float(p["t_hot"]) - t_check),
    )
    assert float(cf_point["frequency_imag"]) < 0.0


def test_frequency_increases_with_temperature(
    symmetric_real_sweep: list[EngineSweepPoint],
) -> None:
    """Real-frequency resonance trend should increase with T_hot."""
    frequencies = np.array([point.result.frequency for point in symmetric_real_sweep])
    assert np.all(np.diff(frequencies) > 0.0)


def test_power_profile_is_physically_reasonable(
    optimized_config: StandingWaveEngineConfig,
) -> None:
    """Optimized benchmark profile should be finite and bounded."""
    point = solve_standing_wave_engine_complex_frequency_with_profiles(
        optimized_config,
        t_hot=650.0,
    )
    power = np.asarray(point["profiles"]["acoustic_power"])
    assert np.all(np.isfinite(power))
    assert np.max(np.abs(power)) < 1e4

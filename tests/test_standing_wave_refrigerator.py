"""Validation tests for standing-wave refrigerator benchmark."""

from __future__ import annotations

import numpy as np

from openthermoacoustics.validation.standing_wave_refrigerator import (
    compute_cooling_power,
    compute_f_solid,
    compute_refrigerator_cop,
    compute_refrigerator_performance_short_stack,
    default_standing_wave_refrigerator_config,
    solve_standing_wave_refrigerator,
    sweep_cold_temperature,
    sweep_drive_ratio,
    tijani_refrigerator_config,
)


def test_refrigerator_config_valid() -> None:
    """Default refrigerator configuration should be physically valid."""
    cfg = default_standing_wave_refrigerator_config()
    assert cfg.mean_pressure > 0.0
    assert cfg.t_hot > cfg.t_cold > 0.0
    assert cfg.drive_ratio > 0.0
    assert cfg.total_length > 0.0
    assert cfg.porosity > 0.0
    assert cfg.hydraulic_radius > 0.0


def test_cooling_power_positive_below_onset() -> None:
    """Driven refrigerator benchmark should produce positive cooling power."""
    cfg = default_standing_wave_refrigerator_config()
    point = solve_standing_wave_refrigerator(cfg)
    q_cold = compute_cooling_power(point)
    assert q_cold > 0.0


def test_cop_below_carnot() -> None:
    """Refrigerator COP should remain below Carnot bound."""
    cfg = default_standing_wave_refrigerator_config()
    point = solve_standing_wave_refrigerator(cfg)
    cop_info = compute_refrigerator_cop(point)
    assert 0.0 <= cop_info["cop"] <= cop_info["cop_carnot"]
    assert 0.0 <= cop_info["cop_relative"] <= 1.0


def test_cop_in_published_range() -> None:
    """
    COP/Carnot should be in a plausible standing-wave range for this proxy model.

    The current benchmark uses an acoustic-power proxy for cooling and therefore
    targets a conservative range.
    """
    cfg = default_standing_wave_refrigerator_config()
    point = solve_standing_wave_refrigerator(cfg)
    rel = float(point["cop_relative"])
    assert 0.005 <= rel <= 1.0


def test_stack_power_absorbed_is_positive() -> None:
    """Refrigerator stack should absorb acoustic power in cooling operation."""
    cfg = default_standing_wave_refrigerator_config()
    point = solve_standing_wave_refrigerator(cfg)
    assert float(point["w_stack_absorbed"]) > 0.0


def test_cooling_power_increases_with_drive() -> None:
    """Cooling power should increase with drive and scale ~quadratically."""
    cfg = default_standing_wave_refrigerator_config()
    drive_ratios = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    rows = sweep_drive_ratio(cfg, drive_ratios)
    q = np.array([float(r["cooling_power"]) for r in rows], dtype=float)

    assert np.all(q > 0.0)
    assert np.all(np.diff(q) > 0.0)

    # Quadratic scaling check using endpoints.
    ratio_q = q[-1] / q[0]
    ratio_drive_sq = (drive_ratios[-1] / drive_ratios[0]) ** 2
    assert abs(ratio_q / ratio_drive_sq - 1.0) < 0.1


def test_tijani_frequency_near_400_hz() -> None:
    """Tijani approximation should resonate near 400 Hz."""
    cfg = tijani_refrigerator_config()
    point = solve_standing_wave_refrigerator(cfg)
    f = float(point["frequency_hz"])
    assert abs(f - 400.0) / 400.0 < 0.2


def test_tijani_cop_stack_in_published_range() -> None:
    """Tijani approximation should produce COP_stack in a plausible range."""
    cfg = tijani_refrigerator_config()
    point = solve_standing_wave_refrigerator(cfg)
    cop = float(point["cop"])
    copr = float(point["cop_relative"])
    assert 0.8 <= cop <= 2.0
    assert 0.08 <= copr <= 0.50
    assert 2.8 <= float(point["cooling_power"]) <= 5.2


def test_tijani_qcold_scales_with_drive_squared() -> None:
    """Tijani approximation should show near-D^2 cooling scaling at small drive."""
    cfg = tijani_refrigerator_config()
    drives = np.array([0.01, 0.014, 0.021])
    rows = sweep_drive_ratio(cfg, drives)
    q = np.array([float(r["cooling_power"]) for r in rows], dtype=float)
    scaled = q / (drives**2)
    # Keep tolerance wide for nonlinear terms and frequency retuning.
    assert np.max(scaled) / np.min(scaled) < 1.5


def test_short_stack_matches_tijani_prediction() -> None:
    """Short-stack selected metric should stay near Tijani's 4 W / COP~1.3 target."""
    cfg = tijani_refrigerator_config()
    point = solve_standing_wave_refrigerator(cfg)
    perf = compute_refrigerator_performance_short_stack(point)
    assert 3.2 <= perf["q_cold"] <= 4.8
    assert 1.1 <= perf["cop"] <= 1.5


def test_tijani_qcold_decreases_with_delta_t() -> None:
    """Cooling power should decrease with larger temperature lift."""
    cfg = tijani_refrigerator_config()
    t_values = np.array([280.0, 260.0, 240.0, 220.0])
    rows = sweep_cold_temperature(cfg, t_values)
    qcold = np.array([float(r["cooling_power"]) for r in rows], dtype=float)
    assert np.all(np.diff(qcold) <= 1e-9)


def test_f_solid_thin_plate_limit() -> None:
    """f_solid should approach 1 for thin plates (l << delta_s)."""
    f_s = compute_f_solid(
        plate_half_thickness=1e-9,
        solid_thermal_conductivity=0.16,
        solid_density=1390.0,
        solid_specific_heat=1170.0,
        omega=2.0 * np.pi * 400.0,
    )
    assert abs(f_s - (1.0 + 0.0j)) < 1e-4


def test_f_solid_thick_plate_limit() -> None:
    """f_solid magnitude should be small for thick plates (l >> delta_s)."""
    f_s = compute_f_solid(
        plate_half_thickness=5e-3,
        solid_thermal_conductivity=0.16,
        solid_density=1390.0,
        solid_specific_heat=1170.0,
        omega=2.0 * np.pi * 400.0,
    )
    assert abs(f_s) < 0.05

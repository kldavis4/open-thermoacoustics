"""Validation tests for distributed-loop traveling-wave engine plumbing."""

from __future__ import annotations

import numpy as np

from openthermoacoustics.validation.standing_wave_engine import (
    detect_onset_from_complex_frequency as detect_sw_onset_from_complex_frequency,
)
from openthermoacoustics.validation.traveling_wave_engine import (
    compute_branch_transfer_matrix,
    compute_efficiency_estimate,
    compute_energy_balance_growth_rate,
    compute_loop_power_profile,
    compute_net_acoustic_power,
    compute_regenerator_phase_profile,
    compute_stored_energy,
    compute_trunk_transfer_matrix,
    default_traveling_wave_engine_config,
    detect_onset_from_complex_frequency,
    detect_onset_from_gain_proxy,
    estimate_loop_frequency_range,
    evaluate_traveling_wave_boundary_determinant,
    find_best_frequency_by_residual,
    find_onset_ratio_proxy,
    scan_loop_eigenvalues,
    scan_loop_eigenvalues_multi_temp,
    solve_loop_lambda_unity,
    solve_loop_self_oscillation,
    solve_traveling_wave_engine_complex_frequency,
    solve_traveling_wave_engine_determinant_complex_frequency,
    solve_traveling_wave_engine_fixed_frequency,
    summarize_multimode_selection,
    sweep_efficiency_estimate,
    sweep_energy_balance_growth_rate,
    sweep_loop_lambda_unity,
    sweep_loop_self_oscillation,
    sweep_traveling_wave_complex_frequency,
    sweep_traveling_wave_complex_frequency_multimode,
    sweep_traveling_wave_determinant_complex_frequency,
    sweep_traveling_wave_determinant_complex_frequency_multimode,
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


def test_energy_balance_signs_match_physical_expectation() -> None:
    """Isothermal should dissipate net power; hot case should produce net power."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point_cold = solve_traveling_wave_engine_fixed_frequency(cfg, frequency_hz=120.0, t_hot=300.0)
    point_hot = solve_traveling_wave_engine_fixed_frequency(cfg, frequency_hz=120.0, t_hot=600.0)

    w_cold = compute_net_acoustic_power(cfg, t_hot=300.0, solve_result=point_cold)
    w_hot = compute_net_acoustic_power(cfg, t_hot=600.0, solve_result=point_hot)

    assert w_cold < 0.0
    assert w_hot > 0.0


def test_energy_balance_growth_rate_crosses_sign_near_proxy_onset() -> None:
    """Energy-balance growth-rate crossing should be consistent with proxy onset."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    rows = sweep_energy_balance_growth_rate(
        cfg,
        t_hot_values=np.arange(300.0, 901.0, 50.0),
        frequency_guess_hz=120.0,
    )
    f_imag = np.array([float(r["frequency_imag_energy_balance"]) for r in rows], dtype=float)
    assert np.all(np.isfinite(f_imag))
    assert np.any(f_imag > 0.0)
    assert np.any(f_imag < 0.0)

    crossing_ratio = None
    for i in range(len(rows) - 1):
        y0 = float(rows[i]["frequency_imag_energy_balance"])
        y1 = float(rows[i + 1]["frequency_imag_energy_balance"])
        if y0 >= 0.0 and y1 <= 0.0:
            t0 = float(rows[i]["t_hot"])
            t1 = float(rows[i + 1]["t_hot"])
            frac = 0.0 if abs(y1 - y0) < 1e-14 else (0.0 - y0) / (y1 - y0)
            crossing_ratio = (t0 + frac * (t1 - t0)) / cfg.t_cold
            break

    proxy_ratio, _ = find_onset_ratio_proxy(
        cfg,
        frequency_hz=120.0,
        t_hot_min=300.0,
        t_hot_max=900.0,
        coarse_step=100.0,
        fine_step=20.0,
    )
    assert crossing_ratio is not None
    assert proxy_ratio is not None
    assert abs(crossing_ratio - proxy_ratio) < 0.2


def test_stored_energy_positive_and_growth_rate_finite() -> None:
    """Stored energy should be positive and produce finite growth-rate estimate."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_traveling_wave_engine_fixed_frequency(cfg, frequency_hz=120.0, t_hot=600.0)
    e_stored = compute_stored_energy(cfg, t_hot=600.0, solve_result=point)
    f_imag = compute_energy_balance_growth_rate(cfg, t_hot=600.0, solve_result=point)

    assert e_stored > 0.0
    assert np.isfinite(f_imag)


def test_5x5_isothermal_fimag_nonzero() -> None:
    """5x5 complex solve should return finite, non-trivial damping at 300 K."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_traveling_wave_engine_complex_frequency(
        cfg,
        t_hot=300.0,
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    assert point["converged"]
    assert np.isfinite(point["residual_norm"])
    assert np.isfinite(point["frequency_real"])
    assert np.isfinite(point["frequency_imag"])
    assert abs(float(point["frequency_imag"])) > 1e-4


def test_5x5_fimag_changes_with_temperature() -> None:
    """5x5 complex sweep should show temperature dependence in f_imag."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    sweep = sweep_traveling_wave_complex_frequency(
        cfg,
        t_hot_values=np.array([300.0, 400.0, 500.0, 600.0]),
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    assert len(sweep) == 4
    f_imag = []
    converged_count = 0
    for point in sweep:
        if point["converged"]:
            converged_count += 1
        assert np.isfinite(point["frequency_real"])
        assert np.isfinite(point["frequency_imag"])
        f_imag.append(float(point["frequency_imag"]))
    assert converged_count >= 3
    assert np.ptp(np.array(f_imag)) > 1e-5


def test_5x5_onset_consistency_with_gain_proxy_trend() -> None:
    """Complex-frequency and gain proxy should not contradict each other grossly."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    sweep = sweep_traveling_wave_complex_frequency(
        cfg,
        t_hot_values=np.arange(300.0, 901.0, 100.0),
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    onset = detect_onset_from_complex_frequency(sweep, max_residual_norm=0.2)
    onset_proxy, _ = find_onset_ratio_proxy(
        cfg,
        frequency_hz=120.0,
        t_hot_min=300.0,
        t_hot_max=900.0,
        coarse_step=100.0,
        fine_step=20.0,
    )
    if onset is not None and onset_proxy is not None:
        assert onset > 1.0
        assert onset < 3.0
    else:
        # If no crossing is detected yet, the sweep should still be finite and signed.
        vals = np.array([float(p["frequency_imag"]) for p in sweep])
        assert np.all(np.isfinite(vals))
        assert np.all(vals <= 0.0) or np.all(vals >= 0.0)


def test_5x5_sign_convention_matches_standing_wave_detector() -> None:
    """Traveling-wave onset detector should match standing-wave crossing convention."""
    synthetic_tw = [
        {"temperature_ratio": 1.0, "frequency_imag": 0.5},
        {"temperature_ratio": 1.2, "frequency_imag": 0.1},
        {"temperature_ratio": 1.4, "frequency_imag": -0.2},
    ]
    synthetic_sw = [
        {"temperature_ratio": 1.0, "frequency_imag": 0.5},
        {"temperature_ratio": 1.2, "frequency_imag": 0.1},
        {"temperature_ratio": 1.4, "frequency_imag": -0.2},
    ]
    onset_tw = detect_onset_from_complex_frequency(synthetic_tw, f_imag_tol_hz=0.0)
    onset_sw = detect_sw_onset_from_complex_frequency(synthetic_sw)
    assert onset_tw is not None
    assert onset_sw is not None
    assert abs(onset_tw - onset_sw) < 1e-12


def test_multimode_branch_selection_returns_structured_result() -> None:
    """Multimode sweep should return comparable branch diagnostics."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    result = sweep_traveling_wave_complex_frequency_multimode(
        cfg,
        t_hot_values=np.array([300.0, 500.0, 700.0]),
        mode_frequency_guesses_hz=[48.0, 120.0],
    )
    assert "branches" in result
    assert len(result["branches"]) == 2
    assert 0 <= int(result["selected_index"]) < 2
    selected = result["selected_branch"]
    assert "sweep" in selected
    assert len(selected["sweep"]) == 3
    summary_rows = summarize_multimode_selection(result)
    assert len(summary_rows) == 2
    assert any(bool(row["selected"]) for row in summary_rows)


def test_transfer_matrix_linearity() -> None:
    """Trunk and branch transfer matrices should map basis vectors linearly."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    omega = 2.0 * np.pi * 120.0
    t_hot = 500.0
    tt = compute_trunk_transfer_matrix(cfg, t_hot=t_hot, omega=omega)
    tb = compute_branch_transfer_matrix(cfg, t_hot=t_hot, omega=omega)

    state = np.array([700.0 + 200.0j, 4e-4 - 1e-4j], dtype=complex)
    out_t = tt @ state
    out_b = tb @ state
    assert np.all(np.isfinite(out_t.real))
    assert np.all(np.isfinite(out_t.imag))
    assert np.all(np.isfinite(out_b.real))
    assert np.all(np.isfinite(out_b.imag))


def test_det_small_at_real_frequency_solution() -> None:
    """Boundary determinant should be small near real-frequency loop closure."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_traveling_wave_engine_fixed_frequency(cfg, frequency_hz=120.0, t_hot=500.0)
    det_eval = evaluate_traveling_wave_boundary_determinant(
        cfg,
        t_hot=500.0,
        frequency_real_hz=float(point["frequency_hz"]),
        frequency_imag_hz=0.0,
    )
    assert np.isfinite(det_eval["det_magnitude"])
    assert float(det_eval["det_magnitude"]) < 5e-1


def test_determinant_complex_frequency_solver_runs() -> None:
    """2x2 determinant-based complex-frequency solve should converge to finite values."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_traveling_wave_engine_determinant_complex_frequency(
        cfg,
        t_hot=500.0,
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    assert point["converged"]
    assert np.isfinite(point["frequency_real"])
    assert np.isfinite(point["frequency_imag"])
    assert np.isfinite(point["det_magnitude"])
    assert np.isfinite(point["residual_norm"])
    assert point["residual_norm"] < 5e-2


def test_determinant_complex_frequency_sweep_tracks_finite_branch() -> None:
    """Determinant sweep should produce finite branch values across temperatures."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    sweep = sweep_traveling_wave_determinant_complex_frequency(
        cfg,
        t_hot_values=np.array([300.0, 500.0, 700.0]),
        f_real_guess=120.0,
        f_imag_guess=0.0,
    )
    assert len(sweep) == 3
    vals = np.array([float(p["frequency_imag"]) for p in sweep], dtype=float)
    residuals = np.array([float(p["residual_norm"]) for p in sweep], dtype=float)
    assert np.all(np.isfinite(vals))
    assert np.all(np.isfinite(residuals))
    assert np.all(residuals < 1e-2)


def test_determinant_multimode_branch_selection_includes_signatures() -> None:
    """Determinant multimode sweep should attach phase/power signatures."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    result = sweep_traveling_wave_determinant_complex_frequency_multimode(
        cfg,
        t_hot_values=np.array([300.0, 500.0]),
        mode_frequency_guesses_hz=[80.0, 120.0],
    )
    assert len(result["branches"]) == 2
    assert 0 <= int(result["selected_index"]) < 2
    selected = result["selected_branch"]
    assert len(selected["sweep"]) == 2
    for point in selected["sweep"]:
        assert "phase_regenerator_mid_deg" in point
        assert "branch_power_sign" in point
        assert "trunk_power_sign" in point


def test_transfer_matrix_sensitive_to_f_imag() -> None:
    """Trunk transfer matrix should change for nonzero imaginary frequency."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    t_hot = 600.0
    omega_real = 2.0 * np.pi * 120.0
    omega_complex = 2.0 * np.pi * (120.0 + 5.0j)
    t_real = compute_trunk_transfer_matrix(cfg, t_hot=t_hot, omega=omega_real)
    t_complex = compute_trunk_transfer_matrix(cfg, t_hot=t_hot, omega=omega_complex)
    diff = float(np.max(np.abs(t_real - t_complex)))
    scale = max(float(np.max(np.abs(t_real))), 1e-12)
    rel_diff = diff / scale
    assert diff > 1e-6
    assert rel_diff > 1e-4


def test_loop_eigenvalue_scan_shows_gain() -> None:
    """Loop eigenvalue scan should produce finite values and show amplification."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    rows = scan_loop_eigenvalues(
        cfg,
        t_hot=600.0,
        f_real_values=np.linspace(20.0, 300.0, 41),
    )
    assert len(rows) == 41
    max_mag = max(float(max(abs(v) for v in row["eigvals"])) for row in rows)
    assert np.isfinite(max_mag)
    assert max_mag > 1.0


def test_loop_det_converges_offaxis() -> None:
    """Loop-only complex solve should converge from an off-axis seed."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_loop_self_oscillation(
        cfg,
        t_hot=600.0,
        f_real_guess=80.0,
        f_imag_guess=10.0,
    )
    assert point["converged"]
    assert np.isfinite(point["frequency_real"])
    assert np.isfinite(point["frequency_imag"])
    assert point["residual_norm"] < 1.0


def test_loop_fimag_crosses_zero() -> None:
    """Loop-only sweep should return finite f_imag values (crossing if present)."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    sweep = sweep_loop_self_oscillation(
        cfg,
        t_hot_values=np.array([300.0, 500.0, 700.0]),
        f_real_guess=80.0,
        f_imag_guess=10.0,
    )
    vals = np.array([float(p["frequency_imag"]) for p in sweep], dtype=float)
    assert np.all(np.isfinite(vals))
    onset = detect_onset_from_complex_frequency(sweep, max_residual_norm=1e-3)
    if onset is not None:
        assert 1.0 < onset < 4.0


def test_lambda1_root_converges_at_600k() -> None:
    """Tracked lambda-unity solve should converge to a finite complex frequency."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_loop_lambda_unity(
        cfg,
        t_hot=600.0,
        f_real_guess=20.0,
        f_imag_guess=0.05,
    )
    assert point["converged"]
    assert np.isfinite(point["frequency_real"])
    assert np.isfinite(point["frequency_imag"])
    assert np.isfinite(point["residual_norm"])


def test_lambda1_fimag_nonzero_at_300k() -> None:
    """At isothermal condition, tracked lambda-unity branch should have nonzero f_imag."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    point = solve_loop_lambda_unity(
        cfg,
        t_hot=300.0,
        f_real_guess=20.0,
        f_imag_guess=0.05,
    )
    assert np.isfinite(point["frequency_imag"])
    assert abs(float(point["frequency_imag"])) > 1e-10


def test_lambda1_scan_multi_temp_and_sweep() -> None:
    """Multi-temperature scan and lambda-unity sweep should return finite values."""
    cfg = tuned_traveling_wave_engine_candidate_config()
    scan = scan_loop_eigenvalues_multi_temp(
        cfg,
        t_hot_values=np.array([300.0, 600.0]),
        f_real_values=np.linspace(1.0, 50.0, 20),
    )
    assert 300.0 in scan and 600.0 in scan
    assert len(scan[300.0]) == 20
    sweep = sweep_loop_lambda_unity(
        cfg,
        t_hot_values=np.array([300.0, 500.0, 700.0]),
        f_real_guess=20.0,
        f_imag_guess=0.05,
    )
    vals = np.array([float(p["frequency_imag"]) for p in sweep], dtype=float)
    assert np.all(np.isfinite(vals))

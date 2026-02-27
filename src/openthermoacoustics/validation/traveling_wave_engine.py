"""Traveling-wave engine helpers using distributed loop propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares, root

from openthermoacoustics import gas, segments
from openthermoacoustics.geometry import WireScreen
from openthermoacoustics.solver import (
    DistributedLoopPropagator,
    TBranchLoopResult,
    TBranchLoopSolver,
)


@dataclass(frozen=True)
class TravelingWaveEngineConfig:
    """Configuration for a simplified Backhaus-Swift-style traveling-wave engine."""

    mean_pressure: float = 3.0e6
    t_cold: float = 300.0
    t_hot: float = 600.0

    # Main flow path (branch loop)
    ahx1_length: float = 0.025
    regenerator_length: float = 0.075
    hhx_length: float = 0.025
    tbt_length: float = 0.25
    ahx2_length: float = 0.025
    feedback_length: float = 0.50

    main_radius: float = 0.044
    feedback_radius: float = 0.022
    resonator_length: float = 1.0
    resonator_radius: float = 0.044

    # Porous media
    ahx_porosity: float = 0.52
    ahx_hydraulic_radius: float = 0.40e-3
    regenerator_porosity: float = 0.72
    regenerator_hydraulic_radius: float = 0.04e-3
    hhx_porosity: float = 0.50
    hhx_hydraulic_radius: float = 0.40e-3

    # Solver controls
    p1_input: complex = 2000.0 + 0.0j
    t_m_start: float = 300.0
    n_points_per_segment: int = 120
    u1_mag_guess: float = 1e-3
    u1_phase_guess: float = 0.0
    zb_real_guess: float = 1e5
    zb_imag_guess: float = -1e5
    tol: float = 1e-8
    maxiter: int = 100

    @property
    def main_area(self) -> float:
        return float(np.pi * self.main_radius**2)


def default_traveling_wave_engine_config() -> TravelingWaveEngineConfig:
    """Default traveling-wave validation configuration."""
    return TravelingWaveEngineConfig()


def tuned_traveling_wave_engine_candidate_config() -> TravelingWaveEngineConfig:
    """
    Return a tuned candidate from focused loop-parameter sweeps.

    This candidate is selected to improve gain-proxy behavior relative to the
    baseline default distributed-loop configuration.
    """
    return TravelingWaveEngineConfig(
        mean_pressure=4.0e6,
        resonator_length=0.8,
        feedback_radius=0.03,
        feedback_length=0.5,
        tbt_length=0.25,
        regenerator_hydraulic_radius=0.12e-3,
    )


def build_traveling_wave_paths(
    config: TravelingWaveEngineConfig,
    *,
    t_hot: float | None = None,
) -> tuple[list[segments.Segment], list[segments.Segment]]:
    """
    Build trunk and loop-branch segment lists for distributed propagation.

    Returns
    -------
    tuple[list[Segment], list[Segment]]
        (trunk_segments, branch_segments)
    """
    hot_temp = config.t_hot if t_hot is None else t_hot

    ahx_geometry = WireScreen(
        porosity=config.ahx_porosity,
        hydraulic_radius=config.ahx_hydraulic_radius,
    )
    regen_geometry = WireScreen(
        porosity=config.regenerator_porosity,
        hydraulic_radius=config.regenerator_hydraulic_radius,
    )
    hhx_geometry = WireScreen(
        porosity=config.hhx_porosity,
        hydraulic_radius=config.hhx_hydraulic_radius,
    )

    branch_segments: list[segments.Segment] = [
        segments.HeatExchanger(
            name="ahx1",
            length=config.ahx1_length,
            porosity=config.ahx_porosity,
            hydraulic_radius=config.ahx_hydraulic_radius,
            temperature=config.t_cold,
            area=config.main_area,
            geometry=ahx_geometry,
        ),
        segments.Stack(
            name="regenerator",
            length=config.regenerator_length,
            porosity=config.regenerator_porosity,
            hydraulic_radius=config.regenerator_hydraulic_radius,
            area=config.main_area,
            geometry=regen_geometry,
            T_cold=config.t_cold,
            T_hot=hot_temp,
        ),
        segments.HeatExchanger(
            name="hhx",
            length=config.hhx_length,
            porosity=config.hhx_porosity,
            hydraulic_radius=config.hhx_hydraulic_radius,
            temperature=hot_temp,
            area=config.main_area,
            geometry=hhx_geometry,
        ),
        segments.Duct(name="tbt", length=config.tbt_length, radius=config.main_radius),
        segments.HeatExchanger(
            name="ahx2",
            length=config.ahx2_length,
            porosity=config.ahx_porosity,
            hydraulic_radius=config.ahx_hydraulic_radius,
            temperature=config.t_cold,
            area=config.main_area,
            geometry=ahx_geometry,
        ),
        segments.Duct(
            name="feedback",
            length=config.feedback_length,
            radius=config.feedback_radius,
        ),
    ]

    trunk_segments: list[segments.Segment] = [
        segments.Duct(
            name="resonator",
            length=config.resonator_length,
            radius=config.resonator_radius,
        ),
        segments.HardEnd(name="hard_end"),
    ]

    return trunk_segments, branch_segments


def _phase_at_regenerator_midpoint(
    branch_result: Any,
) -> float | None:
    """Return phase(p1)-phase(U1) in degrees at regenerator midpoint."""
    if branch_result is None:
        return None

    reg_section = None
    for section in branch_result.sections:
        if section.segment_name == "regenerator":
            reg_section = section
            break
    if reg_section is None or len(reg_section.p1) == 0:
        return None

    i_mid = len(reg_section.p1) // 2
    p1_mid = reg_section.p1[i_mid]
    u1_mid = reg_section.U1[i_mid]
    phase = np.degrees(np.angle(p1_mid) - np.angle(u1_mid))
    # Wrap to [-180, 180]
    phase = (phase + 180.0) % 360.0 - 180.0
    return float(phase)


def _section_power_deltas(chain_result: Any) -> dict[str, float]:
    """Compute per-section acoustic power deltas (out - in)."""
    if chain_result is None:
        return {}
    deltas: dict[str, float] = {}
    for section in chain_result.sections:
        if len(section.acoustic_power) == 0:
            continue
        label = section.segment_name or section.segment_type
        deltas[label] = float(section.acoustic_power[-1] - section.acoustic_power[0])
    return deltas


def _section_boundary_power(chain_result: Any) -> list[dict[str, float | str]]:
    """Return absolute boundary powers for each section."""
    if chain_result is None:
        return []
    rows: list[dict[str, float | str]] = []
    for section in chain_result.sections:
        if len(section.acoustic_power) == 0:
            continue
        label = section.segment_name or section.segment_type
        rows.append(
            {
                "segment": label,
                "x_start": float(section.x_start),
                "x_end": float(section.x_end),
                "w_in": float(section.acoustic_power[0]),
                "w_out": float(section.acoustic_power[-1]),
                "delta_w": float(section.acoustic_power[-1] - section.acoustic_power[0]),
            }
        )
    return rows


def _gain_proxy(
    branch_deltas: dict[str, float],
    trunk_deltas: dict[str, float],
) -> tuple[float | None, float]:
    """
    Compute a simple onset proxy: regenerator gain minus other dissipation.

    Returns
    -------
    tuple[float | None, float]
        (regenerator_delta, net_gain_proxy)
    """
    regen_delta = branch_deltas.get("regenerator")
    if regen_delta is None:
        return None, float("nan")
    losses = 0.0
    for name, value in branch_deltas.items():
        if name == "regenerator":
            continue
        if value < 0.0:
            losses += abs(value)
    for value in trunk_deltas.values():
        if value < 0.0:
            losses += abs(value)
    return float(regen_delta), float(regen_delta - losses)


def _phase_difference_deg(a_deg: float, b_deg: float) -> float:
    """Return wrapped phase difference |a-b| in degrees on [-180, 180]."""
    diff = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return float(abs(diff))


def _dominant_power_sign(rows: list[dict[str, float | str]]) -> int | None:
    """Infer dominant acoustic power-flow sign from boundary rows."""
    signs: list[int] = []
    for row in rows:
        for key in ("w_in", "w_out"):
            val = float(row[key])
            if abs(val) > 1e-10:
                signs.append(int(np.sign(val)))
    if not signs:
        return None
    pos = signs.count(1)
    neg = signs.count(-1)
    if pos == neg:
        return None
    return 1 if pos > neg else -1


def solve_traveling_wave_engine_fixed_frequency(
    config: TravelingWaveEngineConfig,
    *,
    frequency_hz: float,
    t_hot: float | None = None,
    u1_mag_guess: float | None = None,
    u1_phase_guess: float | None = None,
    zb_real_guess: float | None = None,
    zb_imag_guess: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Solve loop closure at a fixed frequency using distributed segments."""
    helium = gas.Helium(mean_pressure=config.mean_pressure)
    omega = 2.0 * np.pi * frequency_hz
    trunk_segments, branch_segments = build_traveling_wave_paths(config, t_hot=t_hot)
    propagator = DistributedLoopPropagator(
        trunk_segments=trunk_segments,
        branch_segments=branch_segments,
        gas=helium,
        omega=omega,
        t_m_start=config.t_m_start,
        n_points_per_segment=config.n_points_per_segment,
    )
    solver = TBranchLoopSolver(
        propagate_func=propagator,
        gas=helium,
        omega=omega,
        T_m=config.t_m_start,
        p1_input=config.p1_input,
    )

    result = solver.solve(
        U1_mag_guess=config.u1_mag_guess if u1_mag_guess is None else u1_mag_guess,
        U1_phase_guess=config.u1_phase_guess if u1_phase_guess is None else u1_phase_guess,
        Zb_real_guess=config.zb_real_guess if zb_real_guess is None else zb_real_guess,
        Zb_imag_guess=config.zb_imag_guess if zb_imag_guess is None else zb_imag_guess,
        tol=config.tol,
        maxiter=config.maxiter,
        verbose=verbose,
    )

    profiles = propagator.latest_profiles()
    branch_deltas = _section_power_deltas(profiles["branch"])
    trunk_deltas = _section_power_deltas(profiles["trunk"])
    regen_delta, net_gain_proxy = _gain_proxy(branch_deltas, trunk_deltas)
    phase_mid = _phase_at_regenerator_midpoint(profiles["branch"])
    return {
        "frequency_hz": float(frequency_hz),
        "omega": omega,
        "result": result,
        "profiles": profiles,
        "branch_power_deltas": branch_deltas,
        "trunk_power_deltas": trunk_deltas,
        "regenerator_power_delta": regen_delta,
        "net_gain_proxy": net_gain_proxy,
        "phase_regenerator_mid_deg": phase_mid,
        "branch_power_profile": _section_boundary_power(profiles["branch"]),
        "trunk_power_profile": _section_boundary_power(profiles["trunk"]),
        "t_hot": float(config.t_hot if t_hot is None else t_hot),
    }


def sweep_traveling_wave_frequency(
    config: TravelingWaveEngineConfig,
    frequencies_hz: np.ndarray,
    *,
    t_hot: float | None = None,
) -> list[dict[str, Any]]:
    """Run fixed-frequency solves across a frequency range with continuation."""
    points: list[dict[str, Any]] = []
    u1_mag = config.u1_mag_guess
    u1_phase = config.u1_phase_guess
    zb_real = config.zb_real_guess
    zb_imag = config.zb_imag_guess

    for f in frequencies_hz:
        point = solve_traveling_wave_engine_fixed_frequency(
            config,
            frequency_hz=float(f),
            t_hot=t_hot,
            u1_mag_guess=u1_mag,
            u1_phase_guess=u1_phase,
            zb_real_guess=zb_real,
            zb_imag_guess=zb_imag,
            verbose=False,
        )
        points.append(point)
        solved: TBranchLoopResult = point["result"]
        u1_mag = solved.U1_magnitude
        u1_phase = solved.U1_phase
        zb_real = solved.Zb_real
        zb_imag = solved.Zb_imag

    return points


def find_best_frequency_by_residual(
    sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the sweep point with minimum residual norm."""
    if not sweep:
        raise ValueError("Frequency sweep is empty.")
    return min(sweep, key=lambda p: float(p["result"].residual_norm))


def sweep_traveling_wave_temperature(
    config: TravelingWaveEngineConfig,
    *,
    frequency_hz: float,
    t_hot_values: np.ndarray,
) -> list[dict[str, Any]]:
    """Sweep hot-side temperature at fixed frequency with continuation."""
    points: list[dict[str, Any]] = []
    u1_mag = config.u1_mag_guess
    u1_phase = config.u1_phase_guess
    zb_real = config.zb_real_guess
    zb_imag = config.zb_imag_guess

    for t_hot in t_hot_values:
        point = solve_traveling_wave_engine_fixed_frequency(
            config,
            frequency_hz=frequency_hz,
            t_hot=float(t_hot),
            u1_mag_guess=u1_mag,
            u1_phase_guess=u1_phase,
            zb_real_guess=zb_real,
            zb_imag_guess=zb_imag,
            verbose=False,
        )
        points.append(point)
        solved: TBranchLoopResult = point["result"]
        u1_mag = solved.U1_magnitude
        u1_phase = solved.U1_phase
        zb_real = solved.Zb_real
        zb_imag = solved.Zb_imag

    return points


def detect_onset_from_gain_proxy(
    sweep: list[dict[str, Any]],
    *,
    t_cold: float,
) -> float | None:
    """
    Detect approximate onset from net-gain proxy crossing zero.

    This is a first-pass proxy for looped topologies until complex-frequency
    growth-rate solving is added for the traveling-wave loop.
    """
    for i in range(len(sweep) - 1):
        g0 = float(sweep[i]["net_gain_proxy"])
        g1 = float(sweep[i + 1]["net_gain_proxy"])
        if not (np.isfinite(g0) and np.isfinite(g1)):
            continue
        if g0 <= 0.0 < g1:
            t0 = float(sweep[i]["t_hot"])
            t1 = float(sweep[i + 1]["t_hot"])
            if abs(g1 - g0) < 1e-15:
                return t1 / t_cold
            t_cross = t0 + (t1 - t0) * ((0.0 - g0) / (g1 - g0))
            return float(t_cross / t_cold)
    return None


def solve_traveling_wave_engine_complex_frequency(
    config: TravelingWaveEngineConfig,
    *,
    t_hot: float,
    f_real_guess: float,
    f_imag_guess: float = 0.0,
    u1_mag_guess: float | None = None,
    u1_phase_fixed_rad: float | None = None,
    zb_real_guess: float | None = None,
    zb_imag_guess: float | None = None,
    p_norm: float | None = None,
    f_real_span_hz: float | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Solve 5x5 complex-frequency loop closure with amplitude normalization.

    Unknowns are `[f_real, f_imag, |U1_input|, Re(Zb), Im(Zb)]` and targets are:
    `Re/Im(p_mismatch)=0`, `Re/Im(U1_hardend)=0`, and
    `|p1_trunk_end| - p_norm = 0`.

    Notes
    -----
    Phase gauge convention: `p1_input = p_norm + 0j`, and `phase(U1_input)` is
    fixed from a converged real-frequency reference solve at the same
    temperature. The 5th residual pins amplitude scale at trunk-end pressure.
    """
    real_reference = None
    if u1_phase_fixed_rad is None or u1_mag_guess is None:
        real_reference = solve_traveling_wave_engine_fixed_frequency(
            config,
            frequency_hz=f_real_guess,
            t_hot=t_hot,
            zb_real_guess=zb_real_guess,
            zb_imag_guess=zb_imag_guess,
            verbose=False,
        )
        ref_result: TBranchLoopResult = real_reference["result"]
        if u1_phase_fixed_rad is None:
            u1_phase_fixed_rad = float(np.angle(ref_result.U1_input))
        if u1_mag_guess is None:
            u1_mag_guess = float(abs(ref_result.U1_input))
        if zb_real_guess is None:
            zb_real_guess = ref_result.Zb_real
        if zb_imag_guess is None:
            zb_imag_guess = ref_result.Zb_imag

    assert u1_phase_fixed_rad is not None
    assert u1_mag_guess is not None

    if zb_real_guess is None:
        zb_real_guess = config.zb_real_guess
    if zb_imag_guess is None:
        zb_imag_guess = config.zb_imag_guess

    if p_norm is None:
        if real_reference is not None:
            p_norm_local = float(np.real(real_reference["result"].p1_union))
        else:
            p_norm_local = float(abs(config.p1_input))
    else:
        p_norm_local = float(p_norm)
    if p_norm_local <= 0.0:
        p_norm_local = 1000.0

    helium = gas.Helium(mean_pressure=config.mean_pressure)
    trunk_segments, branch_segments = build_traveling_wave_paths(config, t_hot=t_hot)
    propagator = DistributedLoopPropagator(
        trunk_segments=trunk_segments,
        branch_segments=branch_segments,
        gas=helium,
        omega=2.0 * np.pi * (f_real_guess + 1j * f_imag_guess),
        t_m_start=config.t_m_start,
        n_points_per_segment=config.n_points_per_segment,
    )

    freq_ref = max(abs(f_real_guess), 1.0)
    freq_span = (
        max(0.1 * freq_ref, 2.0)
        if f_real_span_hz is None
        else max(float(f_real_span_hz), 0.5)
    )
    imag_scale = max(abs(f_imag_guess), 1.0)
    u1_scale = max(abs(u1_mag_guess), 1e-9)
    z_scale = max(abs(zb_real_guess), abs(zb_imag_guess), 1e3)

    def residual(x: np.ndarray) -> np.ndarray:
        # Keep the solve on the continuation branch and avoid mode hopping.
        f_real = freq_ref + freq_span * np.tanh(x[0])
        f_imag = x[1] * imag_scale
        u1_mag = x[2] * u1_scale
        zb_real = x[3] * z_scale
        zb_imag = x[4] * z_scale
        u1_input = u1_mag * np.exp(1j * u1_phase_fixed_rad)
        omega = 2.0 * np.pi * (f_real + 1j * f_imag)
        propagator.omega = omega
        try:
            p1_union, u1_hardend, p_mismatch = propagator(
                p_norm_local + 0j,
                u1_input,
                complex(zb_real, zb_imag),
            )
        except Exception:
            return np.array([1e8, 1e8, 1e8, 1e8, 1e8], dtype=float)

        return np.array(
            [
                p_mismatch.real / p_norm_local,
                p_mismatch.imag / p_norm_local,
                u1_hardend.real / u1_scale,
                u1_hardend.imag / u1_scale,
                (p1_union.real - p_norm_local) / p_norm_local,
            ],
            dtype=float,
        )

    x0 = np.array(
        [
            0.0,
            f_imag_guess / imag_scale,
            u1_mag_guess / u1_scale,
            zb_real_guess / z_scale,
            zb_imag_guess / z_scale,
        ],
        dtype=float,
    )
    tol_local = config.tol if tol is None else tol
    maxiter_local = config.maxiter if maxiter is None else maxiter

    result_hybr = root(
        residual,
        x0,
        method="hybr",
        tol=tol_local,
        options={"maxfev": maxiter_local * 20},
    )
    result_lm = root(
        residual,
        x0,
        method="lm",
        tol=tol_local,
        options={"maxiter": maxiter_local * 20},
    )
    result_ls = least_squares(
        residual,
        x0,
        method="trf",
        max_nfev=maxiter_local * 40,
        xtol=tol_local,
        ftol=tol_local,
        gtol=tol_local,
    )

    candidates: list[tuple[np.ndarray, np.ndarray, bool, str, int]] = [
        (
            np.asarray(result_hybr.x, dtype=float),
            np.asarray(result_hybr.fun, dtype=float),
            bool(result_hybr.success),
            str(result_hybr.message),
            int(getattr(result_hybr, "nfev", 0)),
        ),
        (
            np.asarray(result_lm.x, dtype=float),
            np.asarray(result_lm.fun, dtype=float),
            bool(result_lm.success),
            str(result_lm.message),
            int(getattr(result_lm, "nfev", 0)),
        ),
        (
            np.asarray(result_ls.x, dtype=float),
            np.asarray(result_ls.fun, dtype=float),
            bool(result_ls.success),
            str(result_ls.message),
            int(getattr(result_ls, "nfev", 0)),
        ),
    ]
    best_x, best_fun, best_success, best_message, best_nfev = min(
        candidates,
        key=lambda c: float(np.linalg.norm(c[1])),
    )

    f_real = float(freq_ref + freq_span * np.tanh(best_x[0]))
    f_imag = float(best_x[1] * imag_scale)
    u1_mag = float(best_x[2] * u1_scale)
    zb_real = float(best_x[3] * z_scale)
    zb_imag = float(best_x[4] * z_scale)
    u1_input = u1_mag * np.exp(1j * u1_phase_fixed_rad)
    omega = 2.0 * np.pi * (f_real + 1j * f_imag)
    propagator.omega = omega

    p1_union, u1_hardend, p_mismatch = propagator(
        p_norm_local + 0j,
        u1_input,
        complex(zb_real, zb_imag),
    )
    profiles = propagator.latest_profiles()
    branch_deltas = _section_power_deltas(profiles["branch"])
    trunk_deltas = _section_power_deltas(profiles["trunk"])
    regen_delta, net_gain_proxy = _gain_proxy(branch_deltas, trunk_deltas)
    phase_mid = _phase_at_regenerator_midpoint(profiles["branch"])

    if verbose:
        print(
            "complex solve: "
            f"T_hot={t_hot:.1f} K, "
            f"f={f_real:.3f}+j{f_imag:.3f} Hz, "
            f"|U1|={u1_mag:.3e}, "
            f"residual={np.linalg.norm(best_fun):.3e}"
        )

    residual_norm = float(np.linalg.norm(best_fun))
    converged = bool(best_success) or residual_norm < 0.2

    return {
        "converged": converged,
        "method_success": bool(best_success),
        "message": str(best_message),
        "n_iterations": best_nfev,
        "residual_norm": residual_norm,
        "frequency_real": f_real,
        "frequency_imag": f_imag,
        "omega": omega,
        "u1_magnitude": u1_mag,
        "u1_phase_fixed_rad": float(u1_phase_fixed_rad),
        "zb_real": zb_real,
        "zb_imag": zb_imag,
        "zb": complex(zb_real, zb_imag),
        "p1_input": p_norm_local + 0j,
        "u1_input": u1_input,
        "p1_union": p1_union,
        "u1_hardend": u1_hardend,
        "pressure_mismatch": p_mismatch,
        "normalization_residual": float(p1_union.real - p_norm_local),
        "p_norm": p_norm_local,
        "profiles": profiles,
        "branch_power_deltas": branch_deltas,
        "trunk_power_deltas": trunk_deltas,
        "regenerator_power_delta": regen_delta,
        "net_gain_proxy": net_gain_proxy,
        "phase_regenerator_mid_deg": phase_mid,
        "branch_power_profile": _section_boundary_power(profiles["branch"]),
        "trunk_power_profile": _section_boundary_power(profiles["trunk"]),
        "temperature_ratio": float(t_hot / config.t_cold),
        "t_hot": float(t_hot),
        "u1_reference_from_real": real_reference,
    }


def sweep_traveling_wave_complex_frequency(
    config: TravelingWaveEngineConfig,
    *,
    t_hot_values: np.ndarray,
    f_real_guess: float,
    f_imag_guess: float = 0.0,
    p_norm: float | None = None,
    real_ref_search_span_hz: float = 0.0,
    real_ref_search_step_hz: float = 0.0,
    phase_relaxation_deg: tuple[float, ...] = (0.0, 5.0, -5.0, 15.0, -15.0),
    residual_accept: float = 1e-4,
    mode_lock_max_step_hz: float = 12.0,
    mode_lock_max_step_frac: float = 0.15,
    mode_lock_ref_weight: float = 0.5,
    mode_anchor_search_span_hz: float = 60.0,
    mode_anchor_search_step_hz: float = 10.0,
    mode_lock_anchor_band_hz: float = 20.0,
    mode_lock_anchor_weight: float = 1.0,
    mode_signature_phase_band_deg: float = 25.0,
    mode_signature_phase_weight: float = 0.5,
    mode_signature_power_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Sweep hot temperature using complex-frequency solve with continuation.

    Continuation strategy:
    1. At each temperature, solve the real-frequency loop first to refresh the
       velocity-reference gauge (`U1_input`).
    2. Solve complex frequency using previous `(f_real, f_imag, Zb)` as seeds.
    """
    points: list[dict[str, Any]] = []

    f_real = f_real_guess
    f_imag = f_imag_guess
    u1_mag = config.u1_mag_guess
    u1_phase = config.u1_phase_guess
    zb_real = config.zb_real_guess
    zb_imag = config.zb_imag_guess
    anchor_frequency = float(f_real_guess)
    signature_phase_deg: float | None = None
    signature_branch_sign: int | None = None
    signature_trunk_sign: int | None = None

    if mode_anchor_search_span_hz > 0.0 and mode_anchor_search_step_hz > 0.0:
        t0 = float(t_hot_values[0]) if len(t_hot_values) else float(config.t_hot)
        fmin = max(1.0, anchor_frequency - mode_anchor_search_span_hz)
        fmax = anchor_frequency + mode_anchor_search_span_hz
        freq_grid = np.arange(fmin, fmax + mode_anchor_search_step_hz, mode_anchor_search_step_hz)
        freq_sweep = sweep_traveling_wave_frequency(
            config,
            frequencies_hz=freq_grid,
            t_hot=t0,
        )
        best_seed = find_best_frequency_by_residual(freq_sweep)
        anchor_frequency = float(best_seed["frequency_hz"])
        f_real = anchor_frequency

    for t_hot in t_hot_values:
        ref_frequency = float(f_real)
        if real_ref_search_span_hz > 0.0 and real_ref_search_step_hz > 0.0:
            fmin = max(1.0, ref_frequency - real_ref_search_span_hz)
            fmax = ref_frequency + real_ref_search_span_hz
            freq_grid = np.arange(
                fmin, fmax + real_ref_search_step_hz, real_ref_search_step_hz
            )
            freq_sweep = sweep_traveling_wave_frequency(
                config,
                frequencies_hz=freq_grid,
                t_hot=float(t_hot),
            )
            best_real = find_best_frequency_by_residual(freq_sweep)
            ref_frequency = float(best_real["frequency_hz"])

        real_point = solve_traveling_wave_engine_fixed_frequency(
            config,
            frequency_hz=ref_frequency,
            t_hot=float(t_hot),
            u1_mag_guess=u1_mag,
            u1_phase_guess=u1_phase,
            zb_real_guess=zb_real,
            zb_imag_guess=zb_imag,
            verbose=False,
        )
        real_result: TBranchLoopResult = real_point["result"]
        u1_mag = real_result.U1_magnitude
        u1_phase = real_result.U1_phase

        point = None
        phase_ref = float(np.angle(real_result.U1_input))
        prev_frequency = float(f_real)
        pnorm_use = float(np.real(real_result.p1_union)) if p_norm is None else float(p_norm)
        max_step = max(
            float(mode_lock_max_step_hz),
            float(mode_lock_max_step_frac) * max(prev_frequency, 1.0),
        )
        candidate_centers = [ref_frequency, anchor_frequency]
        if abs(ref_frequency - prev_frequency) > 1e-12:
            candidate_centers.append(prev_frequency)
        scored_candidates: list[tuple[float, dict[str, Any]]] = []

        def candidate_score(candidate: dict[str, Any]) -> float:
            freq_jump = abs(float(candidate["frequency_real"]) - prev_frequency)
            jump_penalty = 10.0 * max(0.0, freq_jump - max_step)
            ref_offset = abs(float(candidate["frequency_real"]) - ref_frequency)
            ref_penalty = float(mode_lock_ref_weight) * max(0.0, ref_offset - max_step)
            anchor_offset = abs(float(candidate["frequency_real"]) - anchor_frequency)
            anchor_penalty = float(mode_lock_anchor_weight) * max(
                0.0, anchor_offset - float(mode_lock_anchor_band_hz)
            )
            phase_penalty = 0.0
            if signature_phase_deg is not None:
                phase_now = candidate.get("phase_regenerator_mid_deg")
                if phase_now is None or not np.isfinite(float(phase_now)):
                    phase_penalty = 10.0
                else:
                    phase_err = _phase_difference_deg(float(phase_now), signature_phase_deg)
                    phase_penalty = float(mode_signature_phase_weight) * max(
                        0.0, phase_err - float(mode_signature_phase_band_deg)
                    ) / 180.0

            power_penalty = 0.0
            branch_sign = _dominant_power_sign(list(candidate.get("branch_power_profile", [])))
            trunk_sign = _dominant_power_sign(list(candidate.get("trunk_power_profile", [])))
            if signature_branch_sign is not None and branch_sign is not None:
                if branch_sign != signature_branch_sign:
                    power_penalty += float(mode_signature_power_weight)
            if signature_trunk_sign is not None and trunk_sign is not None:
                if trunk_sign != signature_trunk_sign:
                    power_penalty += float(mode_signature_power_weight)

            return (
                float(candidate["residual_norm"])
                + jump_penalty
                + ref_penalty
                + anchor_penalty
                + phase_penalty
                + power_penalty
            )

        for phase_offset in phase_relaxation_deg:
            for center in candidate_centers:
                candidate = solve_traveling_wave_engine_complex_frequency(
                    config,
                    t_hot=float(t_hot),
                    f_real_guess=float(center),
                    f_imag_guess=float(f_imag),
                    u1_mag_guess=float(u1_mag),
                    u1_phase_fixed_rad=phase_ref + np.radians(phase_offset),
                    zb_real_guess=real_result.Zb_real,
                    zb_imag_guess=real_result.Zb_imag,
                    p_norm=pnorm_use,
                    f_real_span_hz=max(0.2 * center, max_step, 2.0),
                    verbose=False,
                )
                score = candidate_score(candidate)
                scored_candidates.append((score, candidate))
                if point is None or score < candidate_score(point):
                    point = candidate
            if point is not None and float(point["residual_norm"]) <= residual_accept:
                break

        assert point is not None

        # Hard mode-lock retry if best candidate still jumps too far.
        if abs(float(point["frequency_real"]) - prev_frequency) > max_step:
            locked = solve_traveling_wave_engine_complex_frequency(
                config,
                t_hot=float(t_hot),
                f_real_guess=prev_frequency,
                f_imag_guess=float(f_imag),
                u1_mag_guess=float(u1_mag),
                u1_phase_fixed_rad=phase_ref,
                zb_real_guess=real_result.Zb_real,
                zb_imag_guess=real_result.Zb_imag,
                p_norm=pnorm_use,
                f_real_span_hz=max_step,
                verbose=False,
            )
            if candidate_score(locked) <= candidate_score(point):
                point = locked

        point["real_frequency_reference"] = real_point
        if signature_phase_deg is None:
            phase_val = point.get("phase_regenerator_mid_deg")
            if phase_val is not None and np.isfinite(float(phase_val)):
                signature_phase_deg = float(phase_val)
        if signature_branch_sign is None:
            signature_branch_sign = _dominant_power_sign(
                list(point.get("branch_power_profile", []))
            )
        if signature_trunk_sign is None:
            signature_trunk_sign = _dominant_power_sign(
                list(point.get("trunk_power_profile", []))
            )
        point["mode_signature_phase_deg"] = signature_phase_deg
        point["mode_signature_branch_sign"] = signature_branch_sign
        point["mode_signature_trunk_sign"] = signature_trunk_sign
        if signature_phase_deg is not None and point.get("phase_regenerator_mid_deg") is not None:
            point["mode_phase_error_deg"] = _phase_difference_deg(
                float(point["phase_regenerator_mid_deg"]),
                signature_phase_deg,
            )
        points.append(point)

        f_real = float(point["frequency_real"])
        f_imag = float(point["frequency_imag"])
        u1_mag = float(point["u1_magnitude"])
        zb_real = float(point["zb_real"])
        zb_imag = float(point["zb_imag"])

    return points


def detect_onset_from_complex_frequency(
    sweep: list[dict[str, Any]],
    *,
    f_imag_tol_hz: float = 1e-4,
    max_residual_norm: float | None = None,
) -> float | None:
    """
    Detect onset ratio from `f_imag` sign crossing.

    Convention used: onset is where `f_imag` crosses from positive (damped) to
    non-positive (growing).
    """
    for i in range(len(sweep) - 1):
        if max_residual_norm is not None:
            r0 = float(sweep[i].get("residual_norm", np.inf))
            r1 = float(sweep[i + 1].get("residual_norm", np.inf))
            if r0 > max_residual_norm or r1 > max_residual_norm:
                continue
        fi0 = float(sweep[i]["frequency_imag"])
        fi1 = float(sweep[i + 1]["frequency_imag"])
        if fi0 > f_imag_tol_hz and fi1 < -f_imag_tol_hz:
            r0 = float(sweep[i]["temperature_ratio"])
            r1 = float(sweep[i + 1]["temperature_ratio"])
            if abs(fi1 - fi0) < 1e-15:
                return r1
            return r0 + (r1 - r0) * (fi0 / (fi0 - fi1))
    return None


def sweep_traveling_wave_complex_frequency_multimode(
    config: TravelingWaveEngineConfig,
    *,
    t_hot_values: np.ndarray,
    mode_frequency_guesses_hz: list[float],
    f_imag_guess: float = 0.0,
    p_norm: float | None = None,
    onset_residual_filter: float = 0.2,
    proxy_frequency_hz: float | None = None,
) -> dict[str, Any]:
    """
    Run multiple anchored complex-frequency sweeps and pick a best branch.

    The selected branch is scored by sweep residual quality and optional
    agreement with gain-proxy onset ratio.
    """
    if not mode_frequency_guesses_hz:
        raise ValueError("mode_frequency_guesses_hz must not be empty.")

    proxy_freq = float(mode_frequency_guesses_hz[0] if proxy_frequency_hz is None else proxy_frequency_hz)
    proxy_onset, _proxy_sweep = find_onset_ratio_proxy(
        config,
        frequency_hz=proxy_freq,
        t_hot_min=float(np.min(t_hot_values)),
        t_hot_max=float(np.max(t_hot_values)),
        coarse_step=100.0,
        fine_step=20.0,
    )

    branches: list[dict[str, Any]] = []
    for f_guess in mode_frequency_guesses_hz:
        sweep = sweep_traveling_wave_complex_frequency(
            config,
            t_hot_values=t_hot_values,
            f_real_guess=float(f_guess),
            f_imag_guess=f_imag_guess,
            p_norm=p_norm,
        )
        onset = detect_onset_from_complex_frequency(
            sweep,
            max_residual_norm=onset_residual_filter,
        )
        residuals = np.array([float(p["residual_norm"]) for p in sweep], dtype=float)
        branch: dict[str, Any] = {
            "f_real_guess": float(f_guess),
            "sweep": sweep,
            "onset_ratio_complex": onset,
            "residual_median": float(np.median(residuals)),
            "residual_max": float(np.max(residuals)),
        }
        if onset is None or proxy_onset is None:
            branch["agreement_error"] = float("inf")
        else:
            branch["agreement_error"] = float(abs(onset - proxy_onset) / max(proxy_onset, 1e-12))
        branches.append(branch)

    def _branch_score(branch: dict[str, Any]) -> float:
        score = float(branch["residual_median"])
        if np.isfinite(float(branch["agreement_error"])):
            score += float(branch["agreement_error"])
        else:
            score += 1.0
        return score

    selected_idx = min(range(len(branches)), key=lambda i: _branch_score(branches[i]))
    return {
        "proxy_onset_ratio": proxy_onset,
        "branches": branches,
        "selected_index": int(selected_idx),
        "selected_branch": branches[selected_idx],
    }


def summarize_multimode_selection(result: dict[str, Any]) -> list[dict[str, float | int | bool]]:
    """
    Build compact branch-comparison rows for multimode selection diagnostics.
    """
    rows: list[dict[str, float | int | bool]] = []
    selected_index = int(result["selected_index"])
    proxy_onset = result.get("proxy_onset_ratio")

    for idx, branch in enumerate(result["branches"]):
        onset = branch.get("onset_ratio_complex")
        agreement = branch.get("agreement_error")
        if (
            (agreement is None or not np.isfinite(float(agreement)))
            and onset is not None
            and proxy_onset is not None
            and np.isfinite(float(proxy_onset))
        ):
            agreement = abs(float(onset) - float(proxy_onset)) / max(float(proxy_onset), 1e-12)
        rows.append(
            {
                "branch_index": int(idx),
                "selected": bool(idx == selected_index),
                "f_guess_hz": float(branch["f_real_guess"]),
                "residual_median": float(branch["residual_median"]),
                "residual_max": float(branch["residual_max"]),
                "onset_ratio_complex": float("nan") if onset is None else float(onset),
                "agreement_error": float("nan") if agreement is None else float(agreement),
            }
        )
    return rows


def find_onset_ratio_proxy(
    config: TravelingWaveEngineConfig,
    *,
    frequency_hz: float,
    t_hot_min: float = 300.0,
    t_hot_max: float = 900.0,
    coarse_step: float = 50.0,
    fine_step: float = 10.0,
) -> tuple[float | None, list[dict[str, Any]]]:
    """
    Compute onset ratio from gain proxy using coarse+fine temperature sweeps.
    """
    coarse = np.arange(t_hot_min, t_hot_max + coarse_step, coarse_step)
    coarse_sweep = sweep_traveling_wave_temperature(
        config,
        frequency_hz=frequency_hz,
        t_hot_values=coarse,
    )
    onset = detect_onset_from_gain_proxy(coarse_sweep, t_cold=config.t_cold)
    if onset is None:
        return None, coarse_sweep

    # Refine around nearest coarse transition
    crossing_idx = None
    for i in range(len(coarse_sweep) - 1):
        g0 = float(coarse_sweep[i]["net_gain_proxy"])
        g1 = float(coarse_sweep[i + 1]["net_gain_proxy"])
        if g0 <= 0.0 < g1:
            crossing_idx = i
            break
    if crossing_idx is None:
        return onset, coarse_sweep

    t0 = float(coarse_sweep[crossing_idx]["t_hot"])
    t1 = float(coarse_sweep[crossing_idx + 1]["t_hot"])
    fine = np.arange(t0, t1 + fine_step, fine_step)
    fine_sweep = sweep_traveling_wave_temperature(
        config,
        frequency_hz=frequency_hz,
        t_hot_values=fine,
    )
    onset_fine = detect_onset_from_gain_proxy(fine_sweep, t_cold=config.t_cold)
    return onset_fine, fine_sweep


def compute_regenerator_phase_profile(point: dict[str, Any]) -> dict[str, float]:
    """Return regenerator phase at inlet/mid/outlet in degrees."""
    branch = point["profiles"]["branch"]
    reg_section = None
    for section in branch.sections:
        if section.segment_name == "regenerator":
            reg_section = section
            break
    if reg_section is None or len(reg_section.p1) < 2:
        return {"inlet_deg": float("nan"), "mid_deg": float("nan"), "outlet_deg": float("nan")}

    def phase_idx(i: int) -> float:
        p = reg_section.p1[i]
        u = reg_section.U1[i]
        ph = np.degrees(np.angle(p) - np.angle(u))
        return float((ph + 180.0) % 360.0 - 180.0)

    return {
        "inlet_deg": phase_idx(0),
        "mid_deg": phase_idx(len(reg_section.p1) // 2),
        "outlet_deg": phase_idx(-1),
    }


def compute_loop_power_profile(point: dict[str, Any]) -> dict[str, list[dict[str, float | str]]]:
    """Return absolute boundary power profiles for branch and trunk sections."""
    return {
        "branch": list(point["branch_power_profile"]),
        "trunk": list(point["trunk_power_profile"]),
    }


def compute_efficiency_estimate(
    point: dict[str, Any],
    *,
    t_cold: float,
    t_hot: float,
) -> dict[str, float]:
    """
    Compute a first-order thermal efficiency estimate from power budgets.
    """
    eta_carnot = 1.0 - t_cold / t_hot
    regen_delta = point["regenerator_power_delta"]
    if regen_delta is None or regen_delta <= 0.0:
        return {
            "eta_carnot": float(eta_carnot),
            "eta_thermal_est": 0.0,
            "eta_relative": 0.0,
            "w_regen": float(0.0 if regen_delta is None else regen_delta),
            "w_useful": 0.0,
        }

    # Use resonator dissipation as delivered useful acoustic power proxy.
    trunk_deltas = point["trunk_power_deltas"]
    w_useful = max(0.0, -float(trunk_deltas.get("resonator", 0.0)))
    eta_thermal_est = eta_carnot * (w_useful / float(regen_delta))
    eta_thermal_est = max(0.0, min(float(eta_thermal_est), float(eta_carnot)))
    eta_relative = eta_thermal_est / eta_carnot if eta_carnot > 0 else 0.0
    return {
        "eta_carnot": float(eta_carnot),
        "eta_thermal_est": float(eta_thermal_est),
        "eta_relative": float(eta_relative),
        "w_regen": float(regen_delta),
        "w_useful": float(w_useful),
    }


def sweep_efficiency_estimate(
    config: TravelingWaveEngineConfig,
    *,
    frequency_hz: float,
    t_hot_values: np.ndarray,
) -> list[dict[str, float]]:
    """Sweep temperatures and return efficiency-estimate diagnostics."""
    points = sweep_traveling_wave_temperature(
        config,
        frequency_hz=frequency_hz,
        t_hot_values=t_hot_values,
    )
    rows: list[dict[str, float]] = []
    for point in points:
        t_hot = float(point["t_hot"])
        eff = compute_efficiency_estimate(point, t_cold=config.t_cold, t_hot=t_hot)
        rows.append(
            {
                "t_hot": t_hot,
                "eta_carnot": eff["eta_carnot"],
                "eta_thermal_est": eff["eta_thermal_est"],
                "eta_relative": eff["eta_relative"],
                "w_regen": eff["w_regen"],
                "w_useful": eff["w_useful"],
                "net_gain_proxy": float(point["net_gain_proxy"]),
            }
        )
    return rows


def estimate_loop_frequency_range(config: TravelingWaveEngineConfig) -> dict[str, float]:
    """
    Estimate expected operating frequency from quarter-wave resonator loading.

    The resonator behaves approximately like a quarter-wave section loaded by the
    loop impedance; empirical loading often shifts the frequency downward by
    roughly 30-50%.
    """
    helium = gas.Helium(mean_pressure=config.mean_pressure)
    a = helium.sound_speed(config.t_cold)
    quarter_wave = a / (4.0 * config.resonator_length)
    return {
        "f_quarter_hz": float(quarter_wave),
        "f_expected_low_hz": float(0.5 * quarter_wave),
        "f_expected_high_hz": float(0.7 * quarter_wave),
    }

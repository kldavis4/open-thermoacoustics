"""Traveling-wave engine helpers using distributed loop propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import root

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
    u1_input: complex | None = None,
    zb_real_guess: float | None = None,
    zb_imag_guess: float | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Solve loop closure for complex frequency with fixed input velocity.

    Unknowns are `[f_real, f_imag, Re(Zb), Im(Zb)]` and targets are:
    `Re/Im(p_mismatch)=0`, `Re/Im(U1_hardend)=0`.

    Notes
    -----
    The loop topology needs one gauge choice to close the system when frequency
    is unknown. We use the input velocity from a converged real-frequency solve
    as that gauge and then solve for complex frequency and loop impedance.
    """
    real_reference = None
    if u1_input is None:
        real_reference = solve_traveling_wave_engine_fixed_frequency(
            config,
            frequency_hz=f_real_guess,
            t_hot=t_hot,
            zb_real_guess=zb_real_guess,
            zb_imag_guess=zb_imag_guess,
            verbose=False,
        )
        ref_result: TBranchLoopResult = real_reference["result"]
        u1_input = ref_result.U1_input
        if zb_real_guess is None:
            zb_real_guess = ref_result.Zb_real
        if zb_imag_guess is None:
            zb_imag_guess = ref_result.Zb_imag

    if zb_real_guess is None:
        zb_real_guess = config.zb_real_guess
    if zb_imag_guess is None:
        zb_imag_guess = config.zb_imag_guess

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

    freq_scale = max(abs(f_real_guess), 1.0)
    imag_scale = max(abs(f_imag_guess), 1.0)
    z_scale = max(abs(zb_real_guess), abs(zb_imag_guess), 1e3)
    u_scale = max(abs(u1_input), 1e-9)

    def residual(x: np.ndarray) -> np.ndarray:
        f_real = x[0] * freq_scale
        f_imag = x[1] * imag_scale
        zb_real = x[2] * z_scale
        zb_imag = x[3] * z_scale
        omega = 2.0 * np.pi * (f_real + 1j * f_imag)
        propagator.omega = omega
        try:
            _p1_union, u1_hardend, p_mismatch = propagator(
                config.p1_input,
                u1_input,
                complex(zb_real, zb_imag),
            )
        except Exception:
            return np.array([1e8, 1e8, 1e8, 1e8], dtype=float)

        return np.array(
            [
                p_mismatch.real / 1000.0,
                p_mismatch.imag / 1000.0,
                u1_hardend.real / u_scale,
                u1_hardend.imag / u_scale,
            ],
            dtype=float,
        )

    x0 = np.array(
        [
            f_real_guess / freq_scale,
            f_imag_guess / imag_scale,
            zb_real_guess / z_scale,
            zb_imag_guess / z_scale,
        ],
        dtype=float,
    )
    tol_local = config.tol if tol is None else tol
    maxiter_local = config.maxiter if maxiter is None else maxiter

    result = root(
        residual,
        x0,
        method="hybr",
        tol=tol_local,
        options={"maxfev": maxiter_local * 20},
    )
    if not result.success:
        result = root(
            residual,
            x0,
            method="lm",
            tol=tol_local,
            options={"maxiter": maxiter_local * 20},
        )

    f_real = float(result.x[0] * freq_scale)
    f_imag = float(result.x[1] * imag_scale)
    zb_real = float(result.x[2] * z_scale)
    zb_imag = float(result.x[3] * z_scale)
    omega = 2.0 * np.pi * (f_real + 1j * f_imag)
    propagator.omega = omega

    p1_union, u1_hardend, p_mismatch = propagator(
        config.p1_input,
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
            f"residual={np.linalg.norm(result.fun):.3e}"
        )

    return {
        "converged": bool(result.success),
        "message": str(result.message),
        "n_iterations": int(getattr(result, "nfev", 0)),
        "residual_norm": float(np.linalg.norm(result.fun)),
        "frequency_real": f_real,
        "frequency_imag": f_imag,
        "omega": omega,
        "zb_real": zb_real,
        "zb_imag": zb_imag,
        "zb": complex(zb_real, zb_imag),
        "p1_input": config.p1_input,
        "u1_input": u1_input,
        "p1_union": p1_union,
        "u1_hardend": u1_hardend,
        "pressure_mismatch": p_mismatch,
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

    for t_hot in t_hot_values:
        real_point = solve_traveling_wave_engine_fixed_frequency(
            config,
            frequency_hz=float(f_real),
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

        point = solve_traveling_wave_engine_complex_frequency(
            config,
            t_hot=float(t_hot),
            f_real_guess=float(f_real),
            f_imag_guess=float(f_imag),
            u1_input=real_result.U1_input,
            zb_real_guess=real_result.Zb_real,
            zb_imag_guess=real_result.Zb_imag,
            verbose=False,
        )
        point["real_frequency_reference"] = real_point
        points.append(point)

        f_real = float(point["frequency_real"])
        f_imag = float(point["frequency_imag"])
        zb_real = float(point["zb_real"])
        zb_imag = float(point["zb_imag"])

    return points


def detect_onset_from_complex_frequency(
    sweep: list[dict[str, Any]],
    *,
    f_imag_tol_hz: float = 1e-4,
) -> float | None:
    """
    Detect onset ratio from `f_imag` sign crossing.

    Convention used: onset is where `f_imag` crosses from positive (damped) to
    non-positive (growing).
    """
    for i in range(len(sweep) - 1):
        fi0 = float(sweep[i]["frequency_imag"])
        fi1 = float(sweep[i + 1]["frequency_imag"])
        if fi0 > f_imag_tol_hz and fi1 < -f_imag_tol_hz:
            r0 = float(sweep[i]["temperature_ratio"])
            r1 = float(sweep[i + 1]["temperature_ratio"])
            if abs(fi1 - fi0) < 1e-15:
                return r1
            return r0 + (r1 - r0) * (fi0 / (fi0 - fi1))
    return None


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

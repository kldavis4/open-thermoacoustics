"""Traveling-wave engine helpers using distributed loop propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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

"""Standing-wave thermoacoustic engine validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import root

from openthermoacoustics import gas, segments
from openthermoacoustics.geometry.parallel_plate import ParallelPlate
from openthermoacoustics.solver import NetworkTopology, ShootingSolver
from openthermoacoustics.solver.shooting import SolverResult


@dataclass(frozen=True)
class StandingWaveEngineConfig:
    """Canonical standing-wave engine geometry and operating setup."""

    mean_pressure: float = 1.0e6
    t_cold: float = 300.0
    left_duct_length: float = 0.30
    cold_hx_length: float = 0.02
    stack_length: float = 0.10
    hot_hx_length: float = 0.02
    right_duct_length: float = 0.30
    duct_radius: float = 0.035
    plate_spacing: float = 0.8e-3
    plate_thickness: float = 0.1e-3
    n_points_per_segment: int = 120
    method: str = "lm"
    tol: float = 1e-10
    maxiter: int = 200

    @property
    def porosity(self) -> float:
        """Open-area porosity of the stack/HX matrix."""
        return self.plate_spacing / (self.plate_spacing + self.plate_thickness)

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius for parallel plates (half-gap)."""
        return self.plate_spacing / 2.0

    @property
    def duct_area(self) -> float:
        """Resonator cross-sectional area."""
        return float(np.pi * self.duct_radius**2)

    @property
    def total_length(self) -> float:
        """Total physical length of the modeled resonator."""
        return (
            self.left_duct_length
            + self.cold_hx_length
            + self.stack_length
            + self.hot_hx_length
            + self.right_duct_length
        )

    @property
    def stack_start(self) -> float:
        """Global x-location of the stack inlet."""
        return self.left_duct_length + self.cold_hx_length

    @property
    def stack_end(self) -> float:
        """Global x-location of the stack outlet."""
        return self.stack_start + self.stack_length

    @property
    def right_duct_start(self) -> float:
        """Global x-location of the right duct inlet."""
        return self.stack_end + self.hot_hx_length


@dataclass(frozen=True)
class EngineSweepPoint:
    """One solved operating point in a hot-side temperature sweep."""

    t_hot: float
    result: SolverResult
    stack_power_change: float

    @property
    def temperature_ratio(self) -> float:
        """Hot-to-cold temperature ratio."""
        return self.t_hot / float(self.result.T_m_profile[0])


def default_standing_wave_engine_config() -> StandingWaveEngineConfig:
    """Return default canonical standing-wave engine configuration."""
    return StandingWaveEngineConfig()


def optimized_standing_wave_engine_config() -> StandingWaveEngineConfig:
    """
    Return the current best-performing standing-wave benchmark configuration.

    This configuration was selected from parameter sweeps over stack position,
    plate half-gap, and resonator length, using complex-frequency onset
    crossing as the primary criterion.
    """
    return StandingWaveEngineConfig(
        left_duct_length=1.0,
        right_duct_length=0.2,
        plate_spacing=1.2e-3,   # y0 = 0.6 mm
        plate_thickness=0.14983127109111363e-3,  # keep porosity ~= 0.889
    )


def symmetric_negative_control_config() -> StandingWaveEngineConfig:
    """Return symmetric layout expected to remain below onset <= 800 K."""
    return default_standing_wave_engine_config()


def shifted_negative_control_config() -> StandingWaveEngineConfig:
    """Return shifted layout (0.10/0.50) used as secondary negative control."""
    return StandingWaveEngineConfig(left_duct_length=0.10, right_duct_length=0.50)


def geometry_sensitive_reference_config() -> StandingWaveEngineConfig:
    """Return a geometry-sensitive reference layout with higher onset."""
    return StandingWaveEngineConfig(left_duct_length=0.45, right_duct_length=0.15)


def build_standing_wave_engine_network(
    config: StandingWaveEngineConfig,
    t_hot: float,
) -> NetworkTopology:
    """Build the canonical standing-wave engine network."""
    network = NetworkTopology()
    pore_geometry = ParallelPlate()

    network.add_segment(
        segments.Duct(length=config.left_duct_length, radius=config.duct_radius)
    )
    network.add_segment(
        segments.HeatExchanger(
            length=config.cold_hx_length,
            porosity=config.porosity,
            hydraulic_radius=config.hydraulic_radius,
            temperature=config.t_cold,
            area=config.duct_area,
            geometry=pore_geometry,
            name="cold_hx",
        )
    )
    network.add_segment(
        segments.Stack(
            length=config.stack_length,
            porosity=config.porosity,
            hydraulic_radius=config.hydraulic_radius,
            area=config.duct_area,
            geometry=pore_geometry,
            T_cold=config.t_cold,
            T_hot=t_hot,
            name="stack",
        )
    )
    network.add_segment(
        segments.HeatExchanger(
            length=config.hot_hx_length,
            porosity=config.porosity,
            hydraulic_radius=config.hydraulic_radius,
            temperature=t_hot,
            area=config.duct_area,
            geometry=pore_geometry,
            name="hot_hx",
        )
    )
    network.add_segment(
        segments.Duct(length=config.right_duct_length, radius=config.duct_radius)
    )
    return network


def solve_standing_wave_engine(
    config: StandingWaveEngineConfig,
    t_hot: float,
    *,
    frequency_guess: float | None = None,
    p1_phase_guess: float = 0.0,
) -> EngineSweepPoint:
    """Solve one standing-wave engine operating point."""
    helium = gas.Helium(mean_pressure=config.mean_pressure)
    network = build_standing_wave_engine_network(config, t_hot=t_hot)

    if frequency_guess is None:
        frequency_guess = helium.sound_speed(config.t_cold) / (2.0 * config.total_length)

    solver = ShootingSolver(network, helium)
    result = solver.solve(
        guesses={"frequency": frequency_guess, "p1_phase": p1_phase_guess},
        targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
        options={
            "T_m_start": config.t_cold,
            "n_points_per_segment": config.n_points_per_segment,
            "method": config.method,
            "tol": config.tol,
            "maxiter": config.maxiter,
        },
    )

    stack_power_change = compute_power_change(
        result,
        x_start=config.stack_start,
        x_end=config.stack_end,
    )
    return EngineSweepPoint(
        t_hot=t_hot,
        result=result,
        stack_power_change=stack_power_change,
    )


def sweep_standing_wave_engine(
    config: StandingWaveEngineConfig,
    t_hot_values: Iterable[float],
) -> list[EngineSweepPoint]:
    """Run a continuation sweep over hot-side temperature."""
    sweep: list[EngineSweepPoint] = []

    helium = gas.Helium(mean_pressure=config.mean_pressure)
    frequency_guess = helium.sound_speed(config.t_cold) / (2.0 * config.total_length)
    phase_guess = 0.0

    for t_hot in t_hot_values:
        point = solve_standing_wave_engine(
            config,
            float(t_hot),
            frequency_guess=frequency_guess,
            p1_phase_guess=phase_guess,
        )
        sweep.append(point)
        frequency_guess = point.result.frequency
        phase_guess = float(point.result.guesses_final.get("p1_phase", phase_guess))

    return sweep


def compute_power_change(
    result: SolverResult,
    *,
    x_start: float,
    x_end: float,
) -> float:
    """Compute acoustic-power difference between two x locations."""
    x = result.x_profile
    power = result.acoustic_power
    i_start = int(np.argmin(np.abs(x - x_start)))
    i_end = int(np.argmin(np.abs(x - x_end)))
    return float(power[i_end] - power[i_start])


def solve_standing_wave_engine_complex_frequency(
    config: StandingWaveEngineConfig,
    t_hot: float,
    *,
    frequency_real_guess: float | None = None,
    frequency_imag_guess: float = 0.0,
) -> dict[str, float | bool | str]:
    """
    Solve closed-closed boundary conditions using complex frequency.

    Solves for `(f_real, f_imag)` such that `U1_end_real = U1_end_imag = 0`.
    """
    helium = gas.Helium(mean_pressure=config.mean_pressure)
    network = build_standing_wave_engine_network(config, t_hot=t_hot)

    if frequency_real_guess is None:
        frequency_real_guess = helium.sound_speed(config.t_cold) / (2.0 * config.total_length)

    def residual(vec: np.ndarray) -> np.ndarray:
        f_real = float(vec[0])
        f_imag = float(vec[1])
        omega = 2.0 * np.pi * (f_real + 1j * f_imag)
        try:
            network.propagate_all(
                p1_start=1000.0 + 0.0j,
                U1_start=0.0 + 0.0j,
                T_m_start=config.t_cold,
                omega=omega,  # type: ignore[arg-type]
                gas=helium,
                n_points_per_segment=config.n_points_per_segment,
            )
            endpoint = network.get_endpoint_values()["U1_end"]
            return np.array([endpoint.real, endpoint.imag], dtype=float)
        except Exception:
            return np.array([1e12, 1e12], dtype=float)

    result = root(
        residual,
        x0=np.array([frequency_real_guess, frequency_imag_guess], dtype=float),
        method="hybr",
        tol=config.tol,
    )
    f_real = float(result.x[0])
    f_imag = float(result.x[1])
    omega = 2.0 * np.pi * (f_real + 1j * f_imag)
    network.propagate_all(
        p1_start=1000.0 + 0.0j,
        U1_start=0.0 + 0.0j,
        T_m_start=config.t_cold,
        omega=omega,  # type: ignore[arg-type]
        gas=helium,
        n_points_per_segment=config.n_points_per_segment,
    )
    profiles = network.get_global_profiles()
    i_start = int(np.argmin(np.abs(profiles["x"] - config.stack_start)))
    i_end = int(np.argmin(np.abs(profiles["x"] - config.stack_end)))
    stack_power_change = float(profiles["acoustic_power"][i_end] - profiles["acoustic_power"][i_start])
    u_end = complex(network.get_endpoint_values()["U1_end"])

    return {
        "frequency_real": f_real,
        "frequency_imag": f_imag,
        "converged": bool(result.success),
        "message": str(result.message),
        "residual_norm": float(np.linalg.norm(result.fun)),
        "u1_end_abs": float(abs(u_end)),
        "stack_power_change": stack_power_change,
    }


def solve_standing_wave_engine_complex_frequency_with_profiles(
    config: StandingWaveEngineConfig,
    t_hot: float,
    *,
    frequency_real_guess: float | None = None,
    frequency_imag_guess: float = 0.0,
) -> dict[str, Any]:
    """Solve complex-frequency resonance and return full propagated profiles."""
    helium = gas.Helium(mean_pressure=config.mean_pressure)
    network = build_standing_wave_engine_network(config, t_hot=t_hot)

    if frequency_real_guess is None:
        frequency_real_guess = helium.sound_speed(config.t_cold) / (2.0 * config.total_length)

    def residual(vec: np.ndarray) -> np.ndarray:
        f_real = float(vec[0])
        f_imag = float(vec[1])
        omega = 2.0 * np.pi * (f_real + 1j * f_imag)
        try:
            network.propagate_all(
                p1_start=1000.0 + 0.0j,
                U1_start=0.0 + 0.0j,
                T_m_start=config.t_cold,
                omega=omega,  # type: ignore[arg-type]
                gas=helium,
                n_points_per_segment=config.n_points_per_segment,
            )
            endpoint = network.get_endpoint_values()["U1_end"]
            return np.array([endpoint.real, endpoint.imag], dtype=float)
        except Exception:
            return np.array([1e12, 1e12], dtype=float)

    result = root(
        residual,
        x0=np.array([frequency_real_guess, frequency_imag_guess], dtype=float),
        method="hybr",
        tol=config.tol,
    )
    f_real = float(result.x[0])
    f_imag = float(result.x[1])
    omega = 2.0 * np.pi * (f_real + 1j * f_imag)
    network.propagate_all(
        p1_start=1000.0 + 0.0j,
        U1_start=0.0 + 0.0j,
        T_m_start=config.t_cold,
        omega=omega,  # type: ignore[arg-type]
        gas=helium,
        n_points_per_segment=config.n_points_per_segment,
    )
    profiles = network.get_global_profiles()
    i_start = int(np.argmin(np.abs(profiles["x"] - config.stack_start)))
    i_end = int(np.argmin(np.abs(profiles["x"] - config.stack_end)))
    stack_power_change = float(profiles["acoustic_power"][i_end] - profiles["acoustic_power"][i_start])
    u_end = complex(network.get_endpoint_values()["U1_end"])

    return {
        "frequency_real": f_real,
        "frequency_imag": f_imag,
        "converged": bool(result.success),
        "message": str(result.message),
        "residual_norm": float(np.linalg.norm(result.fun)),
        "u1_end_abs": float(abs(u_end)),
        "stack_power_change": stack_power_change,
        "profiles": profiles,
    }


def sweep_standing_wave_engine_complex_frequency(
    config: StandingWaveEngineConfig,
    t_hot_values: Iterable[float],
    *,
    frequency_real_guess: float | None = None,
    frequency_imag_guess: float = 0.0,
) -> list[dict[str, float | bool | str]]:
    """Run continuation sweep with complex frequency as primary onset metric."""
    if frequency_real_guess is None:
        helium = gas.Helium(mean_pressure=config.mean_pressure)
        frequency_real_guess = helium.sound_speed(config.t_cold) / (2.0 * config.total_length)

    sweep: list[dict[str, float | bool | str]] = []
    fr = frequency_real_guess
    fi = frequency_imag_guess
    for t_hot in t_hot_values:
        point = solve_standing_wave_engine_complex_frequency(
            config,
            t_hot=float(t_hot),
            frequency_real_guess=fr,
            frequency_imag_guess=fi,
        )
        fr = float(point["frequency_real"])
        fi = float(point["frequency_imag"])
        point["t_hot"] = float(t_hot)
        point["temperature_ratio"] = float(t_hot / config.t_cold)
        sweep.append(point)
    return sweep


def detect_onset_from_complex_frequency(
    sweep: list[dict[str, float | bool | str]],
) -> float | None:
    """
    Detect onset temperature ratio from f_imag sign crossing.

    Convention used here: onset occurs when f_imag crosses from positive
    (damped) to negative (growing oscillation).
    """
    for i in range(len(sweep) - 1):
        fi0 = float(sweep[i]["frequency_imag"])
        fi1 = float(sweep[i + 1]["frequency_imag"])
        if fi0 > 0.0 and fi1 <= 0.0:
            r0 = float(sweep[i]["temperature_ratio"])
            r1 = float(sweep[i + 1]["temperature_ratio"])
            if abs(fi1 - fi0) < 1e-15:
                return r1
            return r0 + (r1 - r0) * (fi0 / (fi0 - fi1))
    return None

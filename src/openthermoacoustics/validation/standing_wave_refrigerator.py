"""Standing-wave thermoacoustic refrigerator benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from openthermoacoustics import gas, segments
from openthermoacoustics.geometry.parallel_plate import ParallelPlate
from openthermoacoustics.solver import NetworkTopology, ShootingSolver
from openthermoacoustics.solver.shooting import SolverResult


@dataclass(frozen=True)
class StandingWaveRefrigeratorConfig:
    """Driven standing-wave refrigerator configuration."""

    mean_pressure: float = 1.0e6
    t_hot: float = 300.0
    t_cold: float = 270.0
    drive_ratio: float = 0.03
    left_duct_length: float = 0.50
    hot_hx_length: float = 0.02
    stack_length: float = 0.08
    cold_hx_length: float = 0.02
    right_duct_length: float = 0.10
    duct_radius: float = 0.03
    plate_spacing: float = 0.6e-3
    plate_thickness: float = 0.1e-3
    stack_solid_thermal_conductivity: float = 0.2
    stack_solid_density: float = 1390.0
    stack_solid_specific_heat: float = 1170.0
    streaming_geometry_factor: float = 0.04
    n_points_per_segment: int = 120
    method: str = "lm"
    tol: float = 1e-10
    maxiter: int = 200

    @property
    def porosity(self) -> float:
        """Open-area porosity of stack/HX matrix."""
        return self.plate_spacing / (self.plate_spacing + self.plate_thickness)

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius for parallel plates."""
        return self.plate_spacing / 2.0

    @property
    def plate_half_thickness(self) -> float:
        """Half-thickness of solid plate (l)."""
        return self.plate_thickness / 2.0

    @property
    def duct_area(self) -> float:
        """Resonator cross-sectional area."""
        return float(np.pi * self.duct_radius**2)

    @property
    def total_length(self) -> float:
        """Total resonator physical length."""
        return (
            self.left_duct_length
            + self.hot_hx_length
            + self.stack_length
            + self.cold_hx_length
            + self.right_duct_length
        )

    @property
    def p_drive(self) -> float:
        """Drive pressure amplitude at the left boundary."""
        return float(self.drive_ratio * self.mean_pressure)

    @property
    def segment_boundaries(self) -> dict[str, tuple[float, float]]:
        """Global x-boundaries for each segment."""
        x0 = 0.0
        x1 = x0 + self.left_duct_length
        x2 = x1 + self.hot_hx_length
        x3 = x2 + self.stack_length
        x4 = x3 + self.cold_hx_length
        x5 = x4 + self.right_duct_length
        return {
            "left_duct": (x0, x1),
            "hot_hx": (x1, x2),
            "stack": (x2, x3),
            "cold_hx": (x3, x4),
            "right_duct": (x4, x5),
        }


def default_standing_wave_refrigerator_config() -> StandingWaveRefrigeratorConfig:
    """Return default standing-wave refrigerator benchmark config."""
    return StandingWaveRefrigeratorConfig()


def tijani_refrigerator_config() -> StandingWaveRefrigeratorConfig:
    """
    Return an approximate Tijani et al. (2002) standing-wave refrigerator setup.

    This is a simplified uniform-radius half-wave representation to fit the
    current segment toolkit while preserving the published stack geometry and
    operating point.
    """
    return StandingWaveRefrigeratorConfig(
        mean_pressure=1.0e6,
        t_hot=287.5,
        t_cold=212.5,
        drive_ratio=0.02,
        duct_radius=0.019,
        left_duct_length=0.0375,
        hot_hx_length=0.005,
        stack_length=0.085,
        cold_hx_length=0.0025,
        right_duct_length=1.06,
        plate_spacing=0.3e-3,
        plate_thickness=0.06e-3,
        stack_solid_thermal_conductivity=0.16,
        streaming_geometry_factor=0.04,
        n_points_per_segment=140,
    )


def compute_f_solid(
    plate_half_thickness: float,
    solid_thermal_conductivity: float,
    solid_density: float,
    solid_specific_heat: float,
    omega: float,
) -> complex:
    """Solid thermal correction function for parallel plates."""
    delta_s = np.sqrt(
        2.0
        * solid_thermal_conductivity
        / (solid_density * solid_specific_heat * omega)
    )
    z = (1.0 + 1.0j) * plate_half_thickness / delta_s
    if abs(z) < 1e-10:
        return 1.0 + 0.0j
    return complex(np.tanh(z) / z)


def build_standing_wave_refrigerator_network(
    config: StandingWaveRefrigeratorConfig,
) -> NetworkTopology:
    """Build the driven standing-wave refrigerator network."""
    geometry = ParallelPlate()
    network = NetworkTopology()

    network.add_segment(
        segments.Duct(name="left_duct", length=config.left_duct_length, radius=config.duct_radius)
    )
    network.add_segment(
        segments.HeatExchanger(
            name="hot_hx",
            length=config.hot_hx_length,
            porosity=config.porosity,
            hydraulic_radius=config.hydraulic_radius,
            temperature=config.t_hot,
            area=config.duct_area,
            geometry=geometry,
        )
    )
    # For refrigerator mode, mean temperature decreases from hot to cold along +x.
    network.add_segment(
        segments.Stack(
            name="stack",
            length=config.stack_length,
            porosity=config.porosity,
            hydraulic_radius=config.hydraulic_radius,
            area=config.duct_area,
            geometry=geometry,
            T_cold=config.t_hot,
            T_hot=config.t_cold,
        )
    )
    network.add_segment(
        segments.HeatExchanger(
            name="cold_hx",
            length=config.cold_hx_length,
            porosity=config.porosity,
            hydraulic_radius=config.hydraulic_radius,
            temperature=config.t_cold,
            area=config.duct_area,
            geometry=geometry,
        )
    )
    network.add_segment(
        segments.Duct(name="right_duct", length=config.right_duct_length, radius=config.duct_radius)
    )
    return network


def _segment_power_deltas(
    result: SolverResult,
    boundaries: dict[str, tuple[float, float]],
    scale_power: float,
) -> dict[str, float]:
    x = result.x_profile
    p = result.acoustic_power * scale_power
    deltas: dict[str, float] = {}
    for name, (x_start, x_end) in boundaries.items():
        i0 = int(np.argmin(np.abs(x - x_start)))
        i1 = int(np.argmin(np.abs(x - x_end)))
        deltas[name] = float(p[i1] - p[i0])
    return deltas


def _segment_power_profile(
    result: SolverResult,
    boundaries: dict[str, tuple[float, float]],
    scale_power: float,
) -> list[dict[str, float | str]]:
    """Return per-segment boundary acoustic power values."""
    x = result.x_profile
    p = result.acoustic_power * scale_power
    rows: list[dict[str, float | str]] = []
    for name, (x_start, x_end) in boundaries.items():
        i0 = int(np.argmin(np.abs(x - x_start)))
        i1 = int(np.argmin(np.abs(x - x_end)))
        w_in = float(p[i0])
        w_out = float(p[i1])
        rows.append(
            {
                "name": name,
                "x_start": float(x[i0]),
                "x_end": float(x[i1]),
                "w_in": w_in,
                "w_out": w_out,
                "delta_w": w_out - w_in,
            }
        )
    return rows


def _scaled_profiles(result: SolverResult, p_scale: float) -> dict[str, np.ndarray]:
    return {
        "p1_profile": result.p1_profile * p_scale,
        "U1_profile": result.U1_profile * p_scale,
        "acoustic_power": result.acoustic_power * (p_scale**2),
    }


def _stack_h2_components(
    config: StandingWaveRefrigeratorConfig,
    stack_segment: segments.Stack,
    *,
    p1: complex,
    u1: complex,
    t_m: float,
    d_t_dx: float,
    omega: float,
    helium: gas.Helium,
) -> dict[str, float]:
    """
    Compute H2 components at one stack location.

    H2 = E_dot + H_streaming + Q_conduction
    with streaming and conduction terms following the StackEnergy formulation.
    """
    rho_m = helium.density(t_m)
    cp = helium.specific_heat_cp(t_m)
    k_gas = helium.thermal_conductivity(t_m)
    sigma = helium.prandtl(t_m)
    mu = helium.viscosity(t_m)

    area_gas = config.porosity * config.duct_area
    area_solid = (1.0 - config.porosity) * config.duct_area

    delta_nu = np.sqrt(2.0 * mu / (rho_m * omega))
    delta_kappa = np.sqrt(2.0 * k_gas / (rho_m * cp * omega))
    f_nu = stack_segment.geometry.f_nu(omega, delta_nu, config.hydraulic_radius)
    f_kappa = stack_segment.geometry.f_kappa(omega, delta_kappa, config.hydraulic_radius)

    f_solid = compute_f_solid(
        config.plate_half_thickness,
        config.stack_solid_thermal_conductivity,
        config.stack_solid_density,
        config.stack_solid_specific_heat,
        omega,
    )
    one_minus_f_nu_conj = 1.0 - np.conj(f_nu)
    sigma_denom = 1.0 - sigma + sigma * f_solid

    f_tilde: complex = complex(float("nan"), float("nan"))
    if abs(one_minus_f_nu_conj) < 1e-20 or abs(sigma_denom) < 1e-20:
        streaming_coeff = 0.0
    else:
        f_tilde = (
            f_kappa - f_solid * np.conj(f_nu)
        ) / (one_minus_f_nu_conj * sigma_denom)
        streaming_coeff = (
            config.streaming_geometry_factor
            *
            rho_m
            * cp
            * np.imag(f_tilde)
            / (2.0 * omega * area_gas)
        )

    e_dot = float(0.5 * np.real(p1 * np.conj(u1)))
    h_streaming = float(-streaming_coeff * (abs(u1) ** 2) * d_t_dx)
    q_conduction = float(
        -(k_gas * area_gas + config.stack_solid_thermal_conductivity * area_solid) * d_t_dx
    )
    h2_total = float(e_dot + h_streaming + q_conduction)

    return {
        "E_dot": e_dot,
        "H_streaming": h_streaming,
        "Q_conduction": q_conduction,
        "H2_total": h2_total,
        "f_solid_real": float(np.real(f_solid)),
        "f_solid_imag": float(np.imag(f_solid)),
        "f_tilde_real": float(np.real(f_tilde)),
        "f_tilde_imag": float(np.imag(f_tilde)),
    }


def solve_standing_wave_refrigerator(
    config: StandingWaveRefrigeratorConfig,
    *,
    drive_ratio: float | None = None,
    frequency_guess: float | None = None,
    p1_phase_guess: float = 0.0,
) -> dict[str, Any]:
    """
    Solve driven standing-wave refrigerator mode and return performance metrics.

    Cooling and COP use short-stack (Tijani-style) cooling with acoustic
    power absorbed by the stack as input work. H2 terms are returned for
    diagnostics only.
    """
    cfg = config if drive_ratio is None else replace(config, drive_ratio=float(drive_ratio))
    helium = gas.Helium(mean_pressure=cfg.mean_pressure)
    network = build_standing_wave_refrigerator_network(cfg)
    solver = ShootingSolver(network, helium)

    if frequency_guess is None:
        t_ref = 0.5 * (cfg.t_hot + cfg.t_cold)
        frequency_guess = helium.sound_speed(t_ref) / (2.0 * cfg.total_length)

    result = solver.solve(
        guesses={"frequency": frequency_guess, "p1_phase": p1_phase_guess},
        targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
        options={
            "T_m_start": cfg.t_hot,
            "n_points_per_segment": cfg.n_points_per_segment,
            "method": cfg.method,
            "tol": cfg.tol,
            "maxiter": cfg.maxiter,
        },
    )

    # Base shooting normalization is p1_amplitude=1000 Pa unless guessed otherwise.
    p_scale = cfg.p_drive / 1000.0
    scaled = _scaled_profiles(result, p_scale=p_scale)
    deltas = _segment_power_deltas(
        result,
        boundaries=cfg.segment_boundaries,
        scale_power=p_scale**2,
    )
    power_profile = _segment_power_profile(
        result,
        boundaries=cfg.segment_boundaries,
        scale_power=p_scale**2,
    )

    # H2-based cooling/heating from stack boundaries (streaming + conduction included).
    boundaries = cfg.segment_boundaries
    x = result.x_profile
    i_stack_hot = int(np.argmin(np.abs(x - boundaries["stack"][0])))
    i_stack_cold = int(np.argmin(np.abs(x - boundaries["stack"][1])))
    omega = 2.0 * np.pi * float(result.frequency)
    helium = gas.Helium(mean_pressure=cfg.mean_pressure)
    stack_segment = network.segments[2]
    d_t_dx = (cfg.t_cold - cfg.t_hot) / cfg.stack_length

    p_hot = complex(result.p1_profile[i_stack_hot] * p_scale)
    u_hot = complex(result.U1_profile[i_stack_hot] * p_scale)
    t_hot_local = float(result.T_m_profile[i_stack_hot])
    h2_hot_components = _stack_h2_components(
        cfg,
        stack_segment,
        p1=p_hot,
        u1=u_hot,
        t_m=t_hot_local,
        d_t_dx=d_t_dx,
        omega=omega,
        helium=helium,
    )

    p_cold = complex(result.p1_profile[i_stack_cold] * p_scale)
    u_cold = complex(result.U1_profile[i_stack_cold] * p_scale)
    t_cold_local = float(result.T_m_profile[i_stack_cold])
    h2_cold_components = _stack_h2_components(
        cfg,
        stack_segment,
        p1=p_cold,
        u1=u_cold,
        t_m=t_cold_local,
        d_t_dx=d_t_dx,
        omega=omega,
        helium=helium,
    )

    h2_hot = float(h2_hot_components["H2_total"])
    h2_cold = float(h2_cold_components["H2_total"])

    stack_bounds = boundaries["stack"]
    x_stack_center = 0.5 * (stack_bounds[0] + stack_bounds[1])
    i_stack_center = int(np.argmin(np.abs(x - x_stack_center)))
    p_center = complex(result.p1_profile[i_stack_center] * p_scale)
    u_center = complex(result.U1_profile[i_stack_center] * p_scale)
    t_center = float(result.T_m_profile[i_stack_center])

    w_stack_hot = float(power_profile[1]["w_out"])
    w_stack_cold = float(power_profile[2]["w_out"])
    w_stack_absorbed = float(w_stack_hot - w_stack_cold)
    w_input_system = max(0.0, -float(sum(deltas.values())))
    q_cold_h2 = max(0.0, h2_hot - h2_cold)
    short_stack_terms = _tijani_short_stack_terms(
        p1_center=p_center,
        config=cfg,
        helium=helium,
        omega=omega,
    )
    q_cold_short_stack = max(
        0.0,
        tijani_cooling_power_short_stack(
            p1_center=p_center,
            config=cfg,
            helium=helium,
            omega=omega,
            w_stack_absorbed=w_stack_absorbed,
        ),
    )
    q_cold = q_cold_short_stack
    q_hot = q_cold + w_stack_absorbed
    q_cold_proxy = max(0.0, -float(deltas["cold_hx"]))
    cop = q_cold / w_stack_absorbed if w_stack_absorbed > 0.0 else 0.0
    cop_carnot = cfg.t_cold / max(cfg.t_hot - cfg.t_cold, 1e-12)
    cop_relative = cop / cop_carnot if cop_carnot > 0.0 else 0.0
    cop_system = q_cold / w_input_system if w_input_system > 0.0 else 0.0

    return {
        "config": cfg,
        "result": result,
        "frequency_hz": float(result.frequency),
        "p_drive": float(cfg.p_drive),
        "profiles": scaled,
        "power_deltas": deltas,
        "power_profile": power_profile,
        "w_stack_hot": w_stack_hot,
        "w_stack_cold": w_stack_cold,
        "w_stack_absorbed": w_stack_absorbed,
        "stack_center_x": float(x[i_stack_center]),
        "stack_center_p1": p_center,
        "stack_center_u1": u_center,
        "stack_center_Tm": t_center,
        "gamma_ratio_short_stack": float(short_stack_terms["gamma_ratio"]),
        "cooling_power_short_stack_eq7_raw": float(short_stack_terms["q_cold_eq7"]),
        "cooling_power": float(q_cold),
        "cooling_power_short_stack": float(q_cold_short_stack),
        "cooling_power_h2": float(q_cold_h2),
        "cooling_power_proxy": float(q_cold_proxy),
        "heating_power": float(q_hot),
        "acoustic_input_power": float(w_stack_absorbed),
        "acoustic_input_power_system": float(w_input_system),
        "cop": float(cop),
        "cop_system": float(cop_system),
        "cop_carnot": float(cop_carnot),
        "cop_relative": float(cop_relative),
        "stack_h2_hot": h2_hot,
        "stack_h2_cold": h2_cold,
        "stack_h2_components_hot": h2_hot_components,
        "stack_h2_components_cold": h2_cold_components,
        "stack_dT_dx": float(d_t_dx),
    }


def _tijani_short_stack_terms(
    *,
    p1_center: complex,
    config: StandingWaveRefrigeratorConfig,
    helium: gas.Helium,
    omega: float,
) -> dict[str, float]:
    """
    Estimate cooling power from Tijani et al. (2002) short-stack expression.

    Notes
    -----
    This is an analytical benchmark expression and assumes short-stack
    approximations. It is used as the primary refrigerator cooling metric
    because the current full H2 reconstruction is not energy-conservative.
    """
    t_m = 0.5 * (config.t_hot + config.t_cold)
    a = helium.sound_speed(t_m)
    gamma = helium.gamma(t_m)
    sigma = helium.prandtl(t_m)
    rho_m = helium.density(t_m)
    cp = helium.specific_heat_cp(t_m)
    kappa = helium.thermal_conductivity(t_m)

    d = abs(p1_center) / config.mean_pressure
    y0 = config.hydraulic_radius
    delta_kappa = np.sqrt(2.0 * kappa / (rho_m * cp * omega))
    delta_kappa_n = delta_kappa / y0
    b = config.porosity

    k = omega / a
    x_center = (
        config.left_duct_length
        + config.hot_hx_length
        + 0.5 * config.stack_length
    )
    x_n = k * x_center
    l_s_n = k * config.stack_length

    delta_t = config.t_hot - config.t_cold
    delta_t_n = delta_t / t_m

    sqrt_sigma = float(np.sqrt(sigma))
    k_term = 1.0 - sqrt_sigma * delta_kappa_n + 0.5 * sigma * delta_kappa_n**2
    if abs(k_term) < 1e-16 or abs(l_s_n) < 1e-16:
        return {
            "gamma_ratio": 0.0,
            "q_cold_eq7": 0.0,
            "cop_carnot": float(config.t_cold / max(config.t_hot - config.t_cold, 1e-12)),
        }

    gamma_term = (delta_t_n / ((gamma - 1.0) * b * l_s_n)) * np.tan(x_n)
    bracket = (
        gamma_term * (1.0 + sqrt_sigma + sigma) / (1.0 + sqrt_sigma)
        - 1.0
        + sqrt_sigma
        - sqrt_sigma * delta_kappa_n
    )
    q_cn = (
        -delta_kappa_n
        * d**2
        * np.sin(2.0 * x_n)
        / (8.0 * gamma * (1.0 + sigma) * k_term)
        * bracket
    )
    return {
        "gamma_ratio": float(gamma_term),
        "q_cold_eq7": float(q_cn * config.mean_pressure * a * config.duct_area),
        "cop_carnot": float(config.t_cold / max(config.t_hot - config.t_cold, 1e-12)),
    }


def tijani_cooling_power_short_stack(
    *,
    p1_center: complex,
    config: StandingWaveRefrigeratorConfig,
    helium: gas.Helium,
    omega: float,
    w_stack_absorbed: float,
) -> float:
    """
    Acoustic-power-based short-stack cooling estimate.

    Uses a short-stack Γ-based COP scaling and the solved acoustic power
    absorbed by the stack for robust energy-consistent cooling estimates.
    """
    terms = _tijani_short_stack_terms(
        p1_center=p1_center,
        config=config,
        helium=helium,
        omega=omega,
    )
    gamma_ratio = terms["gamma_ratio"]
    cop_carnot = terms["cop_carnot"]
    cop_stack_short = max(0.0, min(cop_carnot, cop_carnot * (1.0 - gamma_ratio)))
    return float(max(0.0, w_stack_absorbed) * cop_stack_short)


def tijani_acoustic_power_short_stack(
    solve_result: dict[str, Any],
) -> float:
    """
    Acoustic work input to stack for short-stack COP denominator.

    Uses solved acoustic-power absorption across the stack, which is fully
    consistent with the integrated mode shape.
    """
    return float(max(0.0, solve_result["w_stack_absorbed"]))


def compute_refrigerator_performance_short_stack(
    solve_result: dict[str, Any],
) -> dict[str, float]:
    """Return short-stack cooling/COP metrics from a solve result."""
    q_cold = float(solve_result["cooling_power_short_stack"])
    w_stack = tijani_acoustic_power_short_stack(solve_result)
    cop = q_cold / w_stack if w_stack > 0.0 else 0.0
    cop_carnot = float(solve_result["cop_carnot"])
    cop_relative = cop / cop_carnot if cop_carnot > 0.0 else 0.0
    return {
        "q_cold": q_cold,
        "w_stack": w_stack,
        "cop": cop,
        "cop_carnot": cop_carnot,
        "cop_relative": cop_relative,
    }


def compute_cooling_power(solve_result: dict[str, Any]) -> float:
    """Return cooling-power proxy from a solve result."""
    return float(solve_result["cooling_power"])


def compute_refrigerator_cop(solve_result: dict[str, Any]) -> dict[str, float]:
    """Return COP metrics from a solve result."""
    return {
        "cop": float(solve_result["cop"]),
        "cop_system": float(solve_result["cop_system"]),
        "cop_carnot": float(solve_result["cop_carnot"]),
        "cop_relative": float(solve_result["cop_relative"]),
    }


def sweep_drive_ratio(
    config: StandingWaveRefrigeratorConfig,
    drive_ratios: np.ndarray,
) -> list[dict[str, Any]]:
    """Sweep drive ratio using continuation in frequency and phase."""
    rows: list[dict[str, Any]] = []
    frequency_guess = None
    phase_guess = 0.0
    for drive in drive_ratios:
        point = solve_standing_wave_refrigerator(
            config,
            drive_ratio=float(drive),
            frequency_guess=frequency_guess,
            p1_phase_guess=phase_guess,
        )
        rows.append(point)
        frequency_guess = float(point["frequency_hz"])
        phase_guess = float(point["result"].guesses_final.get("p1_phase", phase_guess))
    return rows


def sweep_cold_temperature(
    config: StandingWaveRefrigeratorConfig,
    t_cold_values: np.ndarray,
) -> list[dict[str, Any]]:
    """Sweep cold-side temperature with continuation in frequency and phase."""
    rows: list[dict[str, Any]] = []
    frequency_guess = None
    phase_guess = 0.0
    for t_cold in t_cold_values:
        cfg = replace(config, t_cold=float(t_cold))
        point = solve_standing_wave_refrigerator(
            cfg,
            frequency_guess=frequency_guess,
            p1_phase_guess=phase_guess,
        )
        rows.append(point)
        frequency_guess = float(point["frequency_hz"])
        phase_guess = float(point["result"].guesses_final.get("p1_phase", phase_guess))
    return rows

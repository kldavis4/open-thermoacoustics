"""Stacked-screen heat exchanger segment (SX).

This module implements the SX segment from reference baseline, which models
a heat exchanger made of stacked wire mesh screens. This is commonly used
in Stirling engines and pulse-tube refrigerators.

The implementation follows reference baseline's formulation from equations 
in the published literature, which are identical to STKSCREEN but without temperature
gradient terms (isothermal assumption).

References
----------
[55] Gedeon, D., & Wood, J. G. (1996). JASA 100, 2130. (Referenced in reference baseline guide)
[27] Organ, A. J. (1992). Thermodynamics and Gas Dynamics of the Stirling Cycle Machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from openthermoacoustics.segments.base import Segment
from openthermoacoustics.utils import (
    complex_to_state,
    penetration_depth_thermal,
    state_to_complex,
)

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class ScreenHeatExchanger(Segment):
    """
    Stacked-screen heat exchanger segment (reference baseline's SX).

    Models a heat exchanger made of stacked wire mesh screens. The gas
    flows through the screens at a fixed solid temperature, with heat
    transfer determined by screen geometry and operating conditions.

    This implementation follows reference baseline's equations  which
    are valid when δ_ν, δ_κ >> r_h (the regenerator/small-pore limit).

    Parameters
    ----------
    length : float
        Axial length of the heat exchanger (m).
    porosity : float
        Volumetric porosity (void fraction), 0 < porosity < 1.
        For plain square-weave screen: ϕ ≈ 1 - π*m*d_wire/4
    hydraulic_radius : float
        Hydraulic radius of the mesh (m). For plain square-weave screen:
        rh ≈ d_wire * ϕ / (4 * (1 - ϕ))
    area : float
        Total cross-sectional area including solid (m^2).
    solid_temperature : float
        Fixed temperature of the solid screen material (K).
    heat_in : float, optional
        Heat added to the gas (W). Positive = heat flows from solid to gas.
        Default is 0.0. This is used for energy calculations.
    solid_heat_capacity : float, optional
        Volumetric heat capacity ρ_s * c_s of solid (J/(m³·K)).
        Default is 3.5e6 (copper: 8900 kg/m³ * 390 J/(kg·K)).
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    reference baseline's SX equations assume:
    - δ_ν, δ_κ >> r_h (regenerator limit)
    - Fairly good thermal contact between gas and solid
    - Porosity in range 0.60 < ϕ < 0.77 (empirical fits from Ref. [55])
    - Segment is short enough for single-step integration

    The output gas temperature is set equal to the solid temperature,
    which is an approximation valid for well-designed heat exchangers.
    """

    def __init__(
        self,
        length: float,
        porosity: float,
        hydraulic_radius: float,
        area: float,
        solid_temperature: float,
        heat_in: float = 0.0,
        solid_heat_capacity: float = 3.5e6,
        name: str = "",
    ) -> None:
        if not 0 < porosity < 1:
            raise ValueError(f"Porosity must be in (0, 1), got {porosity}")

        if solid_temperature <= 0:
            raise ValueError(f"Solid temperature must be positive, got {solid_temperature}")

        self._porosity = porosity
        self._hydraulic_radius = hydraulic_radius
        self._solid_temperature = solid_temperature
        self._heat_in = heat_in
        self._solid_heat_capacity = solid_heat_capacity

        super().__init__(name=name, length=length, area=area, geometry=None)

    @property
    def porosity(self) -> float:
        """Volumetric porosity (void fraction)."""
        return self._porosity

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius of the mesh (m)."""
        return self._hydraulic_radius

    @property
    def solid_temperature(self) -> float:
        """Fixed solid temperature (K)."""
        return self._solid_temperature

    @property
    def heat_in(self) -> float:
        """Heat added to gas (W). Positive = solid to gas."""
        return self._heat_in

    @staticmethod
    def _c1(phi: float) -> float:
        """Porosity-dependent viscous coefficient c1(ϕ), governing relations."""
        return 1268.0 - 3545.0 * phi + 2544.0 * phi**2

    @staticmethod
    def _c2(phi: float) -> float:
        """Porosity-dependent viscous coefficient c2(ϕ), governing relations."""
        return -2.82 + 10.7 * phi - 8.6 * phi**2

    @staticmethod
    def _b_phi(phi: float) -> float:
        """Porosity-dependent heat transfer coefficient b(ϕ), governing relations."""
        return 3.81 - 11.29 * phi + 9.47 * phi**2

    @staticmethod
    def _gc_integral(NR1: float) -> float:
        """
        Evaluate gc integral from governing relations.

        gc = (2/π) ∫₀^(π/2) dz / (1 + NR1^(3/5) * cos^(3/5)(z))
        """
        if NR1 < 1e-10:
            return 1.0

        def integrand(z: float) -> float:
            return 1.0 / (1.0 + NR1**0.6 * np.cos(z)**0.6)

        result, _ = quad(integrand, 0, np.pi / 2)
        return 2.0 / np.pi * result

    @staticmethod
    def _gv_integral(NR1: float) -> float:
        """
        Evaluate gv integral from governing relations.

        gv = -(2/π) ∫₀^(π/2) cos(2z) dz / (1 + NR1^(3/5) * cos^(3/5)(z))
        """
        if NR1 < 1e-10:
            return 0.0

        def integrand(z: float) -> float:
            return np.cos(2 * z) / (1.0 + NR1**0.6 * np.cos(z)**0.6)

        result, _ = quad(integrand, 0, np.pi / 2)
        return -2.0 / np.pi * result

    def _compute_parameters(
        self,
        omega: float,
        gas: Gas,
        T_m: float,
        u1_mag: float,
    ) -> dict:
        """
        Compute reference baseline SX parameters at given conditions.

        Returns dictionary with all intermediate parameters needed
        for the momentum and continuity equations.
        """
        phi = self._porosity
        rh = self._hydraulic_radius

        # Gas properties at local temperature
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        gamma = gas.gamma(T_m)
        mu = gas.viscosity(T_m)
        k_gas = gas.thermal_conductivity(T_m)
        cp = gas.specific_heat_cp(T_m)
        sigma = gas.prandtl(T_m)  # Prandtl number
        beta = 1.0 / T_m  # Thermal expansion coefficient for ideal gas

        # Penetration depths
        delta_kappa = penetration_depth_thermal(omega, rho_m, k_gas, cp)

        # Porosity-dependent coefficients, governing relations
        c1 = self._c1(phi)
        c2 = self._c2(phi)
        b = self._b_phi(phi)

        # Reynolds number based on oscillating velocity, governing relations
        NR1 = 4.0 * u1_mag * rh * rho_m / mu

        # Solid-to-gas heat capacity ratio, governing relations
        rho_s_cs = self._solid_heat_capacity
        eps_s = phi * rho_m * cp / ((1 - phi) * rho_s_cs)

        # Heat transfer number, governing relations
        eps_h = 8.0j * rh**2 / (b * sigma**(1.0/3.0) * delta_kappa**2)

        # gc and gv integrals, governing relations
        gc = self._gc_integral(NR1)
        gv = self._gv_integral(NR1)

        # Inertial correction factor for tortuous medium
        # From governing relations: (1 + (1-ϕ)²/(2*(2ϕ-1)))
        if phi > 0.5:
            inertial_factor = 1.0 + (1 - phi)**2 / (2.0 * (2.0 * phi - 1.0))
        else:
            inertial_factor = 1.0 + (1 - phi)**2 / 0.01

        return {
            "rho_m": rho_m,
            "a": a,
            "gamma": gamma,
            "mu": mu,
            "cp": cp,
            "sigma": sigma,
            "beta": beta,
            "c1": c1,
            "c2": c2,
            "NR1": NR1,
            "eps_s": eps_s,
            "eps_h": eps_h,
            "gc": gc,
            "gv": gv,
            "inertial_factor": inertial_factor,
            "delta_kappa": delta_kappa,
        }

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Calculate state derivatives for ODE integration.

        Implements reference baseline's SX equations .
        Unlike STKSCREEN, there is no temperature gradient term.

        Parameters
        ----------
        x : float
            Axial position within the segment (m).
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1)].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K). Uses solid temperature.

        Returns
        -------
        NDArray[np.float64]
            Derivative vector.
        """
        p1, U1 = state_to_complex(y)

        # Use solid temperature for calculations
        T_solid = self._solid_temperature

        phi = self._porosity
        rh = self._hydraulic_radius
        A = self._area

        # Gas area
        A_gas = phi * A

        # Spatial average velocity: <u1> = U1 / (ϕ*A)
        u1 = U1 / A_gas
        u1_mag = np.abs(u1)

        # Get all parameters
        params = self._compute_parameters(omega, gas, T_solid, u1_mag)

        rho_m = params["rho_m"]
        a = params["a"]
        gamma = params["gamma"]
        mu = params["mu"]
        cp = params["cp"]
        beta = params["beta"]
        c1 = params["c1"]
        c2 = params["c2"]
        NR1 = params["NR1"]
        eps_s = params["eps_s"]
        eps_h = params["eps_h"]
        gc = params["gc"]
        gv = params["gv"]
        inertial_factor = params["inertial_factor"]

        # Phase angles, governing relations
        theta_p = np.angle(u1) - np.angle(p1)
        theta_T = theta_p  # Approximation

        # ======================================================================
        # Momentum equation, governing relations
        # dp1/dx = -iωρm * inertial_factor * <u1> - (μ/rh²) * [c1/8 + c2*NR1/(3π)] * <u1>
        # ======================================================================
        viscous_coeff = mu / rh**2 * (c1 / 8.0 + c2 * NR1 / (3.0 * np.pi))
        dp1_dx = -1j * omega * rho_m * inertial_factor * u1 - viscous_coeff * u1

        # ======================================================================
        # Continuity equation, governing relations
        # d<u1>/dx = -iωγ/(ρm*a²)*p1 + iωTmβ²/(ρm*cp) * [thermal term] * p1
        # No temperature gradient term for isothermal heat exchanger
        # ======================================================================

        # Denominator term: 1 + ϵs + (gc + exp(2i*θT)*gv)*ϵh
        denom = 1.0 + eps_s + (gc + np.exp(2j * theta_T) * gv) * eps_h

        # Pressure term coefficient: (ϵs + (gc + exp(2i*θp)*gv)*ϵh) / denom
        p_term_num = eps_s + (gc + np.exp(2j * theta_p) * gv) * eps_h
        p_term = T_solid * beta**2 / (rho_m * cp) * p_term_num / denom

        # Full continuity equation (no temperature gradient term)
        du1_dx = (
            -1j * omega * gamma / (rho_m * a**2) * p1
            + 1j * omega * p_term * p1
        )

        # Convert from <u1> derivatives to U1 derivatives
        dU1_dx = A_gas * du1_dx

        return np.array([dp1_dx.real, dp1_dx.imag, dU1_dx.real, dU1_dx.imag])

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the screen heat exchanger.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m^3/s).
        T_m : float
            Mean temperature at input (K). Ignored - uses solid temperature.
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
            T_m_out equals the solid temperature.
        """
        if self._length == 0:
            return p1_in, U1_in, self._solid_temperature

        y0 = complex_to_state(p1_in, U1_in)

        def ode_func(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.get_derivatives(x, y, omega, gas, self._solid_temperature)

        # reference baseline uses a single RK4 step for SX (short segments)
        # We use RK45 with looser tolerances for efficiency
        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            ode_func,
            (0, self._length),
            y0,
            method="RK45",
            dense_output=False,
            rtol=1e-6,
            atol=1e-8,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        p1_out, U1_out = state_to_complex(sol.y[:, -1])

        # Output temperature is the solid temperature
        T_m_out = self._solid_temperature

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        return (
            f"ScreenHeatExchanger(name='{self._name}', length={self._length}, "
            f"porosity={self._porosity}, rh={self._hydraulic_radius}, "
            f"T_solid={self._solid_temperature})"
        )


# Alias for reference baseline naming convention
SX = ScreenHeatExchanger

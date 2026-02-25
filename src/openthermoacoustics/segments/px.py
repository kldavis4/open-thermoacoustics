"""Power-law heat exchanger segment (PX).

This module implements the PX segment from reference baseline, which models
a heat exchanger with user-specified power-law correlations for friction
factor and heat transfer coefficient.

This is a generalization of SX (screen heat exchanger) that allows any heat
exchanger geometry to be modeled given appropriate friction and heat transfer
correlations.

The correlations are the same as STKPOWERLW:
    f = f_con * N_R^(-f_exp)              (Fanning friction factor)
    N_St * σ^(2/3) = h_con * N_R^(-h_exp)  (Colburn j-factor)

References
----------
[1] published literature, relevant reference, governing relations with power-law parameters
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp, quad

from openthermoacoustics.segments.base import Segment
from openthermoacoustics.utils import (
    complex_to_state,
    penetration_depth_thermal,
    state_to_complex,
)

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class PowerLawHeatExchanger(Segment):
    """
    Power-law heat exchanger segment (reference baseline's PX).

    Models a heat exchanger with user-specified power-law correlations for
    friction factor and heat transfer coefficient. This is a generalization
    of SX for arbitrary heat exchanger geometries.

    Parameters
    ----------
    length : float
        Axial length of the heat exchanger (m).
    porosity : float
        Volumetric porosity (void fraction), 0 < porosity < 1.
    hydraulic_radius : float
        Hydraulic radius of the medium (m).
    area : float
        Total cross-sectional area including solid (m^2).
    solid_temperature : float
        Fixed temperature of the solid material (K).
    f_con : float
        Friction factor coefficient. f = f_con * N_R^(-f_exp)
    f_exp : float
        Friction factor exponent. Typically in range [0.5, 1.0].
    h_con : float
        Heat transfer coefficient. N_St * σ^(2/3) = h_con * N_R^(-h_exp)
    h_exp : float
        Heat transfer exponent. Typically in range [0.3, 0.8].
    heat_in : float, optional
        Heat added to the gas (W). Positive = heat flows from solid to gas.
        Default is 0.0. Used for energy calculations.
    solid_heat_capacity : float, optional
        Volumetric heat capacity ρ_s * c_s of solid (J/(m³·K)).
        Default is 3.5e6 (copper).
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    The power-law correlations are:

    - Friction factor (Fanning): f = f_con * N_R^(-f_exp)
    - Heat transfer (Colburn j-factor): N_St * σ^(2/3) = h_con * N_R^(-h_exp)

    PX is similar to SX but uses user-specified power-law coefficients
    instead of the empirical fits that are valid only for stacked screens.
    This allows modeling of arbitrary heat exchanger geometries such as:

    - Packed spheres
    - Metal foams
    - Finned tubes
    - Random fibers

    The gas temperature at the output is set to the solid temperature,
    which is valid for well-designed heat exchangers.
    """

    def __init__(
        self,
        length: float,
        porosity: float,
        hydraulic_radius: float,
        area: float,
        solid_temperature: float,
        f_con: float,
        f_exp: float,
        h_con: float,
        h_exp: float,
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
        self._f_con = f_con
        self._f_exp = f_exp
        self._h_con = h_con
        self._h_exp = h_exp
        self._heat_in = heat_in
        self._solid_heat_capacity = solid_heat_capacity

        # Precompute the integrals that depend only on exponents
        self._If = self._compute_If(f_exp)
        self._gc_factor = self._compute_gc_factor(h_exp)
        self._gv_factor = self._compute_gv_factor(h_exp)

        super().__init__(name=name, length=length, area=area, geometry=None)

    @property
    def porosity(self) -> float:
        """Volumetric porosity (void fraction)."""
        return self._porosity

    @property
    def hydraulic_radius(self) -> float:
        """Hydraulic radius of the medium (m)."""
        return self._hydraulic_radius

    @property
    def solid_temperature(self) -> float:
        """Fixed solid temperature (K)."""
        return self._solid_temperature

    @property
    def f_con(self) -> float:
        """Friction factor coefficient."""
        return self._f_con

    @property
    def f_exp(self) -> float:
        """Friction factor exponent."""
        return self._f_exp

    @property
    def h_con(self) -> float:
        """Heat transfer coefficient."""
        return self._h_con

    @property
    def h_exp(self) -> float:
        """Heat transfer exponent."""
        return self._h_exp

    @property
    def heat_in(self) -> float:
        """Heat added to gas (W). Positive = solid to gas."""
        return self._heat_in

    @staticmethod
    def _compute_If(f_exp: float) -> float:
        """
        Compute the friction integral I_f (same as STKPOWERLW).

        I_f = (2/π) ∫₀^π sin^(3-f_exp)(z) dz
        """
        def integrand(z: float) -> float:
            return np.sin(z) ** (3 - f_exp)

        result, _ = quad(integrand, 0, np.pi)
        return 2.0 / np.pi * result

    @staticmethod
    def _compute_gc_factor(h_exp: float) -> float:
        """
        Compute the gc integral factor.

        gc_factor = (2/π) ∫₀^(π/2) cos^(h_exp-1)(z) dz
        """
        def integrand(z: float) -> float:
            return np.cos(z) ** (h_exp - 1)

        result, _ = quad(integrand, 0, np.pi / 2)
        return 2.0 / np.pi * result

    @staticmethod
    def _compute_gv_factor(h_exp: float) -> float:
        """
        Compute the gv integral factor.

        gv_factor = -(2/π) ∫₀^(π/2) cos(2z) cos^(h_exp-1)(z) dz
        """
        def integrand(z: float) -> float:
            return np.cos(2 * z) * np.cos(z) ** (h_exp - 1)

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
        Compute PX parameters at given conditions.

        Returns dictionary with all intermediate parameters needed
        for the momentum and continuity equations.
        """
        phi = self._porosity
        rh = self._hydraulic_radius

        # Gas properties at solid temperature
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

        # Reynolds number based on oscillating velocity
        NR1 = 4.0 * u1_mag * rh * rho_m / mu

        # Solid-to-gas heat capacity ratio
        rho_s_cs = self._solid_heat_capacity
        eps_s = phi * rho_m * cp / ((1 - phi) * rho_s_cs) if phi < 1 else 0.0

        # Heat transfer number (using power-law b = h_con)
        b = self._h_con
        eps_h = 8.0j * rh**2 / (b * sigma**(1.0/3.0) * delta_kappa**2)

        # gc and gv from power-law formulas
        if NR1 > 1e-10:
            NR_power = NR1 ** (self._h_exp - 1)
        else:
            NR_power = 1.0
        gc = NR_power * self._gc_factor
        gv = NR_power * self._gv_factor

        return {
            "rho_m": rho_m,
            "a": a,
            "gamma": gamma,
            "mu": mu,
            "cp": cp,
            "sigma": sigma,
            "beta": beta,
            "NR1": NR1,
            "eps_s": eps_s,
            "eps_h": eps_h,
            "gc": gc,
            "gv": gv,
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
        NR1 = params["NR1"]
        eps_s = params["eps_s"]
        eps_h = params["eps_h"]
        gc = params["gc"]
        gv = params["gv"]

        # Phase angles
        theta_p = np.angle(u1) - np.angle(p1)
        theta_T = theta_p  # Approximation

        # ======================================================================
        # Momentum equation (power-law friction)
        # dp1/dx = -iωρm*<u1> - I_f * (μ/8rh²) * f_con * N_R,1^(1-f_exp) * <u1>
        # ======================================================================
        if NR1 > 1e-10:
            viscous_coeff = self._If * mu / (8.0 * rh**2) * self._f_con * NR1**(1 - self._f_exp)
        else:
            viscous_coeff = self._If * mu / (8.0 * rh**2) * self._f_con

        dp1_dx = -1j * omega * rho_m * u1 - viscous_coeff * u1

        # ======================================================================
        # Continuity equation (isothermal - no temperature gradient term)
        # d<u1>/dx = -iωγ/(ρm*a²)*p1 + iωTmβ²/(ρm*cp) * [thermal term] * p1
        # ======================================================================

        # Denominator term: 1 + ϵs + (gc + exp(2i*θT)*gv)*ϵh
        denom = 1.0 + eps_s + (gc + np.exp(2j * theta_T) * gv) * eps_h

        # Pressure term coefficient
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
        Propagate acoustic state through the power-law heat exchanger.

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
            f"PowerLawHeatExchanger(name='{self._name}', length={self._length}, "
            f"porosity={self._porosity}, rh={self._hydraulic_radius}, "
            f"T_solid={self._solid_temperature}, "
            f"f_con={self._f_con}, f_exp={self._f_exp}, "
            f"h_con={self._h_con}, h_exp={self._h_exp})"
        )


# reference baseline alias
PX = PowerLawHeatExchanger

"""Power-law porous medium stack/regenerator segment (STKPOWERLW).

This module implements the STKPOWERLW segment from reference baseline, which models
a porous medium with user-specified power-law correlations for friction
factor and heat transfer coefficient.

This is a generalization of STKSCREEN that allows any regenerator geometry
to be modeled given appropriate friction and heat transfer correlations.

The correlations are:
    f = f_con * N_R^(-f_exp)              (Fanning friction factor)
    N_St * σ^(2/3) = h_con * N_R^(-h_exp)  (Colburn j-factor)

where N_R = 4*|u1|*rh*ρ/(ϕ*A*μ) is the Reynolds number.

References
----------
[1] published literature, relevant reference, governing relations
[54] Kays, W. M., & London, A. L. (1984). Compact Heat Exchangers.
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


class StackPowerLaw(Segment):
    """
    Power-law porous medium stack/regenerator segment.

    Models a porous medium with user-specified power-law correlations for
    friction factor and heat transfer coefficient. This is a generalization
    of STKSCREEN for arbitrary regenerator geometries.

    Parameters
    ----------
    length : float
        Axial length of the regenerator (m).
    porosity : float
        Volumetric porosity (void fraction), 0 < porosity < 1.
    hydraulic_radius : float
        Hydraulic radius of the medium (m).
    area : float
        Total cross-sectional area including solid (m^2).
    f_con : float
        Friction factor coefficient. f = f_con * N_R^(-f_exp)
    f_exp : float
        Friction factor exponent. Typically in range [0.5, 1.0].
    h_con : float
        Heat transfer coefficient. N_St * σ^(2/3) = h_con * N_R^(-h_exp)
    h_exp : float
        Heat transfer exponent. Typically in range [0.3, 0.8].
    ks_frac : float, optional
        Fraction of solid thermal conductivity that contributes to
        axial heat conduction. Default 0.1.
    solid_heat_capacity : float, optional
        Volumetric heat capacity ρ_s * c_s of solid (J/(m³·K)).
        Default is 3.9e6 (stainless steel).
    solid_thermal_conductivity : float, optional
        Thermal conductivity of the solid material (W/(m·K)).
        Default is 15.0 (stainless steel).
    T_hot : float, optional
        Temperature at x=length (K). If provided with T_cold, creates
        a linear temperature profile.
    T_cold : float, optional
        Temperature at x=0 (K).
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    The power-law correlations are:

    - Friction factor (Fanning): f = f_con * N_R^(-f_exp)
    - Heat transfer (Colburn j-factor): N_St * σ^(2/3) = h_con * N_R^(-h_exp)

    Common parameter sets:

    - Packed spheres (Ergun): f_con ≈ 36, f_exp ≈ 1.0 (low Re)
    - Random fibers: f_con ≈ 24-48, f_exp ≈ 0.8-1.0
    - Etched foil: depends on geometry

    For stacked wire screens, use STKSCREEN instead, which has built-in
    empirical correlations valid for 0.60 < ϕ < 0.77.
    """

    def __init__(
        self,
        length: float,
        porosity: float,
        hydraulic_radius: float,
        area: float,
        f_con: float,
        f_exp: float,
        h_con: float,
        h_exp: float,
        ks_frac: float = 0.1,
        solid_heat_capacity: float = 3.9e6,
        solid_thermal_conductivity: float = 15.0,
        T_hot: float | None = None,
        T_cold: float | None = None,
        name: str = "",
    ) -> None:
        if not 0 < porosity < 1:
            raise ValueError(f"Porosity must be in (0, 1), got {porosity}")

        if (T_hot is None) != (T_cold is None):
            raise ValueError("Must provide both T_hot and T_cold, or neither")

        self._porosity = porosity
        self._hydraulic_radius = hydraulic_radius
        self._f_con = f_con
        self._f_exp = f_exp
        self._h_con = h_con
        self._h_exp = h_exp
        self._ks_frac = ks_frac
        self._solid_heat_capacity = solid_heat_capacity
        self._solid_thermal_conductivity = solid_thermal_conductivity
        self._T_hot = T_hot
        self._T_cold = T_cold

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
    def T_hot(self) -> float | None:
        """Temperature at x=length (K)."""
        return self._T_hot

    @property
    def T_cold(self) -> float | None:
        """Temperature at x=0 (K)."""
        return self._T_cold

    def temperature_at(self, x: float, T_m_input: float) -> float:
        """
        Calculate mean temperature at axial position x.

        Parameters
        ----------
        x : float
            Axial position (m), where 0 <= x <= length.
        T_m_input : float
            Input mean temperature (K), used if no gradient is imposed.

        Returns
        -------
        float
            Mean temperature at position x (K).
        """
        if self._T_hot is not None and self._T_cold is not None:
            if self._length == 0:
                return self._T_cold
            return self._T_cold + (self._T_hot - self._T_cold) * x / self._length
        return T_m_input

    def temperature_gradient(self) -> float:
        """
        Calculate the mean temperature gradient dT_m/dx.

        Returns
        -------
        float
            Temperature gradient in K/m. Zero if no gradient imposed.
        """
        if self._T_hot is not None and self._T_cold is not None and self._length > 0:
            return (self._T_hot - self._T_cold) / self._length
        return 0.0

    @staticmethod
    def _compute_If(f_exp: float) -> float:
        """
        Compute the friction integral I_f from governing relations.

        I_f = (2/π) ∫₀^π sin^(3-f_exp)(z) dz
        """
        def integrand(z: float) -> float:
            return np.sin(z) ** (3 - f_exp)

        result, _ = quad(integrand, 0, np.pi)
        return 2.0 / np.pi * result

    @staticmethod
    def _compute_gc_factor(h_exp: float) -> float:
        """
        Compute the gc integral factor from governing relations.

        gc_factor = (2/π) ∫₀^(π/2) cos^(h_exp-1)(z) dz

        Note: gc = N_R,1^(h_exp-1) * gc_factor
        """
        def integrand(z: float) -> float:
            return np.cos(z) ** (h_exp - 1)

        result, _ = quad(integrand, 0, np.pi / 2)
        return 2.0 / np.pi * result

    @staticmethod
    def _compute_gv_factor(h_exp: float) -> float:
        """
        Compute the gv integral factor from governing relations.

        gv_factor = -(2/π) ∫₀^(π/2) cos(2z) cos^(h_exp-1)(z) dz

        Note: gv = N_R,1^(h_exp-1) * gv_factor
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
        Compute STKPOWERLW parameters at given conditions.

        Returns dictionary with all the intermediate parameters needed
        for the momentum and continuity equations.
        """
        phi = self._porosity
        rh = self._hydraulic_radius

        # Gas properties
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

        # Reynolds number based on oscillating velocity, governing relations
        # Note: reference baseline uses 4*U*rh*ρ/(ϕ*A*μ) = 4*|u1|*rh*ρ/μ where u1 = U1/(ϕ*A)
        NR1 = 4.0 * u1_mag * rh * rho_m / mu

        # Solid-to-gas heat capacity ratio
        rho_s_cs = self._solid_heat_capacity
        eps_s = phi * rho_m * cp / ((1 - phi) * rho_s_cs) if phi < 1 else 0.0

        # Heat transfer number (using power-law b = h_con), governing relations
        # ϵh = 8i*rh² / (b*σ^(1/3)*δκ²)
        b = self._h_con
        eps_h = 8.0j * rh**2 / (b * sigma**(1.0/3.0) * delta_kappa**2)

        # gc and gv from power-law formulas, governing relations
        # gc = N_R,1^(h_exp-1) * gc_factor
        # gv = N_R,1^(h_exp-1) * gv_factor
        if NR1 > 1e-10:
            NR_power = NR1 ** (self._h_exp - 1)
        else:
            NR_power = 1.0
        gc = NR_power * self._gc_factor
        gv = NR_power * self._gv_factor

        # Effective solid thermal conductivity
        ks_eff = self._ks_frac * self._solid_thermal_conductivity

        return {
            "rho_m": rho_m,
            "a": a,
            "gamma": gamma,
            "mu": mu,
            "k_gas": k_gas,
            "cp": cp,
            "sigma": sigma,
            "beta": beta,
            "NR1": NR1,
            "eps_s": eps_s,
            "eps_h": eps_h,
            "gc": gc,
            "gv": gv,
            "ks_eff": ks_eff,
            "delta_kappa": delta_kappa,
        }

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m_input: float,
    ) -> NDArray[np.float64]:
        """
        Calculate state derivatives for ODE integration.

        Implements reference baseline's STKPOWERLW equations .

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
        T_m_input : float
            Input mean temperature (K).

        Returns
        -------
        NDArray[np.float64]
            Derivative vector.
        """
        p1, U1 = state_to_complex(y)

        # Local temperature
        T_m = self.temperature_at(x, T_m_input)

        # Temperature gradient
        dT_m_dx = self.temperature_gradient()

        phi = self._porosity
        rh = self._hydraulic_radius
        A = self._area

        # Gas area
        A_gas = phi * A

        # Spatial average velocity: <u1> = U1 / (ϕ*A)
        u1 = U1 / A_gas
        u1_mag = np.abs(u1)

        # Get all parameters
        params = self._compute_parameters(omega, gas, T_m, u1_mag)

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
        theta_T = theta_p  # First approximation

        # ======================================================================
        # Momentum equation, governing relations
        # dp1/dx = -iωρm*<u1> - I_f * (μ/8rh²) * f_con * N_R,1^(1-f_exp) * <u1>
        # ======================================================================
        if NR1 > 1e-10:
            viscous_coeff = self._If * mu / (8.0 * rh**2) * self._f_con * NR1**(1 - self._f_exp)
        else:
            # Low Reynolds number limit: use laminar-like behavior
            viscous_coeff = self._If * mu / (8.0 * rh**2) * self._f_con

        dp1_dx = -1j * omega * rho_m * u1 - viscous_coeff * u1

        # ======================================================================
        # Continuity equation (same form as STKSCREEN with power-law gc, gv)
        # d<u1>/dx = -iωγ/(ρm*a²)*p1 + β*(dTm/dx)*<u1>
        #            + iωβ * [complex thermal term]
        # ======================================================================

        # Denominator term: 1 + ϵs + (gc + exp(2i*θT)*gv)*ϵh
        denom = 1.0 + eps_s + (gc + np.exp(2j * theta_T) * gv) * eps_h

        # Pressure term coefficient
        p_term_num = eps_s + (gc + np.exp(2j * theta_p) * gv) * eps_h
        p_term = T_m * beta / (rho_m * cp) * p_term_num / denom

        # Velocity term coefficient from temperature gradient
        u_term_num = eps_s + (gc - gv) * eps_h
        u_term = (1.0 / (1j * omega)) * (dT_m_dx / T_m) * u_term_num / denom if dT_m_dx != 0 else 0.0

        # Full continuity equation
        du1_dx = (
            -1j * omega * gamma / (rho_m * a**2) * p1
            + beta * dT_m_dx * u1
            + 1j * omega * beta * (p_term * p1 - u_term * u1)
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
        Propagate acoustic state through the power-law porous medium.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m^3/s).
        T_m : float
            Mean temperature at input (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out).
        """
        if self._length == 0:
            return p1_in, U1_in, T_m

        y0 = complex_to_state(p1_in, U1_in)

        def ode_func(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.get_derivatives(x, y, omega, gas, T_m)

        sol = solve_ivp(
            ode_func,
            (0, self._length),
            y0,
            method="RK45",
            dense_output=False,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        p1_out, U1_out = state_to_complex(sol.y[:, -1])
        T_m_out = self.temperature_at(self._length, T_m)

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        temp_info = ""
        if self._T_hot is not None:
            temp_info = f", T_cold={self._T_cold}, T_hot={self._T_hot}"
        return (
            f"StackPowerLaw(name='{self._name}', length={self._length}, "
            f"porosity={self._porosity}, rh={self._hydraulic_radius}, "
            f"f_con={self._f_con}, f_exp={self._f_exp}, "
            f"h_con={self._h_con}, h_exp={self._h_exp}"
            f"{temp_info})"
        )


# reference baseline alias
STKPOWERLW = StackPowerLaw

"""Wire mesh screen stack/regenerator segment (STKSCREEN).

This module implements the STKSCREEN segment from reference baseline, which models
a regenerator made of stacked wire mesh screens. This is commonly used
in Stirling engines and thermoacoustic Stirling heat engines (TASHE).

The implementation follows reference baseline's formulation from equations 
in the published literature, which assumes viscous and thermal penetration depths are
much larger than the hydraulic radius (the "regenerator limit").

References
----------
[55] Gedeon, D., & Wood, J. G. (1996). JASA 100, 2130. (Referenced in reference baseline guide)
[27] Organ, A. J. (1992). Thermodynamics and Gas Dynamics of the Stirling Cycle Machine.
[28] Lewis, M. A., et al. (1999). Cryogenics 39, 215.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from openthermoacoustics.segments.base import Segment
from openthermoacoustics.utils import (
    complex_to_state,
    penetration_depth_thermal,
    penetration_depth_viscous,
    state_to_complex,
)

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class StackScreen(Segment):
    """
    Wire mesh screen stack/regenerator segment.

    Models a regenerator made of stacked wire mesh screens, as used in
    Stirling engines and thermoacoustic Stirling heat engines (TASHE).

    This implementation follows reference baseline's equations  which
    are valid when δ_ν, δ_κ >> r_h (the regenerator/small-pore limit).

    Parameters
    ----------
    length : float
        Axial length of the regenerator (m).
    porosity : float
        Volumetric porosity (void fraction), 0 < porosity < 1.
        For plain square-weave screen: ϕ ≈ 1 - π*m*d_wire/4
        where m is mesh number and d_wire is wire diameter.
    hydraulic_radius : float
        Hydraulic radius of the mesh (m). For plain square-weave screen:
        rh ≈ d_wire * ϕ / (4 * (1 - ϕ))
    area : float
        Total cross-sectional area including solid (m^2).
    ks_frac : float, optional
        Fraction of solid thermal conductivity that contributes to
        axial heat conduction. Accounts for poor thermal contact between
        adjacent screen layers. Default 0.1 (recommended by Lewis et al.
        for metal screens).
    solid_heat_capacity : float, optional
        Volumetric heat capacity ρ_s * c_s of solid (J/(m³·K)).
        Default is 3.9e6 (stainless steel: 8000 kg/m³ * 500 J/(kg·K)).
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
    reference baseline's STKSCREEN equations assume:
    - δ_ν, δ_κ >> r_h (regenerator limit)
    - Reasonably good internal thermal contact
    - Porosity in range 0.60 < ϕ < 0.77 (empirical fits from Ref. [55])

    The momentum equation uses porosity-dependent coefficients c1(ϕ), c2(ϕ)
    derived from steady-flow friction factor correlations for woven screens.

    The continuity equation includes complex terms for phase-dependent
    heat transfer using gc and gv integrals.
    """

    def __init__(
        self,
        length: float,
        porosity: float,
        hydraulic_radius: float,
        area: float,
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
        self._ks_frac = ks_frac
        self._solid_heat_capacity = solid_heat_capacity
        self._solid_thermal_conductivity = solid_thermal_conductivity
        self._T_hot = T_hot
        self._T_cold = T_cold

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
    def ks_frac(self) -> float:
        """Fraction of solid thermal conductivity for axial conduction."""
        return self._ks_frac

    @property
    def solid_thermal_conductivity(self) -> float:
        """Thermal conductivity of solid material (W/(m·K))."""
        return self._solid_thermal_conductivity

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
            # Linear interpolation: T_cold at x=0, T_hot at x=length
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

        Uses numerical integration for accuracy.
        """
        from scipy.integrate import quad

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

        Uses numerical integration for accuracy.
        """
        from scipy.integrate import quad

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
        Compute reference baseline STKSCREEN parameters at given conditions.

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

        # Porosity-dependent coefficients, governing relations
        c1 = self._c1(phi)
        c2 = self._c2(phi)
        b = self._b_phi(phi)

        # Reynolds number based on oscillating velocity, governing relations
        NR1 = 4.0 * u1_mag * rh * rho_m / mu

        # Solid-to-gas heat capacity ratio, governing relations
        # ϵs = ϕ*ρm*cp / ((1-ϕ)*ρs*cs)
        rho_s_cs = self._solid_heat_capacity
        eps_s = phi * rho_m * cp / ((1 - phi) * rho_s_cs)

        # Heat transfer number, governing relations
        # ϵh = 8i*rh² / (b(ϕ)*σ^(1/3)*δκ²)
        eps_h = 8.0j * rh**2 / (b * sigma**(1.0/3.0) * delta_kappa**2)

        # gc and gv integrals, governing relations
        gc = self._gc_integral(NR1)
        gv = self._gv_integral(NR1)

        # Inertial correction factor for tortuous medium
        # From governing relations: (1 + (1-ϕ)²/(2*(2ϕ-1)))
        if phi > 0.5:
            inertial_factor = 1.0 + (1 - phi)**2 / (2.0 * (2.0 * phi - 1.0))
        else:
            # Avoid division by zero for low porosity
            inertial_factor = 1.0 + (1 - phi)**2 / 0.01

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
            "c1": c1,
            "c2": c2,
            "b": b,
            "NR1": NR1,
            "eps_s": eps_s,
            "eps_h": eps_h,
            "gc": gc,
            "gv": gv,
            "inertial_factor": inertial_factor,
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

        Implements reference baseline's STKSCREEN equations .

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

        # Temperature gradient (constant for linear profile)
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
        sigma = params["sigma"]
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

        # For theta_T, we need <T>_u,1 which requires iteration
        # As a first approximation, use theta_T ≈ theta_p
        theta_T = theta_p

        # ======================================================================
        # Momentum equation, governing relations
        # dp1/dx = -iωρm * inertial_factor * <u1> - (μ/rh²) * [c1/8 + c2*NR1/(3π)] * <u1>
        # ======================================================================
        viscous_coeff = mu / rh**2 * (c1 / 8.0 + c2 * NR1 / (3.0 * np.pi))
        dp1_dx = -1j * omega * rho_m * inertial_factor * u1 - viscous_coeff * u1

        # ======================================================================
        # Continuity equation, governing relations
        # d<u1>/dx = -iωγ/(ρm*a²)*p1 + β*(dTm/dx)*<u1>
        #            + iωβ * [complex thermal term]
        # ======================================================================

        # Denominator term: 1 + ϵs + (gc + exp(2i*θT)*gv)*ϵh
        denom = 1.0 + eps_s + (gc + np.exp(2j * theta_T) * gv) * eps_h

        # Pressure term coefficient: (ϵs + (gc + exp(2i*θp)*gv)*ϵh) / denom
        p_term_num = eps_s + (gc + np.exp(2j * theta_p) * gv) * eps_h
        p_term = T_m * beta / (rho_m * cp) * p_term_num / denom

        # Velocity term coefficient from temperature gradient
        # (ϵs + (gc - gv)*ϵh) / denom
        u_term_num = eps_s + (gc - gv) * eps_h
        u_term = (1.0 / (1j * omega)) * (dT_m_dx / T_m) * u_term_num / denom if dT_m_dx != 0 else 0.0

        # Full continuity equation
        du1_dx = (
            -1j * omega * gamma / (rho_m * a**2) * p1
            + beta * dT_m_dx * u1
            + 1j * omega * beta * (p_term * p1 - u_term * u1)
        )

        # Convert from <u1> derivatives to U1 derivatives
        # U1 = A_gas * <u1>, so dU1/dx = A_gas * d<u1>/dx
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
        Propagate acoustic state through the wire mesh regenerator.

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
            f"StackScreen(name='{self._name}', length={self._length}, "
            f"porosity={self._porosity}, rh={self._hydraulic_radius}"
            f"{temp_info})"
        )

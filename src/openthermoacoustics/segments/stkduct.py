"""Boundary-layer stack/pulse tube segment (STKDUCT).

This module implements the STKDUCT segment from reference baseline, which models
a "boundary-layer stack" used for pulse tubes and thermal buffer tubes.
This segment is used when pore size is much greater than thermal and
viscous penetration depths (δ << r).

The implementation follows reference baseline's equations  using
boundary-layer thermoviscous functions.

References
----------
- Swift, G. W. (2002). Thermoacoustics: A unifying perspective.
- Rott, N. (1980). Advances in Applied Mechanics 20.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.special import jv  # Bessel function of the first kind

from openthermoacoustics.segments.base import Segment
from openthermoacoustics.utils import (
    complex_to_state,
    penetration_depth_thermal,
    penetration_depth_viscous,
    state_to_complex,
)

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class StackDuct(Segment):
    """
    Boundary-layer stack / pulse tube segment (reference baseline's STKDUCT).

    Models a duct or tube with lateral dimensions much larger than the
    thermal and viscous penetration depths. Used primarily for:
    - Pulse tubes in pulse-tube refrigerators
    - Thermal buffer tubes in thermoacoustic Stirling engines/refrigerators

    Uses boundary-layer approximation for thermoviscous functions:
        f_j = (1 - i) * Π * δ_j / (2 * A)

    Parameters
    ----------
    length : float
        Axial length of the duct (m).
    area : float
        Cross-sectional area available to gas (m²).
    perimeter : float
        Inside perimeter of the cross-section (m).
    wall_area : float, optional
        Cross-sectional area of wall material (m²). Used for thermal
        conductivity along x. Default is 0.
    solid_thermal_conductivity : float, optional
        Thermal conductivity of wall material (W/(m·K)).
        Default is 15.0 (stainless steel).
    solid_heat_capacity : float, optional
        Volumetric heat capacity ρ_s * c_s of wall (J/(m³·K)).
        Default is 3.9e6 (stainless steel).
    T_hot : float, optional
        Temperature at x=length (K). If provided with T_cold, creates
        a linear temperature profile.
    T_cold : float, optional
        Temperature at x=0 (K).
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    STKDUCT uses the boundary-layer approximation which is valid when
    2A/(Π*δ) > 30. For smaller values, reference baseline transitions to circular
    pipe functions (like STKCIRC).

    The temperature profile can evolve according to energy conservation,
    but for stability, an imposed linear profile is recommended when
    the temperature endpoints are known.

    Examples
    --------
    >>> from openthermoacoustics.segments import StackDuct
    >>> from openthermoacoustics.gas import Helium
    >>> # Circular pulse tube: D = 0.02 m, L = 0.1 m
    >>> pulse_tube = StackDuct(
    ...     length=0.1,
    ...     area=np.pi * 0.01**2,  # π * r²
    ...     perimeter=2 * np.pi * 0.01,  # 2π * r
    ...     wall_area=np.pi * ((0.012)**2 - (0.01)**2),  # annulus
    ...     T_cold=300.0,
    ...     T_hot=80.0,
    ... )
    """

    def __init__(
        self,
        length: float,
        area: float,
        perimeter: float,
        wall_area: float = 0.0,
        solid_thermal_conductivity: float = 15.0,
        solid_heat_capacity: float = 3.9e6,
        T_hot: float | None = None,
        T_cold: float | None = None,
        name: str = "",
    ) -> None:
        if area <= 0:
            raise ValueError(f"Area must be positive, got {area}")
        if perimeter <= 0:
            raise ValueError(f"Perimeter must be positive, got {perimeter}")
        if wall_area < 0:
            raise ValueError(f"Wall area must be non-negative, got {wall_area}")

        if (T_hot is None) != (T_cold is None):
            raise ValueError("Must provide both T_hot and T_cold, or neither")

        self._perimeter = perimeter
        self._wall_area = wall_area
        self._solid_thermal_conductivity = solid_thermal_conductivity
        self._solid_heat_capacity = solid_heat_capacity
        self._T_hot = T_hot
        self._T_cold = T_cold

        super().__init__(name=name, length=length, area=area, geometry=None)

    @property
    def perimeter(self) -> float:
        """Inside perimeter (m)."""
        return self._perimeter

    @property
    def wall_area(self) -> float:
        """Wall cross-sectional area (m²)."""
        return self._wall_area

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

    def _compute_thermoviscous_functions(
        self,
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> tuple[complex, complex, complex]:
        """
        Compute the boundary-layer thermoviscous functions.

        Returns f_nu, f_kappa, and epsilon_s.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Local mean temperature (K).

        Returns
        -------
        tuple[complex, complex, complex]
            Tuple of (f_nu, f_kappa, epsilon_s).
        """
        rho_m = gas.density(T_m)
        mu = gas.viscosity(T_m)
        k_gas = gas.thermal_conductivity(T_m)
        cp = gas.specific_heat_cp(T_m)

        # Penetration depths
        delta_nu = penetration_depth_viscous(omega, rho_m, mu)
        delta_kappa = penetration_depth_thermal(omega, rho_m, k_gas, cp)

        A = self._area
        Pi = self._perimeter

        # Check if boundary layer approximation is valid
        # Valid when 2A/(Π*δ) > 30
        ratio_nu = 2 * A / (Pi * delta_nu)
        ratio_kappa = 2 * A / (Pi * delta_kappa)

        if ratio_nu > 30 and ratio_kappa > 30:
            # Boundary layer approximation, governing relations
            f_nu = (1 - 1j) * Pi * delta_nu / (2 * A)
            f_kappa = (1 - 1j) * Pi * delta_kappa / (2 * A)
        elif ratio_nu < 25 or ratio_kappa < 25:
            # Use circular pipe functions (like STKCIRC)
            r0 = 2 * A / Pi  # Equivalent radius

            # Circular pore functions
            z_nu = r0 * (1j - 1) / delta_nu
            z_kappa = r0 * (1j - 1) / delta_kappa

            f_nu = 2 * jv(1, z_nu) / (z_nu * jv(0, z_nu))
            f_kappa = 2 * jv(1, z_kappa) / (z_kappa * jv(0, z_kappa))
        else:
            # Linear interpolation between 25 and 30
            weight = (min(ratio_nu, ratio_kappa) - 25) / 5

            # Boundary layer values
            f_nu_bl = (1 - 1j) * Pi * delta_nu / (2 * A)
            f_kappa_bl = (1 - 1j) * Pi * delta_kappa / (2 * A)

            # Circular pipe values
            r0 = 2 * A / Pi
            z_nu = r0 * (1j - 1) / delta_nu
            z_kappa = r0 * (1j - 1) / delta_kappa
            f_nu_circ = 2 * jv(1, z_nu) / (z_nu * jv(0, z_nu))
            f_kappa_circ = 2 * jv(1, z_kappa) / (z_kappa * jv(0, z_kappa))

            f_nu = weight * f_nu_bl + (1 - weight) * f_nu_circ
            f_kappa = weight * f_kappa_bl + (1 - weight) * f_kappa_circ

        # Solid correction factor epsilon_s, governing relations
        # ε_s = sqrt(k*ρm*cp / (ks*ρs*cs)) / tanh((1+i)*ℓ/δs)
        # where ℓ = wall cross-sect area / perimeter
        if self._wall_area > 0 and self._solid_heat_capacity > 0:
            ks = self._solid_thermal_conductivity
            rho_s_cs = self._solid_heat_capacity

            # Wall thickness parameter
            ell = self._wall_area / Pi

            # Solid thermal diffusivity
            alpha_s = ks / rho_s_cs

            # Solid penetration depth
            delta_s = np.sqrt(2 * alpha_s / omega)

            # Epsilon_s
            z_s = (1 + 1j) * ell / delta_s
            if np.abs(z_s) > 0.01:
                epsilon_s = np.sqrt(k_gas * rho_m * cp / (ks * rho_s_cs)) / np.tanh(z_s)
            else:
                # Small argument limit
                epsilon_s = np.sqrt(k_gas * rho_m * cp / (ks * rho_s_cs)) * delta_s / ((1 + 1j) * ell)
        else:
            epsilon_s = 0.0

        return complex(f_nu), complex(f_kappa), complex(epsilon_s)

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

        Implements reference baseline's STKDUCT equations .

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

        # Gas properties
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        gamma = gas.gamma(T_m)
        sigma = gas.prandtl(T_m)
        beta = 1.0 / T_m  # Ideal gas

        A_gas = self._area

        # Thermoviscous functions
        f_nu, f_kappa, epsilon_s = self._compute_thermoviscous_functions(omega, gas, T_m)

        # Momentum equation, governing relations
        # dp1/dx = -iωρm / ((1 - f_nu) * A_gas) * U1
        dp1_dx = -1j * omega * rho_m / ((1 - f_nu) * A_gas) * U1

        # Continuity equation, governing relations
        # dU1/dx = -iωA_gas/(ρm*a²) * [1 + (γ-1)*f_κ/(1+ε_s)] * p1
        #          + β*(f_κ - f_ν)/((1-f_ν)*(1-σ)*(1+ε_s)) * dTm/dx * U1
        compressibility_term = 1.0 + (gamma - 1) * f_kappa / (1 + epsilon_s)

        dU1_dx = -1j * omega * A_gas / (rho_m * a**2) * compressibility_term * p1

        # Temperature gradient term
        if dT_m_dx != 0:
            temp_grad_coeff = beta * (f_kappa - f_nu) / ((1 - f_nu) * (1 - sigma) * (1 + epsilon_s))
            dU1_dx += temp_grad_coeff * dT_m_dx * U1

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
        Propagate acoustic state through the STKDUCT.

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
            f"StackDuct(name='{self._name}', length={self._length}, "
            f"area={self._area}, perimeter={self._perimeter}"
            f"{temp_info})"
        )


# Alias for reference baseline naming convention
STKDUCT = StackDuct

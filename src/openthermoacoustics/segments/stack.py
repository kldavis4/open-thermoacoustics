"""Stack/regenerator segment for thermoacoustic heat engines and refrigerators."""

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
    from openthermoacoustics.geometry.base import Geometry


class Stack(Segment):
    """
    Stack/regenerator segment with temperature gradient capability.

    The stack is the heart of a thermoacoustic device, where the interaction
    between oscillating gas and a temperature gradient enables energy conversion.
    This segment implements the full thermoacoustic equations including the
    temperature gradient term in the continuity equation.

    The governing equations are:
        dp1/dx = -(j*omega*rho_m / (A*porosity*(1 - f_nu))) * U1

        dU1/dx = -(j*omega*A*porosity / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1
                 + (f_kappa - f_nu) / ((1 - f_nu)*(1 - sigma)) * (dT_m/dx / T_m) * U1

    where sigma is the Prandtl number and the temperature gradient term enables
    thermoacoustic power conversion.

    Parameters
    ----------
    length : float
        Axial length of the stack (m).
    porosity : float
        Porosity (void fraction) of the stack, 0 < porosity < 1.
    hydraulic_radius : float
        Hydraulic radius of the pores (m).
    geometry : Geometry, optional
        Pore geometry for thermoviscous functions.
    solid_thermal_conductivity : float, optional
        Thermal conductivity of the solid stack material (W/(m*K)).
        Default is 0.0 (negligible solid conduction).
    T_hot : float, optional
        Hot-side temperature (K). If provided with T_cold, creates a
        linear temperature profile.
    T_cold : float, optional
        Cold-side temperature (K). If provided with T_hot, creates a
        linear temperature profile.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    porosity : float
        Porosity of the stack.
    hydraulic_radius : float
        Hydraulic radius of the pores (m).
    solid_thermal_conductivity : float
        Thermal conductivity of the solid material (W/(m*K)).
    T_hot : float or None
        Hot-side temperature for imposed gradient (K).
    T_cold : float or None
        Cold-side temperature for imposed gradient (K).

    Notes
    -----
    The temperature profile can be:
    1. Uniform (T_hot and T_cold not specified): Uses input T_m throughout
    2. Linear (T_hot and T_cold specified): Linear interpolation from T_cold at x=0
       to T_hot at x=length

    Examples
    --------
    >>> from openthermoacoustics.segments import Stack
    >>> from openthermoacoustics.gas import Helium
    >>> stack = Stack(
    ...     length=0.05, porosity=0.7, hydraulic_radius=0.0005,
    ...     T_hot=500.0, T_cold=300.0
    ... )
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = stack.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        length: float,
        porosity: float,
        hydraulic_radius: float,
        area: float | None = None,
        geometry: Geometry | None = None,
        solid_thermal_conductivity: float = 0.0,
        T_hot: float | None = None,
        T_cold: float | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize a stack segment.

        Parameters
        ----------
        length : float
            Axial length of the stack (m).
        porosity : float
            Porosity (void fraction), 0 < porosity < 1.
        hydraulic_radius : float
            Hydraulic radius of the pores (m).
        area : float, optional
            Total cross-sectional area (m^2). If not provided, defaults to
            a single-pore estimate. For realistic simulations, this should
            be provided explicitly.
        geometry : Geometry, optional
            Pore geometry for thermoviscous functions.
        solid_thermal_conductivity : float, optional
            Thermal conductivity of solid material (W/(m*K)). Default is 0.0.
        T_hot : float, optional
            Hot-side temperature (K).
        T_cold : float, optional
            Cold-side temperature (K).
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If porosity is not in (0, 1) or if only one of T_hot/T_cold is provided.
        """
        if not 0 < porosity < 1:
            raise ValueError(f"Porosity must be in (0, 1), got {porosity}")

        if (T_hot is None) != (T_cold is None):
            raise ValueError("Must provide both T_hot and T_cold, or neither")

        self._porosity = porosity
        self._hydraulic_radius = hydraulic_radius
        self._solid_thermal_conductivity = solid_thermal_conductivity
        self._T_hot = T_hot
        self._T_cold = T_cold

        # Reference area (actual flow area is porosity * area)
        if area is None:
            # Default: single-pore estimate (for compatibility/simple cases)
            area = np.pi * hydraulic_radius**2 / porosity

        super().__init__(name=name, length=length, area=area, geometry=geometry)

    @property
    def porosity(self) -> float:
        """
        Porosity (void fraction) of the stack.

        Returns
        -------
        float
            Porosity, dimensionless, in range (0, 1).
        """
        return self._porosity

    @property
    def hydraulic_radius(self) -> float:
        """
        Hydraulic radius of the pores.

        Returns
        -------
        float
            Hydraulic radius in meters.
        """
        return self._hydraulic_radius

    @property
    def solid_thermal_conductivity(self) -> float:
        """
        Thermal conductivity of the solid stack material.

        Returns
        -------
        float
            Thermal conductivity in W/(m*K).
        """
        return self._solid_thermal_conductivity

    @property
    def T_hot(self) -> float | None:
        """
        Hot-side temperature for imposed gradient.

        Returns
        -------
        float or None
            Temperature in K, or None if no gradient is imposed.
        """
        return self._T_hot

    @property
    def T_cold(self) -> float | None:
        """
        Cold-side temperature for imposed gradient.

        Returns
        -------
        float or None
            Temperature in K, or None if no gradient is imposed.
        """
        return self._T_cold

    def temperature_at(self, x: float, T_m_input: float) -> float:
        """
        Calculate mean temperature at axial position x.

        If T_hot and T_cold are specified, returns a linear interpolation.
        Otherwise, returns the input temperature T_m_input.

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
            Temperature gradient in K/m. Zero if no gradient is imposed.
        """
        if self._T_hot is not None and self._T_cold is not None and self._length > 0:
            return (self._T_hot - self._T_cold) / self._length
        return 0.0

    def _compute_thermoviscous_functions(
        self,
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> tuple[complex, complex]:
        """
        Compute the thermoviscous functions f_nu and f_kappa.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        tuple[complex, complex]
            Tuple of (f_nu, f_kappa).
        """
        rho_m = gas.density(T_m)
        mu = gas.viscosity(T_m)
        kappa = gas.thermal_conductivity(T_m)
        cp = gas.specific_heat_cp(T_m)

        delta_nu = penetration_depth_viscous(omega, rho_m, mu)
        delta_kappa = penetration_depth_thermal(omega, rho_m, kappa, cp)

        if self._geometry is not None:
            f_nu = self._geometry.f_nu(omega, delta_nu, self._hydraulic_radius)
            f_kappa = self._geometry.f_kappa(omega, delta_kappa, self._hydraulic_radius)
        else:
            # Boundary layer approximation
            f_nu = (1 - 1j) * delta_nu / self._hydraulic_radius
            f_kappa = (1 - 1j) * delta_kappa / self._hydraulic_radius

        return complex(f_nu), complex(f_kappa)

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

        Implements the full thermoacoustic equations including the
        temperature gradient term:

            dp1/dx = -(j*omega*rho_m / (A_eff*(1 - f_nu))) * U1

            dU1/dx = -(j*omega*A_eff / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1
                     + (f_kappa - f_nu) / ((1 - f_nu)*(1 - sigma)) * (dT_m/dx / T_m) * U1

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
            Input mean temperature (K), used if no gradient is imposed.

        Returns
        -------
        NDArray[np.float64]
            Derivative vector [d(Re(p1))/dx, d(Im(p1))/dx,
                               d(Re(U1))/dx, d(Im(U1))/dx].
        """
        p1, U1 = state_to_complex(y)

        # Local temperature
        T_m = self.temperature_at(x, T_m_input)

        # Gas properties at local temperature
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        gamma = gas.gamma(T_m)
        sigma = gas.prandtl(T_m)

        # Effective flow area (porosity * total area)
        A_eff = self._porosity * self._area

        # Thermoviscous functions
        f_nu, f_kappa = self._compute_thermoviscous_functions(omega, gas, T_m)

        # Temperature gradient
        dT_m_dx = self.temperature_gradient()

        # Momentum equation
        dp1_dx = -1j * omega * rho_m / (A_eff * (1 - f_nu)) * U1

        # Continuity equation with temperature gradient term
        dU1_dx = -1j * omega * A_eff / (rho_m * a**2) * (1 + (gamma - 1) * f_kappa) * p1

        # Add temperature gradient term if gradient exists
        if dT_m_dx != 0:
            gradient_term = (f_kappa - f_nu) / ((1 - f_nu) * (1 - sigma)) * (dT_m_dx / T_m) * U1
            dU1_dx += gradient_term

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
        Propagate acoustic state through the stack.

        Uses scipy.integrate.solve_ivp with RK45 method to integrate
        the governing ODEs from x=0 to x=length.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m^3/s).
        T_m : float
            Mean temperature at input (K). Overridden by T_cold if gradient is imposed.
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out):
            - p1_out: Complex pressure amplitude at output (Pa)
            - U1_out: Complex volumetric velocity amplitude at output (m^3/s)
            - T_m_out: Mean temperature at output (K)
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

        # Output temperature
        T_m_out = self.temperature_at(self._length, T_m)

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        """Return string representation of the stack."""
        temp_info = ""
        if self._T_hot is not None:
            temp_info = f", T_cold={self._T_cold}, T_hot={self._T_hot}"
        return (
            f"Stack(name='{self._name}', length={self._length}, "
            f"porosity={self._porosity}, hydraulic_radius={self._hydraulic_radius}"
            f"{temp_info})"
        )

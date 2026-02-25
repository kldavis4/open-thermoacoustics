"""Duct segment for uniform circular tubes with isothermal walls."""

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


class Duct(Segment):
    """
    Duct segment representing a uniform circular tube with isothermal walls.

    The duct propagates acoustic waves according to the lossy momentum and
    continuity equations, accounting for viscous and thermal losses through
    the thermoviscous functions f_nu and f_kappa.

    The governing equations are:
        dp1/dx = -(j*omega*rho_m / (A*(1 - f_nu))) * U1
        dU1/dx = -(j*omega*A / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1

    For a circular tube with isothermal walls, there is no mean temperature
    gradient (dT_m/dx = 0).

    Parameters
    ----------
    length : float
        Axial length of the duct (m).
    radius : float
        Inner radius of the duct (m).
    geometry : Geometry, optional
        Pore geometry for thermoviscous functions. Defaults to CircularPore
        if not provided (when available).
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    radius : float
        Inner radius of the duct (m).
    hydraulic_radius : float
        Hydraulic radius, equal to the geometric radius for circular tubes (m).

    Examples
    --------
    >>> from openthermoacoustics.segments import Duct
    >>> from openthermoacoustics.gas import Helium
    >>> duct = Duct(length=0.5, radius=0.025)
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = duct.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        length: float,
        radius: float,
        geometry: Geometry | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize a duct segment.

        Parameters
        ----------
        length : float
            Axial length of the duct (m).
        radius : float
            Inner radius of the duct (m).
        geometry : Geometry, optional
            Pore geometry for thermoviscous functions.
        name : str, optional
            Name identifier for the segment.
        """
        self._radius = radius
        area = np.pi * radius**2
        super().__init__(name=name, length=length, area=area, geometry=geometry)

    @property
    def radius(self) -> float:
        """
        Inner radius of the duct.

        Returns
        -------
        float
            Duct radius in meters.
        """
        return self._radius

    @property
    def hydraulic_radius(self) -> float:
        """
        Hydraulic radius of the duct.

        For a circular tube, the hydraulic radius equals the geometric radius.

        Returns
        -------
        float
            Hydraulic radius in meters.
        """
        return self._radius

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
            f_nu = self._geometry.f_nu(omega, delta_nu, self.hydraulic_radius)
            f_kappa = self._geometry.f_kappa(omega, delta_kappa, self.hydraulic_radius)
        else:
            # Boundary layer approximation for wide ducts
            f_nu = (1 - 1j) * delta_nu / self.hydraulic_radius
            f_kappa = (1 - 1j) * delta_kappa / self.hydraulic_radius

        return complex(f_nu), complex(f_kappa)

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

        Implements the momentum and continuity equations:
            dp1/dx = -(j*omega*rho_m / (A*(1 - f_nu))) * U1
            dU1/dx = -(j*omega*A / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1

        Parameters
        ----------
        x : float
            Axial position within the segment (m). Not used for uniform duct.
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1)].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        NDArray[np.float64]
            Derivative vector [d(Re(p1))/dx, d(Im(p1))/dx,
                               d(Re(U1))/dx, d(Im(U1))/dx].
        """
        p1, U1 = state_to_complex(y)

        # Gas properties at mean temperature
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        gamma = gas.gamma(T_m)

        # Thermoviscous functions
        f_nu, f_kappa = self._compute_thermoviscous_functions(omega, gas, T_m)

        # Momentum equation: dp1/dx = -(j*omega*rho_m / (A*(1 - f_nu))) * U1
        dp1_dx = -1j * omega * rho_m / (self._area * (1 - f_nu)) * U1

        # Continuity equation: dU1/dx = -(j*omega*A / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1
        dU1_dx = -1j * omega * self._area / (rho_m * a**2) * (1 + (gamma - 1) * f_kappa) * p1

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
        Propagate acoustic state through the duct.

        Uses scipy.integrate.solve_ivp with RK45 method to integrate
        the governing ODEs from x=0 to x=length.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m^3/s).
        T_m : float
            Mean temperature (K). Constant for isothermal walls.
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
            - T_m_out: Mean temperature at output (K), equal to input for isothermal duct
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

        # Temperature is constant for isothermal walls
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        """Return string representation of the duct."""
        return (
            f"Duct(name='{self._name}', length={self._length}, "
            f"radius={self._radius})"
        )

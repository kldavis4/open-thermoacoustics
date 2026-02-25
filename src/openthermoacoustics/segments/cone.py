"""Cone segment for linearly tapered tubes."""

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


class Cone(Segment):
    """
    Cone segment representing a linearly tapered circular tube.

    The cone has a radius that varies linearly from radius_in at x=0
    to radius_out at x=length. The cross-sectional area and hydraulic
    radius are position-dependent.

    The governing equations are the same as for a duct, but with
    position-dependent area A(x) and hydraulic radius r_h(x):
        dp1/dx = -(j*omega*rho_m / (A(x)*(1 - f_nu))) * U1
        dU1/dx = -(j*omega*A(x) / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1

    Parameters
    ----------
    length : float
        Axial length of the cone (m).
    radius_in : float
        Inner radius at x=0 (inlet) (m).
    radius_out : float
        Inner radius at x=length (outlet) (m).
    geometry : Geometry, optional
        Pore geometry for thermoviscous functions.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    radius_in : float
        Inner radius at inlet (m).
    radius_out : float
        Inner radius at outlet (m).

    Examples
    --------
    >>> from openthermoacoustics.segments import Cone
    >>> from openthermoacoustics.gas import Helium
    >>> cone = Cone(length=0.5, radius_in=0.025, radius_out=0.05)
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = cone.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        length: float,
        radius_in: float,
        radius_out: float,
        geometry: Geometry | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize a cone segment.

        Parameters
        ----------
        length : float
            Axial length of the cone (m).
        radius_in : float
            Inner radius at x=0 (inlet) (m).
        radius_out : float
            Inner radius at x=length (outlet) (m).
        geometry : Geometry, optional
            Pore geometry for thermoviscous functions.
        name : str, optional
            Name identifier for the segment.
        """
        self._radius_in = radius_in
        self._radius_out = radius_out
        # Use inlet area as the nominal area
        area = np.pi * radius_in**2
        super().__init__(name=name, length=length, area=area, geometry=geometry)

    @property
    def radius_in(self) -> float:
        """
        Inner radius at inlet (x=0).

        Returns
        -------
        float
            Inlet radius in meters.
        """
        return self._radius_in

    @property
    def radius_out(self) -> float:
        """
        Inner radius at outlet (x=length).

        Returns
        -------
        float
            Outlet radius in meters.
        """
        return self._radius_out

    def radius_at(self, x: float) -> float:
        """
        Calculate radius at axial position x.

        The radius varies linearly from radius_in to radius_out:
            r(x) = radius_in + (radius_out - radius_in) * x / length

        Parameters
        ----------
        x : float
            Axial position (m), where 0 <= x <= length.

        Returns
        -------
        float
            Radius at position x in meters.
        """
        if self._length == 0:
            return self._radius_in
        return self._radius_in + (self._radius_out - self._radius_in) * x / self._length

    def area_at(self, x: float) -> float:
        """
        Calculate cross-sectional area at axial position x.

        Parameters
        ----------
        x : float
            Axial position (m), where 0 <= x <= length.

        Returns
        -------
        float
            Cross-sectional area at position x in m^2.
        """
        r = self.radius_at(x)
        return np.pi * r**2

    def hydraulic_radius_at(self, x: float) -> float:
        """
        Calculate hydraulic radius at axial position x.

        For a circular cross-section, the hydraulic radius equals
        the geometric radius.

        Parameters
        ----------
        x : float
            Axial position (m), where 0 <= x <= length.

        Returns
        -------
        float
            Hydraulic radius at position x in meters.
        """
        return self.radius_at(x)

    def _compute_thermoviscous_functions(
        self,
        x: float,
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> tuple[complex, complex]:
        """
        Compute the thermoviscous functions f_nu and f_kappa at position x.

        Parameters
        ----------
        x : float
            Axial position (m).
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

        r_h = self.hydraulic_radius_at(x)

        if self._geometry is not None:
            f_nu = self._geometry.f_nu(omega, delta_nu, r_h)
            f_kappa = self._geometry.f_kappa(omega, delta_kappa, r_h)
        else:
            # Boundary layer approximation
            f_nu = (1 - 1j) * delta_nu / r_h
            f_kappa = (1 - 1j) * delta_kappa / r_h

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

        Implements the momentum and continuity equations with
        position-dependent area:
            dp1/dx = -(j*omega*rho_m / (A(x)*(1 - f_nu))) * U1
            dU1/dx = -(j*omega*A(x) / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1

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

        # Position-dependent area
        A_x = self.area_at(x)

        # Thermoviscous functions at this position
        f_nu, f_kappa = self._compute_thermoviscous_functions(x, omega, gas, T_m)

        # Momentum equation
        dp1_dx = -1j * omega * rho_m / (A_x * (1 - f_nu)) * U1

        # Continuity equation
        dU1_dx = -1j * omega * A_x / (rho_m * a**2) * (1 + (gamma - 1) * f_kappa) * p1

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
        Propagate acoustic state through the cone.

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
            - T_m_out: Mean temperature at output (K), equal to input for isothermal cone
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
        """Return string representation of the cone."""
        return (
            f"Cone(name='{self._name}', length={self._length}, "
            f"radius_in={self._radius_in}, radius_out={self._radius_out})"
        )

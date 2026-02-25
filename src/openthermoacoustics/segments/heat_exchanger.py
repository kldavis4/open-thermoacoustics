"""Heat exchanger segment for thermoacoustic systems."""

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


class HeatExchanger(Segment):
    """
    Heat exchanger segment with fixed temperature.

    A heat exchanger is a short porous segment that maintains a fixed
    temperature, typically placed at the ends of a stack to add or remove
    heat from the thermoacoustic cycle. Unlike a stack, there is no
    temperature gradient within the heat exchanger.

    The governing equations are similar to the stack but without the
    temperature gradient term:
        dp1/dx = -(j*omega*rho_m / (A*porosity*(1 - f_nu))) * U1
        dU1/dx = -(j*omega*A*porosity / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1

    Parameters
    ----------
    length : float
        Axial length of the heat exchanger (m).
    porosity : float
        Porosity (void fraction), 0 < porosity < 1.
    hydraulic_radius : float
        Hydraulic radius of the pores (m).
    temperature : float
        Fixed temperature maintained by the heat exchanger (K).
    area : float, optional
        Total cross-sectional area (m^2). If not provided, defaults to
        a single-pore estimate. For realistic simulations, this should
        be provided explicitly.
    geometry : Geometry, optional
        Pore geometry for thermoviscous functions.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    porosity : float
        Porosity of the heat exchanger.
    hydraulic_radius : float
        Hydraulic radius of the pores (m).
    temperature : float
        Fixed temperature of the heat exchanger (K).

    Examples
    --------
    >>> from openthermoacoustics.segments import HeatExchanger
    >>> from openthermoacoustics.gas import Helium
    >>> hx = HeatExchanger(
    ...     length=0.01, porosity=0.5, hydraulic_radius=0.001, temperature=300.0
    ... )
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = hx.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=350.0, omega=628.3, gas=gas
    ... )
    >>> # T_out will be 300.0, the fixed temperature of the heat exchanger
    """

    def __init__(
        self,
        length: float,
        porosity: float,
        hydraulic_radius: float,
        temperature: float,
        area: float | None = None,
        geometry: Geometry | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize a heat exchanger segment.

        Parameters
        ----------
        length : float
            Axial length of the heat exchanger (m).
        porosity : float
            Porosity (void fraction), 0 < porosity < 1.
        hydraulic_radius : float
            Hydraulic radius of the pores (m).
        temperature : float
            Fixed temperature maintained by the heat exchanger (K).
        area : float, optional
            Total cross-sectional area (m^2). If not provided, defaults to
            a single-pore estimate. For realistic simulations, this should
            be provided explicitly.
        geometry : Geometry, optional
            Pore geometry for thermoviscous functions.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If porosity is not in (0, 1) or temperature is not positive.
        """
        if not 0 < porosity < 1:
            raise ValueError(f"Porosity must be in (0, 1), got {porosity}")

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self._porosity = porosity
        self._hydraulic_radius = hydraulic_radius
        self._temperature = temperature

        # Reference area (actual flow area is porosity * area)
        if area is None:
            # Default: single-pore estimate (for compatibility/simple cases)
            area = np.pi * hydraulic_radius**2 / porosity

        super().__init__(name=name, length=length, area=area, geometry=geometry)

    @property
    def porosity(self) -> float:
        """
        Porosity (void fraction) of the heat exchanger.

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
    def temperature(self) -> float:
        """
        Fixed temperature of the heat exchanger.

        Returns
        -------
        float
            Temperature in K.
        """
        return self._temperature

    def _compute_thermoviscous_functions(
        self,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex]:
        """
        Compute the thermoviscous functions f_nu and f_kappa.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex]
            Tuple of (f_nu, f_kappa).
        """
        T_m = self._temperature
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
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Calculate state derivatives for ODE integration.

        Implements the momentum and continuity equations at fixed temperature:
            dp1/dx = -(j*omega*rho_m / (A_eff*(1 - f_nu))) * U1
            dU1/dx = -(j*omega*A_eff / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa] * p1

        Parameters
        ----------
        x : float
            Axial position within the segment (m). Not used for uniform HX.
        y : NDArray[np.float64]
            State vector [Re(p1), Im(p1), Re(U1), Im(U1)].
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Input mean temperature (K). Ignored - uses fixed temperature.

        Returns
        -------
        NDArray[np.float64]
            Derivative vector [d(Re(p1))/dx, d(Im(p1))/dx,
                               d(Re(U1))/dx, d(Im(U1))/dx].
        """
        p1, U1 = state_to_complex(y)

        # Use fixed heat exchanger temperature
        T_hx = self._temperature

        # Gas properties at fixed temperature
        rho_m = gas.density(T_hx)
        a = gas.sound_speed(T_hx)
        gamma = gas.gamma(T_hx)

        # Effective flow area
        A_eff = self._porosity * self._area

        # Thermoviscous functions
        f_nu, f_kappa = self._compute_thermoviscous_functions(omega, gas)

        # Momentum equation
        dp1_dx = -1j * omega * rho_m / (A_eff * (1 - f_nu)) * U1

        # Continuity equation (no temperature gradient term)
        dU1_dx = -1j * omega * A_eff / (rho_m * a**2) * (1 + (gamma - 1) * f_kappa) * p1

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
        Propagate acoustic state through the heat exchanger.

        Uses scipy.integrate.solve_ivp with RK45 method to integrate
        the governing ODEs from x=0 to x=length.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m^3/s).
        T_m : float
            Mean temperature at input (K). Ignored for heat exchanger.
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
            - T_m_out: Fixed temperature of the heat exchanger (K)
        """
        if self._length == 0:
            return p1_in, U1_in, self._temperature

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

        # Output temperature is the fixed heat exchanger temperature
        T_m_out = self._temperature

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        """Return string representation of the heat exchanger."""
        return (
            f"HeatExchanger(name='{self._name}', length={self._length}, "
            f"porosity={self._porosity}, hydraulic_radius={self._hydraulic_radius}, "
            f"temperature={self._temperature})"
        )

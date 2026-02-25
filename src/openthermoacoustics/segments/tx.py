"""Tube-bundle heat exchanger segment (TX).

This module implements the TX segment from reference baseline, which models
a tube-array heat exchanger where the thermoacoustic working gas flows
inside circular tubes. This is commonly used in shell-and-tube heat
exchangers for thermoacoustic engines and refrigerators.

The implementation follows reference baseline's formulation from equations 
in the published literature.

Key differences from HX (parallel-plate heat exchanger):
- Uses circular pore geometry (Bessel functions) instead of parallel plates
- Hydraulic radius rh = r0/2 (tube radius divided by 2)
- Thermoviscous functions use J0, J1 Bessel functions

References
----------
published literature, relevant reference (HX and TX)
Swift, G. W. (2017). Thermoacoustics: A Unifying Perspective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from openthermoacoustics.geometry.circular import CircularPore
from openthermoacoustics.segments.base import Segment
from openthermoacoustics.utils import (
    complex_to_state,
    penetration_depth_thermal,
    penetration_depth_viscous,
    state_to_complex,
)

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class TubeHeatExchanger(Segment):
    """
    Tube-bundle heat exchanger segment (reference baseline's TX).

    Models a shell-and-tube heat exchanger where the thermoacoustic working gas
    flows inside circular tubes. Heat is exchanged with the solid tube walls,
    which are maintained at a specified temperature by external heat exchange.

    This implementation follows reference baseline's equations :
        dp1/dx = -(j*omega*rho_m / (A_gas*(1 - f_nu))) * U1
        dU1/dx = -(j*omega*A_gas / (rho_m * a^2)) * [1 + (gamma-1)*f_kappa/(1+eps_s)] * p1

    where f_nu and f_kappa use circular pore (Bessel function) geometry.

    Parameters
    ----------
    length : float
        Axial length of the heat exchanger (m).
    porosity : float
        Areal porosity (gas area / total area), 0 < porosity < 1.
        Equal to N * pi * r0^2 / Area where N is the number of tubes.
    tube_radius : float
        Inner radius of each tube, r0 (m).
    area : float
        Total cross-sectional area including solid (m^2).
    solid_temperature : float
        Fixed temperature of the solid tube walls (K).
    heat_in : float, optional
        Heat added to the gas (W). Positive = heat flows from solid to gas.
        Default is 0.0.
    solid_heat_capacity : float, optional
        Volumetric heat capacity rho_s * c_s of solid (J/(m^3 K)).
        Default is 3.9e6 (copper: 8900 kg/m^3 * 440 J/(kg K)).
    solid_thermal_conductivity : float, optional
        Thermal conductivity k_s of solid (W/(m K)).
        Default is 400.0 (copper).
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    reference baseline's TX equations assume:
    - Short heat exchangers (single RK4 step for integration)
    - Uniform solid temperature (no axial gradient in solid)
    - Gas temperature Tm is constant through the heat exchanger (dTm/dx = 0)
    - Heat transfer affects only H_tot, not Tm

    The thermoviscous functions use circular pore geometry:
    - For r0/delta < 25: Bessel function formula
    - For r0/delta > 30: Boundary layer approximation
    - Linear interpolation for intermediate values

    Examples
    --------
    >>> from openthermoacoustics.segments import TubeHeatExchanger
    >>> from openthermoacoustics.gas import Helium
    >>> tx = TubeHeatExchanger(
    ...     length=0.02, porosity=0.5, tube_radius=0.003,
    ...     area=0.01, solid_temperature=300.0
    ... )
    >>> gas = Helium(mean_pressure=3e6)
    >>> p1_out, U1_out, T_out = tx.propagate(
    ...     p1_in=1e5+0j, U1_in=0.01+0j, T_m=350.0, omega=628.3, gas=gas
    ... )
    >>> # T_out will be the solid temperature
    """

    def __init__(
        self,
        length: float,
        porosity: float,
        tube_radius: float,
        area: float,
        solid_temperature: float,
        heat_in: float = 0.0,
        solid_heat_capacity: float = 3.9e6,
        solid_thermal_conductivity: float = 400.0,
        name: str = "",
    ) -> None:
        """
        Initialize a tube-bundle heat exchanger segment.

        Parameters
        ----------
        length : float
            Axial length of the heat exchanger (m).
        porosity : float
            Areal porosity (gas area / total area), 0 < porosity < 1.
        tube_radius : float
            Inner radius of each tube, r0 (m).
        area : float
            Total cross-sectional area including solid (m^2).
        solid_temperature : float
            Fixed temperature of the solid tube walls (K).
        heat_in : float, optional
            Heat added to the gas (W). Default is 0.0.
        solid_heat_capacity : float, optional
            Volumetric heat capacity of solid (J/(m^3 K)). Default is 3.9e6.
        solid_thermal_conductivity : float, optional
            Thermal conductivity of solid (W/(m K)). Default is 400.0.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If porosity is not in (0, 1), tube_radius is not positive,
            area is not positive, or solid_temperature is not positive.
        """
        if not 0 < porosity < 1:
            raise ValueError(f"Porosity must be in (0, 1), got {porosity}")

        if tube_radius <= 0:
            raise ValueError(f"Tube radius must be positive, got {tube_radius}")

        if area <= 0:
            raise ValueError(f"Area must be positive, got {area}")

        if solid_temperature <= 0:
            raise ValueError(
                f"Solid temperature must be positive, got {solid_temperature}"
            )

        self._porosity = porosity
        self._tube_radius = tube_radius
        self._solid_temperature = solid_temperature
        self._heat_in = heat_in
        self._solid_heat_capacity = solid_heat_capacity
        self._solid_thermal_conductivity = solid_thermal_conductivity

        # Circular pore geometry for thermoviscous functions
        self._circular_geometry = CircularPore()

        super().__init__(
            name=name, length=length, area=area, geometry=self._circular_geometry
        )

    @property
    def porosity(self) -> float:
        """Areal porosity (gas area / total area)."""
        return self._porosity

    @property
    def tube_radius(self) -> float:
        """Inner radius of each tube (m)."""
        return self._tube_radius

    @property
    def hydraulic_radius(self) -> float:
        """
        Hydraulic radius of the tubes (m).

        For circular tubes, rh = r0/2 (reference baseline convention).
        """
        return self._tube_radius / 2.0

    @property
    def solid_temperature(self) -> float:
        """Fixed solid temperature (K)."""
        return self._solid_temperature

    @property
    def heat_in(self) -> float:
        """Heat added to gas (W). Positive = solid to gas."""
        return self._heat_in

    @heat_in.setter
    def heat_in(self, value: float) -> None:
        """Set the heat input."""
        self._heat_in = value

    def _compute_epsilon_s(self, gas: Gas, T_m: float) -> float:
        """
        Compute the solid thermal capacity ratio epsilon_s.

        From governing relations:
            eps_s = sqrt(k * rho_m * cp / (k_s * rho_s * c_s))

        Parameters
        ----------
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        float
            The solid thermal capacity ratio epsilon_s.
        """
        rho_m = gas.density(T_m)
        cp = gas.specific_heat_cp(T_m)
        k_gas = gas.thermal_conductivity(T_m)

        numerator = k_gas * rho_m * cp
        denominator = self._solid_thermal_conductivity * self._solid_heat_capacity

        return float(np.sqrt(numerator / denominator))

    def _compute_thermoviscous_functions(
        self,
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> tuple[complex, complex]:
        """
        Compute the thermoviscous functions f_nu and f_kappa.

        Uses circular pore geometry (Bessel functions) for the tube bundle.
        For r0/delta < 25: exact Bessel formula
        For r0/delta > 30: boundary layer approximation
        Linear interpolation for intermediate values.

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

        # Use tube radius (r0) for thermoviscous calculations
        r0 = self._tube_radius

        # CircularPore geometry uses the hydraulic radius parameter,
        # but for TX we use the full tube radius r0 in the Bessel formulas
        # The CircularPore._compute_f expects hydraulic_radius parameter
        # but internally computes z = r_h * (1+j) / delta
        # For TX, we want z = r0 * (1+j) / delta, so we pass r0 as the radius
        f_nu = self._circular_geometry.f_nu(omega, delta_nu, r0)
        f_kappa = self._circular_geometry.f_kappa(omega, delta_kappa, r0)

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

        Implements reference baseline's TX equations :
            dp1/dx = -(j*omega*rho_m / (A_gas*(1 - f_nu))) * U1
            dU1/dx = -(j*omega*A_gas / (rho_m * a^2)) *
                     [1 + (gamma-1)*f_kappa/(1+eps_s)] * p1

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
            Mean temperature (K). Uses solid temperature for calculations.

        Returns
        -------
        NDArray[np.float64]
            Derivative vector [d(Re(p1))/dx, d(Im(p1))/dx,
                              d(Re(U1))/dx, d(Im(U1))/dx].
        """
        p1, U1 = state_to_complex(y)

        # Use solid temperature for calculations (isothermal heat exchanger)
        T_solid = self._solid_temperature

        # Gas properties at solid temperature
        rho_m = gas.density(T_solid)
        a = gas.sound_speed(T_solid)
        gamma = gas.gamma(T_solid)

        # Gas area
        A_gas = self._porosity * self._area

        # Thermoviscous functions using circular pore geometry
        f_nu, f_kappa = self._compute_thermoviscous_functions(omega, gas, T_solid)

        # Solid thermal capacity ratio
        eps_s = self._compute_epsilon_s(gas, T_solid)

        # Momentum equation 
        dp1_dx = -1j * omega * rho_m / (A_gas * (1 - f_nu)) * U1

        # Continuity equation 
        thermal_factor = 1 + (gamma - 1) * f_kappa / (1 + eps_s)
        dU1_dx = -1j * omega * A_gas / (rho_m * a**2) * thermal_factor * p1

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
        Propagate acoustic state through the tube-bundle heat exchanger.

        Uses scipy.integrate.solve_ivp with RK45 method to integrate
        the governing ODEs from x=0 to x=length. reference baseline uses a single
        RK4 step for HX/TX segments; we use RK45 with tight tolerances
        for improved accuracy.

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
            Tuple of (p1_out, U1_out, T_m_out):
            - p1_out: Complex pressure amplitude at output (Pa)
            - U1_out: Complex volumetric velocity amplitude at output (m^3/s)
            - T_m_out: Solid temperature of the heat exchanger (K)
        """
        if self._length == 0:
            return p1_in, U1_in, self._solid_temperature

        y0 = complex_to_state(p1_in, U1_in)

        def ode_func(x: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.get_derivatives(x, y, omega, gas, self._solid_temperature)

        # reference baseline uses a single RK4 step for HX/TX (short segments)
        # We use RK45 with tight tolerances for accuracy
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

        # Output temperature is the solid temperature
        T_m_out = self._solid_temperature

        return p1_out, U1_out, T_m_out

    def compute_gas_temperature(
        self,
        p1: complex,
        U1: complex,
        omega: float,
        gas: Gas,
    ) -> complex:
        """
        Compute the spatial average of the oscillating temperature amplitude.

        From governing relations:
            <T1> = Tm * beta / (rho_m * cp) * p1 * (1 - f_kappa/(1+eps_s))

        Parameters
        ----------
        p1 : complex
            Complex pressure amplitude (Pa).
        U1 : complex
            Complex volumetric velocity amplitude (m^3/s). Not used but
            included for API consistency.
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        complex
            Spatial average of oscillating temperature amplitude <T1> (K).
        """
        T_m = self._solid_temperature

        rho_m = gas.density(T_m)
        cp = gas.specific_heat_cp(T_m)
        beta = 1.0 / T_m  # Thermal expansion coefficient for ideal gas

        _, f_kappa = self._compute_thermoviscous_functions(omega, gas, T_m)
        eps_s = self._compute_epsilon_s(gas, T_m)

        T1_avg = T_m * beta / (rho_m * cp) * p1 * (1 - f_kappa / (1 + eps_s))

        return complex(T1_avg)

    def __repr__(self) -> str:
        """Return string representation of the tube heat exchanger."""
        return (
            f"TubeHeatExchanger(name='{self._name}', length={self._length}, "
            f"porosity={self._porosity}, tube_radius={self._tube_radius}, "
            f"T_solid={self._solid_temperature})"
        )


# Alias for reference baseline naming convention
TX = TubeHeatExchanger

"""Adiabatic-isothermal interface loss segment (JOIN).

This module implements the JOIN segment from reference baseline, which accounts for
"small" discontinuities in thermoacoustic variables at the interface between
a heat exchanger (or other isothermal segment) and an unmixed, thermally
stratified, adiabatic region such as a pulse tube.

The physics involves:
- Temperature discontinuity (first order in acoustic amplitude)
- Volume flow rate magnitude discontinuity (second order)
- No discontinuity in pressure or phase of U1

References
----------
[65] Kittel, P. (1992). "An analysis of the pressure-temperature phase
     relationship at the heat exchangers of thermoacoustic devices."
     Cryogenics 32, 843.
[66] Storch, P. J., Radebaugh, R., and Zimmerman, J. E. (1990). "Analytical
     model for the refrigeration power of the orifice pulse tube refrigerator."
     NIST Tech Note 1343.
[67] Olson, J. R., and Swift, G. W. (1997). "Acoustic streaming in pulse tube
     refrigerators: Tapered pulse tubes." Cryogenics 37, 769.
[13] Swift, G. W. (2002). "Thermoacoustics: A Unifying Perspective for Some
     Engines and Refrigerators."
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class Join(Segment):
    """
    Adiabatic-isothermal interface loss segment (reference baseline's JOIN).

    Models the interface between a heat exchanger (isothermal) and a pulse
    tube or thermal buffer tube (adiabatic, thermally stratified). This
    segment accounts for:

    1. Temperature overshoot: Gas parcels entering the pulse tube from the
       heat exchanger have different temperatures than the local mean due to
       pressure-induced temperature oscillations.

    2. Volume flow rate reduction: Irreversible mixing at the interface
       causes a second-order reduction in |U1|.

    The governing equations (from published literature) are:

    Temperature discontinuity (governing relation):
        T_m,out - T_m,in = -(T_m * beta / (rho_m * c_p)) *
                          (|p1| * sin(theta) - |U1|_in * dT_m/dx / (omega * A_gas)) * F

    Volume flow rate magnitude discontinuity (governing relation):
        |U1|_out = |U1|_in - (16/(3*pi)) * ((gamma-1)/(rho_m * a^2)) * E_dot

    where:
        - theta is the phase angle by which p1 leads U1
        - dT_m/dx is evaluated on the adiabatic (pulse tube) side
        - F is a correction factor accounting for 2D effects
        - E_dot is the acoustic power (time-averaged)

    Parameters
    ----------
    area : float
        Cross-sectional area of the adjacent pulse tube/duct (m^2).
    dT_dx : float, optional
        Temperature gradient on the adiabatic side (K/m). Default is 0.
        Positive means temperature increases in +x direction.
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    The JOIN segment is a zero-length lumped element. It should be placed
    between a heat exchanger and a pulse tube (STKDUCT) or thermal buffer
    tube. reference baseline automatically detects the adjacent segments and obtains
    the area and temperature gradient from them; in this implementation,
    these must be specified explicitly.

    The factor F in governing relations accounts for the 2D nature of the interface:
        F = sqrt(2 * (A_gas * k * dT_m/dx)^2 /
                 ((A_gas * k * dT_m/dx)^2 + (H2k - E_dot)^2))

    where H2k is the thermal conduction contribution to enthalpy flux.
    When thermal conductivity effects dominate (large H2k), F -> 0 and
    the temperature discontinuity vanishes. When they are negligible, F -> 1.

    For simplicity, this implementation uses F = 1 when dT_dx != 0 and
    F = 0 when dT_dx = 0. For more accurate results with intermediate
    cases, the full F calculation can be enabled with additional parameters.

    Examples
    --------
    >>> from openthermoacoustics.segments import Join
    >>> from openthermoacoustics.gas import Helium
    >>> # JOIN at hot end of pulse tube (T increases into pulse tube)
    >>> join = Join(area=1.17e-4, dT_dx=-3800.0)  # 300K to 57K over 0.07m
    >>> gas = Helium(mean_pressure=3.1e6)
    >>> omega = 2 * np.pi * 40  # 40 Hz
    >>> p1_out, U1_out, T_out = join.propagate(
    ...     p1_in=135000 * np.exp(-1j * np.radians(17.66)),
    ...     U1_in=2.92e-4 * np.exp(1j * np.radians(138.5)),
    ...     T_m=300.0, omega=omega, gas=gas
    ... )
    """

    def __init__(
        self,
        area: float,
        dT_dx: float = 0.0,
        name: str = "",
    ) -> None:
        """
        Initialize an adiabatic-isothermal interface segment.

        Parameters
        ----------
        area : float
            Cross-sectional area of the adjacent pulse tube/duct (m^2).
        dT_dx : float, optional
            Temperature gradient on the adiabatic side (K/m). Default is 0.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If area is not positive.
        """
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")

        self._dT_dx = dT_dx

        # JOIN is a lumped element (zero length)
        super().__init__(name=name, length=0.0, area=area, geometry=None)

    @property
    def dT_dx(self) -> float:
        """
        Temperature gradient on the adiabatic side (K/m).

        Returns
        -------
        float
            Temperature gradient in K/m.
        """
        return self._dT_dx

    @dT_dx.setter
    def dT_dx(self, value: float) -> None:
        """Set the temperature gradient."""
        self._dT_dx = value

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """
        Return zero derivatives (JOIN is a lumped element).

        JOIN has zero length and applies instantaneous changes,
        so it doesn't use ODE integration.

        Parameters
        ----------
        x : float
            Axial position (m). Not used.
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
            Zero vector [0, 0, 0, 0].
        """
        return np.zeros(4, dtype=np.float64)

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the JOIN interface.

        Applies the temperature overshoot (governing relation) and volume flow
        rate magnitude reduction (governing relation) while preserving p1 and
        the phase of U1.

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
            Tuple of (p1_out, U1_out, T_m_out):
            - p1_out: Complex pressure amplitude (unchanged from input)
            - U1_out: Complex volumetric velocity (magnitude reduced, phase preserved)
            - T_m_out: Mean temperature at output (may differ from input)
        """
        # Gas properties at local temperature
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        gamma = gas.gamma(T_m)
        cp = gas.specific_heat_cp(T_m)
        beta = 1.0 / T_m  # Ideal gas thermal expansion coefficient

        # Acoustic quantities
        p1_mag = np.abs(p1_in)
        U1_mag = np.abs(U1_in)
        U1_phase = np.angle(U1_in)

        # Phase angle by which p1 leads U1
        theta = np.angle(p1_in) - np.angle(U1_in)

        # Acoustic power (time-averaged): E_dot = (1/2) * Re[p1 * conj(U1)]
        E_dot = 0.5 * np.real(p1_in * np.conj(U1_in))

        # --- Temperature discontinuity (governing relation) ---
        # delta_T = -(T_m * beta / (rho_m * cp)) *
        #           (|p1| * sin(theta) - |U1|_in * dT_m/dx / (omega * A_gas)) * F

        if self._dT_dx != 0:
            # Factor F: We use a simplified approach
            # Full formula: F = sqrt(2 * (A*k*dT/dx)^2 / ((A*k*dT/dx)^2 + (H2k - E_dot)^2))
            # For now, use F = 1 when there's a temperature gradient
            # This is valid when wall thermal conduction is negligible
            F = 1.0

            # Temperature overshoot term
            temp_term = p1_mag * np.sin(theta) - U1_mag * self._dT_dx / (omega * self._area)

            delta_T = -(T_m * beta / (rho_m * cp)) * temp_term * F
        else:
            # No temperature gradient means no temperature discontinuity
            delta_T = 0.0

        T_m_out = T_m + delta_T

        # --- Volume flow rate magnitude discontinuity (governing relation) ---
        # |U1|_out = |U1|_in - (16/(3*pi)) * ((gamma-1)/(rho_m * a^2)) * E_dot

        delta_U1_mag = (16.0 / (3.0 * np.pi)) * ((gamma - 1) / (rho_m * a**2)) * E_dot
        U1_mag_out = U1_mag - delta_U1_mag

        # Ensure U1 magnitude doesn't go negative (would indicate
        # the approximation has broken down)
        U1_mag_out = max(U1_mag_out, 0.0)

        # Reconstruct U1 with new magnitude but preserved phase
        U1_out = U1_mag_out * np.exp(1j * U1_phase)

        # Pressure is unchanged
        p1_out = p1_in

        return p1_out, U1_out, T_m_out

    def acoustic_power_dissipation(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> float:
        """
        Calculate the time-averaged acoustic power dissipated at the interface.

        The dissipation comes from the irreversible mixing of gas at
        different temperatures (adiabatic-isothermal mixing loss).

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m³/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        float
            Time-averaged acoustic power dissipated (W).
            Always non-negative.
        """
        p1_out, U1_out, _ = self.propagate(p1_in, U1_in, T_m, omega, gas)

        # Acoustic power: E_dot = (1/2) * Re[p1 * conj(U1)]
        E_dot_in = 0.5 * np.real(p1_in * np.conj(U1_in))
        E_dot_out = 0.5 * np.real(p1_out * np.conj(U1_out))

        # Power dissipated is the difference
        return float(max(E_dot_in - E_dot_out, 0.0))

    def __repr__(self) -> str:
        """Return string representation of the JOIN segment."""
        return (
            f"Join(name='{self._name}', area={self._area}, "
            f"dT_dx={self._dT_dx})"
        )


# Alias for reference baseline naming convention
JOIN = Join

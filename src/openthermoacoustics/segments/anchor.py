"""Thermal mode control segments (ANCHOR and INSULATE).

This module implements the ANCHOR and INSULATE segments from reference baseline,
which control how subsequent segments handle energy accounting.

In the default "thermally insulated" mode, the total power flow H_tot
is independent of x in ducts, cones, etc. (except in heat exchangers
and transducers where heat is added/removed).

In "thermally anchored" mode (set by ANCHOR), segments like DUCT, CONE,
COMPLIANCE, IMPEDANCE, and SURFACE are treated as if immersed in a
thermal bath at T = T_m, so acoustic power dissipation is removed locally
as heat: dH_tot/dx = dE/dx.

References
----------
[1] published literature, relevant reference
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class ThermalMode(Enum):
    """Thermal mode for energy accounting in segments."""

    INSULATED = "insulated"
    """Default mode: H_tot is independent of x (except in HX/transducers)."""

    ANCHORED = "anchored"
    """Thermally anchored: dH_tot/dx = dE/dx (heat removed locally)."""


class Anchor(Segment):
    """
    ANCHOR segment - enables thermally anchored mode.

    This segment sets the thermal mode to "anchored" for subsequent
    segments. In this mode, DUCT, CONE, COMPLIANCE, IMPEDANCE, and
    SURFACE segments are treated as if immersed in a thermal bath
    at T = T_m, so acoustic power dissipation is removed locally as heat.

    The acoustic state (p1, U1) passes through unchanged. Only the
    thermal mode flag is set.

    Parameters
    ----------
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    In thermally anchored mode:
        dH_tot/dx = dE/dx

    This means the acoustic power dissipation shows up as local heat
    removal rather than being carried downstream.

    Use INSULATE to return to the default thermally insulated mode.

    Examples
    --------
    >>> anchor = Anchor(name="water_jacket_start")
    >>> p1_out, U1_out, T_out = anchor.propagate(p1_in, U1_in, T_m, omega, gas)
    >>> # p1_out == p1_in, U1_out == U1_in, T_out == T_m
    >>> assert anchor.thermal_mode == ThermalMode.ANCHORED
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)
        self._thermal_mode = ThermalMode.ANCHORED

    @property
    def thermal_mode(self) -> ThermalMode:
        """The thermal mode set by this segment."""
        return self._thermal_mode

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

        For ANCHOR (zero-length segment), all derivatives are zero.

        Parameters
        ----------
        x : float
            Axial position (not used).
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
            Zero derivative vector.
        """
        return np.zeros(4)

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the ANCHOR segment.

        The acoustic state passes through unchanged. This segment only
        sets the thermal mode flag for energy accounting purposes.

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
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out) = (p1_in, U1_in, T_m).
        """
        return p1_in, U1_in, T_m

    def __repr__(self) -> str:
        return f"Anchor(name='{self._name}')"


class Insulate(Segment):
    """
    INSULATE segment - returns to thermally insulated mode.

    This segment sets the thermal mode back to the default "insulated"
    mode, undoing the effect of a previous ANCHOR segment.

    The acoustic state (p1, U1) passes through unchanged. Only the
    thermal mode flag is set.

    Parameters
    ----------
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    In thermally insulated mode (default):
        dH_tot/dx = 0 (in ducts, cones, etc.)

    This is the default behavior. Use this segment only if you need
    to cancel a previous ANCHOR segment.

    Examples
    --------
    >>> insulate = Insulate(name="end_water_jacket")
    >>> p1_out, U1_out, T_out = insulate.propagate(p1_in, U1_in, T_m, omega, gas)
    >>> assert insulate.thermal_mode == ThermalMode.INSULATED
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)
        self._thermal_mode = ThermalMode.INSULATED

    @property
    def thermal_mode(self) -> ThermalMode:
        """The thermal mode set by this segment."""
        return self._thermal_mode

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

        For INSULATE (zero-length segment), all derivatives are zero.

        Parameters
        ----------
        x : float
            Axial position (not used).
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
            Zero derivative vector.
        """
        return np.zeros(4)

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the INSULATE segment.

        The acoustic state passes through unchanged. This segment only
        sets the thermal mode flag for energy accounting purposes.

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
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out) = (p1_in, U1_in, T_m).
        """
        return p1_in, U1_in, T_m

    def __repr__(self) -> str:
        return f"Insulate(name='{self._name}')"


# reference baseline aliases
ANCHOR = Anchor
INSULATE = Insulate

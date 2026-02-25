"""Abstract base class for acoustic network segments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas
    from openthermoacoustics.geometry.base import Geometry


class Segment(ABC):
    """
    Abstract base class for acoustic network segments.

    A segment represents a component in a thermoacoustic network that can
    propagate acoustic waves. Each segment transforms the acoustic state
    (pressure p1, volumetric velocity U1, and mean temperature T_m) from
    its input to its output.

    All segments must implement:
    - propagate(): Integrate through the segment to find output state
    - get_derivatives(): Return ODE derivatives for numerical integration

    Attributes
    ----------
    _name : str
        Name identifier for the segment.
    _length : float
        Axial length of the segment in meters (0 for lumped elements).
    _area : float
        Cross-sectional area in m^2.
    _geometry : Geometry or None
        Pore geometry for thermoviscous function calculations.

    Notes
    -----
    The state vector for ODE integration is:
        y = [Re(p1), Im(p1), Re(U1), Im(U1)]

    where p1 is complex pressure amplitude (Pa) and U1 is complex
    volumetric velocity amplitude (m^3/s).
    """

    def __init__(
        self,
        name: str = "",
        length: float = 0.0,
        area: float = 0.0,
        geometry: Geometry | None = None,
    ) -> None:
        """
        Initialize the base segment.

        Parameters
        ----------
        name : str, optional
            Name identifier for the segment. Default is empty string.
        length : float, optional
            Axial length in meters. Default is 0.0.
        area : float, optional
            Cross-sectional area in m^2. Default is 0.0.
        geometry : Geometry, optional
            Pore geometry for thermoviscous functions. Default is None.
        """
        self._name = name
        self._length = length
        self._area = area
        self._geometry = geometry

    @property
    def name(self) -> str:
        """
        Name identifier for the segment.

        Returns
        -------
        str
            The segment name.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the segment name."""
        self._name = value

    @property
    def length(self) -> float:
        """
        Axial length of the segment in meters.

        Returns
        -------
        float
            Length in meters. Zero for lumped elements.
        """
        return self._length

    @property
    def area(self) -> float:
        """
        Cross-sectional area of the segment.

        Returns
        -------
        float
            Cross-sectional area in m^2.
        """
        return self._area

    @property
    def geometry(self) -> Geometry | None:
        """
        Pore geometry for thermoviscous function calculations.

        Returns
        -------
        Geometry or None
            The geometry object, or None if not applicable.
        """
        return self._geometry

    @abstractmethod
    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through the segment.

        This method integrates the governing equations from the input
        to the output of the segment, transforming the acoustic state.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m^3/s).
        T_m : float
            Mean temperature at segment input (K).
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
        pass

    @abstractmethod
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

        This method returns the right-hand side of the governing ODEs
        in the form dy/dx = f(x, y).

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
            Mean temperature at position x (K).

        Returns
        -------
        NDArray[np.float64]
            Derivative vector [d(Re(p1))/dx, d(Im(p1))/dx,
                               d(Re(U1))/dx, d(Im(U1))/dx].
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the segment."""
        return f"{self.__class__.__name__}(name='{self._name}', length={self._length})"

"""Compliance (lumped acoustic volume) segment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class Compliance(Segment):
    """
    Compliance segment representing a lumped acoustic volume.

    A compliance is a lumped acoustic element representing a volume that
    stores acoustic energy in the compressibility of the gas. It acts as
    an acoustic capacitor, where pressure is continuous but volumetric
    velocity changes due to gas compression.

    The lumped element relations are:
        p1_out = p1_in  (pressure is continuous)
        U1_out = U1_in - j*omega*V*p1 / (rho_m * a^2)

    This represents the volume velocity absorbed by compression of the
    gas in the compliance volume.

    Parameters
    ----------
    volume : float
        Acoustic volume of the compliance (m^3).
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    volume : float
        Acoustic volume (m^3).

    Notes
    -----
    The compliance is a lumped element with zero length. The ODE
    derivatives method returns zeros since there is no distributed
    propagation; all the physics is captured in the propagate method.

    The acoustic impedance of a compliance is:
        Z_c = 1 / (j*omega*C)
    where C = V / (rho_m * a^2) is the acoustic compliance.

    Examples
    --------
    >>> from openthermoacoustics.segments import Compliance
    >>> from openthermoacoustics.gas import Helium
    >>> comp = Compliance(volume=1e-4)  # 100 cm^3
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = comp.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        volume: float,
        name: str = "",
    ) -> None:
        """
        Initialize a compliance segment.

        Parameters
        ----------
        volume : float
            Acoustic volume (m^3).
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If volume is not positive.
        """
        if volume <= 0:
            raise ValueError(f"Volume must be positive, got {volume}")

        self._volume = volume
        # Lumped element: length = 0, area = 0 (not applicable)
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)

    @property
    def volume(self) -> float:
        """
        Acoustic volume of the compliance.

        Returns
        -------
        float
            Volume in m^3.
        """
        return self._volume

    def acoustic_compliance(self, gas: Gas, T_m: float) -> float:
        """
        Calculate the acoustic compliance C = V / (rho_m * a^2).

        Parameters
        ----------
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        float
            Acoustic compliance in m^3/Pa (or equivalently m^5/(N) or s^2*m^4/kg).
        """
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)
        return self._volume / (rho_m * a**2)

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

        For a lumped element, the derivatives are zero since there is no
        distributed propagation. All physics is captured in the propagate method.

        Parameters
        ----------
        x : float
            Axial position (m). Not used for lumped element.
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
            Zero vector [0, 0, 0, 0] since this is a lumped element.
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
        Propagate acoustic state through the compliance.

        Implements the lumped element relations:
            p1_out = p1_in
            U1_out = U1_in - j*omega*V*p1_in / (rho_m * a^2)

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at segment input (Pa).
        U1_in : complex
            Complex volumetric velocity amplitude at segment input (m^3/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out):
            - p1_out: Complex pressure amplitude at output (Pa), equal to input
            - U1_out: Complex volumetric velocity amplitude at output (m^3/s)
            - T_m_out: Mean temperature at output (K), equal to input
        """
        # Gas properties
        rho_m = gas.density(T_m)
        a = gas.sound_speed(T_m)

        # Pressure is continuous through compliance
        p1_out = p1_in

        # Volume velocity change due to gas compression
        # dU = -j*omega*V*p1 / (rho_m * a^2) = -j*omega*C*p1
        C = self._volume / (rho_m * a**2)
        U1_out = U1_in - 1j * omega * C * p1_in

        # Temperature is unchanged
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        """Return string representation of the compliance."""
        return f"Compliance(name='{self._name}', volume={self._volume})"

"""Inertance (lumped acoustic mass) segment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class Inertance(Segment):
    """
    Inertance segment representing a lumped acoustic mass (short narrow tube).

    An inertance is a lumped acoustic element representing the inertia of
    gas in a short narrow tube. It acts as an acoustic inductor, where
    volumetric velocity is continuous but pressure drops due to gas acceleration.

    The basic lumped element relations are:
        U1_out = U1_in  (volumetric velocity is continuous)
        p1_out = p1_in - j*omega*rho_m*L*U1_in / A

    Optionally, viscous resistance can be included:
        p1_out = p1_in - (j*omega*rho_m*L/A + R) * U1_in

    where R is the acoustic resistance.

    Parameters
    ----------
    length : float
        Length of the narrow tube (m).
    radius : float, optional
        Radius of the tube (m). Either radius or area must be provided.
    area : float, optional
        Cross-sectional area (m^2). Either radius or area must be provided.
    include_resistance : bool, optional
        Whether to include viscous resistance. Default is False.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    radius : float or None
        Tube radius (m), or None if area was specified directly.
    include_resistance : bool
        Whether viscous resistance is included.

    Notes
    -----
    The inertance is treated as a lumped element even though it has a
    physical length. This approximation is valid when the tube length
    is much smaller than the acoustic wavelength.

    The acoustic impedance of an inertance is:
        Z_L = j*omega*L_a + R
    where L_a = rho_m * L / A is the acoustic inertance.

    For a tube with viscous losses (Poiseuille flow approximation):
        R = 8 * mu * L / (pi * r^4) = 8 * mu * L / (A^2 / pi)

    Examples
    --------
    >>> from openthermoacoustics.segments import Inertance
    >>> from openthermoacoustics.gas import Helium
    >>> inert = Inertance(length=0.1, radius=0.005)
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = inert.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        length: float,
        radius: float | None = None,
        area: float | None = None,
        include_resistance: bool = False,
        name: str = "",
    ) -> None:
        """
        Initialize an inertance segment.

        Parameters
        ----------
        length : float
            Length of the narrow tube (m).
        radius : float, optional
            Radius of the tube (m).
        area : float, optional
            Cross-sectional area (m^2).
        include_resistance : bool, optional
            Whether to include viscous resistance. Default is False.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If neither radius nor area is provided, or if both are provided,
            or if length is not positive.
        """
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")

        if radius is None and area is None:
            raise ValueError("Must provide either radius or area")

        if radius is not None and area is not None:
            raise ValueError("Cannot provide both radius and area")

        if radius is not None:
            if radius <= 0:
                raise ValueError(f"Radius must be positive, got {radius}")
            self._radius: float | None = radius
            computed_area = np.pi * radius**2
        else:
            assert area is not None  # for type checker
            if area <= 0:
                raise ValueError(f"Area must be positive, got {area}")
            self._radius = None
            computed_area = area

        self._include_resistance = include_resistance

        # Note: We store the physical length but treat this as lumped (length=0 in base)
        # The _tube_length is used for the inertance calculation
        self._tube_length = length

        super().__init__(name=name, length=0.0, area=computed_area, geometry=None)

    @property
    def tube_length(self) -> float:
        """
        Physical length of the inertance tube.

        Returns
        -------
        float
            Length in meters.
        """
        return self._tube_length

    @property
    def radius(self) -> float | None:
        """
        Radius of the tube if specified.

        Returns
        -------
        float or None
            Radius in meters, or None if area was specified directly.
        """
        return self._radius

    @property
    def include_resistance(self) -> bool:
        """
        Whether viscous resistance is included.

        Returns
        -------
        bool
            True if resistance is included.
        """
        return self._include_resistance

    def acoustic_inertance(self, gas: Gas, T_m: float) -> float:
        """
        Calculate the acoustic inertance L_a = rho_m * L / A.

        Parameters
        ----------
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        float
            Acoustic inertance in kg/m^4 (or equivalently Pa*s^2/m^3).
        """
        rho_m = gas.density(T_m)
        return rho_m * self._tube_length / self._area

    def acoustic_resistance(self, gas: Gas, T_m: float) -> float:
        """
        Calculate the acoustic resistance (Poiseuille flow approximation).

        R = 8 * mu * L / (pi * r^4) for circular tube
        R = 8 * pi * mu * L / A^2 for general area

        Parameters
        ----------
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).

        Returns
        -------
        float
            Acoustic resistance in Pa*s/m^3.
        """
        if not self._include_resistance:
            return 0.0

        mu = gas.viscosity(T_m)
        # Poiseuille resistance for circular tube
        # R = 8 * mu * L / (pi * r^4) = 8 * pi * mu * L / A^2
        return 8 * np.pi * mu * self._tube_length / (self._area**2)

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
        Propagate acoustic state through the inertance.

        Implements the lumped element relations:
            U1_out = U1_in
            p1_out = p1_in - (j*omega*L_a + R) * U1_in

        where L_a = rho_m*L/A is the acoustic inertance and R is the
        optional acoustic resistance.

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
            - p1_out: Complex pressure amplitude at output (Pa)
            - U1_out: Complex volumetric velocity amplitude at output (m^3/s), equal to input
            - T_m_out: Mean temperature at output (K), equal to input
        """
        # Acoustic inertance and resistance
        L_a = self.acoustic_inertance(gas, T_m)
        R = self.acoustic_resistance(gas, T_m)

        # Volumetric velocity is continuous
        U1_out = U1_in

        # Pressure drop due to inertance and resistance
        # dp = -(j*omega*L_a + R) * U1_in
        Z = 1j * omega * L_a + R  # Acoustic impedance
        p1_out = p1_in - Z * U1_in

        # Temperature is unchanged
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        """Return string representation of the inertance."""
        if self._radius is not None:
            return (
                f"Inertance(name='{self._name}', length={self._tube_length}, "
                f"radius={self._radius}, include_resistance={self._include_resistance})"
            )
        return (
            f"Inertance(name='{self._name}', length={self._tube_length}, "
            f"area={self._area}, include_resistance={self._include_resistance})"
        )

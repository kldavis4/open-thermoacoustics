"""Arbitrary user-specified acoustic impedance boundary segment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class Impedance(Segment):
    """
    Arbitrary acoustic impedance boundary segment.

    Models a termination with a user-specified acoustic impedance.
    The impedance can be either a fixed complex value or a callable
    function of angular frequency.

    The boundary condition is:
        p1 = Z * U1

    where Z is the specified acoustic impedance.

    Parameters
    ----------
    impedance : complex, optional
        Fixed acoustic impedance (Pa*s/m^3). Must provide either this
        or impedance_func.
    impedance_func : callable, optional
        Function that takes omega (float) and returns complex impedance.
        Signature: impedance_func(omega: float) -> complex
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    impedance_value : complex or None
        Fixed impedance value, or None if using a function.
    impedance_func : callable or None
        Impedance function, or None if using a fixed value.

    Notes
    -----
    This segment is useful for:
    - Modeling complex terminations from measured impedance data
    - Connecting to external systems with known impedance
    - Parametric studies with varying termination conditions
    - Approximating difficult-to-model acoustic loads

    The impedance boundary condition p1 = Z * U1 means:
    - Z = infinity (or very large): hard end (U1 ~ 0)
    - Z = 0: ideal soft end (p1 ~ 0)
    - Z = rho*c/A: matched termination (no reflection)
    - Z = -j*X: reactive termination (e.g., closed tube, compliance)

    Examples
    --------
    >>> from openthermoacoustics.segments import Impedance
    >>> from openthermoacoustics.gas import Helium
    >>> # Fixed impedance termination
    >>> term = Impedance(impedance=1000+500j)
    >>> gas = Helium(mean_pressure=101325)
    >>> p1_out, U1_out, T_out = term.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )

    >>> # Frequency-dependent impedance
    >>> def my_impedance(omega):
    ...     return 1000 + 1j * 0.5 * omega  # Simple LC-like impedance
    >>> term2 = Impedance(impedance_func=my_impedance)
    """

    def __init__(
        self,
        impedance: complex | None = None,
        impedance_func: Callable[[float], complex] | None = None,
        name: str = "",
    ) -> None:
        """
        Initialize an impedance boundary segment.

        Parameters
        ----------
        impedance : complex, optional
            Fixed acoustic impedance (Pa*s/m^3).
        impedance_func : callable, optional
            Function that takes omega and returns complex impedance.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If neither impedance nor impedance_func is provided,
            or if both are provided.
        """
        if impedance is None and impedance_func is None:
            raise ValueError("Must provide either impedance or impedance_func")
        if impedance is not None and impedance_func is not None:
            raise ValueError("Cannot provide both impedance and impedance_func")

        self._impedance_value: complex | None = impedance
        self._impedance_func: Callable[[float], complex] | None = impedance_func

        # Boundary condition: length = 0, area = 0 (not applicable)
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)

    @property
    def impedance_value(self) -> complex | None:
        """
        Fixed impedance value, if specified.

        Returns
        -------
        complex or None
            Fixed impedance in Pa*s/m^3, or None if using a function.
        """
        return self._impedance_value

    @property
    def impedance_function(self) -> Callable[[float], complex] | None:
        """
        Impedance function, if specified.

        Returns
        -------
        callable or None
            Function that takes omega and returns impedance, or None.
        """
        return self._impedance_func

    def get_impedance(self, omega: float) -> complex:
        """
        Get the acoustic impedance at a given angular frequency.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            Acoustic impedance in Pa*s/m^3.
        """
        if self._impedance_value is not None:
            return self._impedance_value
        else:
            assert self._impedance_func is not None  # for type checker
            return self._impedance_func(omega)

    def reflection_coefficient(self, omega: float, gas: Gas, T_m: float, area: float) -> complex:
        """
        Calculate the pressure reflection coefficient.

        R = (Z - Z_0) / (Z + Z_0)

        where Z_0 = rho*c/A is the characteristic impedance.

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.
        T_m : float
            Mean temperature (K).
        area : float
            Cross-sectional area of the connecting pipe (m^2).

        Returns
        -------
        complex
            Pressure reflection coefficient (dimensionless).
        """
        rho = gas.density(T_m)
        c = gas.sound_speed(T_m)
        Z_0 = rho * c / area
        Z = self.get_impedance(omega)
        return (Z - Z_0) / (Z + Z_0)

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

        For a boundary condition, derivatives are zero (no propagation).

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
        Propagate acoustic state through the impedance boundary.

        Returns the input state unchanged. The impedance boundary is a
        boundary condition marker; use residual() or is_satisfied()
        to check if the boundary condition is met.

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
            Tuple of (p1_out, U1_out, T_m_out), equal to the inputs.
        """
        return p1_in, U1_in, T_m

    def residual(self, p1: complex, U1: complex, omega: float) -> complex:
        """
        Calculate the boundary condition residual.

        For an impedance boundary, the residual is p1 - Z * U1, which
        should be zero when the boundary condition is satisfied.

        Parameters
        ----------
        p1 : complex
            Complex pressure amplitude (Pa).
        U1 : complex
            Complex volumetric velocity amplitude (m^3/s).
        omega : float
            Angular frequency (rad/s).

        Returns
        -------
        complex
            The boundary condition residual.
        """
        Z = self.get_impedance(omega)
        return p1 - Z * U1

    def is_satisfied(
        self,
        p1: complex,
        U1: complex,
        omega: float,
        tolerance: float = 1e-6,
    ) -> bool:
        """
        Check if the impedance boundary condition is satisfied.

        Parameters
        ----------
        p1 : complex
            Complex pressure amplitude (Pa).
        U1 : complex
            Complex volumetric velocity amplitude (m^3/s).
        omega : float
            Angular frequency (rad/s).
        tolerance : float, optional
            Relative tolerance for the boundary condition check. Default is 1e-6.

        Returns
        -------
        bool
            True if the boundary condition is satisfied within tolerance.
        """
        res = self.residual(p1, U1, omega)
        Z = self.get_impedance(omega)
        # Normalize by the larger of |p1| or |Z * U1| for relative tolerance
        norm = max(abs(p1), abs(Z * U1), 1e-20)
        return abs(res) / norm < tolerance

    def __repr__(self) -> str:
        """Return string representation of the impedance boundary."""
        if self._impedance_value is not None:
            return f"Impedance(name='{self._name}', impedance={self._impedance_value})"
        else:
            return f"Impedance(name='{self._name}', impedance_func=<function>)"

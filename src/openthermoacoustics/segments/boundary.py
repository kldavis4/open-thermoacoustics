"""Boundary condition segments for acoustic networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class HardEnd(Segment):
    """
    Hard end (closed end) boundary condition.

    A hard end represents a rigid, impermeable wall where the volumetric
    velocity must be zero (U1 = 0). This is also called a closed end or
    velocity node.

    This boundary condition is used to:
    1. Terminate a network at a closed end
    2. Validate that a computed solution satisfies U1 = 0 at this location

    Parameters
    ----------
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    The hard end is a zero-length segment that validates/enforces
    the boundary condition U1 = 0. When used in propagate(), it
    returns the input state unchanged (for use in validation).

    In a shooting method solver, the hard end condition provides
    one of the boundary conditions that must be satisfied.

    Examples
    --------
    >>> from openthermoacoustics.segments import HardEnd
    >>> from openthermoacoustics.gas import Helium
    >>> hard = HardEnd(name="closed_end")
    >>> gas = Helium(mean_pressure=101325)
    >>> # Validate that U1 is close to zero
    >>> p1, U1, T = hard.propagate(
    ...     p1_in=1000+0j, U1_in=1e-10+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(self, name: str = "") -> None:
        """
        Initialize a hard end boundary condition.

        Parameters
        ----------
        name : str, optional
            Name identifier for the segment.
        """
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)

    def is_satisfied(self, U1: complex, tolerance: float = 1e-10) -> bool:
        """
        Check if the hard end boundary condition is satisfied.

        Parameters
        ----------
        U1 : complex
            Complex volumetric velocity amplitude (m^3/s).
        tolerance : float, optional
            Absolute tolerance for U1 = 0 check. Default is 1e-10.

        Returns
        -------
        bool
            True if |U1| < tolerance.
        """
        return abs(U1) < tolerance

    def residual(self, U1: complex) -> complex:
        """
        Calculate the boundary condition residual.

        For a hard end, the residual is simply U1, which should be zero.
        This is useful for shooting method solvers.

        Parameters
        ----------
        U1 : complex
            Complex volumetric velocity amplitude (m^3/s).

        Returns
        -------
        complex
            The residual (equal to U1).
        """
        return U1

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
        Propagate acoustic state through the hard end.

        Returns the input state unchanged. The hard end is a boundary
        condition marker; use is_satisfied() or residual() to check
        if the boundary condition is met.

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

    def __repr__(self) -> str:
        """Return string representation of the hard end."""
        return f"HardEnd(name='{self._name}')"


class SoftEnd(Segment):
    """
    Soft end (ideal open end) boundary condition.

    A soft end represents an ideal open end where the pressure perturbation
    is zero (p1 = 0). This is also called a pressure release boundary or
    pressure node.

    This boundary condition is used to:
    1. Terminate a network at an ideal open end
    2. Validate that a computed solution satisfies p1 = 0 at this location

    Parameters
    ----------
    name : str, optional
        Name identifier for the segment.

    Notes
    -----
    The soft end is a zero-length segment that validates/enforces
    the boundary condition p1 = 0. When used in propagate(), it
    returns the input state unchanged (for use in validation).

    Real open ends have radiation impedance and are not perfect
    pressure release boundaries. For more accurate modeling of
    open ends, a radiation impedance model should be used instead.

    Examples
    --------
    >>> from openthermoacoustics.segments import SoftEnd
    >>> from openthermoacoustics.gas import Helium
    >>> soft = SoftEnd(name="open_end")
    >>> gas = Helium(mean_pressure=101325)
    >>> # Validate that p1 is close to zero
    >>> p1, U1, T = soft.propagate(
    ...     p1_in=1e-10+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(self, name: str = "") -> None:
        """
        Initialize a soft end boundary condition.

        Parameters
        ----------
        name : str, optional
            Name identifier for the segment.
        """
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)

    def is_satisfied(self, p1: complex, tolerance: float = 1e-10) -> bool:
        """
        Check if the soft end boundary condition is satisfied.

        Parameters
        ----------
        p1 : complex
            Complex pressure amplitude (Pa).
        tolerance : float, optional
            Absolute tolerance for p1 = 0 check. Default is 1e-10.

        Returns
        -------
        bool
            True if |p1| < tolerance.
        """
        return abs(p1) < tolerance

    def residual(self, p1: complex) -> complex:
        """
        Calculate the boundary condition residual.

        For a soft end, the residual is simply p1, which should be zero.
        This is useful for shooting method solvers.

        Parameters
        ----------
        p1 : complex
            Complex pressure amplitude (Pa).

        Returns
        -------
        complex
            The residual (equal to p1).
        """
        return p1

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
        Propagate acoustic state through the soft end.

        Returns the input state unchanged. The soft end is a boundary
        condition marker; use is_satisfied() or residual() to check
        if the boundary condition is met.

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

    def __repr__(self) -> str:
        """Return string representation of the soft end."""
        return f"SoftEnd(name='{self._name}')"

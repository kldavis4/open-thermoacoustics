"""T-branch and Return segments for traveling-wave engine loop topologies.

This module provides segments that enable side branches and feedback loops
needed for traveling-wave engines like the Backhaus-Swift TASHE (Thermoacoustic
Stirling Heat Engine).

The key components are:
- TBranch: T-junction that diverts flow into a side branch
- Return: Reconnects side branch flow back to main duct
- SideBranch: Helper class to model the side branch path
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class TBranch(Segment):
    """
    T-junction segment that diverts flow into a side branch.

    At a T-junction, the acoustic pressure p1 is the same in all three branches
    (main inlet, main outlet, and side branch). The volumetric velocity splits
    according to:
        U1_main_in = U1_main_out + U1_side

    The fraction of flow diverted to the side branch is controlled by the
    `side_fraction` parameter, which can be set for iterative solving of
    loop closure conditions.

    Parameters
    ----------
    side_area : float
        Cross-sectional area of the side branch (m^2).
    side_fraction : float, optional
        Initial fraction of U1 diverted to side branch (0 to 1).
        Default is 0.5.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    side_area : float
        Cross-sectional area of the side branch (m^2).
    side_fraction : float
        Fraction of volumetric velocity diverted to side branch.

    Notes
    -----
    The TBranch is a lumped element with zero length. It stores the side
    branch state (p1, U1_side, T_m) which can be retrieved by a downstream
    Return segment via the `get_side_branch_state()` method.

    For traveling-wave engines, the side branch typically contains the
    regenerator, heat exchangers, and other components that form the
    thermoacoustic core.

    Examples
    --------
    >>> from openthermoacoustics.segments import TBranch
    >>> from openthermoacoustics.gas import Helium
    >>> tbranch = TBranch(side_area=0.001, side_fraction=0.3)
    >>> gas = Helium(mean_pressure=1e6)
    >>> p1_out, U1_out, T_out = tbranch.propagate(
    ...     p1_in=1000+0j, U1_in=0.001+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    >>> p1_side, U1_side, T_side = tbranch.get_side_branch_state()
    """

    def __init__(
        self,
        side_area: float,
        side_fraction: float = 0.5,
        name: str = "",
    ) -> None:
        """
        Initialize a T-branch segment.

        Parameters
        ----------
        side_area : float
            Cross-sectional area of the side branch (m^2).
        side_fraction : float, optional
            Initial fraction of U1 diverted to side branch (0 to 1).
            Default is 0.5.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        ValueError
            If side_area is not positive or side_fraction is not in [0, 1].
        """
        if side_area <= 0:
            raise ValueError(f"side_area must be positive, got {side_area}")
        if not 0 <= side_fraction <= 1:
            raise ValueError(
                f"side_fraction must be in [0, 1], got {side_fraction}"
            )

        self._side_area = side_area
        self._side_fraction = side_fraction

        # State storage for side branch (populated during propagate)
        self._side_p1: complex = 0j
        self._side_U1: complex = 0j
        self._side_T_m: float = 300.0

        # Lumped element: length = 0
        super().__init__(name=name, length=0.0, area=side_area, geometry=None)

    @property
    def side_area(self) -> float:
        """
        Cross-sectional area of the side branch.

        Returns
        -------
        float
            Side branch area in m^2.
        """
        return self._side_area

    @property
    def side_fraction(self) -> float:
        """
        Fraction of volumetric velocity diverted to side branch.

        Returns
        -------
        float
            Fraction between 0 and 1.
        """
        return self._side_fraction

    @property
    def side_U1(self) -> complex:
        """
        Volumetric velocity diverted to side branch.

        Returns
        -------
        complex
            Complex volumetric velocity amplitude (m^3/s).

        Notes
        -----
        This value is only valid after `propagate()` has been called.
        """
        return self._side_U1

    def set_side_fraction(self, fraction: float) -> None:
        """
        Set the fraction of U1 diverted to the side branch.

        Parameters
        ----------
        fraction : float
            Fraction of volumetric velocity to divert (0 to 1).

        Raises
        ------
        ValueError
            If fraction is not in [0, 1].
        """
        if not 0 <= fraction <= 1:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        self._side_fraction = fraction

    def get_side_branch_state(self) -> tuple[complex, complex, float]:
        """
        Get the acoustic state at the side branch inlet.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1, U1_side, T_m):
            - p1: Complex pressure amplitude at junction (Pa)
            - U1_side: Complex volumetric velocity into side branch (m^3/s)
            - T_m: Mean temperature at junction (K)

        Notes
        -----
        This method should be called after `propagate()` to get the
        state that enters the side branch. The Return segment uses
        this to know what flow it needs to return to the main duct.
        """
        return self._side_p1, self._side_U1, self._side_T_m

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
        Propagate acoustic state through the T-branch junction.

        At the junction:
        - Pressure is continuous: p1_out = p1_in = p1_side
        - Velocity splits: U1_in = U1_out + U1_side

        The split ratio is determined by `side_fraction`.

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
            - p1_out: Complex pressure at main outlet (Pa), equal to input
            - U1_out: Complex velocity continuing in main duct (m^3/s)
            - T_m_out: Mean temperature at output (K), equal to input
        """
        # Pressure is continuous through the junction
        p1_out = p1_in

        # Velocity splits according to side_fraction
        self._side_U1 = self._side_fraction * U1_in
        U1_out = U1_in - self._side_U1

        # Store side branch state for later retrieval
        self._side_p1 = p1_in
        self._side_T_m = T_m

        # Temperature is unchanged through the junction
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        """Return string representation of the T-branch."""
        return (
            f"TBranch(name='{self._name}', side_area={self._side_area}, "
            f"side_fraction={self._side_fraction})"
        )


class Return(Segment):
    """
    Return segment that reconnects a side branch back to the main duct.

    At the return junction, the acoustic pressure p1 is the same in all
    branches, and the volumetric velocity combines:
        U1_out = U1_main_in + U1_return

    The Return segment retrieves the returning flow state from the
    associated TBranch (after propagation through the side branch).

    Parameters
    ----------
    tbranch : TBranch
        Reference to the TBranch that started this side branch.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    tbranch : TBranch
        Reference to the associated T-branch junction.

    Notes
    -----
    The Return segment is a lumped element with zero length. It assumes
    that the side branch has been propagated separately and the returning
    state has been set via `set_return_state()`.

    For proper loop closure, the returning pressure must equal the
    junction pressure (eigenvalue condition). This is typically solved
    iteratively by adjusting the side_fraction in the TBranch.

    Examples
    --------
    >>> from openthermoacoustics.segments import TBranch, Return
    >>> tbranch = TBranch(side_area=0.001)
    >>> return_seg = Return(tbranch=tbranch)
    >>> # After propagating through side branch:
    >>> return_seg.set_return_state(p1_return=1000+0j, U1_return=0.0003+0j, T_m_return=350.0)
    >>> p1_out, U1_out, T_out = return_seg.propagate(
    ...     p1_in=1000+0j, U1_in=0.0007+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        tbranch: TBranch,
        name: str = "",
    ) -> None:
        """
        Initialize a Return segment.

        Parameters
        ----------
        tbranch : TBranch
            Reference to the TBranch that started this side branch.
        name : str, optional
            Name identifier for the segment.

        Raises
        ------
        TypeError
            If tbranch is not a TBranch instance.
        """
        if not isinstance(tbranch, TBranch):
            raise TypeError(
                f"tbranch must be a TBranch instance, got {type(tbranch)}"
            )

        self._tbranch = tbranch

        # State storage for returning flow (set after side branch propagation)
        self._return_p1: complex = 0j
        self._return_U1: complex = 0j
        self._return_T_m: float = 300.0
        self._return_state_set: bool = False

        # Lumped element: length = 0
        super().__init__(name=name, length=0.0, area=0.0, geometry=None)

    @property
    def tbranch(self) -> TBranch:
        """
        Reference to the associated T-branch junction.

        Returns
        -------
        TBranch
            The T-branch that started this side branch.
        """
        return self._tbranch

    @property
    def return_U1(self) -> complex:
        """
        Volumetric velocity returning from the side branch.

        Returns
        -------
        complex
            Complex volumetric velocity amplitude (m^3/s).
        """
        return self._return_U1

    def set_return_state(
        self,
        p1_return: complex,
        U1_return: complex,
        T_m_return: float,
    ) -> None:
        """
        Set the acoustic state returning from the side branch.

        This method should be called after propagating through all
        segments in the side branch.

        Parameters
        ----------
        p1_return : complex
            Pressure amplitude at side branch outlet (Pa).
        U1_return : complex
            Volumetric velocity at side branch outlet (m^3/s).
        T_m_return : float
            Mean temperature at side branch outlet (K).
        """
        self._return_p1 = p1_return
        self._return_U1 = U1_return
        self._return_T_m = T_m_return
        self._return_state_set = True

    def get_pressure_mismatch(self, p1_main: complex) -> complex:
        """
        Calculate the pressure mismatch for loop closure.

        For proper loop closure, the returning pressure must equal
        the main duct pressure at the junction.

        Parameters
        ----------
        p1_main : complex
            Pressure in the main duct at the return junction (Pa).

        Returns
        -------
        complex
            Pressure mismatch (p1_return - p1_main). Should be zero
            for proper loop closure.
        """
        return self._return_p1 - p1_main

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
        Propagate acoustic state through the Return junction.

        At the junction:
        - Pressure is continuous (with check for loop closure)
        - Velocity combines: U1_out = U1_in + U1_return

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude in main duct at junction (Pa).
        U1_in : complex
            Complex volumetric velocity in main duct at junction (m^3/s).
        T_m : float
            Mean temperature in main duct (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out):
            - p1_out: Complex pressure at outlet (Pa)
            - U1_out: Complex velocity after combining flows (m^3/s)
            - T_m_out: Mean temperature at output (K)

        Raises
        ------
        RuntimeError
            If return state has not been set via set_return_state().
        """
        if not self._return_state_set:
            raise RuntimeError(
                "Return state not set. Call set_return_state() after "
                "propagating through the side branch."
            )

        # Pressure should be continuous (use main duct pressure)
        # The pressure mismatch can be checked via get_pressure_mismatch()
        p1_out = p1_in

        # Volumetric velocity combines
        U1_out = U1_in + self._return_U1

        # Temperature: use main duct temperature
        # (mixing effects could be added for more accuracy)
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        """Return string representation of the Return segment."""
        return f"Return(name='{self._name}', tbranch={self._tbranch.name!r})"


class SideBranch:
    """
    Helper class to model the side branch path between TBranch and Return.

    A SideBranch wraps a list of segments that form the side branch path.
    It provides a convenient `propagate()` method that chains propagation
    through all contained segments.

    Parameters
    ----------
    segments : list[Segment]
        List of segments forming the side branch, in order from
        TBranch outlet to Return inlet.
    name : str, optional
        Name identifier for the side branch.

    Attributes
    ----------
    segments : list[Segment]
        List of segments in the side branch.
    name : str
        Name identifier.

    Notes
    -----
    The SideBranch is not a Segment itself; it's a container that
    manages propagation through multiple segments. For use in the
    NetworkTopology or LoopNetwork, the individual segments should
    be added separately.

    Examples
    --------
    >>> from openthermoacoustics.segments import Duct, Stack, SideBranch
    >>> side_duct = Duct(length=0.1, radius=0.01)
    >>> stack = Stack(length=0.05, porosity=0.7, hydraulic_radius=0.0005)
    >>> side_branch = SideBranch(segments=[side_duct, stack])
    >>> p1_out, U1_out, T_out = side_branch.propagate(
    ...     p1_in=1000+0j, U1_in=0.0003+0j, T_m=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(
        self,
        segments: list[Segment],
        name: str = "",
    ) -> None:
        """
        Initialize a SideBranch.

        Parameters
        ----------
        segments : list[Segment]
            List of segments forming the side branch.
        name : str, optional
            Name identifier for the side branch.

        Raises
        ------
        ValueError
            If segments list is empty.
        TypeError
            If any element is not a Segment instance.
        """
        if not segments:
            raise ValueError("segments list cannot be empty")

        for i, seg in enumerate(segments):
            if not isinstance(seg, Segment):
                raise TypeError(
                    f"Element {i} must be a Segment instance, got {type(seg)}"
                )

        self._segments = list(segments)
        self._name = name

    @property
    def segments(self) -> list[Segment]:
        """
        List of segments in the side branch.

        Returns
        -------
        list[Segment]
            Segments in order from inlet to outlet.
        """
        return self._segments.copy()

    @property
    def name(self) -> str:
        """
        Name identifier for the side branch.

        Returns
        -------
        str
            The side branch name.
        """
        return self._name

    @property
    def total_length(self) -> float:
        """
        Total length of the side branch.

        Returns
        -------
        float
            Sum of all segment lengths (m).
        """
        return sum(seg.length for seg in self._segments)

    def propagate(
        self,
        p1_in: complex,
        U1_in: complex,
        T_m: float,
        omega: float,
        gas: Gas,
    ) -> tuple[complex, complex, float]:
        """
        Propagate acoustic state through all side branch segments.

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at side branch inlet (Pa).
        U1_in : complex
            Complex volumetric velocity at side branch inlet (m^3/s).
        T_m : float
            Mean temperature at side branch inlet (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object providing thermophysical properties.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_out, U1_out, T_m_out) at side branch outlet.
        """
        p1 = p1_in
        U1 = U1_in
        T = T_m

        for segment in self._segments:
            p1, U1, T = segment.propagate(p1, U1, T, omega, gas)

        return p1, U1, T

    def __len__(self) -> int:
        """Return the number of segments in the side branch."""
        return len(self._segments)

    def __repr__(self) -> str:
        """Return string representation of the side branch."""
        n_segs = len(self._segments)
        return (
            f"SideBranch(name='{self._name}', {n_segs} segments, "
            f"total_length={self.total_length:.4f} m)"
        )

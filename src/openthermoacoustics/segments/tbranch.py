"""reference baseline-compatible TBRANCH and UNION segments for loop topologies.

This module provides segments that match reference baseline's TBRANCH and UNION behavior
for modeling toroidal/looped acoustic networks.

reference baseline's approach:
- TBRANCH splits flow based on branch impedance Zb: U_branch = p1 / Zb
- Main path continues as "trunk"
- Branch path goes through segments then hits SOFTEND
- UNION reconnects by matching the SOFTEND pressure to trunk pressure
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.segments.base import Segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas


class TBranchImpedance(Segment):
    """
    reference baseline-compatible T-branch using branch impedance.

    At a T-junction, the pressure p1 is continuous. The flow splits based
    on the branch impedance Zb:
        U1_branch = p1 / Zb
        U1_trunk = U1_in - U1_branch

    This matches reference baseline's TBRANCH segment.

    Parameters
    ----------
    Zb_real : float
        Real part of branch impedance Re(Zb) in Pa-s/m³.
    Zb_imag : float
        Imaginary part of branch impedance Im(Zb) in Pa-s/m³.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    Zb : complex
        Branch impedance in Pa-s/m³.

    Notes
    -----
    The branch impedance Zb is typically a guess variable that gets
    iterated to satisfy loop closure conditions at the UNION segment.

    For a simple branch with just compliance: Zb ≈ 1/(j*omega*C)
    For a branch with inertance and compliance: Zb ≈ j*omega*L + 1/(j*omega*C)

    Examples
    --------
    >>> tbranch = TBranchImpedance(Zb_real=3.726e5, Zb_imag=-2.701e5)
    >>> p1_trunk, U1_trunk, T_trunk = tbranch.propagate(
    ...     p1_in=2000+0j, U1_in=0.00615+0.0047j, T_m=300.0, omega=377, gas=air
    ... )
    >>> p1_branch, U1_branch, T_branch = tbranch.get_branch_state()
    """

    def __init__(
        self,
        Zb_real: float,
        Zb_imag: float,
        name: str = "",
    ) -> None:
        """
        Initialize a T-branch with impedance-based flow split.

        Parameters
        ----------
        Zb_real : float
            Real part of branch impedance Re(Zb) in Pa-s/m³.
        Zb_imag : float
            Imaginary part of branch impedance Im(Zb) in Pa-s/m³.
        name : str, optional
            Name identifier for the segment.
        """
        self._Zb = complex(Zb_real, Zb_imag)

        # State storage for branch (populated during propagate)
        self._branch_p1: complex = 0j
        self._branch_U1: complex = 0j
        self._branch_T_m: float = 300.0

        # Computed outputs
        self._Edot_trunk: float = 0.0
        self._Edot_branch: float = 0.0
        self._Htot_branch: float = 0.0

        # Lumped element: length = 0
        super().__init__(name=name, length=0.0, area=1.0, geometry=None)

    @property
    def Zb(self) -> complex:
        """Branch impedance in Pa-s/m³."""
        return self._Zb

    @Zb.setter
    def Zb(self, value: complex) -> None:
        """Set branch impedance."""
        self._Zb = complex(value)

    @property
    def Zb_real(self) -> float:
        """Real part of branch impedance."""
        return self._Zb.real

    @property
    def Zb_imag(self) -> float:
        """Imaginary part of branch impedance."""
        return self._Zb.imag

    def set_Zb(self, Zb_real: float, Zb_imag: float) -> None:
        """
        Set the branch impedance.

        Parameters
        ----------
        Zb_real : float
            Real part of branch impedance Re(Zb) in Pa-s/m³.
        Zb_imag : float
            Imaginary part of branch impedance Im(Zb) in Pa-s/m³.
        """
        self._Zb = complex(Zb_real, Zb_imag)

    def get_branch_state(self) -> tuple[complex, complex, float]:
        """
        Get the acoustic state entering the branch.

        Returns
        -------
        tuple[complex, complex, float]
            Tuple of (p1_branch, U1_branch, T_m_branch).

        Notes
        -----
        Call this after propagate() to get the state for the branch path.
        """
        return self._branch_p1, self._branch_U1, self._branch_T_m

    @property
    def Edot_trunk(self) -> float:
        """Acoustic power in trunk after split (W)."""
        return self._Edot_trunk

    @property
    def Edot_branch(self) -> float:
        """Acoustic power entering branch (W)."""
        return self._Edot_branch

    @property
    def Htot_branch(self) -> float:
        """Total enthalpy flux in branch (W). Set externally if needed."""
        return self._Htot_branch

    @Htot_branch.setter
    def Htot_branch(self, value: float) -> None:
        self._Htot_branch = value

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """Return zero derivatives (lumped element)."""
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
        Propagate acoustic state through the T-branch.

        At the junction:
        - Pressure is continuous: p1_trunk = p1_branch = p1_in
        - Branch flow: U1_branch = p1_in / Zb
        - Trunk flow: U1_trunk = U1_in - U1_branch

        Parameters
        ----------
        p1_in : complex
            Complex pressure amplitude at input (Pa).
        U1_in : complex
            Complex volumetric velocity at input (m^3/s).
        T_m : float
            Mean temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.

        Returns
        -------
        tuple[complex, complex, float]
            (p1_trunk, U1_trunk, T_m) - state continuing in trunk.
        """
        # Pressure is continuous
        p1_trunk = p1_in
        self._branch_p1 = p1_in
        self._branch_T_m = T_m

        # Flow splits based on impedance
        if abs(self._Zb) > 1e-10:
            self._branch_U1 = p1_in / self._Zb
        else:
            # Very low impedance - all flow goes to branch
            self._branch_U1 = U1_in

        U1_trunk = U1_in - self._branch_U1

        # Compute acoustic power
        self._Edot_trunk = 0.5 * np.real(p1_trunk * np.conj(U1_trunk))
        self._Edot_branch = 0.5 * np.real(self._branch_p1 * np.conj(self._branch_U1))

        return p1_trunk, U1_trunk, T_m

    def __repr__(self) -> str:
        return (
            f"TBranchImpedance(name='{self._name}', "
            f"Zb={self._Zb.real:.3e}{self._Zb.imag:+.3e}j)"
        )


class Union(Segment):
    """
    reference baseline-compatible UNION segment for reconnecting branches.

    The UNION segment reconnects a branch (terminated by SOFTEND) back
    to the main trunk. For proper loop closure:
    - The branch pressure must equal the trunk pressure
    - The velocities combine: U1_out = U1_trunk + U1_branch

    Parameters
    ----------
    softend_segment : SoftEndWithState
        Reference to the SOFTEND segment at end of branch.
    name : str, optional
        Name identifier for the segment.

    Attributes
    ----------
    pressure_mismatch : complex
        Difference between branch and trunk pressure (should be ~0 for closure).

    Notes
    -----
    In reference baseline, UNION has target constraints that the solver iterates to satisfy.
    Here we compute the mismatch which can be used by an external solver.
    """

    def __init__(
        self,
        name: str = "",
    ) -> None:
        """
        Initialize a UNION segment.

        Parameters
        ----------
        name : str, optional
            Name identifier for the segment.
        """
        # State from branch (set before propagate)
        self._branch_p1: complex = 0j
        self._branch_U1: complex = 0j
        self._branch_T_m: float = 300.0
        self._branch_state_set: bool = False

        # Computed values
        self._pressure_mismatch: complex = 0j

        super().__init__(name=name, length=0.0, area=1.0, geometry=None)

    def set_branch_state(
        self,
        p1_branch: complex,
        U1_branch: complex,
        T_m_branch: float,
    ) -> None:
        """
        Set the state arriving from the branch.

        Parameters
        ----------
        p1_branch : complex
            Pressure at branch end (from SOFTEND) (Pa).
        U1_branch : complex
            Velocity at branch end (m^3/s).
        T_m_branch : float
            Temperature at branch end (K).
        """
        self._branch_p1 = p1_branch
        self._branch_U1 = U1_branch
        self._branch_T_m = T_m_branch
        self._branch_state_set = True

    @property
    def pressure_mismatch(self) -> complex:
        """
        Pressure difference between branch and trunk.

        Returns
        -------
        complex
            p1_branch - p1_trunk. Should be ~0 for proper loop closure.
        """
        return self._pressure_mismatch

    @property
    def pressure_mismatch_magnitude(self) -> float:
        """Magnitude of pressure mismatch (Pa)."""
        return abs(self._pressure_mismatch)

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
        """Return zero derivatives (lumped element)."""
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
        Propagate through UNION, combining trunk and branch flows.

        Parameters
        ----------
        p1_in : complex
            Trunk pressure at junction (Pa).
        U1_in : complex
            Trunk velocity at junction (m^3/s).
        T_m : float
            Trunk temperature (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas object.

        Returns
        -------
        tuple[complex, complex, float]
            (p1_out, U1_out, T_m_out) - combined state.

        Raises
        ------
        RuntimeError
            If branch state not set via set_branch_state().
        """
        if not self._branch_state_set:
            raise RuntimeError(
                "Branch state not set. Call set_branch_state() first."
            )

        # Compute pressure mismatch
        self._pressure_mismatch = self._branch_p1 - p1_in

        # Output pressure is trunk pressure (branch should match)
        p1_out = p1_in

        # Velocities combine
        U1_out = U1_in + self._branch_U1

        # Use trunk temperature
        T_m_out = T_m

        return p1_out, U1_out, T_m_out

    def __repr__(self) -> str:
        return f"Union(name='{self._name}')"


class SoftEndWithState(Segment):
    """
    SOFTEND segment that stores state for UNION reconnection.

    In reference baseline, SOFTEND marks the end of a branch path with a specified
    impedance condition. It stores the final state for the UNION segment.

    Parameters
    ----------
    Re_z : float, optional
        Real part of normalized impedance. Default 0 (pressure release).
    Im_z : float, optional
        Imaginary part of normalized impedance. Default 0.
    name : str, optional
        Name identifier.

    Notes
    -----
    For a pressure release boundary (ideal open end): Re_z = Im_z = 0
    This means p1 = 0 at the boundary.
    """

    def __init__(
        self,
        Re_z: float = 0.0,
        Im_z: float = 0.0,
        name: str = "",
    ) -> None:
        self._Re_z = Re_z
        self._Im_z = Im_z

        # Stored final state
        self._final_p1: complex = 0j
        self._final_U1: complex = 0j
        self._final_T_m: float = 300.0

        super().__init__(name=name, length=0.0, area=1.0, geometry=None)

    @property
    def normalized_impedance(self) -> complex:
        """Normalized impedance z = Z / (rho*a/A)."""
        return complex(self._Re_z, self._Im_z)

    def get_final_state(self) -> tuple[complex, complex, float]:
        """Get the stored final state."""
        return self._final_p1, self._final_U1, self._final_T_m

    def get_derivatives(
        self,
        x: float,
        y: NDArray[np.float64],
        omega: float,
        gas: Gas,
        T_m: float,
    ) -> NDArray[np.float64]:
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
        Store the arriving state and pass through.

        The SOFTEND doesn't change the acoustic state - it just marks
        the end of the branch and stores the state for UNION.
        """
        self._final_p1 = p1_in
        self._final_U1 = U1_in
        self._final_T_m = T_m

        return p1_in, U1_in, T_m

    def __repr__(self) -> str:
        return f"SoftEndWithState(name='{self._name}', z={self._Re_z}+{self._Im_z}j)"

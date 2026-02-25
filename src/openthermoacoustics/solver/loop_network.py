"""Loop network topology for traveling-wave thermoacoustic systems.

This module extends NetworkTopology to handle branching topologies with
side branches and feedback loops, as found in traveling-wave engines
like the Backhaus-Swift TASHE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.solver.network import NetworkTopology, SegmentResult
from openthermoacoustics.segments.branch import TBranch, Return, SideBranch

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas
    from openthermoacoustics.segments.base import Segment


@dataclass
class BranchInfo:
    """
    Information about a side branch in the loop network.

    Attributes
    ----------
    tbranch_index : int
        Index of the TBranch segment in the main network.
    return_index : int
        Index of the Return segment in the main network.
    side_branch : SideBranch
        The SideBranch object containing the side branch segments.
    tbranch : TBranch
        Reference to the TBranch segment.
    return_seg : Return
        Reference to the Return segment.
    """

    tbranch_index: int
    return_index: int
    side_branch: SideBranch
    tbranch: TBranch
    return_seg: Return


class LoopNetwork(NetworkTopology):
    """
    Network topology for systems with side branches and feedback loops.

    This class extends NetworkTopology to handle branching topologies where
    flow can be diverted into side branches (via TBranch segments) and
    return to the main duct (via Return segments). This is essential for
    modeling traveling-wave thermoacoustic engines.

    The key challenge in loop networks is loop closure: the acoustic state
    returning from a side branch must match the state at the return junction.
    This is solved iteratively by adjusting the side branch flow fraction.

    Parameters
    ----------
    None

    Attributes
    ----------
    branches : list[BranchInfo]
        Information about each side branch in the network.

    Notes
    -----
    For proper loop closure:
    1. Pressure must be continuous at both TBranch and Return junctions
    2. Volumetric velocity must split at TBranch and recombine at Return
    3. The fraction of flow going to the side branch must be adjusted
       until the returning pressure matches the junction pressure

    This is an eigenvalue problem that requires iterative solution.

    Examples
    --------
    >>> from openthermoacoustics.solver import LoopNetwork
    >>> from openthermoacoustics.segments import Duct, TBranch, Return, Stack
    >>> # Create main duct segments
    >>> duct1 = Duct(length=0.5, radius=0.05)
    >>> tbranch = TBranch(side_area=0.001)
    >>> duct2 = Duct(length=0.3, radius=0.05)
    >>> return_seg = Return(tbranch=tbranch)
    >>> duct3 = Duct(length=0.5, radius=0.05)
    >>> # Create side branch
    >>> side_duct = Duct(length=0.1, radius=0.015)
    >>> stack = Stack(length=0.05, porosity=0.7, hydraulic_radius=0.0005)
    >>> # Build network
    >>> network = LoopNetwork()
    >>> network.add_segment(duct1)
    >>> network.add_segment(tbranch)
    >>> network.add_segment(duct2)
    >>> network.add_segment(return_seg)
    >>> network.add_segment(duct3)
    >>> network.add_branch(
    ...     tbranch_index=1,
    ...     segments=[side_duct, stack],
    ...     return_index=3
    ... )
    >>> results = network.propagate_all(
    ...     p1_start=1000+0j, U1_start=0.001+0j,
    ...     T_m_start=300.0, omega=628.3, gas=gas
    ... )
    """

    def __init__(self) -> None:
        """Initialize an empty loop network topology."""
        super().__init__()
        self._branches: list[BranchInfo] = []
        self._branch_results: list[list[SegmentResult]] = []
        self._loop_closure_tolerance: float = 1e-6
        self._max_iterations: int = 100

    @property
    def branches(self) -> list[BranchInfo]:
        """
        Information about side branches in the network.

        Returns
        -------
        list[BranchInfo]
            List of BranchInfo objects for each side branch.
        """
        return self._branches.copy()

    @property
    def branch_results(self) -> list[list[SegmentResult]]:
        """
        Results from propagating through side branches.

        Returns
        -------
        list[list[SegmentResult]]
            Results for each segment in each side branch.
        """
        return [r.copy() for r in self._branch_results]

    @property
    def loop_closure_tolerance(self) -> float:
        """
        Tolerance for loop closure convergence.

        Returns
        -------
        float
            Relative tolerance for pressure mismatch.
        """
        return self._loop_closure_tolerance

    @loop_closure_tolerance.setter
    def loop_closure_tolerance(self, value: float) -> None:
        """Set the loop closure tolerance."""
        if value <= 0:
            raise ValueError(f"tolerance must be positive, got {value}")
        self._loop_closure_tolerance = value

    @property
    def max_iterations(self) -> int:
        """
        Maximum iterations for loop closure solving.

        Returns
        -------
        int
            Maximum number of iterations.
        """
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        """Set the maximum number of iterations."""
        if value < 1:
            raise ValueError(f"max_iterations must be >= 1, got {value}")
        self._max_iterations = value

    def add_branch(
        self,
        tbranch_index: int,
        segments: list[Segment],
        return_index: int,
    ) -> None:
        """
        Add a side branch to the network.

        Parameters
        ----------
        tbranch_index : int
            Index of the TBranch segment in the main network.
        segments : list[Segment]
            List of segments forming the side branch.
        return_index : int
            Index of the Return segment in the main network.

        Raises
        ------
        ValueError
            If indices are invalid or segments list is empty.
        TypeError
            If the segment at tbranch_index is not a TBranch,
            or the segment at return_index is not a Return.
        """
        # Validate indices
        n_segments = len(self._segments)
        if not 0 <= tbranch_index < n_segments:
            raise ValueError(
                f"tbranch_index {tbranch_index} out of range [0, {n_segments})"
            )
        if not 0 <= return_index < n_segments:
            raise ValueError(
                f"return_index {return_index} out of range [0, {n_segments})"
            )
        if tbranch_index >= return_index:
            raise ValueError(
                f"tbranch_index ({tbranch_index}) must be less than "
                f"return_index ({return_index})"
            )

        # Validate segment types
        tbranch = self._segments[tbranch_index]
        if not isinstance(tbranch, TBranch):
            raise TypeError(
                f"Segment at index {tbranch_index} must be TBranch, "
                f"got {type(tbranch)}"
            )

        return_seg = self._segments[return_index]
        if not isinstance(return_seg, Return):
            raise TypeError(
                f"Segment at index {return_index} must be Return, "
                f"got {type(return_seg)}"
            )

        # Validate that Return references this TBranch
        if return_seg.tbranch is not tbranch:
            raise ValueError(
                f"Return segment does not reference the TBranch at "
                f"index {tbranch_index}"
            )

        # Create SideBranch
        side_branch = SideBranch(segments, name=f"branch_{len(self._branches)}")

        # Store branch info
        branch_info = BranchInfo(
            tbranch_index=tbranch_index,
            return_index=return_index,
            side_branch=side_branch,
            tbranch=tbranch,
            return_seg=return_seg,
        )
        self._branches.append(branch_info)

    def clear(self) -> None:
        """Remove all segments and branches from the network."""
        super().clear()
        self._branches.clear()
        self._branch_results.clear()

    def propagate_all(
        self,
        p1_start: complex,
        U1_start: complex,
        T_m_start: float,
        omega: float,
        gas: Gas,
        n_points_per_segment: int = 100,
    ) -> list[SegmentResult]:
        """
        Propagate acoustic waves through the loop network.

        For networks with side branches, this method iteratively solves
        for loop closure by adjusting the side branch flow fractions.

        Parameters
        ----------
        p1_start : complex
            Complex pressure amplitude at network inlet (Pa).
        U1_start : complex
            Complex volumetric velocity amplitude at network inlet (m^3/s).
        T_m_start : float
            Mean temperature at network inlet (K).
        omega : float
            Angular frequency (rad/s).
        gas : Gas
            Gas properties object.
        n_points_per_segment : int, optional
            Number of output points per segment, by default 100.

        Returns
        -------
        list[SegmentResult]
            Results for each segment in the main network.

        Raises
        ------
        ValueError
            If the network has no segments.
        RuntimeError
            If loop closure iteration does not converge.

        Notes
        -----
        The iteration procedure:
        1. Propagate through main network with current side fractions
        2. Propagate through each side branch
        3. Check loop closure (pressure match at Return junctions)
        4. Adjust side fractions and repeat until converged

        The side branch results are stored in `branch_results`.
        """
        if not self._segments:
            raise ValueError("Network has no segments. Add segments first.")

        # If no branches, use simple propagation
        if not self._branches:
            return super().propagate_all(
                p1_start, U1_start, T_m_start, omega, gas, n_points_per_segment
            )

        # Iterative loop closure
        converged = False
        iteration = 0

        while not converged and iteration < self._max_iterations:
            iteration += 1

            # Propagate through main network and side branches
            self._propagate_with_branches(
                p1_start, U1_start, T_m_start, omega, gas, n_points_per_segment
            )

            # Check loop closure for all branches
            max_mismatch = 0.0
            for branch_info in self._branches:
                # Get pressure at Return junction from main network
                return_idx = branch_info.return_index
                if return_idx > 0:
                    # Pressure entering the Return (from previous segment)
                    p1_main = self._results[return_idx - 1].p1[-1]
                else:
                    p1_main = p1_start

                # Get pressure mismatch
                mismatch = branch_info.return_seg.get_pressure_mismatch(p1_main)
                rel_mismatch = abs(mismatch) / max(abs(p1_main), 1e-10)
                max_mismatch = max(max_mismatch, rel_mismatch)

            # Check convergence
            if max_mismatch < self._loop_closure_tolerance:
                converged = True
            else:
                # Adjust side fractions
                self._adjust_side_fractions(gas, omega)

        if not converged:
            raise RuntimeError(
                f"Loop closure did not converge after {self._max_iterations} "
                f"iterations. Final mismatch: {max_mismatch:.2e}"
            )

        return self._results.copy()

    def _propagate_with_branches(
        self,
        p1_start: complex,
        U1_start: complex,
        T_m_start: float,
        omega: float,
        gas: Gas,
        n_points_per_segment: int,
    ) -> None:
        """
        Propagate through main network and all side branches.

        This internal method handles the complex propagation through
        a network with side branches, ensuring proper state handling
        at TBranch and Return junctions.
        """
        from openthermoacoustics.solver.integrator import integrate_segment

        self._results.clear()
        self._branch_results = [[] for _ in self._branches]

        x_cumulative = 0.0
        p1_in = p1_start
        U1_in = U1_start
        T_m_in = T_m_start

        # Build lookup for branch positions
        tbranch_indices = {b.tbranch_index: i for i, b in enumerate(self._branches)}
        return_indices = {b.return_index: i for i, b in enumerate(self._branches)}

        for seg_idx, segment in enumerate(self._segments):
            # Check if this is a TBranch
            if seg_idx in tbranch_indices:
                branch_idx = tbranch_indices[seg_idx]
                branch_info = self._branches[branch_idx]

                # Propagate through TBranch
                p1_out, U1_out, T_m_out = segment.propagate(
                    p1_in, U1_in, T_m_in, omega, gas
                )

                # Store result for TBranch (lumped element)
                seg_result = SegmentResult(
                    segment=segment,
                    x=np.array([0.0]),
                    x_global=np.array([x_cumulative]),
                    p1=np.array([p1_out]),
                    U1=np.array([U1_out]),
                    T_m=np.array([T_m_out]),
                    acoustic_power=np.array(
                        [0.5 * np.real(p1_out * np.conj(U1_out))]
                    ),
                )
                self._results.append(seg_result)

                # Propagate through side branch
                p1_side, U1_side, T_m_side = branch_info.tbranch.get_side_branch_state()
                side_results = []

                for side_seg in branch_info.side_branch.segments:
                    if side_seg.length > 0:
                        result_dict = integrate_segment(
                            segment=side_seg,
                            p1_in=p1_side,
                            U1_in=U1_side,
                            T_m=T_m_side,
                            omega=omega,
                            gas=gas,
                            n_points=n_points_per_segment,
                        )
                        side_result = SegmentResult(
                            segment=side_seg,
                            x=result_dict["x"],
                            x_global=result_dict["x"],  # Relative to side branch
                            p1=result_dict["p1"],
                            U1=result_dict["U1"],
                            T_m=result_dict["T_m"],
                            acoustic_power=result_dict["acoustic_power"],
                        )
                        side_results.append(side_result)
                        p1_side = result_dict["p1"][-1]
                        U1_side = result_dict["U1"][-1]
                        T_m_side = result_dict["T_m"][-1]
                    else:
                        # Lumped element
                        p1_side, U1_side, T_m_side = side_seg.propagate(
                            p1_side, U1_side, T_m_side, omega, gas
                        )
                        side_result = SegmentResult(
                            segment=side_seg,
                            x=np.array([0.0]),
                            x_global=np.array([0.0]),
                            p1=np.array([p1_side]),
                            U1=np.array([U1_side]),
                            T_m=np.array([T_m_side]),
                            acoustic_power=np.array(
                                [0.5 * np.real(p1_side * np.conj(U1_side))]
                            ),
                        )
                        side_results.append(side_result)

                self._branch_results[branch_idx] = side_results

                # Set return state
                branch_info.return_seg.set_return_state(
                    p1_return=p1_side,
                    U1_return=U1_side,
                    T_m_return=T_m_side,
                )

                # Continue in main duct
                p1_in = p1_out
                U1_in = U1_out
                T_m_in = T_m_out

            elif seg_idx in return_indices:
                # This is a Return segment
                # Propagate through Return (combines flows)
                p1_out, U1_out, T_m_out = segment.propagate(
                    p1_in, U1_in, T_m_in, omega, gas
                )

                # Store result for Return (lumped element)
                seg_result = SegmentResult(
                    segment=segment,
                    x=np.array([0.0]),
                    x_global=np.array([x_cumulative]),
                    p1=np.array([p1_out]),
                    U1=np.array([U1_out]),
                    T_m=np.array([T_m_out]),
                    acoustic_power=np.array(
                        [0.5 * np.real(p1_out * np.conj(U1_out))]
                    ),
                )
                self._results.append(seg_result)

                p1_in = p1_out
                U1_in = U1_out
                T_m_in = T_m_out

            else:
                # Regular segment - use standard propagation
                if segment.length > 0:
                    result_dict = integrate_segment(
                        segment=segment,
                        p1_in=p1_in,
                        U1_in=U1_in,
                        T_m=T_m_in,
                        omega=omega,
                        gas=gas,
                        n_points=n_points_per_segment,
                    )

                    x_local = result_dict["x"]
                    x_global = x_local + x_cumulative

                    seg_result = SegmentResult(
                        segment=segment,
                        x=x_local,
                        x_global=x_global,
                        p1=result_dict["p1"],
                        U1=result_dict["U1"],
                        T_m=result_dict["T_m"],
                        acoustic_power=result_dict["acoustic_power"],
                    )
                    self._results.append(seg_result)

                    x_cumulative += segment.length
                    p1_in = result_dict["p1"][-1]
                    U1_in = result_dict["U1"][-1]
                    T_m_in = result_dict["T_m"][-1]
                else:
                    # Lumped element
                    p1_out, U1_out, T_m_out = segment.propagate(
                        p1_in, U1_in, T_m_in, omega, gas
                    )

                    seg_result = SegmentResult(
                        segment=segment,
                        x=np.array([0.0]),
                        x_global=np.array([x_cumulative]),
                        p1=np.array([p1_out]),
                        U1=np.array([U1_out]),
                        T_m=np.array([T_m_out]),
                        acoustic_power=np.array(
                            [0.5 * np.real(p1_out * np.conj(U1_out))]
                        ),
                    )
                    self._results.append(seg_result)

                    p1_in = p1_out
                    U1_in = U1_out
                    T_m_in = T_m_out

    def _adjust_side_fractions(self, gas: Gas, omega: float) -> None:
        """
        Adjust side branch flow fractions to improve loop closure.

        This uses a simple gradient-based adjustment. More sophisticated
        methods (Newton-Raphson, secant) could be implemented for faster
        convergence.
        """
        step_size = 0.1  # Fraction adjustment step

        for branch_info in self._branches:
            # Get current mismatch
            return_idx = branch_info.return_index
            if return_idx > 0 and return_idx <= len(self._results):
                p1_main = self._results[return_idx - 1].p1[-1]
            else:
                # Use a reference pressure
                p1_main = self._results[0].p1[0] if self._results else 1000.0

            mismatch = branch_info.return_seg.get_pressure_mismatch(p1_main)

            # Adjust side fraction based on mismatch phase
            # This is a heuristic; more sophisticated algorithms exist
            current_fraction = branch_info.tbranch.side_fraction

            # If pressure is too high in return, reduce side fraction
            # If pressure is too low, increase side fraction
            phase = np.angle(mismatch)
            adjustment = -step_size * np.sign(np.real(mismatch))

            new_fraction = current_fraction + adjustment
            new_fraction = max(0.01, min(0.99, new_fraction))  # Keep in bounds

            branch_info.tbranch.set_side_fraction(new_fraction)

    def get_branch_profiles(
        self,
        branch_index: int = 0,
    ) -> dict[str, NDArray[Any]]:
        """
        Get concatenated profiles for a side branch.

        Parameters
        ----------
        branch_index : int, optional
            Index of the branch to get profiles for, by default 0.

        Returns
        -------
        dict[str, NDArray[Any]]
            Dictionary with concatenated arrays:
            - 'x': position along branch (m)
            - 'p1': complex pressure amplitude (Pa)
            - 'U1': complex volumetric velocity amplitude (m^3/s)
            - 'T_m': mean temperature (K)
            - 'acoustic_power': time-averaged acoustic power (W)

        Raises
        ------
        ValueError
            If branch_index is out of range or no results available.
        """
        if branch_index >= len(self._branch_results):
            raise ValueError(
                f"branch_index {branch_index} out of range "
                f"[0, {len(self._branch_results)})"
            )

        results = self._branch_results[branch_index]
        if not results:
            raise ValueError(
                "No branch results available. Call propagate_all first."
            )

        # Concatenate arrays from all segments
        x_arrays = []
        p1_arrays = []
        U1_arrays = []
        T_m_arrays = []
        power_arrays = []

        x_cumulative = 0.0
        for i, result in enumerate(results):
            if i == 0:
                x_arrays.append(result.x + x_cumulative)
                p1_arrays.append(result.p1)
                U1_arrays.append(result.U1)
                T_m_arrays.append(result.T_m)
                power_arrays.append(result.acoustic_power)
            else:
                # Skip first point (duplicate)
                x_arrays.append(result.x[1:] + x_cumulative)
                p1_arrays.append(result.p1[1:])
                U1_arrays.append(result.U1[1:])
                T_m_arrays.append(result.T_m[1:])
                power_arrays.append(result.acoustic_power[1:])

            x_cumulative += result.segment.length

        return {
            "x": np.concatenate(x_arrays) if x_arrays else np.array([]),
            "p1": np.concatenate(p1_arrays) if p1_arrays else np.array([], dtype=complex),
            "U1": np.concatenate(U1_arrays) if U1_arrays else np.array([], dtype=complex),
            "T_m": np.concatenate(T_m_arrays) if T_m_arrays else np.array([]),
            "acoustic_power": np.concatenate(power_arrays) if power_arrays else np.array([]),
        }

    def verify_mass_conservation(self) -> dict[str, float]:
        """
        Verify mass conservation through the network.

        Checks that volumetric velocity is properly split at TBranch
        junctions and recombined at Return junctions.

        Returns
        -------
        dict[str, float]
            Dictionary with conservation checks:
            - 'max_split_error': Maximum error at TBranch junctions
            - 'max_combine_error': Maximum error at Return junctions

        Raises
        ------
        ValueError
            If no results are available.
        """
        if not self._results:
            raise ValueError(
                "No results available. Call propagate_all first."
            )

        max_split_error = 0.0
        max_combine_error = 0.0

        for branch_info in self._branches:
            # Check TBranch
            tbranch_idx = branch_info.tbranch_index
            tbranch = branch_info.tbranch

            # Get U1 before TBranch
            if tbranch_idx > 0:
                U1_before = self._results[tbranch_idx - 1].U1[-1]
            else:
                U1_before = self._results[tbranch_idx].U1[0]

            # Get U1 after TBranch (main) and side
            U1_main_after = self._results[tbranch_idx].U1[-1]
            U1_side = tbranch.side_U1

            # Check split: U1_before = U1_main_after + U1_side
            split_error = abs(U1_before - (U1_main_after + U1_side))
            rel_split_error = split_error / max(abs(U1_before), 1e-10)
            max_split_error = max(max_split_error, rel_split_error)

            # Check Return
            return_idx = branch_info.return_index
            return_seg = branch_info.return_seg

            # Get U1 before Return (main) and returning
            if return_idx > 0:
                U1_main_before = self._results[return_idx - 1].U1[-1]
            else:
                U1_main_before = 0j

            U1_return = return_seg.return_U1

            # Get U1 after Return
            U1_after = self._results[return_idx].U1[-1]

            # Check combine: U1_after = U1_main_before + U1_return
            combine_error = abs(U1_after - (U1_main_before + U1_return))
            rel_combine_error = combine_error / max(abs(U1_after), 1e-10)
            max_combine_error = max(max_combine_error, rel_combine_error)

        return {
            "max_split_error": max_split_error,
            "max_combine_error": max_combine_error,
        }

    def __repr__(self) -> str:
        """Return a string representation of the loop network."""
        n_segments = len(self._segments)
        n_branches = len(self._branches)
        total_len = self.total_length
        return (
            f"LoopNetwork({n_segments} segments, {n_branches} branches, "
            f"total_length={total_len:.4f} m)"
        )

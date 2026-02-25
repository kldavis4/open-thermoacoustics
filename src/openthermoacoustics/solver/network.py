"""Network topology and assembly for thermoacoustic systems.

This module provides classes for building and analyzing thermoacoustic
network topologies, managing the connection between segments and
propagating acoustic waves through the entire system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from openthermoacoustics.solver.integrator import integrate_segment
from openthermoacoustics.utils import acoustic_power

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas
    from openthermoacoustics.segments.base import Segment


@dataclass
class SegmentResult:
    """
    Results from integrating through a single segment.

    Attributes
    ----------
    segment : Segment
        The segment that was integrated.
    x : NDArray[np.float64]
        Position array relative to segment start (m).
    x_global : NDArray[np.float64]
        Position array relative to network start (m).
    p1 : NDArray[np.complex128]
        Complex pressure amplitude at each position (Pa).
    U1 : NDArray[np.complex128]
        Complex volumetric velocity amplitude at each position (m^3/s).
    T_m : NDArray[np.float64]
        Mean temperature at each position (K).
    acoustic_power : NDArray[np.float64]
        Time-averaged acoustic power at each position (W).
    """

    segment: Segment
    x: NDArray[np.float64]
    x_global: NDArray[np.float64]
    p1: NDArray[np.complex128]
    U1: NDArray[np.complex128]
    T_m: NDArray[np.float64]
    acoustic_power: NDArray[np.float64]


class NetworkTopology:
    """
    Manages the topology and propagation through a thermoacoustic network.

    This class handles sequential (linear) network topologies where segments
    are connected end-to-end. At area discontinuities between segments,
    pressure and volumetric velocity (U1) are both continuous, following the
    standard thermoacoustic "lumped" approximation for mass conservation.

    Parameters
    ----------
    None

    Attributes
    ----------
    segments : list[Segment]
        List of segments in the network, in order.
    results : list[SegmentResult]
        Results from the most recent propagation.

    Examples
    --------
    >>> from openthermoacoustics.segments import Duct, Stack
    >>> from openthermoacoustics.gas import Helium
    >>> network = NetworkTopology()
    >>> network.add_segment(Duct(length=0.1, diameter=0.05))
    >>> network.add_segment(Stack(length=0.03, ...))
    >>> network.add_segment(Duct(length=0.1, diameter=0.05))
    >>> gas = Helium(mean_pressure=1e6)
    >>> results = network.propagate_all(
    ...     p1_start=1000+0j, U1_start=0.001+0j,
    ...     T_m_start=300.0, omega=2*np.pi*100, gas=gas
    ... )
    """

    def __init__(self) -> None:
        """Initialize an empty network topology."""
        self._segments: list[Segment] = []
        self._results: list[SegmentResult] = []

    @property
    def segments(self) -> list[Segment]:
        """
        List of segments in the network.

        Returns
        -------
        list[Segment]
            Segments in sequential order.
        """
        return self._segments.copy()

    @property
    def results(self) -> list[SegmentResult]:
        """
        Results from the most recent propagation.

        Returns
        -------
        list[SegmentResult]
            Results for each segment.
        """
        return self._results.copy()

    @property
    def total_length(self) -> float:
        """
        Total length of the network.

        Returns
        -------
        float
            Sum of all segment lengths (m).
        """
        return sum(seg.length for seg in self._segments)

    def add_segment(self, segment: Segment) -> None:
        """
        Add a segment to the end of the network.

        Parameters
        ----------
        segment : Segment
            The segment to add. Must have 'length' and 'area' attributes.

        Raises
        ------
        TypeError
            If the segment does not have required attributes.
        """
        # Validate segment has required interface
        if not hasattr(segment, "length"):
            raise TypeError(
                f"Segment {segment!r} must have a 'length' attribute"
            )

        self._segments.append(segment)

    def clear(self) -> None:
        """Remove all segments from the network."""
        self._segments.clear()
        self._results.clear()

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
        Propagate acoustic waves through all segments in the network.

        This method integrates the acoustic wave equations through each
        segment sequentially, applying appropriate boundary conditions
        at segment interfaces (area changes).

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
            Results for each segment in the network.

        Raises
        ------
        ValueError
            If the network has no segments.
        RuntimeError
            If integration through any segment fails.

        Notes
        -----
        At area changes between segments, both pressure and volumetric velocity
        (U1) are continuous. This is the standard thermoacoustic "lumped"
        approximation where mass flux (ρ * U1) is conserved at the interface.
        """
        if not self._segments:
            raise ValueError("Network has no segments. Add segments first.")

        self._results.clear()
        x_cumulative = 0.0

        # Current state at segment inlet
        p1_in = p1_start
        U1_in = U1_start
        T_m_in = T_m_start

        for i, segment in enumerate(self._segments):
            # Integrate through this segment
            result_dict = integrate_segment(
                segment=segment,
                p1_in=p1_in,
                U1_in=U1_in,
                T_m=T_m_in,
                omega=omega,
                gas=gas,
                n_points=n_points_per_segment,
            )

            # Compute global x positions
            x_local = result_dict["x"]
            x_global = x_local + x_cumulative

            # Create result object
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

            # Update cumulative position
            x_cumulative += segment.length

            # Get exit state for next segment
            p1_out = result_dict["p1"][-1]
            U1_out = result_dict["U1"][-1]
            T_m_out = result_dict["T_m"][-1]

            # Apply interface conditions for next segment
            if i < len(self._segments) - 1:
                next_segment = self._segments[i + 1]
                p1_in, U1_in = self._apply_interface_conditions(
                    p1_out, U1_out, segment, next_segment
                )
                T_m_in = T_m_out
            # (else: last segment, no interface to apply)

        return self._results.copy()

    def _apply_interface_conditions(
        self,
        p1: complex,
        U1: complex,
        segment_out: Segment,
        segment_in: Segment,
    ) -> tuple[complex, complex]:
        """
        Apply interface conditions between two segments.

        At an area change, both pressure and volumetric velocity (U1) are
        continuous. This follows the standard thermoacoustic "lumped"
        approximation where mass flux is conserved at the interface.

        Parameters
        ----------
        p1 : complex
            Pressure amplitude at exit of outgoing segment (Pa).
        U1 : complex
            Volumetric velocity amplitude at exit of outgoing segment (m^3/s).
        segment_out : Segment
            The segment being exited.
        segment_in : Segment
            The segment being entered.

        Returns
        -------
        tuple[complex, complex]
            (p1_new, U1_new) at the inlet of the incoming segment.

        Notes
        -----
        The "lumped" approximation assumes the area change occurs over a
        length much shorter than the acoustic wavelength. In this limit,
        mass conservation (ρ * U1 = const) requires U1 continuity for a
        uniform-density gas. For more accurate modeling of area changes,
        minor losses or the Karal model for wave scattering may be needed.
        """
        # Pressure is continuous
        p1_new = p1

        # Volumetric velocity is continuous (mass conservation in lumped limit)
        U1_new = U1

        return p1_new, U1_new

    def _get_segment_inlet_area(self, segment: Segment) -> float:
        """
        Get the inlet area of a segment.

        Parameters
        ----------
        segment : Segment
            The segment.

        Returns
        -------
        float
            Inlet area (m^2).
        """
        if callable(getattr(segment, "area", None)):
            return float(segment.area(0.0))
        if hasattr(segment, "area"):
            return float(segment.area)
        if hasattr(segment, "inlet_area"):
            return float(segment.inlet_area)
        raise AttributeError(
            f"Segment {segment!r} has no 'area' or 'inlet_area' attribute"
        )

    def _get_segment_exit_area(self, segment: Segment) -> float:
        """
        Get the exit area of a segment.

        Parameters
        ----------
        segment : Segment
            The segment.

        Returns
        -------
        float
            Exit area (m^2).
        """
        if callable(getattr(segment, "area", None)):
            return float(segment.area(segment.length))
        if hasattr(segment, "area"):
            return float(segment.area)
        if hasattr(segment, "exit_area"):
            return float(segment.exit_area)
        raise AttributeError(
            f"Segment {segment!r} has no 'area' or 'exit_area' attribute"
        )

    def get_global_profiles(
        self,
    ) -> dict[str, NDArray[Any]]:
        """
        Get concatenated profiles across all segments.

        Returns
        -------
        dict[str, NDArray[Any]]
            Dictionary with concatenated arrays:
            - 'x': global position (m)
            - 'p1': complex pressure amplitude (Pa)
            - 'U1': complex volumetric velocity amplitude (m^3/s)
            - 'T_m': mean temperature (K)
            - 'acoustic_power': time-averaged acoustic power (W)

        Raises
        ------
        ValueError
            If no propagation results are available.
        """
        if not self._results:
            raise ValueError(
                "No propagation results available. Call propagate_all first."
            )

        # Concatenate arrays from all segments
        # Avoid duplicating interface points
        x_arrays = []
        p1_arrays = []
        U1_arrays = []
        T_m_arrays = []
        power_arrays = []

        for i, result in enumerate(self._results):
            if i == 0:
                # Include all points from first segment
                x_arrays.append(result.x_global)
                p1_arrays.append(result.p1)
                U1_arrays.append(result.U1)
                T_m_arrays.append(result.T_m)
                power_arrays.append(result.acoustic_power)
            else:
                # Skip first point (duplicate of previous segment's last point)
                x_arrays.append(result.x_global[1:])
                p1_arrays.append(result.p1[1:])
                U1_arrays.append(result.U1[1:])
                T_m_arrays.append(result.T_m[1:])
                power_arrays.append(result.acoustic_power[1:])

        return {
            "x": np.concatenate(x_arrays),
            "p1": np.concatenate(p1_arrays),
            "U1": np.concatenate(U1_arrays),
            "T_m": np.concatenate(T_m_arrays),
            "acoustic_power": np.concatenate(power_arrays),
        }

    def get_endpoint_values(self) -> dict[str, complex | float]:
        """
        Get the values at the network inlet and outlet.

        Returns
        -------
        dict[str, complex | float]
            Dictionary with endpoint values:
            - 'p1_start': pressure at inlet (Pa)
            - 'U1_start': volumetric velocity at inlet (m^3/s)
            - 'T_m_start': temperature at inlet (K)
            - 'p1_end': pressure at outlet (Pa)
            - 'U1_end': volumetric velocity at outlet (m^3/s)
            - 'T_m_end': temperature at outlet (K)

        Raises
        ------
        ValueError
            If no propagation results are available.
        """
        if not self._results:
            raise ValueError(
                "No propagation results available. Call propagate_all first."
            )

        first = self._results[0]
        last = self._results[-1]

        return {
            "p1_start": first.p1[0],
            "U1_start": first.U1[0],
            "T_m_start": first.T_m[0],
            "p1_end": last.p1[-1],
            "U1_end": last.U1[-1],
            "T_m_end": last.T_m[-1],
        }

    def __len__(self) -> int:
        """Return the number of segments in the network."""
        return len(self._segments)

    def __repr__(self) -> str:
        """Return a string representation of the network."""
        n_segments = len(self._segments)
        total_len = self.total_length
        return (
            f"NetworkTopology({n_segments} segments, "
            f"total_length={total_len:.4f} m)"
        )

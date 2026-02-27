"""Distributed-segment propagation bridge for TBRANCH/UNION loop solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from openthermoacoustics.solver.integrator import integrate_segment

if TYPE_CHECKING:
    from openthermoacoustics.gas.base import Gas
    from openthermoacoustics.segments.base import Segment


@dataclass
class SegmentChainSection:
    """Per-segment profile section from chain propagation."""

    segment_name: str
    segment_type: str
    x_start: float
    x_end: float
    x: np.ndarray
    p1: np.ndarray
    U1: np.ndarray
    T_m: np.ndarray
    acoustic_power: np.ndarray


@dataclass
class SegmentChainResult:
    """Aggregated profile/result for sequential segment propagation."""

    x: np.ndarray
    p1: np.ndarray
    U1: np.ndarray
    T_m: np.ndarray
    acoustic_power: np.ndarray
    sections: list[SegmentChainSection]
    p1_end: complex
    U1_end: complex
    T_m_end: float


def integrate_segment_chain(
    segments: list[Segment],
    p1_start: complex,
    u1_start: complex,
    t_m_start: float,
    omega: complex,
    gas: Gas,
    n_points_per_segment: int = 120,
) -> SegmentChainResult:
    """
    Integrate through a list of distributed segments sequentially.

    Uses the same `integrate_segment` ODE machinery as linear network solves.
    """
    p1_in = p1_start
    u1_in = u1_start
    t_m_in = t_m_start
    x_offset = 0.0

    x_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    u_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    sections: list[SegmentChainSection] = []

    for i, segment in enumerate(segments):
        result = integrate_segment(
            segment=segment,
            p1_in=p1_in,
            U1_in=u1_in,
            T_m=t_m_in,
            omega=omega,
            gas=gas,
            n_points=n_points_per_segment,
        )

        x_local = np.asarray(result["x"], dtype=float)
        p_local = np.asarray(result["p1"], dtype=complex)
        u_local = np.asarray(result["U1"], dtype=complex)
        t_local = np.asarray(result["T_m"], dtype=float)
        w_local = np.asarray(result["acoustic_power"], dtype=float)
        x_global = x_local + x_offset

        # Avoid duplicate boundary points when concatenating contiguous segments.
        if i > 0 and len(x_global) > 0:
            x_global = x_global[1:]
            p_local = p_local[1:]
            u_local = u_local[1:]
            t_local = t_local[1:]
            w_local = w_local[1:]

        x_parts.append(x_global)
        p_parts.append(p_local)
        u_parts.append(u_local)
        t_parts.append(t_local)
        w_parts.append(w_local)

        section = SegmentChainSection(
            segment_name=getattr(segment, "name", "") or "",
            segment_type=type(segment).__name__,
            x_start=x_offset,
            x_end=x_offset + float(segment.length),
            x=x_global,
            p1=p_local,
            U1=u_local,
            T_m=t_local,
            acoustic_power=w_local,
        )
        sections.append(section)

        p1_in = complex(result["p1"][-1])
        u1_in = complex(result["U1"][-1])
        t_m_in = float(result["T_m"][-1])
        x_offset += float(segment.length)

    if x_parts:
        x = np.concatenate(x_parts)
        p1 = np.concatenate(p_parts)
        u1 = np.concatenate(u_parts)
        t_m = np.concatenate(t_parts)
        w = np.concatenate(w_parts)
    else:
        x = np.array([0.0])
        p1 = np.array([p1_start], dtype=complex)
        u1 = np.array([u1_start], dtype=complex)
        t_m = np.array([t_m_start], dtype=float)
        w = np.array([0.5 * np.real(p1_start * np.conj(u1_start))], dtype=float)

    return SegmentChainResult(
        x=x,
        p1=p1,
        U1=u1,
        T_m=t_m,
        acoustic_power=w,
        sections=sections,
        p1_end=complex(p1_in),
        U1_end=complex(u1_in),
        T_m_end=float(t_m_in),
    )


class DistributedLoopPropagator:
    """
    TBRANCH/UNION distributed propagation adapter for `TBranchLoopSolver`.

    Topology:
    - Trunk path: TBRANCH -> ... -> UNION/HARDEND
    - Branch path: TBRANCH -> ... -> UNION
    """

    def __init__(
        self,
        trunk_segments: list[Segment],
        branch_segments: list[Segment],
        gas: Gas,
        omega: complex,
        t_m_start: float,
        n_points_per_segment: int = 120,
    ) -> None:
        self.trunk_segments = trunk_segments
        self.branch_segments = branch_segments
        self.gas = gas
        self.omega = omega
        self.t_m_start = t_m_start
        self.n_points_per_segment = n_points_per_segment
        self.last_trunk: SegmentChainResult | None = None
        self.last_branch: SegmentChainResult | None = None

    def __call__(
        self,
        p1_input: complex,
        u1_input: complex,
        zb: complex,
    ) -> tuple[complex, complex, complex]:
        """
        Propagate both paths and assemble loop-closure residual quantities.
        """
        if abs(zb) < 1e-18:
            raise ValueError("Branch impedance Zb is too close to zero.")

        # TBRANCH split
        u1_branch = p1_input / zb
        u1_trunk = u1_input - u1_branch

        # Propagate trunk and branch independently from shared TBRANCH state.
        trunk = integrate_segment_chain(
            self.trunk_segments,
            p1_start=p1_input,
            u1_start=u1_trunk,
            t_m_start=self.t_m_start,
            omega=self.omega,
            gas=self.gas,
            n_points_per_segment=self.n_points_per_segment,
        )
        branch = integrate_segment_chain(
            self.branch_segments,
            p1_start=p1_input,
            u1_start=u1_branch,
            t_m_start=self.t_m_start,
            omega=self.omega,
            gas=self.gas,
            n_points_per_segment=self.n_points_per_segment,
        )
        self.last_trunk = trunk
        self.last_branch = branch

        # UNION/HARDEND conditions.
        p1_union = trunk.p1_end
        u1_hardend = trunk.U1_end + branch.U1_end
        p_mismatch = branch.p1_end - trunk.p1_end
        return p1_union, u1_hardend, p_mismatch

    def latest_profiles(self) -> dict[str, Any]:
        """Return latest trunk/branch profiles after a propagation call."""
        return {"trunk": self.last_trunk, "branch": self.last_branch}

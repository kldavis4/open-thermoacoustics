"""Solver modules for thermoacoustic network analysis."""

from openthermoacoustics.solver.distributed_loop import (
    DistributedLoopPropagator,
    SegmentChainResult,
    SegmentChainSection,
    integrate_segment_chain,
)
from openthermoacoustics.solver.integrator import integrate_segment
from openthermoacoustics.solver.loop_network import LoopNetwork
from openthermoacoustics.solver.network import NetworkTopology, SegmentResult
from openthermoacoustics.solver.shooting import ShootingSolver, SolverResult
from openthermoacoustics.solver.tbranch_loop_solver import (
    TBranchLoopResult,
    TBranchLoopSolver,
    solve_lrc1_loop,
)

__all__ = [
    "integrate_segment",
    "integrate_segment_chain",
    "DistributedLoopPropagator",
    "SegmentChainResult",
    "SegmentChainSection",
    "NetworkTopology",
    "SegmentResult",
    "ShootingSolver",
    "SolverResult",
    "LoopNetwork",
    "TBranchLoopSolver",
    "TBranchLoopResult",
    "solve_lrc1_loop",
]

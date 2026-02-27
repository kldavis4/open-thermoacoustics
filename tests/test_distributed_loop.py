"""Tests for distributed TBRANCH/UNION propagation glue."""

from __future__ import annotations

import numpy as np

from openthermoacoustics import gas, segments
from openthermoacoustics.solver import (
    DistributedLoopPropagator,
    NetworkTopology,
    integrate_segment_chain,
)


def test_integrate_segment_chain_matches_linear_network_endpoint() -> None:
    """Chain integration endpoint should match NetworkTopology propagation."""
    helium = gas.Helium(mean_pressure=1.0e6)
    omega = 2.0 * np.pi * 300.0
    segs = [
        segments.Duct(length=0.20, radius=0.02),
        segments.Duct(length=0.10, radius=0.02),
    ]
    p1_start = 1200.0 + 40.0j
    u1_start = 2.0e-4 - 3.0e-4j

    chain = integrate_segment_chain(
        segs,
        p1_start=p1_start,
        u1_start=u1_start,
        t_m_start=300.0,
        omega=omega,
        gas=helium,
        n_points_per_segment=60,
    )

    network = NetworkTopology()
    for seg in segs:
        network.add_segment(seg)
    network.propagate_all(
        p1_start=p1_start,
        U1_start=u1_start,
        T_m_start=300.0,
        omega=omega,
        gas=helium,
        n_points_per_segment=60,
    )
    endpoint = network.get_endpoint_values()

    assert np.isclose(chain.p1_end, endpoint["p1_end"], rtol=1e-10, atol=1e-10)
    assert np.isclose(chain.U1_end, endpoint["U1_end"], rtol=1e-10, atol=1e-10)
    assert np.all(np.isfinite(chain.acoustic_power))


def test_distributed_loop_propagator_splits_and_recombines_paths() -> None:
    """Distributed loop propagator should produce finite mismatch/endpoint values."""
    helium = gas.Helium(mean_pressure=1.0e6)
    omega = 2.0 * np.pi * 180.0
    trunk = [segments.Duct(length=0.25, radius=0.03), segments.HardEnd()]
    branch = [segments.Duct(length=0.20, radius=0.02)]

    propagator = DistributedLoopPropagator(
        trunk_segments=trunk,
        branch_segments=branch,
        gas=helium,
        omega=omega,
        t_m_start=300.0,
        n_points_per_segment=50,
    )
    p1_union, u1_hardend, p_mismatch = propagator(
        p1_input=1500.0 + 0.0j,
        u1_input=1.0e-3 + 2.0e-4j,
        zb=2.5e5 - 1.0e5j,
    )

    assert np.isfinite(p1_union.real)
    assert np.isfinite(u1_hardend.real)
    assert np.isfinite(p_mismatch.real)
    assert propagator.last_trunk is not None
    assert propagator.last_branch is not None

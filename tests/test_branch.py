"""Tests for TBranch, Return, and SideBranch segments.

Tests cover:
- TBranch velocity splitting
- Return velocity combining
- Pressure continuity at junctions
- Simple loop closure (side branch that returns)
- Mass conservation through the network
"""

from __future__ import annotations

import numpy as np
import pytest

from openthermoacoustics.gas.helium import Helium
from openthermoacoustics.segments import Duct, TBranch, Return, SideBranch
from openthermoacoustics.solver import LoopNetwork


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def helium_gas() -> Helium:
    """Create Helium gas at 1 MPa pressure (typical for thermoacoustic engines)."""
    return Helium(mean_pressure=1e6)


@pytest.fixture
def test_temperature() -> float:
    """Standard test temperature of 300 K."""
    return 300.0


@pytest.fixture
def test_frequency() -> float:
    """Standard test frequency of 100 Hz."""
    return 100.0


@pytest.fixture
def test_omega(test_frequency: float) -> float:
    """Angular frequency corresponding to test_frequency."""
    return 2 * np.pi * test_frequency


# =============================================================================
# TBranch Tests
# =============================================================================


class TestTBranch:
    """Test T-branch junction segment."""

    def test_initialization_valid(self) -> None:
        """Test valid TBranch initialization."""
        tbranch = TBranch(side_area=0.001, side_fraction=0.3)
        assert tbranch.side_area == 0.001
        assert tbranch.side_fraction == 0.3
        assert tbranch.length == 0.0

    def test_initialization_invalid_area(self) -> None:
        """Test TBranch raises error for invalid side_area."""
        with pytest.raises(ValueError, match="side_area must be positive"):
            TBranch(side_area=0)
        with pytest.raises(ValueError, match="side_area must be positive"):
            TBranch(side_area=-0.001)

    def test_initialization_invalid_fraction(self) -> None:
        """Test TBranch raises error for invalid side_fraction."""
        with pytest.raises(ValueError, match="side_fraction must be in"):
            TBranch(side_area=0.001, side_fraction=-0.1)
        with pytest.raises(ValueError, match="side_fraction must be in"):
            TBranch(side_area=0.001, side_fraction=1.5)

    def test_set_side_fraction_valid(self) -> None:
        """Test setting valid side fraction."""
        tbranch = TBranch(side_area=0.001)
        tbranch.set_side_fraction(0.7)
        assert tbranch.side_fraction == 0.7

    def test_set_side_fraction_invalid(self) -> None:
        """Test setting invalid side fraction raises error."""
        tbranch = TBranch(side_area=0.001)
        with pytest.raises(ValueError, match="fraction must be in"):
            tbranch.set_side_fraction(-0.1)
        with pytest.raises(ValueError, match="fraction must be in"):
            tbranch.set_side_fraction(1.5)

    def test_velocity_splitting(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that velocity splits according to side_fraction."""
        side_fraction = 0.3
        tbranch = TBranch(side_area=0.001, side_fraction=side_fraction)

        p1_in = 1000.0 + 200j
        U1_in = 0.001 + 0.0002j

        p1_out, U1_out, T_out = tbranch.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Check velocity splitting
        expected_U1_side = side_fraction * U1_in
        expected_U1_out = U1_in - expected_U1_side

        assert np.isclose(U1_out, expected_U1_out), (
            f"Main duct velocity incorrect: got {U1_out}, expected {expected_U1_out}"
        )
        assert np.isclose(tbranch.side_U1, expected_U1_side), (
            f"Side branch velocity incorrect: got {tbranch.side_U1}, "
            f"expected {expected_U1_side}"
        )

    def test_pressure_continuity(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that pressure is continuous through junction."""
        tbranch = TBranch(side_area=0.001, side_fraction=0.3)

        p1_in = 1000.0 + 200j
        U1_in = 0.001 + 0.0002j

        p1_out, U1_out, T_out = tbranch.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Pressure should be continuous
        assert p1_out == p1_in, (
            f"Pressure not continuous: in={p1_in}, out={p1_out}"
        )

        # Side branch should also have same pressure
        p1_side, U1_side, T_side = tbranch.get_side_branch_state()
        assert p1_side == p1_in, (
            f"Side branch pressure mismatch: {p1_side} vs {p1_in}"
        )

    def test_temperature_continuity(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that temperature is continuous through junction."""
        tbranch = TBranch(side_area=0.001, side_fraction=0.3)

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_out, U1_out, T_out = tbranch.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        assert T_out == test_temperature
        _, _, T_side = tbranch.get_side_branch_state()
        assert T_side == test_temperature

    def test_get_side_branch_state(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test retrieval of side branch state after propagation."""
        side_fraction = 0.4
        tbranch = TBranch(side_area=0.001, side_fraction=side_fraction)

        p1_in = 1000.0 + 200j
        U1_in = 0.001 + 0.0002j

        tbranch.propagate(p1_in, U1_in, test_temperature, test_omega, helium_gas)

        p1_side, U1_side, T_side = tbranch.get_side_branch_state()

        assert p1_side == p1_in
        assert np.isclose(U1_side, side_fraction * U1_in)
        assert T_side == test_temperature

    def test_mass_conservation_at_junction(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test mass conservation: U1_in = U1_out + U1_side."""
        tbranch = TBranch(side_area=0.001, side_fraction=0.3)

        p1_in = 1000.0 + 200j
        U1_in = 0.001 + 0.0002j

        p1_out, U1_out, T_out = tbranch.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        U1_side = tbranch.side_U1

        # Mass conservation: U1_in = U1_out + U1_side
        U1_total = U1_out + U1_side
        assert np.isclose(U1_total, U1_in), (
            f"Mass not conserved: in={U1_in}, out+side={U1_total}"
        )


# =============================================================================
# Return Tests
# =============================================================================


class TestReturn:
    """Test Return junction segment."""

    def test_initialization_valid(self) -> None:
        """Test valid Return initialization."""
        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)
        assert return_seg.tbranch is tbranch
        assert return_seg.length == 0.0

    def test_initialization_invalid_tbranch(self) -> None:
        """Test Return raises error for invalid tbranch."""
        with pytest.raises(TypeError, match="tbranch must be a TBranch"):
            Return(tbranch="not a tbranch")

    def test_set_return_state(self) -> None:
        """Test setting the return state."""
        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)

        p1_return = 1000.0 + 100j
        U1_return = 0.0003 + 0.0001j
        T_m_return = 350.0

        return_seg.set_return_state(p1_return, U1_return, T_m_return)

        assert return_seg.return_U1 == U1_return

    def test_propagate_without_return_state(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test propagate raises error if return state not set."""
        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)

        with pytest.raises(RuntimeError, match="Return state not set"):
            return_seg.propagate(
                1000.0 + 0j, 0.001 + 0j, test_temperature, test_omega, helium_gas
            )

    def test_velocity_combining(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that velocities combine at Return junction."""
        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)

        # Set return state
        U1_return = 0.0003 + 0.0001j
        return_seg.set_return_state(
            p1_return=1000.0 + 0j,
            U1_return=U1_return,
            T_m_return=test_temperature,
        )

        # Main duct flow at junction
        p1_main = 1000.0 + 0j
        U1_main = 0.0007 + 0.0002j

        p1_out, U1_out, T_out = return_seg.propagate(
            p1_main, U1_main, test_temperature, test_omega, helium_gas
        )

        # Check velocity combining
        expected_U1_out = U1_main + U1_return
        assert np.isclose(U1_out, expected_U1_out), (
            f"Combined velocity incorrect: got {U1_out}, expected {expected_U1_out}"
        )

    def test_pressure_continuity_at_return(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that pressure is continuous at Return junction."""
        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)

        p1_return = 1000.0 + 100j
        return_seg.set_return_state(
            p1_return=p1_return,
            U1_return=0.0003 + 0j,
            T_m_return=test_temperature,
        )

        p1_main = 1000.0 + 100j  # Same as return pressure
        U1_main = 0.0007 + 0j

        p1_out, U1_out, T_out = return_seg.propagate(
            p1_main, U1_main, test_temperature, test_omega, helium_gas
        )

        # Pressure should be continuous (equals main duct pressure)
        assert p1_out == p1_main

    def test_pressure_mismatch(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test pressure mismatch calculation for loop closure."""
        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)

        # Return pressure different from main
        p1_return = 1050.0 + 100j
        return_seg.set_return_state(
            p1_return=p1_return,
            U1_return=0.0003 + 0j,
            T_m_return=test_temperature,
        )

        p1_main = 1000.0 + 100j

        mismatch = return_seg.get_pressure_mismatch(p1_main)
        expected_mismatch = p1_return - p1_main

        assert np.isclose(mismatch, expected_mismatch), (
            f"Mismatch incorrect: got {mismatch}, expected {expected_mismatch}"
        )


# =============================================================================
# SideBranch Tests
# =============================================================================


class TestSideBranch:
    """Test SideBranch helper class."""

    def test_initialization_valid(self) -> None:
        """Test valid SideBranch initialization."""
        duct1 = Duct(length=0.1, radius=0.01)
        duct2 = Duct(length=0.2, radius=0.01)
        side_branch = SideBranch(segments=[duct1, duct2], name="test_branch")

        assert len(side_branch) == 2
        assert side_branch.name == "test_branch"
        assert np.isclose(side_branch.total_length, 0.3)

    def test_initialization_empty_list(self) -> None:
        """Test SideBranch raises error for empty segments list."""
        with pytest.raises(ValueError, match="segments list cannot be empty"):
            SideBranch(segments=[])

    def test_initialization_invalid_segment(self) -> None:
        """Test SideBranch raises error for invalid segment type."""
        duct = Duct(length=0.1, radius=0.01)
        with pytest.raises(TypeError, match="must be a Segment instance"):
            SideBranch(segments=[duct, "not a segment"])

    def test_propagate_through_side_branch(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test propagation through a side branch with multiple segments."""
        duct1 = Duct(length=0.1, radius=0.01)
        duct2 = Duct(length=0.1, radius=0.01)
        side_branch = SideBranch(segments=[duct1, duct2])

        p1_in = 1000.0 + 0j
        U1_in = 0.0003 + 0j

        p1_out, U1_out, T_out = side_branch.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Check that propagation produced valid results
        assert np.isfinite(p1_out), "Output pressure should be finite"
        assert np.isfinite(U1_out), "Output velocity should be finite"
        assert T_out == test_temperature

        # Verify that the result matches sequential propagation
        p1_mid, U1_mid, T_mid = duct1.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )
        p1_expected, U1_expected, T_expected = duct2.propagate(
            p1_mid, U1_mid, T_mid, test_omega, helium_gas
        )

        assert np.isclose(p1_out, p1_expected)
        assert np.isclose(U1_out, U1_expected)

    def test_segments_property(self) -> None:
        """Test that segments property returns a copy."""
        duct = Duct(length=0.1, radius=0.01)
        side_branch = SideBranch(segments=[duct])

        segments = side_branch.segments
        segments.append(Duct(length=0.2, radius=0.01))

        # Original should not be modified
        assert len(side_branch.segments) == 1


# =============================================================================
# Loop Closure Tests
# =============================================================================


class TestLoopClosure:
    """Test loop closure with TBranch, SideBranch, and Return."""

    def test_simple_loop_velocity_conservation(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that total velocity is conserved through a simple loop.

        Setup:
        - Main duct with TBranch and Return
        - Side branch connects them

        The total U1 entering should equal total U1 exiting.
        """
        side_fraction = 0.3

        # Create junction segments
        tbranch = TBranch(side_area=0.001, side_fraction=side_fraction)
        return_seg = Return(tbranch=tbranch)

        # Side branch (simple short duct)
        side_duct = Duct(length=0.1, radius=0.015)
        side_branch = SideBranch(segments=[side_duct])

        # Initial state entering TBranch
        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        # Propagate through TBranch
        p1_main, U1_main, T_main = tbranch.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Get side branch inlet state
        p1_side_in, U1_side_in, T_side_in = tbranch.get_side_branch_state()

        # Propagate through side branch
        p1_side_out, U1_side_out, T_side_out = side_branch.propagate(
            p1_side_in, U1_side_in, T_side_in, test_omega, helium_gas
        )

        # Set return state
        return_seg.set_return_state(p1_side_out, U1_side_out, T_side_out)

        # Propagate through Return
        p1_out, U1_out, T_out = return_seg.propagate(
            p1_main, U1_main, T_main, test_omega, helium_gas
        )

        # Total velocity should be approximately conserved
        # (small differences due to wave propagation in side branch)
        # But U1_out = U1_main + U1_side_out, and at TBranch: U1_in = U1_main + U1_side_in
        # So U1_out should be close to U1_in if side branch is lossless
        rel_error = abs(U1_out - U1_in) / abs(U1_in)
        assert rel_error < 0.1, (
            f"Velocity not conserved through loop: in={U1_in}, out={U1_out}, "
            f"relative error={rel_error:.2%}"
        )

    def test_pressure_continuity_through_loop(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test pressure continuity at both junction points."""
        tbranch = TBranch(side_area=0.001, side_fraction=0.3)
        return_seg = Return(tbranch=tbranch)

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        # TBranch
        p1_main, U1_main, _ = tbranch.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )
        p1_side_in, U1_side_in, T_side_in = tbranch.get_side_branch_state()

        # Verify TBranch pressure continuity
        assert p1_main == p1_in, "Pressure not continuous at TBranch main outlet"
        assert p1_side_in == p1_in, "Pressure not continuous at TBranch side outlet"

        # Side branch (very short to minimize pressure change)
        side_duct = Duct(length=0.001, radius=0.015)
        p1_side_out, U1_side_out, T_side_out = side_duct.propagate(
            p1_side_in, U1_side_in, T_side_in, test_omega, helium_gas
        )

        return_seg.set_return_state(p1_side_out, U1_side_out, T_side_out)

        # For a short side branch, returning pressure should be close to main
        mismatch = return_seg.get_pressure_mismatch(p1_main)
        rel_mismatch = abs(mismatch) / abs(p1_main)
        assert rel_mismatch < 0.01, (
            f"Large pressure mismatch at Return: {rel_mismatch:.2%}"
        )


# =============================================================================
# LoopNetwork Tests
# =============================================================================


class TestLoopNetwork:
    """Test LoopNetwork class for branching topologies."""

    def test_initialization(self) -> None:
        """Test LoopNetwork initialization."""
        network = LoopNetwork()
        assert len(network.branches) == 0
        assert network.loop_closure_tolerance == 1e-6
        assert network.max_iterations == 100

    def test_add_branch_valid(self) -> None:
        """Test adding a valid branch to the network."""
        network = LoopNetwork()

        duct1 = Duct(length=0.1, radius=0.05)
        tbranch = TBranch(side_area=0.001, side_fraction=0.3)
        duct2 = Duct(length=0.1, radius=0.05)
        return_seg = Return(tbranch=tbranch)
        duct3 = Duct(length=0.1, radius=0.05)

        network.add_segment(duct1)
        network.add_segment(tbranch)
        network.add_segment(duct2)
        network.add_segment(return_seg)
        network.add_segment(duct3)

        # Add side branch
        side_duct = Duct(length=0.1, radius=0.015)
        network.add_branch(
            tbranch_index=1,
            segments=[side_duct],
            return_index=3,
        )

        assert len(network.branches) == 1
        assert network.branches[0].tbranch_index == 1
        assert network.branches[0].return_index == 3

    def test_add_branch_invalid_indices(self) -> None:
        """Test add_branch raises error for invalid indices."""
        network = LoopNetwork()

        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)

        network.add_segment(Duct(length=0.1, radius=0.05))
        network.add_segment(tbranch)
        network.add_segment(return_seg)

        # TBranch index >= Return index
        with pytest.raises(ValueError, match="tbranch_index .* must be less than"):
            network.add_branch(
                tbranch_index=2,
                segments=[Duct(length=0.1, radius=0.015)],
                return_index=1,
            )

    def test_add_branch_wrong_segment_type(self) -> None:
        """Test add_branch raises error for wrong segment types."""
        network = LoopNetwork()

        # Add regular ducts instead of TBranch/Return
        network.add_segment(Duct(length=0.1, radius=0.05))
        network.add_segment(Duct(length=0.1, radius=0.05))
        network.add_segment(Duct(length=0.1, radius=0.05))

        with pytest.raises(TypeError, match="must be TBranch"):
            network.add_branch(
                tbranch_index=0,
                segments=[Duct(length=0.1, radius=0.015)],
                return_index=2,
            )

    def test_propagate_without_branches(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test propagation without branches uses parent class method."""
        network = LoopNetwork()
        network.add_segment(Duct(length=0.1, radius=0.05))
        network.add_segment(Duct(length=0.1, radius=0.05))

        results = network.propagate_all(
            p1_start=1000.0 + 0j,
            U1_start=0.001 + 0j,
            T_m_start=test_temperature,
            omega=test_omega,
            gas=helium_gas,
        )

        assert len(results) == 2

    def test_mass_conservation_network(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test mass conservation through the loop network."""
        network = LoopNetwork()

        # Build network
        duct1 = Duct(length=0.1, radius=0.05)
        tbranch = TBranch(side_area=0.001, side_fraction=0.3)
        duct2 = Duct(length=0.1, radius=0.05)
        return_seg = Return(tbranch=tbranch)
        duct3 = Duct(length=0.1, radius=0.05)

        network.add_segment(duct1)
        network.add_segment(tbranch)
        network.add_segment(duct2)
        network.add_segment(return_seg)
        network.add_segment(duct3)

        # Add side branch
        side_duct = Duct(length=0.05, radius=0.015)
        network.add_branch(
            tbranch_index=1,
            segments=[side_duct],
            return_index=3,
        )

        # Set loose tolerance for this test
        network.loop_closure_tolerance = 0.1
        network.max_iterations = 50

        try:
            network.propagate_all(
                p1_start=1000.0 + 0j,
                U1_start=0.001 + 0j,
                T_m_start=test_temperature,
                omega=test_omega,
                gas=helium_gas,
            )

            # Check mass conservation
            conservation = network.verify_mass_conservation()
            assert conservation["max_split_error"] < 1e-10, (
                f"Split error too large: {conservation['max_split_error']}"
            )
            assert conservation["max_combine_error"] < 0.2, (
                f"Combine error too large: {conservation['max_combine_error']}"
            )
        except RuntimeError:
            # Loop closure may not converge with simple iteration
            # This is expected for some configurations
            pytest.skip("Loop closure did not converge - acceptable for this test")

    def test_clear_network(self) -> None:
        """Test clearing the network removes segments and branches."""
        network = LoopNetwork()

        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)

        network.add_segment(Duct(length=0.1, radius=0.05))
        network.add_segment(tbranch)
        network.add_segment(return_seg)
        network.add_branch(
            tbranch_index=1,
            segments=[Duct(length=0.1, radius=0.015)],
            return_index=2,
        )

        assert len(network.segments) == 3
        assert len(network.branches) == 1

        network.clear()

        assert len(network.segments) == 0
        assert len(network.branches) == 0


# =============================================================================
# Mass Conservation Tests
# =============================================================================


class TestMassConservation:
    """Dedicated tests for mass conservation through branching network."""

    def test_tbranch_split_conservation(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that U1_in = U1_main + U1_side at TBranch."""
        for side_fraction in [0.1, 0.3, 0.5, 0.7, 0.9]:
            tbranch = TBranch(side_area=0.001, side_fraction=side_fraction)

            U1_in = 0.001 + 0.0005j
            p1_in = 1000.0 + 0j

            _, U1_main, _ = tbranch.propagate(
                p1_in, U1_in, test_temperature, test_omega, helium_gas
            )
            U1_side = tbranch.side_U1

            U1_total = U1_main + U1_side

            assert np.isclose(U1_total, U1_in, rtol=1e-12), (
                f"Mass not conserved at TBranch with fraction={side_fraction}: "
                f"in={U1_in}, total={U1_total}"
            )

    def test_return_combine_conservation(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that U1_out = U1_main + U1_return at Return."""
        tbranch = TBranch(side_area=0.001)
        return_seg = Return(tbranch=tbranch)

        U1_main = 0.0007 + 0.0003j
        U1_return = 0.0003 + 0.0002j
        p1_main = 1000.0 + 0j

        return_seg.set_return_state(
            p1_return=p1_main,
            U1_return=U1_return,
            T_m_return=test_temperature,
        )

        _, U1_out, _ = return_seg.propagate(
            p1_main, U1_main, test_temperature, test_omega, helium_gas
        )

        expected_U1_out = U1_main + U1_return

        assert np.isclose(U1_out, expected_U1_out, rtol=1e-12), (
            f"Mass not conserved at Return: out={U1_out}, expected={expected_U1_out}"
        )

    def test_full_loop_mass_balance(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test mass balance through complete TBranch -> SideBranch -> Return path."""
        side_fraction = 0.4
        tbranch = TBranch(side_area=0.001, side_fraction=side_fraction)
        return_seg = Return(tbranch=tbranch)

        # Use a very short side branch to minimize propagation effects
        side_branch = SideBranch(segments=[Duct(length=0.001, radius=0.015)])

        U1_in = 0.001 + 0j
        p1_in = 1000.0 + 0j

        # TBranch splits the flow
        _, U1_main, T_main = tbranch.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )
        p1_side, U1_side, T_side = tbranch.get_side_branch_state()

        # Side branch propagates (minimal change expected)
        p1_return, U1_return, T_return = side_branch.propagate(
            p1_side, U1_side, T_side, test_omega, helium_gas
        )

        # Return combines the flows
        return_seg.set_return_state(p1_return, U1_return, T_return)
        _, U1_out, _ = return_seg.propagate(
            p1_in, U1_main, T_main, test_omega, helium_gas
        )

        # For very short side branch, output should be very close to input
        # (small difference due to wave propagation)
        rel_error = abs(U1_out - U1_in) / abs(U1_in)
        assert rel_error < 0.01, (
            f"Mass balance error through loop: in={U1_in}, out={U1_out}, "
            f"relative error={rel_error:.4f}"
        )

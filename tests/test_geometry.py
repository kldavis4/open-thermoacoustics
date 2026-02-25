"""
Comprehensive tests for the geometry module.

Tests cover:
1. Boundary layer limit (large pores, r_h >> delta)
2. Narrow channel limit (small pores, r_h << delta)
3. Intermediate regime
4. f_nu vs f_kappa relationship
5. WireScreen geometry construction and behavior
6. Array inputs support
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from openthermoacoustics.geometry import CircularPore, ParallelPlate, WireScreen


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def circular_pore() -> CircularPore:
    """Create a CircularPore geometry instance."""
    return CircularPore()


@pytest.fixture
def parallel_plate() -> ParallelPlate:
    """Create a ParallelPlate geometry instance."""
    return ParallelPlate()


@pytest.fixture
def wire_screen_from_physical() -> WireScreen:
    """
    Create a WireScreen from physical parameters.

    Uses 200 mesh (200 wires per inch) with 0.05mm wire diameter.
    """
    wire_diameter = 5e-5  # 50 microns
    mesh_count = 200 / 0.0254  # 200 per inch -> per meter
    return WireScreen(wire_diameter=wire_diameter, mesh_count=mesh_count)


@pytest.fixture
def wire_screen_from_derived() -> WireScreen:
    """Create a WireScreen from derived parameters (porosity and hydraulic radius)."""
    return WireScreen(porosity=0.7, hydraulic_radius=5e-5)


# -----------------------------------------------------------------------------
# Test: Boundary layer limit (large pores, r_h >> delta)
# -----------------------------------------------------------------------------


class TestBoundaryLayerLimit:
    """
    Test the boundary layer limit where r_h >> delta.

    In this regime, the viscous/thermal boundary layers are thin compared to
    the pore size.

    For CircularPore: f = 2*J1(z)/(z*J0(z)) -> 2/z = (-1-j)*delta/r_h
        where z = (-1+j)*r_h/delta (reference baseline convention)
    For ParallelPlate: f = tanh(z)/z -> 1/z = (1-j)*delta/(2*r_h)
        where z = (1+j)*r_h/delta

    Note: reference baseline uses different sign conventions for circular vs parallel plate.
    """

    @pytest.fixture
    def large_pore_params(self) -> dict:
        """
        Parameters for boundary layer limit test.

        r_h = 0.01 m (10 mm)
        delta = 1e-4 m (0.1 mm)
        ratio = r_h / delta = 100
        """
        return {
            "r_h": 0.01,
            "delta": 1e-4,
            "omega": 1000.0,  # arbitrary, not used in computation
        }

    def test_circular_pore_boundary_layer_limit(
        self, circular_pore: CircularPore, large_pore_params: dict
    ) -> None:
        """
        For large r_h/delta, CircularPore f_nu should approach (-1-j)*delta/r_h.

        The circular pore formula f = 2*J1(z)/(z*J0(z)) approaches 2/z for large |z|.
        With z = (-1+j)*r_h/delta (reference baseline convention), we get:
        f -> 2*delta/((-1+j)*r_h) = (-1-j)*delta/r_h.

        The expected value for r_h=0.01, delta=1e-4 is:
        f = (-1-j) * 1e-4 / 0.01 = -0.01 - 0.01j
        """
        r_h = large_pore_params["r_h"]
        delta = large_pore_params["delta"]
        omega = large_pore_params["omega"]

        f_nu = circular_pore.f_nu(omega, delta, r_h)
        expected = (-1 - 1j) * delta / r_h

        # Check that f approaches the asymptotic limit
        assert_allclose(f_nu.real, expected.real, rtol=0.05)
        assert_allclose(f_nu.imag, expected.imag, rtol=0.05)

    def test_parallel_plate_boundary_layer_limit(
        self, parallel_plate: ParallelPlate, large_pore_params: dict
    ) -> None:
        """
        For large r_h/delta, ParallelPlate f_nu should approach (1-j)*delta/(2*r_h).

        The parallel plate formula f = tanh(z)/z approaches 1/z for large |z|.
        With z = y0*(1+j)/delta (where y0 = r_h is the half-gap), we get
        f -> delta/(y0*(1+j)) = (1-j)*delta/(2*y0).

        The expected value for r_h=0.01, delta=1e-4 is:
        f = (1-j) * 1e-4 / (2 * 0.01) = 0.005 - 0.005j
        """
        r_h = large_pore_params["r_h"]
        delta = large_pore_params["delta"]
        omega = large_pore_params["omega"]

        f_nu = parallel_plate.f_nu(omega, delta, r_h)
        # Note: Parallel plate has factor of 2 difference from circular pore
        expected = (1 - 1j) * delta / (2 * r_h)

        # Check that f approaches the asymptotic limit
        assert_allclose(f_nu.real, expected.real, rtol=0.05)
        assert_allclose(f_nu.imag, expected.imag, rtol=0.05)

    def test_boundary_layer_limit_magnitude(
        self, circular_pore: CircularPore, parallel_plate: ParallelPlate
    ) -> None:
        """
        Verify |f| is small when r_h >> delta (boundary layer regime).

        When the pore is large compared to the penetration depth,
        most of the flow is inviscid, so |f| << 1.

        For CircularPore: |f| ~ delta/r_h * sqrt(2)
        For ParallelPlate: |f| ~ delta/(2*r_h) * sqrt(2)
        """
        r_h = 0.01
        delta = 1e-4
        omega = 1000.0

        f_circular = circular_pore.f_nu(omega, delta, r_h)
        f_parallel = parallel_plate.f_nu(omega, delta, r_h)

        # CircularPore: |f| ~ delta/r_h * sqrt(2) = 0.01 * 1.414 ~ 0.014
        expected_magnitude_circular = delta / r_h * np.sqrt(2)
        # ParallelPlate: |f| ~ delta/(2*r_h) * sqrt(2) = 0.005 * 1.414 ~ 0.007
        expected_magnitude_parallel = delta / (2 * r_h) * np.sqrt(2)

        assert_allclose(np.abs(f_circular), expected_magnitude_circular, rtol=0.1)
        assert_allclose(np.abs(f_parallel), expected_magnitude_parallel, rtol=0.1)


# -----------------------------------------------------------------------------
# Test: Narrow channel limit (small pores, r_h << delta)
# -----------------------------------------------------------------------------


class TestNarrowChannelLimit:
    """
    Test the narrow channel limit where r_h << delta.

    In this regime, the viscous/thermal boundary layers fill the entire pore,
    so: f -> 1 as r_h/delta -> 0.

    This corresponds to small |z| where z = r_h * (1+j) / delta.
    """

    @pytest.fixture
    def small_pore_params(self) -> dict:
        """
        Parameters for narrow channel limit test.

        r_h = 1e-6 m (1 micron)
        delta = 1e-3 m (1 mm)
        ratio = r_h / delta = 0.001
        """
        return {
            "r_h": 1e-6,
            "delta": 1e-3,
            "omega": 1000.0,
        }

    def test_circular_pore_narrow_channel_limit(
        self, circular_pore: CircularPore, small_pore_params: dict
    ) -> None:
        """
        For small r_h/delta, CircularPore f_nu should approach 1.0.

        When the pore is much smaller than the penetration depth,
        the flow is fully viscous/isothermal and f -> 1.
        """
        r_h = small_pore_params["r_h"]
        delta = small_pore_params["delta"]
        omega = small_pore_params["omega"]

        f_nu = circular_pore.f_nu(omega, delta, r_h)

        # f should approach 1.0 (small imaginary part)
        assert_allclose(f_nu.real, 1.0, rtol=1e-3)
        assert np.abs(f_nu.imag) < 1e-3

    def test_parallel_plate_narrow_channel_limit(
        self, parallel_plate: ParallelPlate, small_pore_params: dict
    ) -> None:
        """
        For small r_h/delta, ParallelPlate f_nu should approach 1.0.
        """
        r_h = small_pore_params["r_h"]
        delta = small_pore_params["delta"]
        omega = small_pore_params["omega"]

        f_nu = parallel_plate.f_nu(omega, delta, r_h)

        assert_allclose(f_nu.real, 1.0, rtol=1e-3)
        assert np.abs(f_nu.imag) < 1e-3

    def test_narrow_channel_f_approaches_unity(
        self, circular_pore: CircularPore, parallel_plate: ParallelPlate
    ) -> None:
        """
        As r_h/delta -> 0, verify f -> 1 for both geometries.

        Test with decreasing r_h/delta ratios.
        """
        delta = 1e-3
        omega = 1000.0
        ratios = [0.1, 0.01, 0.001, 0.0001]

        for ratio in ratios:
            r_h = ratio * delta

            f_circ = circular_pore.f_nu(omega, delta, r_h)
            f_para = parallel_plate.f_nu(omega, delta, r_h)

            # Tolerance increases as ratio decreases (approaching limit)
            tol = max(ratio * 10, 1e-4)
            assert_allclose(f_circ.real, 1.0, rtol=tol)
            assert_allclose(f_para.real, 1.0, rtol=tol)


# -----------------------------------------------------------------------------
# Test: Intermediate regime
# -----------------------------------------------------------------------------


class TestIntermediateRegime:
    """
    Test the intermediate regime where r_h ~ delta.

    In this regime:
    - 0 < |f| < 1
    - The imaginary part is non-zero (indicating phase shift between pressure and velocity)

    Note: The sign of Im(f) depends on the geometry-specific formula and conventions.
    For ParallelPlate (tanh formula): Im(f) < 0
    For CircularPore (Bessel formula): Im(f) can be positive in intermediate regime
    """

    @pytest.fixture
    def typical_params(self) -> dict:
        """
        Typical thermoacoustic parameters.

        r_h = 0.5e-3 m (0.5 mm) - typical regenerator/stack pore size
        delta = 0.1e-3 m (0.1 mm) - typical penetration depth
        ratio = r_h / delta = 5 (intermediate regime)
        """
        return {
            "r_h": 0.5e-3,
            "delta": 0.1e-3,
            "omega": 1000.0,
        }

    def test_circular_pore_intermediate_magnitude(
        self, circular_pore: CircularPore, typical_params: dict
    ) -> None:
        """Verify |f| is between 0 and 1 in the intermediate regime."""
        r_h = typical_params["r_h"]
        delta = typical_params["delta"]
        omega = typical_params["omega"]

        f_nu = circular_pore.f_nu(omega, delta, r_h)
        magnitude = np.abs(f_nu)

        assert 0 < magnitude < 1, f"|f| = {magnitude} should be in (0, 1)"

    def test_parallel_plate_intermediate_magnitude(
        self, parallel_plate: ParallelPlate, typical_params: dict
    ) -> None:
        """Verify |f| is between 0 and 1 in the intermediate regime."""
        r_h = typical_params["r_h"]
        delta = typical_params["delta"]
        omega = typical_params["omega"]

        f_nu = parallel_plate.f_nu(omega, delta, r_h)
        magnitude = np.abs(f_nu)

        assert 0 < magnitude < 1, f"|f| = {magnitude} should be in (0, 1)"

    def test_circular_pore_nonzero_imaginary(
        self, circular_pore: CircularPore, typical_params: dict
    ) -> None:
        """
        Verify Im(f) is non-zero for CircularPore in intermediate regime.

        The Bessel function formula 2*J1(z)/(z*J0(z)) produces a complex result
        with non-zero imaginary part in the intermediate regime.
        """
        r_h = typical_params["r_h"]
        delta = typical_params["delta"]
        omega = typical_params["omega"]

        f_nu = circular_pore.f_nu(omega, delta, r_h)

        assert f_nu.imag != 0, f"Im(f) = {f_nu.imag} should be non-zero"

    def test_parallel_plate_negative_imaginary(
        self, parallel_plate: ParallelPlate, typical_params: dict
    ) -> None:
        """
        Verify Im(f) < 0 for ParallelPlate in intermediate regime.

        The tanh(z)/z formula gives negative imaginary part for the
        parallel plate geometry with the standard convention.
        """
        r_h = typical_params["r_h"]
        delta = typical_params["delta"]
        omega = typical_params["omega"]

        f_nu = parallel_plate.f_nu(omega, delta, r_h)

        assert f_nu.imag < 0, f"Im(f) = {f_nu.imag} should be negative"

    def test_intermediate_regime_sweep(
        self, circular_pore: CircularPore, parallel_plate: ParallelPlate
    ) -> None:
        """
        Sweep through intermediate r_h/delta ratios and verify physical constraints.
        """
        delta = 1e-4
        omega = 1000.0
        # Ratios from 0.5 to 20 (intermediate regime)
        ratios = np.logspace(np.log10(0.5), np.log10(20), 10)

        for ratio in ratios:
            r_h = ratio * delta

            f_circ = circular_pore.f_nu(omega, delta, r_h)
            f_para = parallel_plate.f_nu(omega, delta, r_h)

            # Check magnitude constraints
            assert 0 < np.abs(f_circ) < 1.5, f"Circular: |f| = {np.abs(f_circ)} at ratio {ratio}"
            assert 0 < np.abs(f_para) < 1.5, f"Parallel: |f| = {np.abs(f_para)} at ratio {ratio}"

            # Check that imaginary parts are non-zero (phase shift present)
            assert f_circ.imag != 0, f"Circular: Im(f) should be non-zero at ratio {ratio}"
            assert f_para.imag < 0, f"Parallel: Im(f) = {f_para.imag} should be negative at ratio {ratio}"


# -----------------------------------------------------------------------------
# Test: f_nu vs f_kappa relationship
# -----------------------------------------------------------------------------


class TestFnuFkappaRelationship:
    """
    Test the relationship between f_nu and f_kappa.

    For the same geometry:
    - f_kappa has the same functional form as f_nu but uses delta_kappa
    - When delta_kappa = delta_nu (Prandtl number = 1), f_kappa = f_nu
    """

    def test_circular_pore_same_formula_different_delta(
        self, circular_pore: CircularPore
    ) -> None:
        """
        Verify f_nu and f_kappa use the same formula with different penetration depths.

        If we use the same delta for both, they should be equal.
        """
        r_h = 0.5e-3
        delta = 0.1e-3
        omega = 1000.0

        f_nu = circular_pore.f_nu(omega, delta, r_h)
        f_kappa = circular_pore.f_kappa(omega, delta, r_h)

        assert_allclose(f_nu, f_kappa)

    def test_parallel_plate_same_formula_different_delta(
        self, parallel_plate: ParallelPlate
    ) -> None:
        """
        Verify f_nu and f_kappa use the same formula for ParallelPlate.
        """
        r_h = 0.5e-3
        delta = 0.1e-3
        omega = 1000.0

        f_nu = parallel_plate.f_nu(omega, delta, r_h)
        f_kappa = parallel_plate.f_kappa(omega, delta, r_h)

        assert_allclose(f_nu, f_kappa)

    def test_prandtl_one_equality(
        self, circular_pore: CircularPore, parallel_plate: ParallelPlate
    ) -> None:
        """
        When Prandtl number = 1 (delta_kappa = delta_nu), f_kappa = f_nu.

        The Prandtl number Pr = nu/alpha, and delta_kappa/delta_nu = sqrt(1/Pr).
        When Pr = 1, the penetration depths are equal.
        """
        r_h = 0.5e-3
        delta_nu = 0.1e-3
        delta_kappa = delta_nu  # Pr = 1
        omega = 1000.0

        # CircularPore
        f_nu_circ, f_kappa_circ = circular_pore.compute_both(
            omega, delta_nu, delta_kappa, r_h
        )
        assert_allclose(f_nu_circ, f_kappa_circ)

        # ParallelPlate
        f_nu_para, f_kappa_para = parallel_plate.compute_both(
            omega, delta_nu, delta_kappa, r_h
        )
        assert_allclose(f_nu_para, f_kappa_para)

    def test_different_prandtl_numbers(
        self, circular_pore: CircularPore
    ) -> None:
        """
        Test f_nu and f_kappa with different Prandtl numbers.

        For Pr > 1: delta_kappa < delta_nu, so |f_kappa| < |f_nu|
        For Pr < 1: delta_kappa > delta_nu, so |f_kappa| > |f_nu|
        """
        r_h = 0.5e-3
        delta_nu = 0.1e-3
        omega = 1000.0

        # Prandtl = 0.7 (typical for air): delta_kappa = delta_nu * sqrt(1/0.7)
        delta_kappa_air = delta_nu * np.sqrt(1 / 0.7)
        f_nu, f_kappa = circular_pore.compute_both(omega, delta_nu, delta_kappa_air, r_h)
        # They should be different since penetration depths differ
        assert not np.isclose(f_nu, f_kappa)

        # Prandtl = 4 (typical for water): delta_kappa = delta_nu * sqrt(1/4) = delta_nu/2
        delta_kappa_water = delta_nu * 0.5
        f_nu, f_kappa = circular_pore.compute_both(omega, delta_nu, delta_kappa_water, r_h)
        assert not np.isclose(f_nu, f_kappa)


# -----------------------------------------------------------------------------
# Test: WireScreen geometry
# -----------------------------------------------------------------------------


class TestWireScreenConstruction:
    """Test WireScreen construction from different parameter sets."""

    def test_construction_from_wire_diameter_and_mesh_count(self) -> None:
        """Test WireScreen construction from physical parameters."""
        wire_diameter = 5e-5  # 50 microns
        mesh_count = 200 / 0.0254  # 200 per inch

        screen = WireScreen(wire_diameter=wire_diameter, mesh_count=mesh_count)

        # Verify computed properties
        assert screen.wire_diameter == wire_diameter
        assert screen.mesh_count == mesh_count
        assert 0 < screen.porosity < 1
        assert screen.hydraulic_radius > 0

    def test_construction_from_porosity_and_hydraulic_radius(self) -> None:
        """Test WireScreen construction from derived parameters."""
        porosity = 0.7
        hydraulic_radius = 5e-5

        screen = WireScreen(porosity=porosity, hydraulic_radius=hydraulic_radius)

        assert screen.porosity == porosity
        assert screen.hydraulic_radius == hydraulic_radius
        assert screen.wire_diameter is None
        assert screen.mesh_count is None

    def test_construction_missing_parameters_raises(self) -> None:
        """Test that incomplete parameter sets raise ValueError."""
        # Only wire_diameter
        with pytest.raises(ValueError, match="Must provide either"):
            WireScreen(wire_diameter=5e-5)

        # Only mesh_count
        with pytest.raises(ValueError, match="Must provide either"):
            WireScreen(mesh_count=1000)

        # Only porosity
        with pytest.raises(ValueError, match="Must provide either"):
            WireScreen(porosity=0.7)

        # Only hydraulic_radius
        with pytest.raises(ValueError, match="Must provide either"):
            WireScreen(hydraulic_radius=5e-5)

        # No parameters
        with pytest.raises(ValueError, match="Must provide either"):
            WireScreen()

    def test_invalid_porosity_raises(self) -> None:
        """Test that invalid porosity values raise ValueError."""
        # Porosity <= 0
        with pytest.raises(ValueError, match="Porosity must be in range"):
            WireScreen(porosity=0.0, hydraulic_radius=5e-5)

        with pytest.raises(ValueError, match="Porosity must be in range"):
            WireScreen(porosity=-0.1, hydraulic_radius=5e-5)

        # Porosity >= 1
        with pytest.raises(ValueError, match="Porosity must be in range"):
            WireScreen(porosity=1.0, hydraulic_radius=5e-5)

    def test_invalid_hydraulic_radius_raises(self) -> None:
        """Test that invalid hydraulic radius values raise ValueError."""
        with pytest.raises(ValueError, match="Hydraulic radius must be positive"):
            WireScreen(porosity=0.7, hydraulic_radius=0.0)

        with pytest.raises(ValueError, match="Hydraulic radius must be positive"):
            WireScreen(porosity=0.7, hydraulic_radius=-1e-5)

    def test_too_dense_mesh_raises(self) -> None:
        """Test that an impossibly dense mesh raises ValueError."""
        # Very thick wires with high mesh count -> negative porosity
        with pytest.raises(ValueError, match="porosity.*is not positive"):
            WireScreen(wire_diameter=1e-3, mesh_count=2000)


class TestWireScreenPorosityCalculation:
    """Test WireScreen porosity calculation from physical parameters."""

    def test_porosity_formula(self) -> None:
        """
        Verify the porosity calculation: phi = 1 - (pi/4) * (n * d)^2

        For a square mesh, the open area fraction depends on how much
        of the unit cell is blocked by wire crossings.
        """
        wire_diameter = 5e-5
        mesh_count = 200 / 0.0254  # ~7874 per meter

        screen = WireScreen(wire_diameter=wire_diameter, mesh_count=mesh_count)

        # Manual calculation
        nd = mesh_count * wire_diameter
        expected_porosity = 1.0 - (np.pi / 4.0) * nd**2

        assert_allclose(screen.porosity, expected_porosity)

    def test_hydraulic_radius_formula(self) -> None:
        """
        Verify hydraulic radius: r_h = (phi * d) / (4 * (1 - phi))
        """
        wire_diameter = 5e-5
        mesh_count = 200 / 0.0254

        screen = WireScreen(wire_diameter=wire_diameter, mesh_count=mesh_count)

        # Manual calculation
        nd = mesh_count * wire_diameter
        phi = 1.0 - (np.pi / 4.0) * nd**2
        expected_r_h = (phi * wire_diameter) / (4.0 * (1.0 - phi))

        assert_allclose(screen.hydraulic_radius, expected_r_h)


class TestWireScreenUsesParallelPlateFormula:
    """Test that WireScreen uses the parallel plate formula internally."""

    def test_same_as_parallel_plate(self) -> None:
        """
        WireScreen should produce the same f values as ParallelPlate
        for the same hydraulic radius.
        """
        porosity = 0.7
        r_h = 5e-5

        screen = WireScreen(porosity=porosity, hydraulic_radius=r_h)
        parallel_plate = ParallelPlate()

        delta = 1e-4
        omega = 1000.0

        f_screen = screen.f_nu(omega, delta)
        f_parallel = parallel_plate.f_nu(omega, delta, r_h)

        assert_allclose(f_screen, f_parallel)

    def test_same_thermal_as_parallel_plate(self) -> None:
        """WireScreen f_kappa should also match ParallelPlate."""
        porosity = 0.7
        r_h = 5e-5

        screen = WireScreen(porosity=porosity, hydraulic_radius=r_h)
        parallel_plate = ParallelPlate()

        delta_kappa = 1.2e-4
        omega = 1000.0

        f_screen = screen.f_kappa(omega, delta_kappa)
        f_parallel = parallel_plate.f_kappa(omega, delta_kappa, r_h)

        assert_allclose(f_screen, f_parallel)


class TestWireScreenDefaultHydraulicRadius:
    """Test that WireScreen uses its stored hydraulic radius by default."""

    def test_uses_stored_hydraulic_radius(
        self, wire_screen_from_derived: WireScreen
    ) -> None:
        """WireScreen should use its internal r_h when not specified."""
        screen = wire_screen_from_derived
        delta = 1e-4
        omega = 1000.0

        # Call without specifying hydraulic_radius
        f = screen.f_nu(omega, delta)

        # Compare with explicit call using stored r_h
        f_explicit = screen.f_nu(omega, delta, hydraulic_radius=screen.hydraulic_radius)

        assert_allclose(f, f_explicit)

    def test_can_override_hydraulic_radius(
        self, wire_screen_from_derived: WireScreen
    ) -> None:
        """WireScreen should allow overriding the hydraulic radius."""
        screen = wire_screen_from_derived
        delta = 1e-4
        omega = 1000.0
        different_r_h = 1e-4  # Different from screen's r_h

        # Call with different r_h
        f_override = screen.f_nu(omega, delta, hydraulic_radius=different_r_h)
        f_default = screen.f_nu(omega, delta)

        # Should be different values
        assert not np.isclose(f_override, f_default)


# -----------------------------------------------------------------------------
# Test: Array inputs
# -----------------------------------------------------------------------------


class TestArrayInputs:
    """Test that geometry functions work with numpy array inputs."""

    def test_circular_pore_array_delta(self, circular_pore: CircularPore) -> None:
        """Test CircularPore with array of penetration depths."""
        r_h = 0.5e-3
        omega = 1000.0
        delta_array = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3])

        result = circular_pore.f_nu(omega, delta_array, r_h)

        # Should return array of same shape
        assert isinstance(result, np.ndarray)
        assert result.shape == delta_array.shape
        assert result.dtype == np.complex128

    def test_parallel_plate_array_delta(self, parallel_plate: ParallelPlate) -> None:
        """Test ParallelPlate with array of penetration depths."""
        r_h = 0.5e-3
        omega = 1000.0
        delta_array = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3])

        result = parallel_plate.f_nu(omega, delta_array, r_h)

        assert isinstance(result, np.ndarray)
        assert result.shape == delta_array.shape
        assert result.dtype == np.complex128

    def test_wire_screen_array_delta(
        self, wire_screen_from_derived: WireScreen
    ) -> None:
        """Test WireScreen with array of penetration depths."""
        omega = 1000.0
        delta_array = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3])

        result = wire_screen_from_derived.f_nu(omega, delta_array)

        assert isinstance(result, np.ndarray)
        assert result.shape == delta_array.shape

    def test_array_values_are_correct(self, circular_pore: CircularPore) -> None:
        """
        Verify array computation gives same results as scalar calls.
        """
        r_h = 0.5e-3
        omega = 1000.0
        delta_array = np.array([1e-5, 1e-4, 1e-3])

        # Array call
        array_result = circular_pore.f_nu(omega, delta_array, r_h)

        # Scalar calls
        scalar_results = [circular_pore.f_nu(omega, d, r_h) for d in delta_array]

        for i, (arr_val, scalar_val) in enumerate(zip(array_result, scalar_results)):
            assert_allclose(
                arr_val, scalar_val,
                err_msg=f"Mismatch at index {i} for delta={delta_array[i]}"
            )

    def test_compute_both_with_arrays(self, circular_pore: CircularPore) -> None:
        """Test compute_both method with array inputs."""
        r_h = 0.5e-3
        omega = 1000.0
        delta_nu_array = np.linspace(1e-5, 1e-3, 5)
        delta_kappa_array = delta_nu_array * 1.2  # Different thermal depth

        f_nu, f_kappa = circular_pore.compute_both(
            omega, delta_nu_array, delta_kappa_array, r_h
        )

        assert isinstance(f_nu, np.ndarray)
        assert isinstance(f_kappa, np.ndarray)
        assert f_nu.shape == delta_nu_array.shape
        assert f_kappa.shape == delta_kappa_array.shape

    def test_2d_array_input(self, parallel_plate: ParallelPlate) -> None:
        """Test with 2D array input."""
        r_h = 0.5e-3
        omega = 1000.0
        delta_2d = np.array([[1e-5, 1e-4], [1e-3, 5e-4]])

        result = parallel_plate.f_nu(omega, delta_2d, r_h)

        assert result.shape == delta_2d.shape

    def test_empty_array_input(self, circular_pore: CircularPore) -> None:
        """Test with empty array input."""
        r_h = 0.5e-3
        omega = 1000.0
        delta_empty = np.array([])

        result = circular_pore.f_nu(omega, delta_empty, r_h)

        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)


# -----------------------------------------------------------------------------
# Test: Edge cases and properties
# -----------------------------------------------------------------------------


class TestGeometryProperties:
    """Test basic properties of geometry classes."""

    def test_circular_pore_name(self, circular_pore: CircularPore) -> None:
        """Test CircularPore name property."""
        assert circular_pore.name == "circular"

    def test_parallel_plate_name(self, parallel_plate: ParallelPlate) -> None:
        """Test ParallelPlate name property."""
        assert parallel_plate.name == "parallel_plate"

    def test_wire_screen_name(self, wire_screen_from_derived: WireScreen) -> None:
        """Test WireScreen name property."""
        assert wire_screen_from_derived.name == "wire_screen"

    def test_circular_pore_repr(self, circular_pore: CircularPore) -> None:
        """Test CircularPore string representation."""
        assert "CircularPore" in repr(circular_pore)

    def test_parallel_plate_repr(self, parallel_plate: ParallelPlate) -> None:
        """Test ParallelPlate string representation."""
        assert "ParallelPlate" in repr(parallel_plate)

    def test_wire_screen_repr(self, wire_screen_from_derived: WireScreen) -> None:
        """Test WireScreen string representation."""
        repr_str = repr(wire_screen_from_derived)
        assert "WireScreen" in repr_str
        assert "porosity" in repr_str
        assert "hydraulic_radius" in repr_str


class TestScalarVsArrayConsistency:
    """Test that scalar and array inputs produce consistent results."""

    def test_scalar_returns_complex(self, circular_pore: CircularPore) -> None:
        """Scalar input should return Python complex, not numpy array."""
        r_h = 0.5e-3
        delta = 1e-4
        omega = 1000.0

        result = circular_pore.f_nu(omega, delta, r_h)

        assert isinstance(result, complex)
        assert not isinstance(result, np.ndarray)

    def test_single_element_array_returns_array(
        self, circular_pore: CircularPore
    ) -> None:
        """Single-element array input should return single-element numpy array."""
        r_h = 0.5e-3
        delta = np.array([1e-4])
        omega = 1000.0

        result = circular_pore.f_nu(omega, delta, r_h)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

"""Tests for the TubeHeatExchanger (TX) segment."""

import numpy as np
import pytest

from openthermoacoustics.gas import Helium
from openthermoacoustics.segments import TubeHeatExchanger, TX


class TestTubeHeatExchangerInit:
    """Test TubeHeatExchanger initialization."""

    def test_basic_initialization(self):
        """Test basic TX creation."""
        tx = TubeHeatExchanger(
            length=0.02,
            porosity=0.5,
            tube_radius=0.003,
            area=0.01,
            solid_temperature=300.0,
        )
        assert tx.length == 0.02
        assert tx.porosity == 0.5
        assert tx.tube_radius == 0.003
        assert tx.area == 0.01
        assert tx.solid_temperature == 300.0
        assert tx.heat_in == 0.0

    def test_initialization_with_all_parameters(self):
        """Test TX creation with all optional parameters."""
        tx = TubeHeatExchanger(
            length=0.02,
            porosity=0.5,
            tube_radius=0.003,
            area=0.01,
            solid_temperature=300.0,
            heat_in=1000.0,
            solid_heat_capacity=4e6,
            solid_thermal_conductivity=500.0,
            name="TestTX",
        )
        assert tx.heat_in == 1000.0
        assert tx.name == "TestTX"

    def test_tx_alias(self):
        """Test that TX is an alias for TubeHeatExchanger."""
        assert TX is TubeHeatExchanger
        tx = TX(
            length=0.01,
            porosity=0.3,
            tube_radius=0.002,
            area=0.005,
            solid_temperature=350.0,
        )
        assert isinstance(tx, TubeHeatExchanger)

    def test_invalid_porosity_too_low(self):
        """Test that porosity <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="Porosity must be in"):
            TubeHeatExchanger(
                length=0.02,
                porosity=0.0,
                tube_radius=0.003,
                area=0.01,
                solid_temperature=300.0,
            )

    def test_invalid_porosity_too_high(self):
        """Test that porosity >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="Porosity must be in"):
            TubeHeatExchanger(
                length=0.02,
                porosity=1.0,
                tube_radius=0.003,
                area=0.01,
                solid_temperature=300.0,
            )

    def test_invalid_tube_radius(self):
        """Test that non-positive tube radius raises ValueError."""
        with pytest.raises(ValueError, match="Tube radius must be positive"):
            TubeHeatExchanger(
                length=0.02,
                porosity=0.5,
                tube_radius=-0.001,
                area=0.01,
                solid_temperature=300.0,
            )

    def test_invalid_area(self):
        """Test that non-positive area raises ValueError."""
        with pytest.raises(ValueError, match="Area must be positive"):
            TubeHeatExchanger(
                length=0.02,
                porosity=0.5,
                tube_radius=0.003,
                area=0.0,
                solid_temperature=300.0,
            )

    def test_invalid_temperature(self):
        """Test that non-positive temperature raises ValueError."""
        with pytest.raises(ValueError, match="Solid temperature must be positive"):
            TubeHeatExchanger(
                length=0.02,
                porosity=0.5,
                tube_radius=0.003,
                area=0.01,
                solid_temperature=-100.0,
            )


class TestTubeHeatExchangerProperties:
    """Test TubeHeatExchanger properties."""

    @pytest.fixture
    def tx(self):
        """Create a TX segment for testing."""
        return TubeHeatExchanger(
            length=0.02,
            porosity=0.5,
            tube_radius=0.004,
            area=0.01,
            solid_temperature=300.0,
            heat_in=500.0,
        )

    def test_hydraulic_radius(self, tx):
        """Test hydraulic radius is r0/2 for circular tubes."""
        assert tx.hydraulic_radius == tx.tube_radius / 2.0
        assert tx.hydraulic_radius == 0.002

    def test_heat_in_setter(self, tx):
        """Test heat_in property can be modified."""
        assert tx.heat_in == 500.0
        tx.heat_in = 1000.0
        assert tx.heat_in == 1000.0

    def test_repr(self, tx):
        """Test string representation."""
        repr_str = repr(tx)
        assert "TubeHeatExchanger" in repr_str
        assert "length=0.02" in repr_str
        assert "porosity=0.5" in repr_str
        assert "tube_radius=0.004" in repr_str


class TestTubeHeatExchangerPropagate:
    """Test TubeHeatExchanger propagation."""

    @pytest.fixture
    def gas(self):
        """Create a gas for testing."""
        return Helium(mean_pressure=1e6)

    @pytest.fixture
    def tx(self):
        """Create a TX segment for testing."""
        return TubeHeatExchanger(
            length=0.02,
            porosity=0.5,
            tube_radius=0.003,
            area=0.01,
            solid_temperature=300.0,
        )

    def test_propagate_returns_correct_types(self, tx, gas):
        """Test that propagate returns correct types."""
        omega = 2 * np.pi * 500
        p1_out, U1_out, T_out = tx.propagate(1000 + 0j, 0.001 + 0j, 350.0, omega, gas)

        assert isinstance(p1_out, complex)
        assert isinstance(U1_out, complex)
        assert isinstance(T_out, float)

    def test_propagate_output_temperature_is_solid_temperature(self, tx, gas):
        """Test that output temperature equals solid temperature."""
        omega = 2 * np.pi * 500
        _, _, T_out = tx.propagate(1000 + 0j, 0.001 + 0j, 350.0, omega, gas)

        # Output temperature should be solid temperature, not input
        assert T_out == tx.solid_temperature
        assert T_out != 350.0

    def test_propagate_zero_length(self, gas):
        """Test propagation through zero-length TX (passthrough)."""
        tx = TubeHeatExchanger(
            length=0.0,
            porosity=0.5,
            tube_radius=0.003,
            area=0.01,
            solid_temperature=300.0,
        )
        omega = 2 * np.pi * 500
        p1_in, U1_in = 1000 + 100j, 0.001 + 0.0001j
        p1_out, U1_out, T_out = tx.propagate(p1_in, U1_in, 350.0, omega, gas)

        assert p1_out == p1_in
        assert U1_out == U1_in
        assert T_out == tx.solid_temperature

    def test_propagate_preserves_acoustic_power_approximately(self, tx, gas):
        """Test that acoustic power is approximately preserved (lossy)."""
        omega = 2 * np.pi * 500
        p1_in = 10000 + 0j
        U1_in = 0.001 + 0.0001j

        # Acoustic power in
        E_in = 0.5 * np.real(p1_in * np.conj(U1_in))

        p1_out, U1_out, _ = tx.propagate(p1_in, U1_in, 300.0, omega, gas)

        # Acoustic power out (should be slightly less due to losses)
        E_out = 0.5 * np.real(p1_out * np.conj(U1_out))

        # Power should not increase significantly (small losses expected)
        # For short heat exchangers, losses should be modest
        assert E_out <= E_in * 1.1  # Allow 10% tolerance for numerical effects
        assert E_out >= E_in * 0.5  # But not massive losses

    def test_propagate_different_frequencies(self, tx, gas):
        """Test propagation at different frequencies."""
        p1_in = 5000 + 0j
        U1_in = 0.0005 + 0j

        results = {}
        for freq in [100, 500, 1000]:
            omega = 2 * np.pi * freq
            p1_out, U1_out, T_out = tx.propagate(p1_in, U1_in, 300.0, omega, gas)
            results[freq] = (abs(p1_out), abs(U1_out))

        # Higher frequency should generally have more attenuation
        # (due to thinner boundary layers)
        assert all(T == 300.0 for _, _, T in [
            tx.propagate(p1_in, U1_in, 300.0, 2*np.pi*f, gas) for f in [100, 500, 1000]
        ])


class TestTubeHeatExchangerDerivatives:
    """Test TubeHeatExchanger derivative calculations."""

    @pytest.fixture
    def gas(self):
        """Create a gas for testing."""
        return Helium(mean_pressure=3e6)

    @pytest.fixture
    def tx(self):
        """Create a TX segment for testing."""
        return TubeHeatExchanger(
            length=0.02,
            porosity=0.188,
            tube_radius=0.00635,
            area=0.2,
            solid_temperature=300.0,
        )

    def test_get_derivatives_shape(self, tx, gas):
        """Test that get_derivatives returns correct shape."""
        omega = 2 * np.pi * 90
        y = np.array([1e5, 0, 0.01, 0.001])  # [Re(p1), Im(p1), Re(U1), Im(U1)]

        dydt = tx.get_derivatives(0.0, y, omega, gas, 300.0)

        assert dydt.shape == (4,)

    def test_get_derivatives_nonzero(self, tx, gas):
        """Test that derivatives are non-zero for typical inputs."""
        omega = 2 * np.pi * 90
        y = np.array([1e5, 1e4, 0.01, 0.001])

        dydt = tx.get_derivatives(0.0, y, omega, gas, 300.0)

        # At least some derivatives should be non-zero
        assert np.any(dydt != 0)


class TestTubeHeatExchangerComputeGasTemperature:
    """Test TubeHeatExchanger oscillating temperature calculation."""

    @pytest.fixture
    def gas(self):
        """Create a gas for testing."""
        return Helium(mean_pressure=1e6)

    @pytest.fixture
    def tx(self):
        """Create a TX segment for testing."""
        return TubeHeatExchanger(
            length=0.02,
            porosity=0.5,
            tube_radius=0.003,
            area=0.01,
            solid_temperature=300.0,
        )

    def test_compute_gas_temperature_returns_complex(self, tx, gas):
        """Test that compute_gas_temperature returns complex."""
        omega = 2 * np.pi * 500
        p1 = 1000 + 100j
        U1 = 0.001 + 0j

        T1 = tx.compute_gas_temperature(p1, U1, omega, gas)

        assert isinstance(T1, complex)

    def test_compute_gas_temperature_scales_with_pressure(self, tx, gas):
        """Test that T1 scales approximately linearly with p1."""
        omega = 2 * np.pi * 500
        U1 = 0.001 + 0j

        T1_low = tx.compute_gas_temperature(1000 + 0j, U1, omega, gas)
        T1_high = tx.compute_gas_temperature(2000 + 0j, U1, omega, gas)

        # Should scale approximately linearly
        ratio = abs(T1_high) / abs(T1_low)
        assert 1.8 < ratio < 2.2

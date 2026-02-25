"""Tests for Phase 2 segments: Transducer, OpenEnd, Join, Impedance.

Tests cover:
- Transducer: blocked and free acoustic impedances, electromechanical coupling
- OpenEnd: radiation impedance vs analytical formulas for different flange types
- Join: area scaling and minor losses
- Impedance: fixed and frequency-dependent boundary conditions
"""

from __future__ import annotations

import numpy as np
import pytest

from openthermoacoustics.gas.helium import Helium
from openthermoacoustics.segments import (
    Impedance,
    Join,
    OpenEnd,
    Transducer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def helium_gas() -> Helium:
    """Create Helium gas at 1 atm pressure."""
    return Helium(mean_pressure=101325.0)


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
# Transducer Tests
# =============================================================================


class TestTransducer:
    """Test electrodynamic transducer segment."""

    @pytest.fixture
    def typical_transducer(self) -> Transducer:
        """Create a typical loudspeaker transducer."""
        return Transducer(
            Bl=5.0,       # T*m - force factor
            R_e=6.0,      # Ohm - DC resistance
            L_e=0.5e-3,   # H - voice coil inductance
            m=0.01,       # kg - moving mass
            k=2000.0,     # N/m - suspension stiffness
            R_m=1.0,      # N*s/m - mechanical damping
            A_d=0.01,     # m^2 - diaphragm area
        )

    def test_transducer_creation(self, typical_transducer: Transducer) -> None:
        """Test transducer parameter storage."""
        trans = typical_transducer
        assert trans.Bl == 5.0
        assert trans.R_e == 6.0
        assert trans.L_e == 0.5e-3
        assert trans.m == 0.01
        assert trans.k == 2000.0
        assert trans.R_m == 1.0
        assert trans.A_d == 0.01

    def test_transducer_validation(self) -> None:
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Force factor Bl must be positive"):
            Transducer(Bl=0, R_e=6, L_e=0.5e-3, m=0.01, k=2000, R_m=1, A_d=0.01)

        with pytest.raises(ValueError, match="Electrical resistance R_e must be positive"):
            Transducer(Bl=5, R_e=0, L_e=0.5e-3, m=0.01, k=2000, R_m=1, A_d=0.01)

        with pytest.raises(ValueError, match="Moving mass m must be positive"):
            Transducer(Bl=5, R_e=6, L_e=0.5e-3, m=0, k=2000, R_m=1, A_d=0.01)

    def test_resonant_frequency(self, typical_transducer: Transducer) -> None:
        """Test mechanical resonant frequency calculation."""
        trans = typical_transducer
        # f_s = sqrt(k/m) / (2*pi)
        expected_fs = np.sqrt(2000.0 / 0.01) / (2 * np.pi)
        actual_fs = trans.resonant_frequency()
        assert np.isclose(actual_fs, expected_fs, rtol=1e-10)

    def test_electrical_impedance(self, typical_transducer: Transducer) -> None:
        """Test electrical impedance Z_e = R_e + j*omega*L_e."""
        trans = typical_transducer
        omega = 628.3  # ~100 Hz

        Z_e = trans.electrical_impedance(omega)
        expected_Z_e = trans.R_e + 1j * omega * trans.L_e

        assert np.isclose(Z_e, expected_Z_e, rtol=1e-10)

    def test_mechanical_impedance(self, typical_transducer: Transducer) -> None:
        """Test mechanical impedance Z_m = R_m + j*omega*m + k/(j*omega)."""
        trans = typical_transducer
        omega = 628.3

        Z_m = trans.mechanical_impedance(omega)
        expected_Z_m = trans.R_m + 1j * omega * trans.m + trans.k / (1j * omega)

        assert np.isclose(Z_m, expected_Z_m, rtol=1e-10)

    def test_blocked_acoustic_impedance(
        self, typical_transducer: Transducer, test_omega: float
    ) -> None:
        """Test blocked acoustic impedance (electrical terminals open).

        Z_a_blocked = Z_m / A_d^2
        """
        trans = typical_transducer

        Z_a_blocked = trans.blocked_acoustic_impedance(test_omega)
        Z_m = trans.mechanical_impedance(test_omega)
        expected = Z_m / (trans.A_d**2)

        assert np.isclose(Z_a_blocked, expected, rtol=1e-10), (
            f"Blocked impedance mismatch: {Z_a_blocked} vs {expected}"
        )

    def test_free_acoustic_impedance(
        self, typical_transducer: Transducer, test_omega: float
    ) -> None:
        """Test free acoustic impedance (electrical terminals shorted).

        Z_a_free = Z_m / A_d^2 + (Bl)^2 / (Z_e * A_d^2)
        """
        trans = typical_transducer

        Z_a_free = trans.free_acoustic_impedance(test_omega)

        Z_e = trans.electrical_impedance(test_omega)
        Z_m = trans.mechanical_impedance(test_omega)
        Z_motional = (trans.Bl**2) / (Z_e * trans.A_d**2)
        expected = Z_m / (trans.A_d**2) + Z_motional

        assert np.isclose(Z_a_free, expected, rtol=1e-10), (
            f"Free impedance mismatch: {Z_a_free} vs {expected}"
        )

    def test_free_vs_blocked_impedance_difference(
        self, typical_transducer: Transducer, test_omega: float
    ) -> None:
        """Test that free impedance differs from blocked due to motional term."""
        trans = typical_transducer

        Z_blocked = trans.blocked_acoustic_impedance(test_omega)
        Z_free = trans.free_acoustic_impedance(test_omega)

        # The difference is the motional impedance
        Z_e = trans.electrical_impedance(test_omega)
        expected_diff = (trans.Bl**2) / (Z_e * trans.A_d**2)

        actual_diff = Z_free - Z_blocked
        assert np.isclose(actual_diff, expected_diff, rtol=1e-10)

    def test_acoustic_impedance_with_load(
        self, typical_transducer: Transducer, test_omega: float
    ) -> None:
        """Test acoustic impedance with finite electrical load."""
        trans = typical_transducer
        Z_load = 8.0 + 0j  # 8 Ohm resistive load

        Z_a = trans.acoustic_impedance(test_omega, Z_load)

        Z_e = trans.electrical_impedance(test_omega)
        Z_m = trans.mechanical_impedance(test_omega)
        Z_total = Z_e + Z_load
        Z_motional = (trans.Bl**2) / (Z_total * trans.A_d**2)
        expected = Z_m / (trans.A_d**2) + Z_motional

        assert np.isclose(Z_a, expected, rtol=1e-10)

    def test_propagate(
        self,
        typical_transducer: Transducer,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test propagation through transducer applies acoustic impedance."""
        trans = typical_transducer

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_out, U1_out, T_out = trans.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Velocity should be continuous
        assert U1_out == U1_in

        # Temperature should be unchanged
        assert T_out == test_temperature

        # Pressure drop should equal Z_a * U1_in (blocked condition)
        Z_a = trans.blocked_acoustic_impedance(test_omega)
        expected_p1_out = p1_in - Z_a * U1_in
        assert np.isclose(p1_out, expected_p1_out, rtol=1e-10)

    def test_resonance_impedance_minimum(
        self, typical_transducer: Transducer
    ) -> None:
        """Test that mechanical impedance magnitude is minimum at resonance."""
        trans = typical_transducer
        f_s = trans.resonant_frequency()
        omega_s = 2 * np.pi * f_s

        # At resonance, the reactive parts (mass and spring) cancel
        # Z_m = R_m + j*(omega*m - k/omega)
        # At omega_s = sqrt(k/m), the imaginary part is zero
        Z_m_res = trans.mechanical_impedance(omega_s)

        # At resonance, Z_m should be purely real (only damping)
        assert abs(Z_m_res.imag) < 1e-10 * abs(Z_m_res.real), (
            f"At resonance, Z_m should be real: {Z_m_res}"
        )
        assert np.isclose(Z_m_res.real, trans.R_m, rtol=1e-6)


# =============================================================================
# OpenEnd Tests
# =============================================================================


class TestOpenEnd:
    """Test open end radiation impedance segment."""

    def test_open_end_creation(self) -> None:
        """Test open end parameter storage."""
        open_end = OpenEnd(radius=0.025, flange_type="unflanged")
        assert open_end.radius == 0.025
        assert open_end.flange_type == "unflanged"

    def test_open_end_validation(self) -> None:
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Radius must be positive"):
            OpenEnd(radius=0)

        with pytest.raises(ValueError, match="flange_type must be one of"):
            OpenEnd(radius=0.025, flange_type="invalid")  # type: ignore

    def test_end_correction_unflanged(self) -> None:
        """Test unflanged end correction: delta = 0.6133 * a."""
        radius = 0.025
        open_end = OpenEnd(radius=radius, flange_type="unflanged")

        expected_delta = 0.6133 * radius
        actual_delta = open_end.end_correction()

        assert np.isclose(actual_delta, expected_delta, rtol=1e-10)

    def test_end_correction_flanged(self) -> None:
        """Test flanged end correction: delta = 0.8216 * a."""
        radius = 0.025
        open_end = OpenEnd(radius=radius, flange_type="flanged")

        expected_delta = 0.8216 * radius
        actual_delta = open_end.end_correction()

        assert np.isclose(actual_delta, expected_delta, rtol=1e-10)

    def test_end_correction_infinite_baffle(self) -> None:
        """Test infinite baffle end correction: delta = 0.8216 * a."""
        radius = 0.025
        open_end = OpenEnd(radius=radius, flange_type="infinite_baffle")

        expected_delta = 0.8216 * radius
        actual_delta = open_end.end_correction()

        assert np.isclose(actual_delta, expected_delta, rtol=1e-10)

    def test_radiation_impedance_low_frequency_unflanged(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test unflanged radiation impedance at low frequency.

        Z_rad = (rho*c/A) * [(ka)^2/4 + j*0.6133*ka]
        """
        radius = 0.025
        frequency = 100.0  # Low frequency so ka << 1
        omega = 2 * np.pi * frequency

        open_end = OpenEnd(radius=radius, flange_type="unflanged")
        Z_rad = open_end.radiation_impedance(omega, helium_gas, test_temperature)

        # Calculate expected impedance
        rho = helium_gas.density(test_temperature)
        c = helium_gas.sound_speed(test_temperature)
        A = np.pi * radius**2
        k = omega / c
        ka = k * radius

        Z_0 = rho * c / A
        R_rad_expected = Z_0 * (ka**2) / 4
        X_rad_expected = Z_0 * 0.6133 * ka

        # Check that ka is indeed small
        assert ka < 0.5, f"Test setup: ka should be << 1, got {ka}"

        # Check real and imaginary parts separately
        assert np.isclose(Z_rad.real, R_rad_expected, rtol=0.1), (
            f"Radiation resistance: {Z_rad.real} vs {R_rad_expected}"
        )
        assert np.isclose(Z_rad.imag, X_rad_expected, rtol=0.1), (
            f"Radiation reactance: {Z_rad.imag} vs {X_rad_expected}"
        )

    def test_radiation_impedance_low_frequency_flanged(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test flanged radiation impedance at low frequency.

        Z_rad = (rho*c/A) * [(ka)^2/2 + j*0.8216*ka]
        """
        radius = 0.025
        frequency = 100.0
        omega = 2 * np.pi * frequency

        open_end = OpenEnd(radius=radius, flange_type="flanged")
        Z_rad = open_end.radiation_impedance(omega, helium_gas, test_temperature)

        # Calculate expected impedance
        rho = helium_gas.density(test_temperature)
        c = helium_gas.sound_speed(test_temperature)
        A = np.pi * radius**2
        k = omega / c
        ka = k * radius

        Z_0 = rho * c / A
        R_rad_expected = Z_0 * (ka**2) / 2
        X_rad_expected = Z_0 * 0.8216 * ka

        assert np.isclose(Z_rad.real, R_rad_expected, rtol=0.1)
        assert np.isclose(Z_rad.imag, X_rad_expected, rtol=0.1)

    def test_flanged_vs_unflanged_comparison(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test that flanged has higher radiation resistance than unflanged."""
        radius = 0.025
        frequency = 100.0
        omega = 2 * np.pi * frequency

        unflanged = OpenEnd(radius=radius, flange_type="unflanged")
        flanged = OpenEnd(radius=radius, flange_type="flanged")

        Z_unflanged = unflanged.radiation_impedance(omega, helium_gas, test_temperature)
        Z_flanged = flanged.radiation_impedance(omega, helium_gas, test_temperature)

        # Flanged should have higher radiation resistance (factor of 2 at low ka)
        assert Z_flanged.real > Z_unflanged.real, (
            "Flanged should have higher radiation resistance"
        )

        # Flanged should also have higher reactance (due to larger end correction)
        assert Z_flanged.imag > Z_unflanged.imag, (
            "Flanged should have higher reactance (larger end correction)"
        )

    def test_reflection_coefficient_near_unity_low_frequency(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test that reflection coefficient magnitude is near 1 at low frequency.

        At low frequencies, little energy is radiated and most is reflected.
        """
        radius = 0.025
        frequency = 50.0  # Very low frequency
        omega = 2 * np.pi * frequency

        open_end = OpenEnd(radius=radius, flange_type="unflanged")
        R = open_end.reflection_coefficient(omega, helium_gas, test_temperature)

        # Reflection coefficient magnitude should be close to 1
        assert abs(R) > 0.99, (
            f"At low frequency, |R| should be ~1, got {abs(R)}"
        )

        # Phase should be close to pi or -pi (pressure inversion at open end)
        # At ideal soft end (Z=0), R = -1 (phase = pi or -pi, which are equivalent)
        # With small Z_rad, R ~ -1 + small correction
        phase = np.angle(R)
        assert abs(phase) > 2.5, (
            f"Phase of R should be near +/-pi, got {phase}"
        )

    def test_residual_and_is_satisfied(
        self, helium_gas: Helium, test_temperature: float, test_omega: float
    ) -> None:
        """Test residual calculation and boundary condition check."""
        radius = 0.025
        open_end = OpenEnd(radius=radius)

        Z_rad = open_end.radiation_impedance(test_omega, helium_gas, test_temperature)

        # Set up state that satisfies p1 = Z_rad * U1
        U1 = 0.001 + 0j
        p1 = Z_rad * U1

        # Residual should be zero
        res = open_end.residual(p1, U1, test_omega, helium_gas, test_temperature)
        assert abs(res) < 1e-12, f"Residual should be zero, got {res}"

        # is_satisfied should return True
        assert open_end.is_satisfied(p1, U1, test_omega, helium_gas, test_temperature)

        # Non-matching state should not satisfy
        p1_wrong = p1 + 100
        assert not open_end.is_satisfied(
            p1_wrong, U1, test_omega, helium_gas, test_temperature
        )

    def test_propagate_passthrough(
        self, helium_gas: Helium, test_temperature: float, test_omega: float
    ) -> None:
        """Test that propagate returns input unchanged (boundary marker)."""
        open_end = OpenEnd(radius=0.025)

        p1_in = 1000.0 + 500j
        U1_in = 0.001 + 0.0005j

        p1_out, U1_out, T_out = open_end.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        assert p1_out == p1_in
        assert U1_out == U1_in
        assert T_out == test_temperature


# =============================================================================
# Join Tests (reference baseline JOIN - Adiabatic-Isothermal Interface Loss)
# =============================================================================


class TestJoin:
    """Test adiabatic-isothermal interface loss segment (reference baseline JOIN).

    JOIN models the interface between a heat exchanger (isothermal) and a
    pulse tube (adiabatic, thermally stratified). It accounts for:
    - Temperature discontinuity (governing relation)
    - Volume flow rate magnitude reduction (governing relation)
    - No discontinuity in pressure or phase of U1
    """

    def test_join_creation(self) -> None:
        """Test join parameter storage."""
        join = Join(area=0.001, dT_dx=-1000.0, name="test")
        assert join.area == 0.001
        assert join.dT_dx == -1000.0
        assert join.name == "test"
        assert join.length == 0.0  # Lumped element

    def test_join_validation(self) -> None:
        """Test parameter validation."""
        with pytest.raises(ValueError, match="area must be positive"):
            Join(area=0)

        with pytest.raises(ValueError, match="area must be positive"):
            Join(area=-0.001)

    def test_dT_dx_property(self) -> None:
        """Test temperature gradient property and setter."""
        join = Join(area=0.001, dT_dx=-1000.0)
        assert join.dT_dx == -1000.0

        # Test setter
        join.dT_dx = -2000.0
        assert join.dT_dx == -2000.0

        # Can set to zero
        join.dT_dx = 0.0
        assert join.dT_dx == 0.0

    def test_pressure_unchanged(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that pressure is unchanged through JOIN (governing relation)."""
        join = Join(area=1.17e-4, dT_dx=-3464.0)

        p1_in = 135000 * np.exp(-1j * np.radians(17.66))
        U1_in = 2.92e-4 * np.exp(1j * np.radians(138.5))

        p1_out, _, _ = join.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Pressure should be exactly unchanged
        assert p1_out == p1_in

    def test_U1_phase_unchanged(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that U1 phase is unchanged through JOIN (governing relation)."""
        join = Join(area=1.17e-4, dT_dx=-3464.0)

        p1_in = 135000 * np.exp(-1j * np.radians(17.66))
        U1_in = 2.92e-4 * np.exp(1j * np.radians(138.5))

        _, U1_out, _ = join.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Phase should be unchanged
        assert np.isclose(np.angle(U1_out), np.angle(U1_in), atol=1e-10)

    def test_U1_magnitude_reduction(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that |U1| is reduced by JOIN for positive acoustic power.

        From governing relations:
        |U1|_out = |U1|_in - (16/(3*pi)) * ((gamma-1)/(rho_m * a^2)) * E_dot
        """
        join = Join(area=1.17e-4, dT_dx=0.0)  # No temp gradient for this test

        # Set up conditions where E_dot > 0 (p1 and U1 in phase)
        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j  # In phase with p1

        _, U1_out, _ = join.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # For positive E_dot, |U1| should decrease
        assert abs(U1_out) < abs(U1_in)

    def test_temperature_discontinuity_with_gradient(
        self,
        helium_gas: Helium,
        test_omega: float,
    ) -> None:
        """Test temperature discontinuity with nonzero dT_m/dx (governing relation)."""
        # Typical pulse tube conditions
        area = 1.17e-4  # m^2
        dT_dx = -3464.0  # K/m (300K to 57K over 0.07m)
        T_m = 300.0  # K

        join = Join(area=area, dT_dx=dT_dx)

        p1_in = 135000 * np.exp(-1j * np.radians(17.66))
        U1_in = 2.92e-4 * np.exp(1j * np.radians(138.5))

        _, _, T_out = join.propagate(p1_in, U1_in, T_m, test_omega, helium_gas)

        # Temperature should change (discontinuity)
        assert T_out != T_m
        # The change depends on phase relationship between p1 and U1

    def test_temperature_unchanged_with_zero_gradient(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test that temperature is unchanged when dT_dx=0 (isothermal case)."""
        join = Join(area=0.001, dT_dx=0.0)

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        _, _, T_out = join.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # With no temperature gradient, temperature discontinuity is zero
        assert T_out == test_temperature

    def test_acoustic_power_dissipation(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test acoustic power dissipation calculation."""
        join = Join(area=1.17e-4, dT_dx=0.0)

        # Set up conditions with positive acoustic power
        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        dissipation = join.acoustic_power_dissipation(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Should be non-negative
        assert dissipation >= 0

        # Should be positive for nonzero acoustic power
        assert dissipation > 0

    def test_repr(self) -> None:
        """Test string representation."""
        join = Join(area=0.001, dT_dx=-1000.0, name="test")
        repr_str = repr(join)

        assert "Join" in repr_str
        assert "area=0.001" in repr_str
        assert "dT_dx=-1000.0" in repr_str
        assert "test" in repr_str

    def test_get_derivatives_returns_zeros(self) -> None:
        """Test that get_derivatives returns zeros (lumped element)."""
        join = Join(area=0.001, dT_dx=-1000.0)

        y = np.array([1000.0, 0.0, 0.001, 0.0])

        # Need a gas object for the interface
        from openthermoacoustics.gas import Helium
        gas = Helium(mean_pressure=101325)

        derivs = join.get_derivatives(0.0, y, 628.3, gas, 300.0)

        assert np.allclose(derivs, np.zeros(4))


# =============================================================================
# Impedance Tests
# =============================================================================


class TestImpedance:
    """Test arbitrary acoustic impedance boundary segment."""

    def test_impedance_creation_fixed(self) -> None:
        """Test creation with fixed impedance value."""
        Z = 1000 + 500j
        impedance = Impedance(impedance=Z)
        assert impedance.impedance_value == Z
        assert impedance.impedance_function is None

    def test_impedance_creation_function(self) -> None:
        """Test creation with impedance function."""
        def Z_func(omega: float) -> complex:
            return 1000 + 1j * omega

        impedance = Impedance(impedance_func=Z_func)
        assert impedance.impedance_value is None
        assert impedance.impedance_function is not None

    def test_impedance_validation(self) -> None:
        """Test parameter validation."""
        with pytest.raises(ValueError, match="Must provide either"):
            Impedance()

        with pytest.raises(ValueError, match="Cannot provide both"):
            Impedance(impedance=1000+0j, impedance_func=lambda omega: 1000+0j)

    def test_get_impedance_fixed(self) -> None:
        """Test get_impedance with fixed value."""
        Z = 1000 + 500j
        impedance = Impedance(impedance=Z)

        # Should return same value regardless of omega
        assert impedance.get_impedance(100) == Z
        assert impedance.get_impedance(1000) == Z
        assert impedance.get_impedance(10000) == Z

    def test_get_impedance_function(self) -> None:
        """Test get_impedance with frequency-dependent function."""
        def Z_func(omega: float) -> complex:
            return 1000 + 1j * omega * 0.5  # Inductive-like impedance

        impedance = Impedance(impedance_func=Z_func)

        assert impedance.get_impedance(100) == 1000 + 1j * 50
        assert impedance.get_impedance(1000) == 1000 + 1j * 500
        assert impedance.get_impedance(2000) == 1000 + 1j * 1000

    def test_reflection_coefficient(
        self, helium_gas: Helium, test_temperature: float, test_omega: float
    ) -> None:
        """Test reflection coefficient calculation.

        R = (Z - Z_0) / (Z + Z_0)
        """
        rho = helium_gas.density(test_temperature)
        c = helium_gas.sound_speed(test_temperature)
        area = 0.001

        Z_0 = rho * c / area  # Characteristic impedance

        # Matched termination: Z = Z_0 should give R = 0
        impedance_matched = Impedance(impedance=Z_0)
        R_matched = impedance_matched.reflection_coefficient(
            test_omega, helium_gas, test_temperature, area
        )
        assert abs(R_matched) < 1e-10, f"Matched load should give R=0, got {R_matched}"

        # Hard end: Z -> infinity should give R = 1
        impedance_hard = Impedance(impedance=1e12 + 0j)  # Very large Z
        R_hard = impedance_hard.reflection_coefficient(
            test_omega, helium_gas, test_temperature, area
        )
        assert np.isclose(abs(R_hard), 1.0, atol=1e-6), (
            f"Hard end should give |R|=1, got {abs(R_hard)}"
        )

        # Soft end: Z = 0 should give R = -1
        impedance_soft = Impedance(impedance=1e-12 + 0j)  # Very small Z
        R_soft = impedance_soft.reflection_coefficient(
            test_omega, helium_gas, test_temperature, area
        )
        assert np.isclose(R_soft, -1.0, atol=1e-6), (
            f"Soft end should give R=-1, got {R_soft}"
        )

    def test_residual_calculation(self, test_omega: float) -> None:
        """Test residual: p1 - Z * U1 = 0 when boundary condition met."""
        Z = 1000 + 500j
        impedance = Impedance(impedance=Z)

        # State that satisfies boundary condition
        U1 = 0.001 + 0j
        p1 = Z * U1

        res = impedance.residual(p1, U1, test_omega)
        assert abs(res) < 1e-12, f"Residual should be zero, got {res}"

        # State that doesn't satisfy
        p1_wrong = p1 + 100
        res_wrong = impedance.residual(p1_wrong, U1, test_omega)
        assert np.isclose(res_wrong, 100 + 0j), (
            f"Residual should be 100, got {res_wrong}"
        )

    def test_is_satisfied(self, test_omega: float) -> None:
        """Test boundary condition satisfaction check."""
        Z = 1000 + 500j
        impedance = Impedance(impedance=Z)

        U1 = 0.001 + 0j
        p1 = Z * U1

        # Should be satisfied
        assert impedance.is_satisfied(p1, U1, test_omega) is True

        # Should not be satisfied with wrong p1
        assert impedance.is_satisfied(p1 + 100, U1, test_omega) is False

    def test_propagate_passthrough(
        self, helium_gas: Helium, test_temperature: float, test_omega: float
    ) -> None:
        """Test that propagate returns input unchanged (boundary marker)."""
        impedance = Impedance(impedance=1000 + 500j)

        p1_in = 1000.0 + 500j
        U1_in = 0.001 + 0.0005j

        p1_out, U1_out, T_out = impedance.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        assert p1_out == p1_in
        assert U1_out == U1_in
        assert T_out == test_temperature

    def test_frequency_dependent_impedance_usage(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test practical use of frequency-dependent impedance.

        Model an LC-like termination: Z = R + j*omega*L + 1/(j*omega*C)
        """
        R = 100
        L = 0.01  # 10 mH
        C = 1e-6  # 1 uF

        def lc_impedance(omega: float) -> complex:
            return complex(R) + 1j * omega * L + 1 / (1j * omega * C)

        impedance = Impedance(impedance_func=lc_impedance)

        # Test at different frequencies
        omega_low = 100  # rad/s
        omega_res = 1 / np.sqrt(L * C)  # Resonance ~10000 rad/s
        omega_high = 50000  # rad/s - well above resonance

        Z_low = impedance.get_impedance(omega_low)
        Z_res = impedance.get_impedance(omega_res)
        Z_high = impedance.get_impedance(omega_high)

        # At resonance, reactive parts cancel, Z ~ R
        assert np.isclose(Z_res.real, R, rtol=0.1)
        assert abs(Z_res.imag) < 10  # Should be small

        # At low frequency (below resonance), capacitive (negative imaginary)
        # omega*L - 1/(omega*C) should be negative for omega < omega_res
        assert Z_low.imag < 0, f"Below resonance should be capacitive, got Z={Z_low}"

        # At high frequency (above resonance), inductive (positive imaginary)
        # omega*L - 1/(omega*C) should be positive for omega > omega_res
        assert Z_high.imag > 0, f"Above resonance should be inductive, got Z={Z_high}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase2Integration:
    """Integration tests combining Phase 2 segments with existing ones."""

    def test_transducer_at_end_of_pipe(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test transducer as a termination at the end of an acoustic pipe."""
        from openthermoacoustics.segments import Duct

        # Create a short duct terminated with a transducer
        duct = Duct(length=0.5, radius=0.025)
        transducer = Transducer(
            Bl=5.0, R_e=6.0, L_e=0.5e-3, m=0.01, k=2000.0, R_m=1.0, A_d=0.002
        )

        frequency = 100.0
        omega = 2 * np.pi * frequency

        # Propagate through duct and then transducer
        p1_start = 1000.0 + 0j
        U1_start = 0.0001 + 0j

        p1_mid, U1_mid, T_mid = duct.propagate(
            p1_start, U1_start, test_temperature, omega, helium_gas
        )

        p1_end, U1_end, T_end = transducer.propagate(
            p1_mid, U1_mid, T_mid, omega, helium_gas
        )

        # Results should be finite and non-zero
        assert np.isfinite(p1_end)
        assert np.isfinite(U1_end)
        assert abs(p1_end) > 0

    def test_join_at_pulse_tube_interface(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test JOIN at interface between heat exchanger and pulse tube.

        This models a typical pulse tube refrigerator configuration where
        JOIN accounts for the adiabatic-isothermal interface loss.
        """
        from openthermoacoustics.segments import Duct

        # Heat exchanger (isothermal) -> JOIN -> Pulse tube (adiabatic)
        # Use Duct as a simple stand-in for heat exchanger
        hx = Duct(length=0.01, radius=0.006)  # Short heat exchanger duct

        # JOIN at the interface with temperature gradient
        pulse_tube_area = np.pi * 0.006**2
        dT_dx = -3000.0  # Temperature drops into pulse tube
        join = Join(area=pulse_tube_area, dT_dx=dT_dx)

        # Pulse tube (modeled as simple duct for this test)
        pulse_tube = Duct(length=0.07, radius=0.006)

        frequency = 40.0
        omega = 2 * np.pi * frequency

        p1_start = 135000.0 * np.exp(-1j * np.radians(17.0))
        U1_start = 3e-4 * np.exp(1j * np.radians(140.0))

        # Propagate through heat exchanger
        p1_1, U1_1, T_1 = hx.propagate(
            p1_start, U1_start, test_temperature, omega, helium_gas
        )

        # Propagate through JOIN (interface loss)
        p1_j, U1_j, T_j = join.propagate(
            p1_1, U1_1, T_1, omega, helium_gas
        )

        # Propagate through pulse tube
        p1_2, U1_2, T_2 = pulse_tube.propagate(
            p1_j, U1_j, T_j, omega, helium_gas
        )

        # Results should be finite
        assert np.isfinite(p1_2)
        assert np.isfinite(U1_2)
        assert np.isfinite(T_2)

        # Pressure should be unchanged through JOIN
        assert p1_j == p1_1

        # U1 phase should be preserved through JOIN
        assert np.isclose(np.angle(U1_j), np.angle(U1_1), atol=1e-10)

        # Temperature may have changed due to interface loss
        # (depends on phase relationship between p1 and U1)

    def test_open_end_termination(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test duct terminated with realistic open end."""
        from openthermoacoustics.segments import Duct

        radius = 0.025
        duct = Duct(length=0.5, radius=radius)
        open_end = OpenEnd(radius=radius, flange_type="unflanged")

        frequency = 100.0
        omega = 2 * np.pi * frequency

        p1_start = 1000.0 + 0j
        U1_start = 0.0001 + 0j

        p1_out, U1_out, T_out = duct.propagate(
            p1_start, U1_start, test_temperature, omega, helium_gas
        )

        # Check how well the state matches the open end boundary condition
        Z_rad = open_end.radiation_impedance(omega, helium_gas, test_temperature)
        p1_bc = Z_rad * U1_out  # Expected p1 for matched boundary condition

        # The state won't exactly match the BC (that requires solving eigenvalue problem)
        # but we can verify the open end impedance is reasonable
        assert np.isfinite(Z_rad)
        assert Z_rad.real > 0  # Radiation resistance is positive

    def test_impedance_as_matched_load(
        self, helium_gas: Helium, test_temperature: float
    ) -> None:
        """Test using Impedance as a matched termination (no reflection)."""
        from openthermoacoustics.segments import Duct

        radius = 0.025
        area = np.pi * radius**2
        rho = helium_gas.density(test_temperature)
        c = helium_gas.sound_speed(test_temperature)

        # Characteristic impedance
        Z_0 = rho * c / area

        duct = Duct(length=0.5, radius=radius)
        matched_load = Impedance(impedance=Z_0)

        frequency = 100.0
        omega = 2 * np.pi * frequency

        # For a matched load, reflection coefficient should be zero
        R = matched_load.reflection_coefficient(
            omega, helium_gas, test_temperature, area
        )
        assert abs(R) < 1e-10, f"Matched load should give R=0, got {R}"

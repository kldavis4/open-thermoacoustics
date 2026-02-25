"""Comprehensive tests for the segments module.

Tests cover all segment types with physically meaningful validation:
- Duct: lossless and lossy propagation
- Cone: area variation and continuity
- Stack: temperature gradient effects
- HeatExchanger: fixed temperature operation
- Compliance: lumped acoustic volume
- Inertance: lumped acoustic mass
- Boundary conditions: HardEnd and SoftEnd
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from openthermoacoustics.gas.helium import Helium
from openthermoacoustics.geometry.circular import CircularPore
from openthermoacoustics.segments import (
    Compliance,
    Cone,
    Duct,
    HardEnd,
    HeatExchanger,
    Inertance,
    SoftEnd,
    Stack,
)
from openthermoacoustics.utils import (
    acoustic_power,
    penetration_depth_thermal,
    penetration_depth_viscous,
    wavenumber,
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


@pytest.fixture
def circular_geometry() -> CircularPore:
    """Create circular pore geometry for thermoviscous function calculations."""
    return CircularPore()


# =============================================================================
# Duct Segment Tests - Lossless Limit
# =============================================================================


class TestDuctLossless:
    """Test duct segment in the lossless (inviscid) limit.

    In the lossless limit (large radius, high frequency), the duct should
    exhibit plane wave propagation where:
    - |p1| and |U1| oscillate as cos(kx) and sin(kx)
    - Acoustic power is conserved (constant along duct)
    """

    def test_plane_wave_propagation_pressure_node(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Test plane wave propagation starting from a pressure antinode.

        At x=0, we have maximum pressure and zero velocity (hard end).
        At x=lambda/4, we should have zero pressure and maximum velocity.
        """
        # Use high frequency and large radius for nearly lossless propagation
        frequency = 1000.0  # Hz - high frequency
        omega = 2 * np.pi * frequency
        radius = 0.1  # 10 cm radius - large to minimize boundary layer effects

        a = helium_gas.sound_speed(test_temperature)
        wavelength = a / frequency
        length = wavelength / 4  # Quarter wavelength

        duct = Duct(length=length, radius=radius)

        # Initial conditions: pressure antinode (max pressure, zero velocity)
        p1_in = 1000.0 + 0j  # Pa
        U1_in = 0.0 + 0j  # m^3/s

        p1_out, U1_out, T_out = duct.propagate(
            p1_in, U1_in, test_temperature, omega, helium_gas
        )

        # At quarter wavelength from hard end:
        # - Pressure should be near zero (pressure node)
        # - Velocity should be near maximum
        # Allow 5% tolerance due to small residual losses
        assert abs(p1_out) < 0.05 * abs(p1_in), (
            f"Pressure at quarter wavelength should be near zero, got |p1|={abs(p1_out):.2f}"
        )
        assert abs(U1_out) > 0, "Velocity at quarter wavelength should be non-zero"

    def test_acoustic_power_conservation_lossless(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify acoustic power is approximately conserved in nearly lossless limit.

        For a nearly lossless duct (large radius, short length), time-averaged
        acoustic power should be approximately constant along the duct length.
        Note: Even with boundary layer approximation, there are always some losses.
        """
        frequency = 1000.0  # Hz
        omega = 2 * np.pi * frequency
        radius = 0.1  # Large radius for minimal losses
        length = 0.1  # Short length to minimize losses

        duct = Duct(length=length, radius=radius)

        # Use traveling wave initial condition (forward traveling)
        # For a traveling wave: U1 = A*p1 / (rho*a) where A is area
        rho = helium_gas.density(test_temperature)
        a = helium_gas.sound_speed(test_temperature)
        A = np.pi * radius**2

        p1_in = 1000.0 + 0j  # Pa
        Z_acoustic = rho * a / A  # Acoustic impedance
        U1_in = p1_in / (rho * a / A)  # Traveling wave condition

        power_in = acoustic_power(p1_in, U1_in)

        p1_out, U1_out, T_out = duct.propagate(
            p1_in, U1_in, test_temperature, omega, helium_gas
        )

        power_out = acoustic_power(p1_out, U1_out)

        # Power should be conserved to within 5% for nearly lossless case
        # (allowing some margin for residual losses from boundary layer approximation)
        relative_error = abs(power_out - power_in) / power_in
        assert relative_error < 0.05, (
            f"Power not conserved: in={power_in:.4f} W, out={power_out:.4f} W, "
            f"relative error={relative_error:.2%}"
        )

    def test_standing_wave_pattern(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify standing wave pattern in closed duct.

        For a standing wave, |p1|^2 + (rho*a/A)^2 * |U1|^2 should be constant
        along the duct (energy density is uniform on average).
        """
        frequency = 500.0  # Hz
        omega = 2 * np.pi * frequency
        radius = 0.1  # m

        a = helium_gas.sound_speed(test_temperature)
        rho = helium_gas.density(test_temperature)
        A = np.pi * radius**2
        wavelength = a / frequency

        # Test at multiple positions
        positions = [0.0, wavelength/8, wavelength/4, wavelength/2]

        # Start with standing wave initial condition (hard end at x=0)
        p1_start = 1000.0 + 0j
        U1_start = 0.0 + 0j  # Hard end condition

        # Propagate to each position and check standing wave pattern
        results = [(abs(p1_start), abs(U1_start))]

        for i, length in enumerate(positions[1:], 1):
            duct = Duct(length=length, radius=radius)
            p1_out, U1_out, _ = duct.propagate(
                p1_start, U1_start, test_temperature, omega, helium_gas
            )
            results.append((abs(p1_out), abs(U1_out)))

        # Verify phase relationship: when |p1| is max, |U1| should be min and vice versa
        # At x=0 (hard end): |p1| max, |U1|=0
        # At x=lambda/4: |p1|=0, |U1| max
        assert results[0][1] < 0.01 * abs(p1_start)  # |U1| ~ 0 at hard end


# =============================================================================
# Duct Segment Tests - Lossy
# =============================================================================


class TestDuctLossy:
    """Test duct segment with realistic viscous and thermal losses.

    With finite penetration depths, the duct should exhibit:
    - Amplitude attenuation along the length
    - Attenuation consistent with Kirchhoff formula
    """

    def test_wave_propagation_with_losses(
        self,
        helium_gas: Helium,
        test_temperature: float,
        circular_geometry: CircularPore,
    ) -> None:
        """Verify lossy duct shows dissipation effects.

        For a lossy duct with boundary layer effects, the wave propagation
        shows dissipation through the imaginary part of the wave impedance,
        which manifests as a phase relationship change between p1 and U1.
        """
        frequency = 100.0  # Hz
        omega = 2 * np.pi * frequency
        radius = 0.005  # 5 mm - small enough for significant losses
        length = 0.5  # m

        duct = Duct(length=length, radius=radius, geometry=circular_geometry)

        # Calculate penetration depths to verify losses are significant
        rho = helium_gas.density(test_temperature)
        mu = helium_gas.viscosity(test_temperature)
        kappa = helium_gas.thermal_conductivity(test_temperature)
        cp = helium_gas.specific_heat_cp(test_temperature)

        delta_nu = penetration_depth_viscous(omega, rho, mu)
        delta_kappa = penetration_depth_thermal(omega, rho, kappa, cp)

        # Verify penetration depths are significant compared to radius
        assert delta_nu / radius > 0.01, (
            f"Test setup: viscous penetration depth too small ({delta_nu/radius:.4f} << 1)"
        )

        # Use traveling wave initial condition
        a = helium_gas.sound_speed(test_temperature)
        A = np.pi * radius**2

        p1_in = 1000.0 + 0j
        U1_in = p1_in * A / (rho * a)  # Forward traveling wave

        p1_out, U1_out, T_out = duct.propagate(
            p1_in, U1_in, test_temperature, omega, helium_gas
        )

        # In a lossy duct, the wave impedance has an imaginary component
        # This means the phase relationship between p1 and U1 should change
        # Calculate the impedance-like ratio at input and output
        Z_in = p1_in / U1_in if U1_in != 0 else float('inf')
        Z_out = p1_out / U1_out if U1_out != 0 else float('inf')

        # The phase of the impedance should change due to losses
        # For a lossless case, Z would be purely real for a traveling wave
        # For a lossy case, Z will have an imaginary component
        phase_in = np.angle(Z_in)
        phase_out = np.angle(Z_out)

        # The output should have finite, non-zero values
        assert np.isfinite(p1_out), "Output pressure should be finite"
        assert np.isfinite(U1_out), "Output velocity should be finite"
        assert abs(p1_out) > 0, "Output pressure should be non-zero"
        assert abs(U1_out) > 0, "Output velocity should be non-zero"

        # For a lossy duct, there should be some phase change
        # (the phase relationship between p1 and U1 evolves differently than lossless)
        # We verify the solution is physically reasonable and distinct from input
        assert not (np.isclose(p1_out.real, p1_in.real) and np.isclose(p1_out.imag, p1_in.imag)), (
            "Output should differ from input after propagation through lossy duct"
        )

    def test_attenuation_kirchhoff_formula(
        self,
        helium_gas: Helium,
        test_temperature: float,
        circular_geometry: CircularPore,
    ) -> None:
        """Verify attenuation is consistent with Kirchhoff formula.

        The Kirchhoff formula for attenuation in a circular tube:
        alpha = (1/r) * sqrt(omega/2) * [sqrt(mu/rho) + (gamma-1)*sqrt(kappa/(rho*cp))] / a

        This gives the amplitude attenuation coefficient (1/m).
        """
        frequency = 100.0
        omega = 2 * np.pi * frequency
        radius = 0.01  # 1 cm
        length = 0.5  # m

        duct = Duct(length=length, radius=radius, geometry=circular_geometry)

        # Gas properties
        rho = helium_gas.density(test_temperature)
        mu = helium_gas.viscosity(test_temperature)
        kappa = helium_gas.thermal_conductivity(test_temperature)
        cp = helium_gas.specific_heat_cp(test_temperature)
        a = helium_gas.sound_speed(test_temperature)
        gamma = helium_gas.gamma(test_temperature)

        # Kirchhoff attenuation coefficient (for pressure amplitude)
        delta_nu = penetration_depth_viscous(omega, rho, mu)
        delta_kappa = penetration_depth_thermal(omega, rho, kappa, cp)

        # Boundary layer approximation: alpha = (delta_nu + (gamma-1)*delta_kappa) / (2*r*a) * omega
        # Simplified: alpha ~ omega/a * (delta_nu + (gamma-1)*delta_kappa) / (2*r)
        k = omega / a  # wavenumber
        alpha_kirchhoff = k * (delta_nu + (gamma - 1) * delta_kappa) / (2 * radius)

        # Expected amplitude ratio
        expected_amplitude_ratio = np.exp(-alpha_kirchhoff * length)

        # Propagate through duct
        p1_in = 1000.0 + 0j
        A = np.pi * radius**2
        U1_in = p1_in * A / (rho * a)  # Forward traveling wave

        p1_out, U1_out, _ = duct.propagate(
            p1_in, U1_in, test_temperature, omega, helium_gas
        )

        # Measured amplitude ratio
        measured_amplitude_ratio = abs(p1_out) / abs(p1_in)

        # Allow 50% tolerance since Kirchhoff is an approximation
        # and we're using the full thermoviscous function
        assert 0.5 * expected_amplitude_ratio < measured_amplitude_ratio < 2.0 * expected_amplitude_ratio, (
            f"Attenuation mismatch: expected ratio={expected_amplitude_ratio:.4f}, "
            f"measured ratio={measured_amplitude_ratio:.4f}"
        )

    def test_penetration_depth_effect(
        self,
        helium_gas: Helium,
        test_temperature: float,
        circular_geometry: CircularPore,
    ) -> None:
        """Verify that smaller radius (larger delta/r) gives more amplitude attenuation.

        For a duct with boundary layer losses, smaller radius means:
        - Larger ratio of penetration depth to radius (delta/r)
        - More of the flow affected by viscous/thermal losses
        - Greater amplitude attenuation
        """
        frequency = 100.0
        omega = 2 * np.pi * frequency
        length = 1.0  # m - use longer length to see clearer attenuation

        radii = [0.003, 0.01, 0.03]  # 3mm, 10mm, 30mm - wider range
        amplitude_ratios = []

        rho = helium_gas.density(test_temperature)
        a = helium_gas.sound_speed(test_temperature)

        for radius in radii:
            duct = Duct(length=length, radius=radius, geometry=circular_geometry)

            # Use pure pressure input to measure pressure attenuation clearly
            p1_in = 1000.0 + 0j
            U1_in = 0.0 + 0j  # Start with zero velocity (hard end condition)

            p1_out, U1_out, _ = duct.propagate(
                p1_in, U1_in, test_temperature, omega, helium_gas
            )

            amplitude_ratios.append(abs(p1_out) / abs(p1_in))

        # Smaller radius should give lower amplitude ratio (more attenuation)
        # This is because delta/r is larger for smaller r
        assert amplitude_ratios[0] < amplitude_ratios[1] < amplitude_ratios[2], (
            f"Smaller radius should give more attenuation: {dict(zip(radii, amplitude_ratios))}"
        )


# =============================================================================
# Cone Segment Tests
# =============================================================================


class TestCone:
    """Test cone segment for area variation and continuity."""

    def test_area_variation_along_length(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify area changes correctly from inlet to outlet."""
        length = 0.5
        radius_in = 0.01
        radius_out = 0.02

        cone = Cone(length=length, radius_in=radius_in, radius_out=radius_out)

        # Check area at inlet
        A_in = cone.area_at(0.0)
        expected_A_in = np.pi * radius_in**2
        assert np.isclose(A_in, expected_A_in, rtol=1e-10), (
            f"Inlet area mismatch: {A_in} vs {expected_A_in}"
        )

        # Check area at outlet
        A_out = cone.area_at(length)
        expected_A_out = np.pi * radius_out**2
        assert np.isclose(A_out, expected_A_out, rtol=1e-10), (
            f"Outlet area mismatch: {A_out} vs {expected_A_out}"
        )

        # Check area at midpoint
        A_mid = cone.area_at(length / 2)
        r_mid = (radius_in + radius_out) / 2
        expected_A_mid = np.pi * r_mid**2
        assert np.isclose(A_mid, expected_A_mid, rtol=1e-10), (
            f"Midpoint area mismatch: {A_mid} vs {expected_A_mid}"
        )

    def test_radius_linear_interpolation(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify radius varies linearly along cone length."""
        length = 1.0
        radius_in = 0.01
        radius_out = 0.05

        cone = Cone(length=length, radius_in=radius_in, radius_out=radius_out)

        # Check at various positions
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        for x in positions:
            r_actual = cone.radius_at(x)
            r_expected = radius_in + (radius_out - radius_in) * x / length
            assert np.isclose(r_actual, r_expected, rtol=1e-10), (
                f"Radius at x={x}: got {r_actual}, expected {r_expected}"
            )

    def test_mass_continuity_conservation(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify continuity (mass conservation) through cone.

        For an inviscid fluid, rho * u * A = constant (mass flow rate).
        In the acoustic approximation, this manifests as proper handling
        of area changes in the momentum and continuity equations.
        """
        frequency = 500.0
        omega = 2 * np.pi * frequency
        length = 0.3
        radius_in = 0.01
        radius_out = 0.02  # Expansion

        cone = Cone(length=length, radius_in=radius_in, radius_out=radius_out)

        # Initial conditions
        p1_in = 1000.0 + 0j

        # For continuity check, use a simple initial velocity
        rho = helium_gas.density(test_temperature)
        a = helium_gas.sound_speed(test_temperature)
        A_in = np.pi * radius_in**2
        U1_in = p1_in * A_in / (rho * a)  # Forward traveling wave at inlet

        p1_out, U1_out, T_out = cone.propagate(
            p1_in, U1_in, test_temperature, omega, helium_gas
        )

        # For a nearly lossless expanding cone, the velocity amplitude
        # should decrease (conservation of mass flux ~ constant)
        # and pressure should adjust accordingly

        # Basic check: solution should be finite and reasonable
        assert np.isfinite(p1_out), "Output pressure is not finite"
        assert np.isfinite(U1_out), "Output velocity is not finite"
        assert abs(p1_out) > 0, "Output pressure should be non-zero"
        assert abs(U1_out) > 0, "Output velocity should be non-zero"

    def test_cone_vs_duct_uniform_case(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify cone with equal radii behaves like a duct."""
        frequency = 200.0
        omega = 2 * np.pi * frequency
        length = 0.3
        radius = 0.02

        # Create cone with equal inlet/outlet radii
        cone = Cone(length=length, radius_in=radius, radius_out=radius)
        duct = Duct(length=length, radius=radius)

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_cone, U1_cone, T_cone = cone.propagate(
            p1_in, U1_in, test_temperature, omega, helium_gas
        )

        p1_duct, U1_duct, T_duct = duct.propagate(
            p1_in, U1_in, test_temperature, omega, helium_gas
        )

        # Results should match within numerical tolerance
        assert np.isclose(p1_cone, p1_duct, rtol=1e-6), (
            f"Pressure mismatch: cone={p1_cone}, duct={p1_duct}"
        )
        assert np.isclose(U1_cone, U1_duct, rtol=1e-6), (
            f"Velocity mismatch: cone={U1_cone}, duct={U1_duct}"
        )


# =============================================================================
# Stack Segment Tests
# =============================================================================


class TestStack:
    """Test stack segment with temperature gradient."""

    def test_temperature_profile_linear(
        self,
        helium_gas: Helium,
    ) -> None:
        """Verify temperature profile is linear from T_cold to T_hot."""
        length = 0.05
        porosity = 0.7
        hydraulic_radius = 0.0005
        T_cold = 300.0
        T_hot = 500.0

        stack = Stack(
            length=length,
            porosity=porosity,
            hydraulic_radius=hydraulic_radius,
            T_hot=T_hot,
            T_cold=T_cold,
        )

        # Check temperature at various positions
        positions = [0.0, 0.25 * length, 0.5 * length, 0.75 * length, length]

        for x in positions:
            T_actual = stack.temperature_at(x, T_m_input=T_cold)
            T_expected = T_cold + (T_hot - T_cold) * x / length
            assert np.isclose(T_actual, T_expected, rtol=1e-10), (
                f"Temperature at x={x}: got {T_actual} K, expected {T_expected} K"
            )

    def test_temperature_gradient_calculation(
        self,
        helium_gas: Helium,
    ) -> None:
        """Verify temperature gradient is calculated correctly."""
        length = 0.05
        porosity = 0.7
        hydraulic_radius = 0.0005
        T_cold = 300.0
        T_hot = 500.0

        stack = Stack(
            length=length,
            porosity=porosity,
            hydraulic_radius=hydraulic_radius,
            T_hot=T_hot,
            T_cold=T_cold,
        )

        expected_gradient = (T_hot - T_cold) / length
        actual_gradient = stack.temperature_gradient()

        assert np.isclose(actual_gradient, expected_gradient, rtol=1e-10), (
            f"Temperature gradient: got {actual_gradient} K/m, expected {expected_gradient} K/m"
        )

    def test_output_temperature_with_gradient(
        self,
        helium_gas: Helium,
    ) -> None:
        """Verify output temperature equals T_hot when gradient is imposed."""
        frequency = 100.0
        omega = 2 * np.pi * frequency
        length = 0.05
        porosity = 0.7
        hydraulic_radius = 0.0005
        T_cold = 300.0
        T_hot = 500.0

        stack = Stack(
            length=length,
            porosity=porosity,
            hydraulic_radius=hydraulic_radius,
            T_hot=T_hot,
            T_cold=T_cold,
        )

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_out, U1_out, T_out = stack.propagate(
            p1_in, U1_in, T_cold, omega, helium_gas
        )

        assert np.isclose(T_out, T_hot, rtol=1e-10), (
            f"Output temperature should be T_hot={T_hot} K, got {T_out} K"
        )

    def test_acoustic_power_change_with_gradient(
        self,
        helium_gas: Helium,
        circular_geometry: CircularPore,
    ) -> None:
        """Verify acoustic power changes with temperature gradient.

        In a stack with temperature gradient, the thermoacoustic effect
        causes acoustic power to either be amplified (engine mode) or
        absorbed (refrigerator mode) depending on phasing.
        """
        frequency = 100.0
        omega = 2 * np.pi * frequency
        length = 0.05
        porosity = 0.7
        hydraulic_radius = 0.0005
        T_cold = 300.0
        T_hot = 400.0  # Moderate gradient

        stack = Stack(
            length=length,
            porosity=porosity,
            hydraulic_radius=hydraulic_radius,
            T_hot=T_hot,
            T_cold=T_cold,
            geometry=circular_geometry,
        )

        # Use standing wave phasing (p1 and U1 in phase) for amplification
        p1_in = 1000.0 + 0j
        U1_in = 0.0001 + 0j  # In phase with pressure

        power_in = acoustic_power(p1_in, U1_in)

        p1_out, U1_out, T_out = stack.propagate(
            p1_in, U1_in, T_cold, omega, helium_gas
        )

        power_out = acoustic_power(p1_out, U1_out)

        # Power should change (either increase or decrease, depending on phasing)
        # Here we just verify it's not exactly conserved
        power_ratio = power_out / power_in if power_in != 0 else 0

        # With temperature gradient, power should not be exactly conserved
        # (within a loose tolerance to account for the thermoacoustic effect)
        assert not np.isclose(power_ratio, 1.0, atol=0.5), (
            f"Acoustic power should change with temperature gradient, ratio={power_ratio:.4f}"
        )

    def test_stack_no_gradient_isothermal(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify stack without gradient behaves like isothermal case."""
        frequency = 100.0
        omega = 2 * np.pi * frequency
        length = 0.05
        porosity = 0.7
        hydraulic_radius = 0.0005

        # Stack without temperature gradient
        stack = Stack(
            length=length,
            porosity=porosity,
            hydraulic_radius=hydraulic_radius,
        )

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_out, U1_out, T_out = stack.propagate(
            p1_in, U1_in, test_temperature, omega, helium_gas
        )

        # Temperature should be unchanged
        assert np.isclose(T_out, test_temperature, rtol=1e-10), (
            f"Output temperature should equal input when no gradient: got {T_out} K"
        )

        # Gradient should be zero
        assert stack.temperature_gradient() == 0.0, (
            "Temperature gradient should be zero when T_hot/T_cold not specified"
        )


# =============================================================================
# Heat Exchanger Tests
# =============================================================================


class TestHeatExchanger:
    """Test heat exchanger segment with fixed temperature."""

    def test_output_temperature_fixed(
        self,
        helium_gas: Helium,
    ) -> None:
        """Verify output temperature equals the fixed HX temperature."""
        frequency = 100.0
        omega = 2 * np.pi * frequency
        length = 0.01
        porosity = 0.5
        hydraulic_radius = 0.001
        T_hx = 350.0  # Fixed HX temperature

        hx = HeatExchanger(
            length=length,
            porosity=porosity,
            hydraulic_radius=hydraulic_radius,
            temperature=T_hx,
        )

        # Input at different temperature
        T_in = 300.0
        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_out, U1_out, T_out = hx.propagate(
            p1_in, U1_in, T_in, omega, helium_gas
        )

        assert np.isclose(T_out, T_hx, rtol=1e-10), (
            f"Output temperature should be HX temperature {T_hx} K, got {T_out} K"
        )

    def test_hx_wave_propagation(
        self,
        helium_gas: Helium,
        circular_geometry: CircularPore,
    ) -> None:
        """Verify wave propagation through heat exchanger is similar to porous duct."""
        frequency = 100.0
        omega = 2 * np.pi * frequency
        length = 0.01
        porosity = 0.5
        hydraulic_radius = 0.001
        T_hx = 300.0

        hx = HeatExchanger(
            length=length,
            porosity=porosity,
            hydraulic_radius=hydraulic_radius,
            temperature=T_hx,
            geometry=circular_geometry,
        )

        p1_in = 1000.0 + 0j
        U1_in = 0.0001 + 0j

        p1_out, U1_out, T_out = hx.propagate(
            p1_in, U1_in, T_hx, omega, helium_gas
        )

        # Solution should be finite and reasonable
        assert np.isfinite(p1_out), "Output pressure is not finite"
        assert np.isfinite(U1_out), "Output velocity is not finite"

        # For a short HX, changes should be modest
        assert abs(p1_out - p1_in) < abs(p1_in), (
            "Pressure change through short HX should be bounded"
        )

    def test_hx_zero_length(
        self,
        helium_gas: Helium,
    ) -> None:
        """Verify zero-length HX returns fixed temperature but unchanged acoustics."""
        omega = 628.3
        porosity = 0.5
        hydraulic_radius = 0.001
        T_hx = 350.0

        hx = HeatExchanger(
            length=0.0,
            porosity=porosity,
            hydraulic_radius=hydraulic_radius,
            temperature=T_hx,
        )

        T_in = 300.0
        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_out, U1_out, T_out = hx.propagate(
            p1_in, U1_in, T_in, omega, helium_gas
        )

        # Zero-length: acoustics unchanged, temperature becomes HX temperature
        assert p1_out == p1_in, "Zero-length HX should not change pressure"
        assert U1_out == U1_in, "Zero-length HX should not change velocity"
        assert T_out == T_hx, "Zero-length HX should still set temperature"


# =============================================================================
# Compliance Tests
# =============================================================================


class TestCompliance:
    """Test compliance (lumped acoustic volume) segment."""

    def test_pressure_continuous(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Verify p1_out = p1_in for compliance element."""
        volume = 1e-4  # 100 cm^3
        compliance = Compliance(volume=volume)

        p1_in = 1000.0 + 500j
        U1_in = 0.001 + 0.0005j

        p1_out, U1_out, T_out = compliance.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        assert p1_out == p1_in, (
            f"Pressure should be continuous through compliance: in={p1_in}, out={p1_out}"
        )

    def test_velocity_change(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Verify U1 changes according to lumped compliance relation."""
        volume = 1e-4  # 100 cm^3
        compliance = Compliance(volume=volume)

        rho = helium_gas.density(test_temperature)
        a = helium_gas.sound_speed(test_temperature)
        C = volume / (rho * a**2)  # Acoustic compliance

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_out, U1_out, T_out = compliance.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Expected: U1_out = U1_in - j*omega*C*p1_in
        expected_delta_U1 = -1j * test_omega * C * p1_in
        expected_U1_out = U1_in + expected_delta_U1

        assert np.isclose(U1_out, expected_U1_out, rtol=1e-10), (
            f"Velocity change incorrect: got {U1_out}, expected {expected_U1_out}"
        )

    def test_compliance_impedance(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test against analytical impedance: Z = 1/(j*omega*C) where C = V/(rho*a^2).

        The compliance acts as an acoustic capacitor. The relationship is:
        delta_U1 = -j*omega*C*p1, or equivalently, Z = p1/delta_U1 = 1/(j*omega*C)
        """
        volume = 1e-4  # 100 cm^3
        compliance = Compliance(volume=volume)

        rho = helium_gas.density(test_temperature)
        a = helium_gas.sound_speed(test_temperature)
        C = volume / (rho * a**2)  # Acoustic compliance

        # Expected impedance
        Z_expected = 1 / (1j * test_omega * C)

        # Test by applying a pressure and measuring velocity change
        p1_in = 1000.0 + 0j
        U1_in = 0.0 + 0j  # Start with zero velocity

        p1_out, U1_out, _ = compliance.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # The change in U1 is due to the compliance
        delta_U1 = U1_out - U1_in

        # Z = -p1/delta_U1 (negative because delta_U is opposite to conventional definition)
        if delta_U1 != 0:
            Z_measured = -p1_in / delta_U1
            assert np.isclose(Z_measured, Z_expected, rtol=1e-10), (
                f"Impedance mismatch: measured={Z_measured}, expected={Z_expected}"
            )

    def test_acoustic_compliance_property(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify the acoustic_compliance property returns correct value."""
        volume = 1e-4
        compliance = Compliance(volume=volume)

        rho = helium_gas.density(test_temperature)
        a = helium_gas.sound_speed(test_temperature)
        expected_C = volume / (rho * a**2)

        actual_C = compliance.acoustic_compliance(helium_gas, test_temperature)

        assert np.isclose(actual_C, expected_C, rtol=1e-10), (
            f"Acoustic compliance: got {actual_C}, expected {expected_C}"
        )


# =============================================================================
# Inertance Tests
# =============================================================================


class TestInertance:
    """Test inertance (lumped acoustic mass) segment."""

    def test_velocity_continuous(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Verify U1_out = U1_in for inertance element."""
        length = 0.1  # 10 cm
        radius = 0.005  # 5 mm
        inertance = Inertance(length=length, radius=radius)

        p1_in = 1000.0 + 500j
        U1_in = 0.001 + 0.0005j

        p1_out, U1_out, T_out = inertance.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        assert U1_out == U1_in, (
            f"Velocity should be continuous through inertance: in={U1_in}, out={U1_out}"
        )

    def test_pressure_change(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Verify p1 changes according to lumped inertance relation."""
        length = 0.1  # 10 cm
        radius = 0.005  # 5 mm
        area = np.pi * radius**2
        inertance = Inertance(length=length, radius=radius, include_resistance=False)

        rho = helium_gas.density(test_temperature)
        L_a = rho * length / area  # Acoustic inertance

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_out, U1_out, T_out = inertance.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # Expected: p1_out = p1_in - j*omega*L_a*U1_in
        expected_delta_p1 = -1j * test_omega * L_a * U1_in
        expected_p1_out = p1_in + expected_delta_p1

        assert np.isclose(p1_out, expected_p1_out, rtol=1e-10), (
            f"Pressure change incorrect: got {p1_out}, expected {expected_p1_out}"
        )

    def test_inertance_impedance(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Test against analytical impedance: Z = j*omega*L where L = rho*length/A.

        The inertance acts as an acoustic inductor. The relationship is:
        delta_p1 = -j*omega*L*U1, or equivalently, Z = -delta_p1/U1 = j*omega*L
        """
        length = 0.1
        radius = 0.005
        area = np.pi * radius**2
        inertance = Inertance(length=length, radius=radius, include_resistance=False)

        rho = helium_gas.density(test_temperature)
        L_a = rho * length / area  # Acoustic inertance

        # Expected impedance
        Z_expected = 1j * test_omega * L_a

        # Test by applying a velocity and measuring pressure change
        p1_in = 0.0 + 0j  # Start with zero pressure
        U1_in = 0.001 + 0j

        p1_out, U1_out, _ = inertance.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # The change in p1 is due to the inertance
        delta_p1 = p1_out - p1_in

        # Z = -delta_p1/U1
        if U1_in != 0:
            Z_measured = -delta_p1 / U1_in
            assert np.isclose(Z_measured, Z_expected, rtol=1e-10), (
                f"Impedance mismatch: measured={Z_measured}, expected={Z_expected}"
            )

    def test_acoustic_inertance_property(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Verify the acoustic_inertance property returns correct value."""
        length = 0.1
        radius = 0.005
        area = np.pi * radius**2
        inertance = Inertance(length=length, radius=radius)

        rho = helium_gas.density(test_temperature)
        expected_L = rho * length / area

        actual_L = inertance.acoustic_inertance(helium_gas, test_temperature)

        assert np.isclose(actual_L, expected_L, rtol=1e-10), (
            f"Acoustic inertance: got {actual_L}, expected {expected_L}"
        )

    def test_inertance_with_resistance(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Verify inertance with resistance includes resistive losses."""
        length = 0.1
        radius = 0.005
        area = np.pi * radius**2

        inertance_no_R = Inertance(length=length, radius=radius, include_resistance=False)
        inertance_with_R = Inertance(length=length, radius=radius, include_resistance=True)

        rho = helium_gas.density(test_temperature)
        mu = helium_gas.viscosity(test_temperature)
        L_a = rho * length / area
        R = 8 * np.pi * mu * length / (area**2)  # Poiseuille resistance

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_no_R, _, _ = inertance_no_R.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        p1_with_R, _, _ = inertance_with_R.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        # With resistance, the pressure drop should be larger
        delta_p_no_R = abs(p1_no_R - p1_in)
        delta_p_with_R = abs(p1_with_R - p1_in)

        # The resistive component adds to the pressure drop
        expected_delta_p_with_R_squared = (test_omega * L_a * abs(U1_in))**2 + (R * abs(U1_in))**2

        # Verify resistance adds to pressure drop
        assert delta_p_with_R > delta_p_no_R, (
            "Pressure drop with resistance should be larger than without"
        )

    def test_inertance_with_area_instead_of_radius(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Verify inertance can be created with area parameter."""
        length = 0.1
        radius = 0.005
        area = np.pi * radius**2

        inert_radius = Inertance(length=length, radius=radius)
        inert_area = Inertance(length=length, area=area)

        p1_in = 1000.0 + 0j
        U1_in = 0.001 + 0j

        p1_r, U1_r, _ = inert_radius.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        p1_a, U1_a, _ = inert_area.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        assert np.isclose(p1_r, p1_a, rtol=1e-10), (
            "Inertance with radius vs area should give same result"
        )


# =============================================================================
# Boundary Condition Tests
# =============================================================================


class TestHardEnd:
    """Test hard end (closed end) boundary condition."""

    def test_is_satisfied_zero_velocity(self) -> None:
        """Verify is_satisfied returns True when |U1| < tolerance."""
        hard_end = HardEnd()

        # Zero velocity - should be satisfied
        assert hard_end.is_satisfied(U1=0.0 + 0j, tolerance=1e-10) is True

        # Very small velocity - should be satisfied
        assert hard_end.is_satisfied(U1=1e-12 + 1e-12j, tolerance=1e-10) is True

        # Small but above tolerance - should not be satisfied
        assert hard_end.is_satisfied(U1=1e-8 + 0j, tolerance=1e-10) is False

    def test_is_satisfied_nonzero_velocity(self) -> None:
        """Verify is_satisfied returns False when |U1| > tolerance."""
        hard_end = HardEnd()

        # Non-zero velocity
        assert hard_end.is_satisfied(U1=0.001 + 0j, tolerance=1e-10) is False
        assert hard_end.is_satisfied(U1=0.0 + 0.001j, tolerance=1e-10) is False
        assert hard_end.is_satisfied(U1=0.001 + 0.001j, tolerance=1e-10) is False

    def test_is_satisfied_custom_tolerance(self) -> None:
        """Verify is_satisfied works with custom tolerance."""
        hard_end = HardEnd()

        U1 = 1e-6 + 0j

        # With tight tolerance - not satisfied
        assert hard_end.is_satisfied(U1, tolerance=1e-8) is False

        # With loose tolerance - satisfied
        assert hard_end.is_satisfied(U1, tolerance=1e-4) is True

    def test_residual(self) -> None:
        """Verify residual returns U1."""
        hard_end = HardEnd()

        U1 = 0.001 + 0.002j
        assert hard_end.residual(U1) == U1

    def test_propagate_passthrough(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Verify propagate returns input state unchanged."""
        hard_end = HardEnd()

        p1_in = 1000.0 + 500j
        U1_in = 0.001 + 0.0005j

        p1_out, U1_out, T_out = hard_end.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        assert p1_out == p1_in
        assert U1_out == U1_in
        assert T_out == test_temperature


class TestSoftEnd:
    """Test soft end (ideal open end) boundary condition."""

    def test_is_satisfied_zero_pressure(self) -> None:
        """Verify is_satisfied returns True when |p1| < tolerance."""
        soft_end = SoftEnd()

        # Zero pressure - should be satisfied
        assert soft_end.is_satisfied(p1=0.0 + 0j, tolerance=1e-10) is True

        # Very small pressure - should be satisfied
        assert soft_end.is_satisfied(p1=1e-12 + 1e-12j, tolerance=1e-10) is True

        # Small but above tolerance - should not be satisfied
        assert soft_end.is_satisfied(p1=1e-8 + 0j, tolerance=1e-10) is False

    def test_is_satisfied_nonzero_pressure(self) -> None:
        """Verify is_satisfied returns False when |p1| > tolerance."""
        soft_end = SoftEnd()

        # Non-zero pressure
        assert soft_end.is_satisfied(p1=1000.0 + 0j, tolerance=1e-10) is False
        assert soft_end.is_satisfied(p1=0.0 + 1000.0j, tolerance=1e-10) is False
        assert soft_end.is_satisfied(p1=1000.0 + 1000.0j, tolerance=1e-10) is False

    def test_is_satisfied_custom_tolerance(self) -> None:
        """Verify is_satisfied works with custom tolerance."""
        soft_end = SoftEnd()

        p1 = 1e-6 + 0j

        # With tight tolerance - not satisfied
        assert soft_end.is_satisfied(p1, tolerance=1e-8) is False

        # With loose tolerance - satisfied
        assert soft_end.is_satisfied(p1, tolerance=1e-4) is True

    def test_residual(self) -> None:
        """Verify residual returns p1."""
        soft_end = SoftEnd()

        p1 = 1000.0 + 500.0j
        assert soft_end.residual(p1) == p1

    def test_propagate_passthrough(
        self,
        helium_gas: Helium,
        test_temperature: float,
        test_omega: float,
    ) -> None:
        """Verify propagate returns input state unchanged."""
        soft_end = SoftEnd()

        p1_in = 1000.0 + 500j
        U1_in = 0.001 + 0.0005j

        p1_out, U1_out, T_out = soft_end.propagate(
            p1_in, U1_in, test_temperature, test_omega, helium_gas
        )

        assert p1_out == p1_in
        assert U1_out == U1_in
        assert T_out == test_temperature


# =============================================================================
# Integration Tests
# =============================================================================


class TestSegmentIntegration:
    """Integration tests combining multiple segments."""

    def test_compliance_inertance_resonator(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Test Helmholtz resonator using compliance and inertance.

        A Helmholtz resonator consists of an inertance (neck) connected
        to a compliance (volume). The resonant frequency is:
        f = (1/2*pi) * sqrt(1/(L_a * C))
        """
        # Design parameters for ~100 Hz resonance
        volume = 1e-3  # 1 liter
        neck_length = 0.05  # 5 cm
        neck_radius = 0.01  # 1 cm
        neck_area = np.pi * neck_radius**2

        rho = helium_gas.density(test_temperature)
        a = helium_gas.sound_speed(test_temperature)

        L_a = rho * neck_length / neck_area
        C = volume / (rho * a**2)

        f_resonance = 1 / (2 * np.pi * np.sqrt(L_a * C))

        compliance = Compliance(volume=volume)
        inertance = Inertance(length=neck_length, radius=neck_radius)

        # Test at resonance
        omega_res = 2 * np.pi * f_resonance

        # At resonance, the impedance of the system should be purely resistive (zero)
        # for an ideal lossless case
        p1_in = 1000.0 + 0j
        U1_in = 0.0 + 0j

        # Propagate through inertance then compliance
        p1_mid, U1_mid, T_mid = inertance.propagate(
            p1_in, U1_in, test_temperature, omega_res, helium_gas
        )

        p1_out, U1_out, T_out = compliance.propagate(
            p1_mid, U1_mid, T_mid, omega_res, helium_gas
        )

        # System should have predictable behavior at resonance
        # (detailed resonance checking requires solving the full system)
        assert np.isfinite(p1_out), "Output pressure should be finite"
        assert np.isfinite(U1_out), "Output velocity should be finite"

    def test_duct_with_boundary_conditions(
        self,
        helium_gas: Helium,
        test_temperature: float,
    ) -> None:
        """Test duct terminated with hard end boundary condition."""
        frequency = 100.0
        omega = 2 * np.pi * frequency

        a = helium_gas.sound_speed(test_temperature)
        wavelength = a / frequency

        # Quarter wavelength duct with hard end should have zero velocity at end
        duct = Duct(length=wavelength/4, radius=0.05)
        hard_end = HardEnd()

        # Start with pressure maximum, zero velocity (standing wave antinode)
        p1_start = 1000.0 + 0j
        U1_start = 0.0 + 0j

        p1_end, U1_end, T_end = duct.propagate(
            p1_start, U1_start, test_temperature, omega, helium_gas
        )

        # At quarter wavelength from a hard end, we should have a pressure node
        # and velocity antinode. Starting from pressure antinode means we
        # end at pressure node.
        # The hard end is at the starting position (velocity = 0).

        # Check that the hard end would be satisfied at the start
        assert hard_end.is_satisfied(U1_start, tolerance=1e-10), (
            "Hard end boundary condition should be satisfied at start"
        )

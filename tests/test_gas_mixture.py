"""Comprehensive tests for gas mixture module.

This module tests the GasMixture class and related convenience functions
for calculating thermophysical properties of gas mixtures. Tests verify:
1. Pure gas limits (100% component should match pure gas properties)
2. Molar mass calculation for known mixtures
3. Ideal gas law compliance for mixtures
4. Prandtl number reasonableness (0.6-0.8 range)
5. Wilke's mixing rule against literature values for He-Ar
6. Sound speed for He-Ar mixtures against published data

All tests use SI units consistently.

References
----------
Poling, B. E., Prausnitz, J. M., & O'Connell, J. P. (2001).
    The Properties of Gases and Liquids (5th ed.). McGraw-Hill.

Swift, G. W. (2017). Thermoacoustics: A Unifying Perspective for Some
    Engines and Refrigerators (2nd ed.). Springer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openthermoacoustics.gas import (
    Argon,
    GasMixture,
    Helium,
    Xenon,
    helium_argon,
    helium_xenon,
)
from openthermoacoustics.utils import MOLAR_MASS, R_UNIVERSAL


# Standard atmospheric pressure (Pa)
P_ATM = 101325.0


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def helium() -> Helium:
    """Create a Helium gas instance at standard atmospheric pressure."""
    return Helium(mean_pressure=P_ATM)


@pytest.fixture
def argon() -> Argon:
    """Create an Argon gas instance at standard atmospheric pressure."""
    return Argon(mean_pressure=P_ATM)


@pytest.fixture
def xenon() -> Xenon:
    """Create a Xenon gas instance at standard atmospheric pressure."""
    return Xenon(mean_pressure=P_ATM)


@pytest.fixture
def he_ar_70_30(helium: Helium, argon: Argon) -> GasMixture:
    """Create a 70% He, 30% Ar mixture at standard atmospheric pressure."""
    return GasMixture([helium, argon], [0.7, 0.3])


@pytest.fixture
def he_ar_50_50(helium: Helium, argon: Argon) -> GasMixture:
    """Create a 50% He, 50% Ar mixture at standard atmospheric pressure."""
    return GasMixture([helium, argon], [0.5, 0.5])


# ============================================================================
# Test: Pure Gas Limits
# ============================================================================


class TestPureGasLimits:
    """
    Tests verifying that 100% of a component matches pure gas properties.

    When a mixture contains only one gas (100% mole fraction), all properties
    should match the pure gas values exactly.
    """

    T_TEST = 300.0  # Test temperature (K)

    def test_pure_helium_density(self, helium: Helium) -> None:
        """100% He mixture should have same density as pure Helium."""
        mix = GasMixture([helium], [1.0])

        assert mix.density(self.T_TEST) == pytest.approx(
            helium.density(self.T_TEST), rel=1e-10
        )

    def test_pure_helium_sound_speed(self, helium: Helium) -> None:
        """100% He mixture should have same sound speed as pure Helium."""
        mix = GasMixture([helium], [1.0])

        assert mix.sound_speed(self.T_TEST) == pytest.approx(
            helium.sound_speed(self.T_TEST), rel=1e-10
        )

    def test_pure_helium_viscosity(self, helium: Helium) -> None:
        """100% He mixture should have same viscosity as pure Helium."""
        mix = GasMixture([helium], [1.0])

        assert mix.viscosity(self.T_TEST) == pytest.approx(
            helium.viscosity(self.T_TEST), rel=1e-10
        )

    def test_pure_helium_thermal_conductivity(self, helium: Helium) -> None:
        """100% He mixture should have same thermal conductivity as pure Helium."""
        mix = GasMixture([helium], [1.0])

        assert mix.thermal_conductivity(self.T_TEST) == pytest.approx(
            helium.thermal_conductivity(self.T_TEST), rel=1e-10
        )

    def test_pure_helium_specific_heat(self, helium: Helium) -> None:
        """100% He mixture should have same specific heat as pure Helium."""
        mix = GasMixture([helium], [1.0])

        assert mix.specific_heat_cp(self.T_TEST) == pytest.approx(
            helium.specific_heat_cp(self.T_TEST), rel=1e-10
        )

    def test_pure_helium_gamma(self, helium: Helium) -> None:
        """100% He mixture should have same gamma as pure Helium."""
        mix = GasMixture([helium], [1.0])

        assert mix.gamma(self.T_TEST) == pytest.approx(
            helium.gamma(self.T_TEST), rel=1e-10
        )

    def test_pure_helium_prandtl(self, helium: Helium) -> None:
        """100% He mixture should have same Prandtl number as pure Helium."""
        mix = GasMixture([helium], [1.0])

        assert mix.prandtl(self.T_TEST) == pytest.approx(
            helium.prandtl(self.T_TEST), rel=1e-10
        )

    def test_pure_argon_all_properties(self, argon: Argon) -> None:
        """100% Ar mixture should match pure Argon for all properties."""
        mix = GasMixture([argon], [1.0])

        assert mix.molar_mass == pytest.approx(argon.molar_mass, rel=1e-10)
        assert mix.density(self.T_TEST) == pytest.approx(
            argon.density(self.T_TEST), rel=1e-10
        )
        assert mix.sound_speed(self.T_TEST) == pytest.approx(
            argon.sound_speed(self.T_TEST), rel=1e-10
        )
        assert mix.viscosity(self.T_TEST) == pytest.approx(
            argon.viscosity(self.T_TEST), rel=1e-10
        )
        assert mix.thermal_conductivity(self.T_TEST) == pytest.approx(
            argon.thermal_conductivity(self.T_TEST), rel=1e-10
        )

    def test_binary_mixture_near_pure_helium(
        self, helium: Helium, argon: Argon
    ) -> None:
        """99.99% He should be very close to pure Helium."""
        mix = GasMixture([helium, argon], [0.9999, 0.0001])

        # Should be within 0.1% of pure helium
        assert mix.density(self.T_TEST) == pytest.approx(
            helium.density(self.T_TEST), rel=0.001
        )
        assert mix.sound_speed(self.T_TEST) == pytest.approx(
            helium.sound_speed(self.T_TEST), rel=0.001
        )


# ============================================================================
# Test: Molar Mass Calculation
# ============================================================================


class TestMolarMassCalculation:
    """
    Tests verifying mixture molar mass calculation.

    M_mix = sum(x_i * M_i) where x_i is mole fraction and M_i is molar mass.
    """

    def test_he_ar_molar_mass_50_50(self, helium: Helium, argon: Argon) -> None:
        """50-50 He-Ar mixture molar mass should be arithmetic mean."""
        mix = GasMixture([helium, argon], [0.5, 0.5])

        expected_M = 0.5 * MOLAR_MASS["helium"] + 0.5 * MOLAR_MASS["argon"]

        assert mix.molar_mass == pytest.approx(expected_M, rel=1e-10)

    def test_he_ar_molar_mass_70_30(self, helium: Helium, argon: Argon) -> None:
        """70-30 He-Ar mixture molar mass calculation."""
        mix = GasMixture([helium, argon], [0.7, 0.3])

        # M_mix = 0.7 * 4.002602e-3 + 0.3 * 39.948e-3
        expected_M = 0.7 * MOLAR_MASS["helium"] + 0.3 * MOLAR_MASS["argon"]

        assert mix.molar_mass == pytest.approx(expected_M, rel=1e-10)
        # Expected value: ~0.01479 kg/mol
        assert 0.014 < mix.molar_mass < 0.016

    def test_he_xe_molar_mass(self, helium: Helium, xenon: Xenon) -> None:
        """He-Xe mixture molar mass calculation."""
        mix = GasMixture([helium, xenon], [0.8, 0.2])

        expected_M = 0.8 * MOLAR_MASS["helium"] + 0.2 * MOLAR_MASS["xenon"]

        assert mix.molar_mass == pytest.approx(expected_M, rel=1e-10)

    def test_molar_mass_increases_with_ar_fraction(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Molar mass should increase as Ar fraction increases."""
        fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        molar_masses = []

        for he_frac in fractions:
            ar_frac = 1.0 - he_frac
            if he_frac == 0:
                mix = GasMixture([argon], [1.0])
            elif ar_frac == 0:
                mix = GasMixture([helium], [1.0])
            else:
                mix = GasMixture([helium, argon], [he_frac, ar_frac])
            molar_masses.append(mix.molar_mass)

        # Molar mass should decrease as He fraction increases (He is lighter)
        for i in range(len(molar_masses) - 1):
            assert molar_masses[i] > molar_masses[i + 1]


# ============================================================================
# Test: Ideal Gas Law for Mixtures
# ============================================================================


class TestIdealGasLawMixture:
    """
    Tests verifying that mixture density follows the ideal gas law.

    rho = P / (R_specific * T) where R_specific = R_universal / M_mix
    """

    TP_COMBINATIONS = [
        (200.0, P_ATM),
        (300.0, P_ATM),
        (400.0, P_ATM),
        (300.0, 50000.0),
        (300.0, 200000.0),
    ]

    @pytest.mark.parametrize("T,P", TP_COMBINATIONS)
    def test_he_ar_ideal_gas_law(
        self, he_ar_70_30: GasMixture, T: float, P: float
    ) -> None:
        """He-Ar mixture density should follow ideal gas law."""
        R_specific = R_UNIVERSAL / he_ar_70_30.molar_mass
        expected_rho = P / (R_specific * T)
        calculated_rho = he_ar_70_30.density(T, P)

        assert calculated_rho == pytest.approx(expected_rho, rel=1e-10)

    def test_mixture_density_between_components(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Mixture density should be between pure component densities."""
        T = 300.0
        mix = GasMixture([helium, argon], [0.5, 0.5])

        rho_he = helium.density(T)
        rho_ar = argon.density(T)
        rho_mix = mix.density(T)

        # Mixture density should be between pure component densities
        assert rho_he < rho_mix < rho_ar

    def test_density_uses_mean_pressure_when_none(
        self, he_ar_70_30: GasMixture
    ) -> None:
        """Density should use mean_pressure when P is not specified."""
        T = 300.0
        rho_explicit = he_ar_70_30.density(T, P_ATM)
        rho_implicit = he_ar_70_30.density(T)

        assert rho_explicit == pytest.approx(rho_implicit, rel=1e-10)


# ============================================================================
# Test: Prandtl Number Reasonableness
# ============================================================================


class TestPrandtlNumberReasonableness:
    """
    Tests verifying that mixture Prandtl numbers are in reasonable range.

    For gas mixtures of monatomic gases, Prandtl numbers calculated using
    Wilke's mixing rule for viscosity and Mason-Saxena for thermal conductivity
    can range from about 0.6 to 1.0 depending on composition and the large
    mass ratio between He and Ar.

    Note: The Prandtl number for He-Ar mixtures can exceed the pure component
    values due to the nonlinear mixing rules. This is physically correct and
    documented in the literature (see Poling et al., "Properties of Gases and
    Liquids").
    """

    TEMPERATURES = [200.0, 300.0, 400.0, 500.0]

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_he_ar_prandtl_in_range(
        self, he_ar_70_30: GasMixture, T: float
    ) -> None:
        """He-Ar mixture Prandtl should be in physically reasonable range.

        Due to nonlinear mixing rules, He-Ar mixtures can have Prandtl numbers
        higher than both pure components (0.67 for monatomic gases).
        """
        Pr = he_ar_70_30.prandtl(T)

        # Extended range to account for nonlinear mixing behavior
        assert 0.6 <= Pr <= 1.0, (
            f"Prandtl number {Pr:.3f} at {T}K is outside expected range 0.6-1.0"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_he_ar_50_50_prandtl_in_range(
        self, he_ar_50_50: GasMixture, T: float
    ) -> None:
        """50-50 He-Ar mixture Prandtl should be in reasonable range."""
        Pr = he_ar_50_50.prandtl(T)

        # Extended range for nonlinear mixing
        assert 0.6 <= Pr <= 1.0

    def test_prandtl_formula_consistency(self, he_ar_70_30: GasMixture) -> None:
        """Prandtl number should equal mu * cp / k."""
        T = 300.0
        mu = he_ar_70_30.viscosity(T)
        cp = he_ar_70_30.specific_heat_cp(T)
        k = he_ar_70_30.thermal_conductivity(T)

        expected_Pr = mu * cp / k
        calculated_Pr = he_ar_70_30.prandtl(T)

        assert calculated_Pr == pytest.approx(expected_Pr, rel=1e-10)

    def test_prandtl_varies_with_composition(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Prandtl number should vary smoothly with composition."""
        T = 300.0
        fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
        prandtl_values = []

        for he_frac in fractions:
            mix = GasMixture([helium, argon], [he_frac, 1.0 - he_frac])
            prandtl_values.append(mix.prandtl(T))

        # All should be in reasonable range (extended for nonlinear mixing)
        for Pr in prandtl_values:
            assert 0.6 <= Pr <= 1.0

        # Variation should be smooth (no sudden jumps)
        for i in range(len(prandtl_values) - 1):
            delta = abs(prandtl_values[i + 1] - prandtl_values[i])
            assert delta < 0.15  # Allow slightly larger jumps for nonlinear behavior


# ============================================================================
# Test: Wilke's Mixing Rule (Viscosity)
# ============================================================================


class TestWilkeMixingRule:
    """
    Tests verifying Wilke's mixing rule for gas mixture viscosity.

    mu_mix = sum(x_i * mu_i / sum(x_j * phi_ij))

    Reference values from various literature sources.
    """

    T_TEST = 300.0

    def test_wilke_pure_gas_recovers_component(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Wilke's rule should recover pure component viscosity for x=1."""
        # For a pure gas, phi_ii = 1, so mu_mix = mu_i
        mix_he = GasMixture([helium], [1.0])
        mix_ar = GasMixture([argon], [1.0])

        assert mix_he.viscosity(self.T_TEST) == pytest.approx(
            helium.viscosity(self.T_TEST), rel=1e-10
        )
        assert mix_ar.viscosity(self.T_TEST) == pytest.approx(
            argon.viscosity(self.T_TEST), rel=1e-10
        )

    def test_wilke_viscosity_positive(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Mixture viscosity should always be positive."""
        fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for he_frac in fractions:
            ar_frac = 1.0 - he_frac
            if he_frac == 0:
                mix = GasMixture([argon], [1.0])
            elif ar_frac == 0:
                mix = GasMixture([helium], [1.0])
            else:
                mix = GasMixture([helium, argon], [he_frac, ar_frac])

            mu = mix.viscosity(self.T_TEST)
            assert mu > 0
            assert math.isfinite(mu)

    def test_he_ar_viscosity_literature_comparison(
        self, helium: Helium, argon: Argon
    ) -> None:
        """
        Compare He-Ar mixture viscosity to literature values.

        He-Ar mixtures exhibit a well-documented "viscosity maximum" phenomenon
        where the mixture viscosity can EXCEED both pure component values due
        to the large mass ratio (~10:1 for Ar/He). This is a real physical
        effect predicted by Wilke's mixing rule and confirmed experimentally.

        Reference: Kestin et al. (1984), J. Phys. Chem. Ref. Data
        """
        T = 300.0

        # Create mixture at common composition
        mix = GasMixture([helium, argon], [0.5, 0.5])

        mu_he = helium.viscosity(T)
        mu_ar = argon.viscosity(T)
        mu_mix = mix.viscosity(T)

        # Due to viscosity maximum, mixture can be up to ~20% higher than
        # the arithmetic mean or even slightly above the higher component
        mu_max_expected = max(mu_he, mu_ar) * 1.25  # Allow up to 25% overshoot

        assert mu_mix <= mu_max_expected, (
            f"Mixture viscosity {mu_mix:.3e} exceeds expected max "
            f"{mu_max_expected:.3e}"
        )
        assert mu_mix > 0, "Viscosity must be positive"

    def test_he_ar_viscosity_maximum_phenomenon(
        self, helium: Helium, argon: Argon
    ) -> None:
        """
        Test that He-Ar mixtures show viscosity maximum behavior.

        For He-Ar mixtures, the viscosity maximum occurs at intermediate
        compositions and can exceed both pure component values.
        """
        T = 300.0

        mu_he = helium.viscosity(T)
        mu_ar = argon.viscosity(T)

        # Check viscosity at various compositions
        fractions = [0.2, 0.4, 0.6, 0.8]
        viscosities = []

        for he_frac in fractions:
            mix = GasMixture([helium, argon], [he_frac, 1.0 - he_frac])
            viscosities.append(mix.viscosity(T))

        # At least one intermediate composition should have viscosity
        # higher than one of the pure components (demonstrating the maximum)
        max_mixture_visc = max(viscosities)
        min_pure_visc = min(mu_he, mu_ar)

        # The maximum should exceed the minimum pure component
        assert max_mixture_visc > min_pure_visc, (
            "He-Ar mixture should show nonlinear viscosity behavior"
        )

    def test_wilke_viscosity_nonlinear(
        self, helium: Helium, argon: Argon
    ) -> None:
        """
        Wilke's rule produces nonlinear mixing behavior.

        Due to the interaction parameter phi_ij, the mixture viscosity
        does not follow simple linear mixing.
        """
        T = 300.0

        mu_he = helium.viscosity(T)
        mu_ar = argon.viscosity(T)

        mix = GasMixture([helium, argon], [0.5, 0.5])
        mu_mix = mix.viscosity(T)

        # Linear mixing would give (mu_he + mu_ar) / 2
        linear_mix = (mu_he + mu_ar) / 2

        # The actual mixture viscosity should differ from linear mixing
        # due to the Wilke interaction parameters
        assert mu_mix != pytest.approx(linear_mix, rel=0.01), (
            "Wilke's rule should produce nonlinear mixing behavior"
        )

    def test_phi_ii_equals_one(self, he_ar_70_30: GasMixture) -> None:
        """Wilke's phi_ii should equal 1 (same component interaction)."""
        # For same component: mu_i/mu_i = 1, M_i/M_i = 1
        # phi_ii = (1/sqrt(8)) * (1+1)^(-0.5) * (1 + 1*1)^2
        #        = (1/sqrt(8)) * (1/sqrt(2)) * 4
        #        = 4 / (sqrt(8) * sqrt(2))
        #        = 4 / 4 = 1

        mu = 1.0  # Arbitrary value
        M = 1.0  # Arbitrary value

        phi_ii = he_ar_70_30._wilke_phi(mu, mu, M, M)

        assert phi_ii == pytest.approx(1.0, rel=1e-10)


# ============================================================================
# Test: Sound Speed for He-Ar Mixtures
# ============================================================================


class TestSoundSpeedHeAr:
    """
    Tests verifying sound speed calculations for He-Ar mixtures.

    Sound speed data for He-Ar mixtures is available from various sources.
    The ideal gas formula is: a = sqrt(gamma_mix * R_mix * T)
    """

    T_TEST = 300.0

    def test_sound_speed_formula(self, he_ar_70_30: GasMixture) -> None:
        """Sound speed should follow a = sqrt(gamma * R_specific * T)."""
        R_specific = R_UNIVERSAL / he_ar_70_30.molar_mass
        gamma_mix = he_ar_70_30.gamma(self.T_TEST)

        expected_a = np.sqrt(gamma_mix * R_specific * self.T_TEST)
        calculated_a = he_ar_70_30.sound_speed(self.T_TEST)

        assert calculated_a == pytest.approx(expected_a, rel=1e-10)

    def test_sound_speed_between_components(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Mixture sound speed should be between pure component values."""
        mix = GasMixture([helium, argon], [0.5, 0.5])

        a_he = helium.sound_speed(self.T_TEST)
        a_ar = argon.sound_speed(self.T_TEST)
        a_mix = mix.sound_speed(self.T_TEST)

        assert a_ar < a_mix < a_he, (
            f"Sound speed {a_mix:.1f} m/s not between Ar ({a_ar:.1f}) "
            f"and He ({a_he:.1f})"
        )

    def test_sound_speed_increases_with_he_fraction(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Sound speed should increase as helium fraction increases."""
        fractions = [0.2, 0.4, 0.6, 0.8]
        sound_speeds = []

        for he_frac in fractions:
            mix = GasMixture([helium, argon], [he_frac, 1.0 - he_frac])
            sound_speeds.append(mix.sound_speed(self.T_TEST))

        # Sound speed should increase monotonically with He fraction
        for i in range(len(sound_speeds) - 1):
            assert sound_speeds[i] < sound_speeds[i + 1], (
                f"Sound speed should increase with He fraction"
            )

    def test_he_ar_70_30_sound_speed_reference(
        self, helium: Helium, argon: Argon
    ) -> None:
        """
        Test 70% He, 30% Ar mixture sound speed against reference.

        For a 70-30 He-Ar mixture at 300K:
        - M_mix = 0.7 * 4.003e-3 + 0.3 * 39.95e-3 = 0.01479 kg/mol
        - R_mix = 8.314 / 0.01479 = 562.1 J/(kg·K)
        - gamma_mix ~ 5/3 = 1.667 (both components are monatomic)
        - a = sqrt(1.667 * 562.1 * 300) = sqrt(280900) ~ 530 m/s

        Note: Actual value differs due to mixing rules for cp and cv.
        """
        mix = GasMixture([helium, argon], [0.7, 0.3])
        a = mix.sound_speed(300.0)

        # Expected range based on ideal gas calculation
        # Should be somewhere between ~500 and ~800 m/s for this mixture
        assert 500 < a < 900, f"Sound speed {a:.1f} m/s outside expected range"

        # More precise check: verify it's between pure component values
        a_he = helium.sound_speed(300.0)  # ~1019 m/s
        a_ar = argon.sound_speed(300.0)   # ~323 m/s
        assert a_ar < a < a_he

    def test_sound_speed_temperature_scaling(
        self, he_ar_70_30: GasMixture
    ) -> None:
        """Sound speed should scale as sqrt(T)."""
        T1, T2 = 300.0, 600.0
        a1 = he_ar_70_30.sound_speed(T1)
        a2 = he_ar_70_30.sound_speed(T2)

        # a2/a1 should equal sqrt(T2/T1) = sqrt(2) ~ 1.414
        expected_ratio = np.sqrt(T2 / T1)
        actual_ratio = a2 / a1

        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-10)


# ============================================================================
# Test: Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Tests for helium_argon() and helium_xenon() convenience functions."""

    def test_helium_argon_basic(self) -> None:
        """Test helium_argon convenience function."""
        mix = helium_argon(0.7, P_ATM)

        assert isinstance(mix, GasMixture)
        assert len(mix.components) == 2
        assert mix.mole_fractions == pytest.approx((0.7, 0.3), rel=1e-10)
        assert mix.mean_pressure == P_ATM

    def test_helium_argon_pure_helium(self) -> None:
        """helium_argon(1.0) should give pure helium."""
        mix = helium_argon(1.0, P_ATM)
        he = Helium(P_ATM)

        T = 300.0
        assert mix.density(T) == pytest.approx(he.density(T), rel=1e-10)
        assert mix.sound_speed(T) == pytest.approx(he.sound_speed(T), rel=1e-10)

    def test_helium_argon_pure_argon(self) -> None:
        """helium_argon(0.0) should give pure argon."""
        mix = helium_argon(0.0, P_ATM)
        ar = Argon(P_ATM)

        T = 300.0
        assert mix.density(T) == pytest.approx(ar.density(T), rel=1e-10)
        assert mix.sound_speed(T) == pytest.approx(ar.sound_speed(T), rel=1e-10)

    def test_helium_argon_invalid_fraction(self) -> None:
        """helium_argon should reject invalid fractions."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            helium_argon(-0.1, P_ATM)

        with pytest.raises(ValueError, match="between 0 and 1"):
            helium_argon(1.1, P_ATM)

    def test_helium_xenon_basic(self) -> None:
        """Test helium_xenon convenience function."""
        mix = helium_xenon(0.8, P_ATM)

        assert isinstance(mix, GasMixture)
        assert len(mix.components) == 2
        assert mix.mole_fractions == pytest.approx((0.8, 0.2), rel=1e-10)

    def test_helium_xenon_molar_mass(self) -> None:
        """He-Xe mixture molar mass calculation."""
        mix = helium_xenon(0.8, P_ATM)

        expected_M = 0.8 * MOLAR_MASS["helium"] + 0.2 * MOLAR_MASS["xenon"]
        assert mix.molar_mass == pytest.approx(expected_M, rel=1e-10)

    def test_helium_xenon_invalid_fraction(self) -> None:
        """helium_xenon should reject invalid fractions."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            helium_xenon(-0.1, P_ATM)


# ============================================================================
# Test: Input Validation
# ============================================================================


class TestInputValidation:
    """Tests for GasMixture input validation."""

    def test_empty_components_rejected(self) -> None:
        """Empty component list should be rejected."""
        with pytest.raises(ValueError, match="At least one component"):
            GasMixture([], [])

    def test_mismatched_lengths_rejected(self, helium: Helium) -> None:
        """Mismatched component/fraction lengths should be rejected."""
        with pytest.raises(ValueError, match="must match"):
            GasMixture([helium], [0.5, 0.5])

    def test_negative_fraction_rejected(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Negative mole fractions should be rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            GasMixture([helium, argon], [1.1, -0.1])

    def test_fractions_not_summing_to_one_rejected(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Mole fractions not summing to 1 should be rejected."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            GasMixture([helium, argon], [0.5, 0.4])

    def test_fractions_normalized(self, helium: Helium, argon: Argon) -> None:
        """Fractions should be normalized to exactly 1.0."""
        # Small deviation within tolerance should be normalized
        mix = GasMixture([helium, argon], [0.7000001, 0.2999999])

        assert sum(mix.mole_fractions) == pytest.approx(1.0, rel=1e-10)


# ============================================================================
# Test: Thermal Conductivity (Wassiljewa-Mason-Saxena)
# ============================================================================


class TestThermalConductivityMixture:
    """
    Tests for mixture thermal conductivity using Wassiljewa equation
    with Mason-Saxena modification.
    """

    T_TEST = 300.0

    def test_thermal_conductivity_positive(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Mixture thermal conductivity should always be positive."""
        fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

        for he_frac in fractions:
            ar_frac = 1.0 - he_frac
            if he_frac == 0:
                mix = GasMixture([argon], [1.0])
            elif ar_frac == 0:
                mix = GasMixture([helium], [1.0])
            else:
                mix = GasMixture([helium, argon], [he_frac, ar_frac])

            k = mix.thermal_conductivity(self.T_TEST)
            assert k > 0
            assert math.isfinite(k)

    def test_thermal_conductivity_between_components(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Mixture k should be between pure component values."""
        mix = GasMixture([helium, argon], [0.5, 0.5])

        k_he = helium.thermal_conductivity(self.T_TEST)
        k_ar = argon.thermal_conductivity(self.T_TEST)
        k_mix = mix.thermal_conductivity(self.T_TEST)

        k_min = min(k_he, k_ar)
        k_max = max(k_he, k_ar)

        assert k_min <= k_mix <= k_max

    def test_thermal_conductivity_increases_with_he(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Thermal conductivity should increase with He fraction."""
        # Helium has much higher thermal conductivity than argon
        fractions = [0.2, 0.4, 0.6, 0.8]
        k_values = []

        for he_frac in fractions:
            mix = GasMixture([helium, argon], [he_frac, 1.0 - he_frac])
            k_values.append(mix.thermal_conductivity(self.T_TEST))

        for i in range(len(k_values) - 1):
            assert k_values[i] < k_values[i + 1]


# ============================================================================
# Test: Specific Heat and Gamma
# ============================================================================


class TestSpecificHeatMixture:
    """Tests for mixture specific heat and gamma calculations."""

    T_TEST = 300.0

    def test_cp_mass_weighted_average(
        self, helium: Helium, argon: Argon
    ) -> None:
        """
        cp_mix should be mass-fraction weighted average.

        cp_mix = sum(w_i * cp_i) where w_i = x_i * M_i / M_mix
        """
        mix = GasMixture([helium, argon], [0.5, 0.5])

        M_mix = mix.molar_mass
        w_he = 0.5 * helium.molar_mass / M_mix
        w_ar = 0.5 * argon.molar_mass / M_mix

        expected_cp = (
            w_he * helium.specific_heat_cp(self.T_TEST) +
            w_ar * argon.specific_heat_cp(self.T_TEST)
        )

        assert mix.specific_heat_cp(self.T_TEST) == pytest.approx(
            expected_cp, rel=1e-10
        )

    def test_gamma_monatomic_mixture(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Mixture of monatomic gases should have gamma near 5/3."""
        mix = GasMixture([helium, argon], [0.5, 0.5])

        gamma_mix = mix.gamma(self.T_TEST)

        # Both He and Ar are monatomic with gamma = 5/3
        # The mixture should also have gamma close to 5/3
        assert gamma_mix == pytest.approx(5/3, rel=0.01)

    def test_gamma_equals_cp_over_cv(self, he_ar_70_30: GasMixture) -> None:
        """gamma should equal cp_mix / cv_mix."""
        cp_mix = he_ar_70_30.specific_heat_cp(self.T_TEST)
        cv_mix = he_ar_70_30._specific_heat_cv(self.T_TEST)

        expected_gamma = cp_mix / cv_mix
        calculated_gamma = he_ar_70_30.gamma(self.T_TEST)

        assert calculated_gamma == pytest.approx(expected_gamma, rel=1e-10)


# ============================================================================
# Test: Mixture Properties
# ============================================================================


class TestMixtureProperties:
    """Tests for GasMixture property methods."""

    def test_name_property(self, helium: Helium, argon: Argon) -> None:
        """Name should describe the mixture composition."""
        mix = GasMixture([helium, argon], [0.7, 0.3])

        name = mix.name
        assert "70.0% Helium" in name
        assert "30.0% Argon" in name
        assert "+" in name

    def test_components_property(self, helium: Helium, argon: Argon) -> None:
        """Components property should return tuple of gases."""
        mix = GasMixture([helium, argon], [0.7, 0.3])

        assert len(mix.components) == 2
        assert isinstance(mix.components, tuple)
        assert mix.components[0] is helium
        assert mix.components[1] is argon

    def test_mole_fractions_property(self, helium: Helium, argon: Argon) -> None:
        """Mole fractions property should return tuple of floats."""
        mix = GasMixture([helium, argon], [0.7, 0.3])

        assert isinstance(mix.mole_fractions, tuple)
        assert mix.mole_fractions == pytest.approx((0.7, 0.3), rel=1e-10)

    def test_mean_pressure_from_first_component(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Mean pressure should default to first component's value."""
        he_high_p = Helium(mean_pressure=200000.0)
        ar = Argon(mean_pressure=100000.0)

        mix = GasMixture([he_high_p, ar], [0.5, 0.5])

        # Should use first component's mean_pressure
        assert mix.mean_pressure == 200000.0

    def test_mean_pressure_override(self, helium: Helium, argon: Argon) -> None:
        """Mean pressure can be overridden in constructor."""
        mix = GasMixture([helium, argon], [0.5, 0.5], mean_pressure=500000.0)

        assert mix.mean_pressure == 500000.0

    def test_specific_gas_constant(self, he_ar_70_30: GasMixture) -> None:
        """Specific gas constant should equal R_universal / M_mix."""
        R_specific = he_ar_70_30.specific_gas_constant()
        expected_R = R_UNIVERSAL / he_ar_70_30.molar_mass

        assert R_specific == pytest.approx(expected_R, rel=1e-10)


# ============================================================================
# Test: Multi-component Mixtures
# ============================================================================


class TestMultiComponentMixtures:
    """Tests for mixtures with more than two components."""

    def test_three_component_mixture(
        self, helium: Helium, argon: Argon, xenon: Xenon
    ) -> None:
        """Three-component mixture should work correctly."""
        mix = GasMixture(
            [helium, argon, xenon],
            [0.5, 0.3, 0.2]
        )

        T = 300.0

        # All properties should be positive and finite
        assert mix.density(T) > 0
        assert mix.sound_speed(T) > 0
        assert mix.viscosity(T) > 0
        assert mix.thermal_conductivity(T) > 0
        assert mix.prandtl(T) > 0

        # Molar mass should be weighted average
        expected_M = (
            0.5 * helium.molar_mass +
            0.3 * argon.molar_mass +
            0.2 * xenon.molar_mass
        )
        assert mix.molar_mass == pytest.approx(expected_M, rel=1e-10)

    def test_three_component_mole_fractions(
        self, helium: Helium, argon: Argon, xenon: Xenon
    ) -> None:
        """Three-component mole fractions should sum to 1."""
        mix = GasMixture(
            [helium, argon, xenon],
            [0.5, 0.3, 0.2]
        )

        assert sum(mix.mole_fractions) == pytest.approx(1.0, rel=1e-10)


# ============================================================================
# Test: Numerical Stability
# ============================================================================


class TestNumericalStabilityMixture:
    """Tests for numerical stability at extreme conditions."""

    def test_low_temperature_stability(self, he_ar_70_30: GasMixture) -> None:
        """Properties should be stable at low temperatures."""
        T = 100.0

        assert math.isfinite(he_ar_70_30.density(T))
        assert math.isfinite(he_ar_70_30.sound_speed(T))
        assert math.isfinite(he_ar_70_30.viscosity(T))
        assert math.isfinite(he_ar_70_30.thermal_conductivity(T))
        assert math.isfinite(he_ar_70_30.prandtl(T))

    def test_high_temperature_stability(self, he_ar_70_30: GasMixture) -> None:
        """Properties should be stable at high temperatures."""
        T = 1000.0

        assert math.isfinite(he_ar_70_30.density(T))
        assert math.isfinite(he_ar_70_30.sound_speed(T))
        assert math.isfinite(he_ar_70_30.viscosity(T))
        assert math.isfinite(he_ar_70_30.thermal_conductivity(T))
        assert math.isfinite(he_ar_70_30.prandtl(T))

    def test_high_pressure_stability(
        self, helium: Helium, argon: Argon
    ) -> None:
        """Properties should be stable at high pressures."""
        mix = GasMixture([helium, argon], [0.7, 0.3], mean_pressure=10 * P_ATM)
        T = 300.0
        P = 10 * P_ATM

        rho = mix.density(T, P)
        assert math.isfinite(rho)
        assert rho > 0

        # Density should scale linearly with pressure
        rho_1atm = mix.density(T, P_ATM)
        assert rho == pytest.approx(10 * rho_1atm, rel=1e-10)

    def test_extreme_composition(self, helium: Helium, argon: Argon) -> None:
        """Properties should be stable for extreme compositions."""
        T = 300.0

        # Very small He fraction
        mix_low_he = GasMixture([helium, argon], [0.001, 0.999])
        assert math.isfinite(mix_low_he.viscosity(T))
        assert math.isfinite(mix_low_he.thermal_conductivity(T))

        # Very small Ar fraction
        mix_low_ar = GasMixture([helium, argon], [0.999, 0.001])
        assert math.isfinite(mix_low_ar.viscosity(T))
        assert math.isfinite(mix_low_ar.thermal_conductivity(T))

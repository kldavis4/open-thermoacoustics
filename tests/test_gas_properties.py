"""Comprehensive tests for gas properties module.

This module tests the thermophysical properties of gases (Helium, Argon, Nitrogen, Air)
used in thermoacoustic calculations. Tests verify:
1. Ideal gas law compliance
2. Sound speed calculations
3. Temperature dependence of transport properties
4. Prandtl number consistency
5. Specific heat relationships
6. Comparison against NIST reference values

All tests use SI units consistently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openthermoacoustics.gas import Air, Argon, Helium, Nitrogen
from openthermoacoustics.utils import GAMMA, MOLAR_MASS, R_UNIVERSAL


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
def nitrogen() -> Nitrogen:
    """Create a Nitrogen gas instance at standard atmospheric pressure."""
    return Nitrogen(mean_pressure=P_ATM)


@pytest.fixture
def air() -> Air:
    """Create an Air gas instance at standard atmospheric pressure."""
    return Air(mean_pressure=P_ATM)


# ============================================================================
# Test: Ideal Gas Law (rho = P / (R_specific * T))
# ============================================================================


class TestIdealGasLaw:
    """
    Tests verifying that density follows the ideal gas law: rho = P / (R_specific * T).

    The ideal gas law is fundamental to thermoacoustic calculations. These tests
    verify that each gas class correctly implements density calculations for
    various temperature and pressure combinations.
    """

    # Test cases: (temperature K, pressure Pa)
    TP_COMBINATIONS = [
        (200.0, P_ATM),        # Low temperature, atmospheric pressure
        (300.0, P_ATM),        # Room temperature, atmospheric pressure
        (400.0, P_ATM),        # Elevated temperature, atmospheric pressure
        (500.0, P_ATM),        # High temperature, atmospheric pressure
        (300.0, 50000.0),      # Room temperature, low pressure
        (300.0, 200000.0),     # Room temperature, high pressure
        (250.0, 150000.0),     # Mixed conditions
        (350.0, 75000.0),      # Mixed conditions
    ]

    @pytest.mark.parametrize("T,P", TP_COMBINATIONS)
    def test_helium_ideal_gas_law(self, helium: Helium, T: float, P: float) -> None:
        """
        Verify helium density follows rho = P / (R_specific * T).

        Helium behaves as an ideal gas over a wide range of conditions due to
        its small atomic size and weak intermolecular forces.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["helium"]
        expected_rho = P / (R_specific * T)
        calculated_rho = helium.density(T, P)

        assert calculated_rho == pytest.approx(expected_rho, rel=1e-10), (
            f"Helium density at T={T}K, P={P}Pa: "
            f"expected {expected_rho:.6f}, got {calculated_rho:.6f}"
        )

    @pytest.mark.parametrize("T,P", TP_COMBINATIONS)
    def test_argon_ideal_gas_law(self, argon: Argon, T: float, P: float) -> None:
        """
        Verify argon density follows rho = P / (R_specific * T).

        Argon is a noble gas that closely follows ideal gas behavior at
        standard conditions.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["argon"]
        expected_rho = P / (R_specific * T)
        calculated_rho = argon.density(T, P)

        assert calculated_rho == pytest.approx(expected_rho, rel=1e-10), (
            f"Argon density at T={T}K, P={P}Pa: "
            f"expected {expected_rho:.6f}, got {calculated_rho:.6f}"
        )

    @pytest.mark.parametrize("T,P", TP_COMBINATIONS)
    def test_nitrogen_ideal_gas_law(self, nitrogen: Nitrogen, T: float, P: float) -> None:
        """
        Verify nitrogen density follows rho = P / (R_specific * T).

        Nitrogen exhibits ideal gas behavior at moderate temperatures and
        pressures typical of thermoacoustic applications.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["nitrogen"]
        expected_rho = P / (R_specific * T)
        calculated_rho = nitrogen.density(T, P)

        assert calculated_rho == pytest.approx(expected_rho, rel=1e-10), (
            f"Nitrogen density at T={T}K, P={P}Pa: "
            f"expected {expected_rho:.6f}, got {calculated_rho:.6f}"
        )

    @pytest.mark.parametrize("T,P", TP_COMBINATIONS)
    def test_air_ideal_gas_law(self, air: Air, T: float, P: float) -> None:
        """
        Verify air density follows rho = P / (R_specific * T).

        Air, being primarily nitrogen and oxygen, follows ideal gas behavior
        at standard conditions with an effective molar mass.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["air"]
        expected_rho = P / (R_specific * T)
        calculated_rho = air.density(T, P)

        assert calculated_rho == pytest.approx(expected_rho, rel=1e-10), (
            f"Air density at T={T}K, P={P}Pa: "
            f"expected {expected_rho:.6f}, got {calculated_rho:.6f}"
        )

    def test_density_uses_mean_pressure_when_none(self, helium: Helium) -> None:
        """Verify that density uses mean_pressure when P is not specified."""
        T = 300.0
        rho_explicit = helium.density(T, P_ATM)
        rho_implicit = helium.density(T)  # Should use mean_pressure

        assert rho_explicit == pytest.approx(rho_implicit, rel=1e-10), (
            "Density with P=None should use mean_pressure"
        )


# ============================================================================
# Test: Sound Speed (a = sqrt(gamma * R_specific * T))
# ============================================================================


class TestSoundSpeed:
    """
    Tests verifying that sound speed follows a = sqrt(gamma * R_specific * T).

    For an ideal gas, the speed of sound depends only on temperature and the
    gas's specific heat ratio (gamma) and molar mass. It is independent of
    pressure.
    """

    TEMPERATURES = [200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 600.0]

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_helium_sound_speed(self, helium: Helium, T: float) -> None:
        """
        Verify helium sound speed follows a = sqrt(gamma * R_specific * T).

        Helium has a high sound speed due to its low molar mass, making it
        useful for high-frequency thermoacoustic devices.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["helium"]
        gamma = GAMMA["helium"]
        expected_a = math.sqrt(gamma * R_specific * T)
        calculated_a = helium.sound_speed(T)

        assert calculated_a == pytest.approx(expected_a, rel=1e-10), (
            f"Helium sound speed at T={T}K: "
            f"expected {expected_a:.2f} m/s, got {calculated_a:.2f} m/s"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_argon_sound_speed(self, argon: Argon, T: float) -> None:
        """
        Verify argon sound speed follows a = sqrt(gamma * R_specific * T).

        Argon has a lower sound speed than helium due to its higher molar mass.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["argon"]
        gamma = GAMMA["argon"]
        expected_a = math.sqrt(gamma * R_specific * T)
        calculated_a = argon.sound_speed(T)

        assert calculated_a == pytest.approx(expected_a, rel=1e-10), (
            f"Argon sound speed at T={T}K: "
            f"expected {expected_a:.2f} m/s, got {calculated_a:.2f} m/s"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_nitrogen_sound_speed(self, nitrogen: Nitrogen, T: float) -> None:
        """
        Verify nitrogen sound speed follows a = sqrt(gamma * R_specific * T).

        Nitrogen, as a diatomic gas, has gamma = 1.4.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["nitrogen"]
        gamma = GAMMA["nitrogen"]
        expected_a = math.sqrt(gamma * R_specific * T)
        calculated_a = nitrogen.sound_speed(T)

        assert calculated_a == pytest.approx(expected_a, rel=1e-10), (
            f"Nitrogen sound speed at T={T}K: "
            f"expected {expected_a:.2f} m/s, got {calculated_a:.2f} m/s"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_air_sound_speed(self, air: Air, T: float) -> None:
        """
        Verify air sound speed follows a = sqrt(gamma * R_specific * T).

        Air is treated as a diatomic gas with gamma = 1.4 and an effective
        molar mass.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["air"]
        gamma = GAMMA["air"]
        expected_a = math.sqrt(gamma * R_specific * T)
        calculated_a = air.sound_speed(T)

        assert calculated_a == pytest.approx(expected_a, rel=1e-10), (
            f"Air sound speed at T={T}K: "
            f"expected {expected_a:.2f} m/s, got {calculated_a:.2f} m/s"
        )

    def test_sound_speed_independent_of_pressure(self, helium: Helium) -> None:
        """
        Verify that sound speed is independent of pressure for ideal gases.

        This is a fundamental property of ideal gases where the sound speed
        depends only on temperature through: a = sqrt(gamma * R_specific * T).
        """
        T = 300.0
        pressures = [50000.0, 101325.0, 200000.0, 500000.0]

        reference_a = helium.sound_speed(T, pressures[0])
        for P in pressures[1:]:
            calculated_a = helium.sound_speed(T, P)
            assert calculated_a == pytest.approx(reference_a, rel=1e-10), (
                f"Sound speed should be independent of pressure: "
                f"at P={P}Pa got {calculated_a:.2f} m/s, expected {reference_a:.2f} m/s"
            )


# ============================================================================
# Test: Temperature Dependence of Transport Properties
# ============================================================================


class TestTemperatureDependence:
    """
    Tests verifying that viscosity and thermal conductivity increase with temperature.

    For gases, both viscosity and thermal conductivity increase with temperature
    due to increased molecular motion and momentum/energy transfer. This is
    opposite to the behavior of liquids.
    """

    TEMPERATURE_PAIRS = [
        (200.0, 300.0),
        (300.0, 400.0),
        (400.0, 500.0),
        (250.0, 350.0),
        (350.0, 450.0),
    ]

    @pytest.mark.parametrize("T_low,T_high", TEMPERATURE_PAIRS)
    def test_helium_viscosity_increases_with_temperature(
        self, helium: Helium, T_low: float, T_high: float
    ) -> None:
        """
        Verify helium viscosity increases with temperature.

        Uses power-law correlation: mu(T) = mu_ref * (T/T_ref)^n where n > 0.
        """
        mu_low = helium.viscosity(T_low)
        mu_high = helium.viscosity(T_high)

        assert mu_high > mu_low, (
            f"Helium viscosity should increase with temperature: "
            f"mu({T_low}K)={mu_low:.3e} Pa.s, mu({T_high}K)={mu_high:.3e} Pa.s"
        )

    @pytest.mark.parametrize("T_low,T_high", TEMPERATURE_PAIRS)
    def test_helium_thermal_conductivity_increases_with_temperature(
        self, helium: Helium, T_low: float, T_high: float
    ) -> None:
        """
        Verify helium thermal conductivity increases with temperature.

        Uses power-law correlation: kappa(T) = kappa_ref * (T/T_ref)^n where n > 0.
        """
        kappa_low = helium.thermal_conductivity(T_low)
        kappa_high = helium.thermal_conductivity(T_high)

        assert kappa_high > kappa_low, (
            f"Helium thermal conductivity should increase with temperature: "
            f"kappa({T_low}K)={kappa_low:.4f} W/(m.K), kappa({T_high}K)={kappa_high:.4f} W/(m.K)"
        )

    @pytest.mark.parametrize("T_low,T_high", TEMPERATURE_PAIRS)
    def test_argon_viscosity_increases_with_temperature(
        self, argon: Argon, T_low: float, T_high: float
    ) -> None:
        """Verify argon viscosity increases with temperature."""
        mu_low = argon.viscosity(T_low)
        mu_high = argon.viscosity(T_high)

        assert mu_high > mu_low, (
            f"Argon viscosity should increase with temperature: "
            f"mu({T_low}K)={mu_low:.3e} Pa.s, mu({T_high}K)={mu_high:.3e} Pa.s"
        )

    @pytest.mark.parametrize("T_low,T_high", TEMPERATURE_PAIRS)
    def test_argon_thermal_conductivity_increases_with_temperature(
        self, argon: Argon, T_low: float, T_high: float
    ) -> None:
        """Verify argon thermal conductivity increases with temperature."""
        kappa_low = argon.thermal_conductivity(T_low)
        kappa_high = argon.thermal_conductivity(T_high)

        assert kappa_high > kappa_low, (
            f"Argon thermal conductivity should increase with temperature: "
            f"kappa({T_low}K)={kappa_low:.4f} W/(m.K), kappa({T_high}K)={kappa_high:.4f} W/(m.K)"
        )

    @pytest.mark.parametrize("T_low,T_high", TEMPERATURE_PAIRS)
    def test_nitrogen_viscosity_increases_with_temperature(
        self, nitrogen: Nitrogen, T_low: float, T_high: float
    ) -> None:
        """Verify nitrogen viscosity increases with temperature."""
        mu_low = nitrogen.viscosity(T_low)
        mu_high = nitrogen.viscosity(T_high)

        assert mu_high > mu_low, (
            f"Nitrogen viscosity should increase with temperature: "
            f"mu({T_low}K)={mu_low:.3e} Pa.s, mu({T_high}K)={mu_high:.3e} Pa.s"
        )

    @pytest.mark.parametrize("T_low,T_high", TEMPERATURE_PAIRS)
    def test_nitrogen_thermal_conductivity_increases_with_temperature(
        self, nitrogen: Nitrogen, T_low: float, T_high: float
    ) -> None:
        """Verify nitrogen thermal conductivity increases with temperature."""
        kappa_low = nitrogen.thermal_conductivity(T_low)
        kappa_high = nitrogen.thermal_conductivity(T_high)

        assert kappa_high > kappa_low, (
            f"Nitrogen thermal conductivity should increase with temperature: "
            f"kappa({T_low}K)={kappa_low:.4f} W/(m.K), kappa({T_high}K)={kappa_high:.4f} W/(m.K)"
        )

    @pytest.mark.parametrize("T_low,T_high", TEMPERATURE_PAIRS)
    def test_air_viscosity_increases_with_temperature(
        self, air: Air, T_low: float, T_high: float
    ) -> None:
        """
        Verify air viscosity increases with temperature.

        Air uses Sutherland's law, which provides accurate viscosity values
        over a wide temperature range.
        """
        mu_low = air.viscosity(T_low)
        mu_high = air.viscosity(T_high)

        assert mu_high > mu_low, (
            f"Air viscosity should increase with temperature: "
            f"mu({T_low}K)={mu_low:.3e} Pa.s, mu({T_high}K)={mu_high:.3e} Pa.s"
        )

    @pytest.mark.parametrize("T_low,T_high", TEMPERATURE_PAIRS)
    def test_air_thermal_conductivity_increases_with_temperature(
        self, air: Air, T_low: float, T_high: float
    ) -> None:
        """Verify air thermal conductivity increases with temperature."""
        kappa_low = air.thermal_conductivity(T_low)
        kappa_high = air.thermal_conductivity(T_high)

        assert kappa_high > kappa_low, (
            f"Air thermal conductivity should increase with temperature: "
            f"kappa({T_low}K)={kappa_low:.4f} W/(m.K), kappa({T_high}K)={kappa_high:.4f} W/(m.K)"
        )


# ============================================================================
# Test: Prandtl Number Consistency
# ============================================================================


class TestPrandtlNumber:
    """
    Tests verifying that Prandtl number is approximately constant across temperatures.

    The Prandtl number Pr = mu * cp / kappa characterizes the relative importance
    of momentum diffusion to thermal diffusion. For gases:
    - Monatomic gases (He, Ar): Pr ~ 0.67
    - Diatomic gases (N2, Air): Pr ~ 0.71

    The Prandtl number should be nearly constant across a wide temperature range
    because viscosity and thermal conductivity scale similarly with temperature.
    """

    TEMPERATURES = [200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 600.0]

    # Tolerances for Prandtl number constancy and expected values
    MONATOMIC_PR_EXPECTED = 0.67
    DIATOMIC_PR_EXPECTED = 0.71
    PR_TOLERANCE = 0.05  # 5% relative tolerance for value
    PR_CONSTANCY_TOLERANCE = 0.10  # 10% variation across temperature range

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_helium_prandtl_approximately_constant(
        self, helium: Helium, T: float
    ) -> None:
        """
        Verify helium Prandtl number is approximately 0.67 (monatomic gas).

        For monatomic ideal gases, kinetic theory predicts Pr = 2/3 ~ 0.67.
        """
        Pr = helium.prandtl(T)

        assert Pr == pytest.approx(self.MONATOMIC_PR_EXPECTED, rel=self.PR_TOLERANCE), (
            f"Helium Prandtl at T={T}K: expected ~{self.MONATOMIC_PR_EXPECTED}, got {Pr:.3f}"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_argon_prandtl_approximately_constant(
        self, argon: Argon, T: float
    ) -> None:
        """
        Verify argon Prandtl number is approximately 0.67 (monatomic gas).
        """
        Pr = argon.prandtl(T)

        assert Pr == pytest.approx(self.MONATOMIC_PR_EXPECTED, rel=self.PR_TOLERANCE), (
            f"Argon Prandtl at T={T}K: expected ~{self.MONATOMIC_PR_EXPECTED}, got {Pr:.3f}"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_nitrogen_prandtl_approximately_constant(
        self, nitrogen: Nitrogen, T: float
    ) -> None:
        """
        Verify nitrogen Prandtl number is approximately 0.71 (diatomic gas).

        Diatomic gases have additional rotational degrees of freedom that
        affect the relationship between viscosity and thermal conductivity.
        """
        Pr = nitrogen.prandtl(T)

        assert Pr == pytest.approx(self.DIATOMIC_PR_EXPECTED, rel=self.PR_TOLERANCE), (
            f"Nitrogen Prandtl at T={T}K: expected ~{self.DIATOMIC_PR_EXPECTED}, got {Pr:.3f}"
        )

    @pytest.mark.parametrize("T", [200.0, 250.0, 300.0, 350.0, 400.0])
    def test_air_prandtl_approximately_constant(self, air: Air, T: float) -> None:
        """
        Verify air Prandtl number is approximately 0.71 (primarily diatomic gas).

        Note: At higher temperatures (T > 400K), the Prandtl number deviates
        from 0.71 because Sutherland's law for viscosity and the power-law
        correlation for thermal conductivity have different temperature
        dependencies. This is a known limitation of the correlations.
        """
        Pr = air.prandtl(T)

        assert Pr == pytest.approx(self.DIATOMIC_PR_EXPECTED, rel=self.PR_TOLERANCE), (
            f"Air Prandtl at T={T}K: expected ~{self.DIATOMIC_PR_EXPECTED}, got {Pr:.3f}"
        )

    def test_helium_prandtl_constant_across_temperature_range(
        self, helium: Helium
    ) -> None:
        """
        Verify helium Prandtl number variation is small across temperature range.

        The Prandtl number should not vary by more than ~10% across a wide
        temperature range because mu and kappa scale similarly with T.
        """
        Pr_values = [helium.prandtl(T) for T in self.TEMPERATURES]
        Pr_mean = sum(Pr_values) / len(Pr_values)

        for i, (T, Pr) in enumerate(zip(self.TEMPERATURES, Pr_values)):
            relative_deviation = abs(Pr - Pr_mean) / Pr_mean
            assert relative_deviation < self.PR_CONSTANCY_TOLERANCE, (
                f"Helium Prandtl varies too much: at T={T}K, Pr={Pr:.3f}, "
                f"mean={Pr_mean:.3f}, deviation={relative_deviation*100:.1f}%"
            )

    def test_prandtl_formula_consistency(self, helium: Helium) -> None:
        """
        Verify Prandtl number is calculated as Pr = mu * cp / kappa.

        This test ensures the implementation follows the standard definition.
        """
        T = 300.0
        mu = helium.viscosity(T)
        cp = helium.specific_heat_cp(T)
        kappa = helium.thermal_conductivity(T)

        expected_Pr = mu * cp / kappa
        calculated_Pr = helium.prandtl(T)

        assert calculated_Pr == pytest.approx(expected_Pr, rel=1e-10), (
            f"Prandtl formula inconsistency: mu*cp/kappa={expected_Pr:.6f}, "
            f"prandtl()={calculated_Pr:.6f}"
        )


# ============================================================================
# Test: Specific Heat Relationship for Ideal Gases
# ============================================================================


class TestSpecificHeat:
    """
    Tests verifying that cp = gamma * R_specific / (gamma - 1) for ideal gases.

    For ideal gases, the specific heats are related to the specific gas constant
    by the equations:
    - cp - cv = R_specific
    - gamma = cp / cv

    These combine to give: cp = gamma * R_specific / (gamma - 1)

    For monatomic gases (He, Ar): cp = (5/2) * R_specific
    For diatomic gases (N2, Air): cp = (7/2) * R_specific
    """

    TEMPERATURES = [200.0, 300.0, 400.0, 500.0]

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_helium_specific_heat(self, helium: Helium, T: float) -> None:
        """
        Verify helium cp = gamma * R_specific / (gamma - 1).

        For monatomic ideal gas: cp = (5/3)/((5/3)-1) * R = (5/2) * R_specific
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["helium"]
        gamma = GAMMA["helium"]
        expected_cp = gamma * R_specific / (gamma - 1)
        calculated_cp = helium.specific_heat_cp(T)

        assert calculated_cp == pytest.approx(expected_cp, rel=1e-10), (
            f"Helium cp at T={T}K: expected {expected_cp:.2f} J/(kg.K), "
            f"got {calculated_cp:.2f} J/(kg.K)"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_argon_specific_heat(self, argon: Argon, T: float) -> None:
        """
        Verify argon cp = gamma * R_specific / (gamma - 1).

        For monatomic ideal gas: cp = (5/2) * R_specific
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["argon"]
        gamma = GAMMA["argon"]
        expected_cp = gamma * R_specific / (gamma - 1)
        calculated_cp = argon.specific_heat_cp(T)

        assert calculated_cp == pytest.approx(expected_cp, rel=1e-10), (
            f"Argon cp at T={T}K: expected {expected_cp:.2f} J/(kg.K), "
            f"got {calculated_cp:.2f} J/(kg.K)"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_nitrogen_specific_heat(self, nitrogen: Nitrogen, T: float) -> None:
        """
        Verify nitrogen cp = gamma * R_specific / (gamma - 1).

        For diatomic ideal gas: cp = 1.4/0.4 * R = (7/2) * R_specific
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["nitrogen"]
        gamma = GAMMA["nitrogen"]
        expected_cp = gamma * R_specific / (gamma - 1)
        calculated_cp = nitrogen.specific_heat_cp(T)

        assert calculated_cp == pytest.approx(expected_cp, rel=1e-10), (
            f"Nitrogen cp at T={T}K: expected {expected_cp:.2f} J/(kg.K), "
            f"got {calculated_cp:.2f} J/(kg.K)"
        )

    @pytest.mark.parametrize("T", TEMPERATURES)
    def test_air_specific_heat(self, air: Air, T: float) -> None:
        """
        Verify air cp = gamma * R_specific / (gamma - 1).

        Air is treated as a diatomic gas with gamma = 1.4.
        """
        R_specific = R_UNIVERSAL / MOLAR_MASS["air"]
        gamma = GAMMA["air"]
        expected_cp = gamma * R_specific / (gamma - 1)
        calculated_cp = air.specific_heat_cp(T)

        assert calculated_cp == pytest.approx(expected_cp, rel=1e-10), (
            f"Air cp at T={T}K: expected {expected_cp:.2f} J/(kg.K), "
            f"got {calculated_cp:.2f} J/(kg.K)"
        )

    def test_specific_heat_independent_of_temperature_for_ideal_gas(
        self, helium: Helium
    ) -> None:
        """
        Verify that cp is constant (independent of T) for ideal gases.

        For ideal gases modeled with constant gamma, the specific heat cp
        depends only on the gas constant and gamma, not on temperature.
        """
        temperatures = [200.0, 300.0, 400.0, 500.0, 600.0]
        cp_values = [helium.specific_heat_cp(T) for T in temperatures]

        cp_reference = cp_values[0]
        for T, cp in zip(temperatures, cp_values):
            assert cp == pytest.approx(cp_reference, rel=1e-10), (
                f"Helium cp should be constant: cp({T}K)={cp:.2f} != "
                f"cp({temperatures[0]}K)={cp_reference:.2f}"
            )

    def test_gamma_method_returns_correct_value(self, helium: Helium) -> None:
        """Verify that the gamma() method returns the expected ratio of specific heats."""
        T = 300.0
        expected_gamma = GAMMA["helium"]
        calculated_gamma = helium.gamma(T)

        assert calculated_gamma == pytest.approx(expected_gamma, rel=1e-10), (
            f"Helium gamma: expected {expected_gamma}, got {calculated_gamma}"
        )


# ============================================================================
# Test: Reference Values (NIST Comparison at 300K, 1 atm)
# ============================================================================


class TestNISTReferenceValues:
    """
    Tests comparing calculated properties against NIST reference values at 300K, 1 atm.

    Reference values are from NIST Chemistry WebBook and other standard sources.
    These tests verify that the gas property correlations are calibrated to
    match real-world measurements.

    Tolerances:
    - Density and sound speed: 1% (these are calculated from fundamental relations)
    - Transport properties: 5% (empirical correlations have larger uncertainty)
    """

    T_REF = 300.0  # Reference temperature (K)
    P_REF = 101325.0  # Reference pressure (Pa, 1 atm)

    # Tolerances
    DENSITY_TOL = 0.01  # 1% for density
    SOUND_SPEED_TOL = 0.01  # 1% for sound speed
    TRANSPORT_TOL = 0.05  # 5% for transport properties

    def test_helium_density_nist(self, helium: Helium) -> None:
        """
        Verify helium density at 300K, 1 atm matches NIST value.

        NIST reference: rho ~ 0.164 kg/m^3 at 300K, 101325 Pa
        """
        nist_rho = 0.164  # kg/m^3
        calculated_rho = helium.density(self.T_REF, self.P_REF)

        assert calculated_rho == pytest.approx(nist_rho, rel=self.DENSITY_TOL), (
            f"Helium density at 300K, 1 atm: NIST={nist_rho} kg/m^3, "
            f"calculated={calculated_rho:.4f} kg/m^3"
        )

    def test_helium_sound_speed_nist(self, helium: Helium) -> None:
        """
        Verify helium sound speed at 300K matches NIST value.

        NIST reference: a ~ 1019 m/s at 300K (for ideal monatomic gas with gamma=5/3).
        Note: The value 1008 m/s sometimes cited includes real gas effects;
        the ideal gas value using gamma=5/3 is approximately 1019 m/s.
        """
        # Using ideal gas formula: a = sqrt(gamma * R_specific * T)
        # For He at 300K with gamma=5/3: a ~ 1019 m/s
        nist_a = 1019.0  # m/s (ideal gas value)
        calculated_a = helium.sound_speed(self.T_REF)

        assert calculated_a == pytest.approx(nist_a, rel=self.SOUND_SPEED_TOL), (
            f"Helium sound speed at 300K: NIST={nist_a} m/s, "
            f"calculated={calculated_a:.1f} m/s"
        )

    def test_helium_viscosity_nist(self, helium: Helium) -> None:
        """
        Verify helium viscosity at 300K matches NIST value.

        NIST reference: mu ~ 1.96e-5 Pa.s at 300K
        """
        nist_mu = 1.96e-5  # Pa.s
        calculated_mu = helium.viscosity(self.T_REF)

        assert calculated_mu == pytest.approx(nist_mu, rel=self.TRANSPORT_TOL), (
            f"Helium viscosity at 300K: NIST={nist_mu:.3e} Pa.s, "
            f"calculated={calculated_mu:.3e} Pa.s"
        )

    def test_air_density_nist(self, air: Air) -> None:
        """
        Verify air density at 300K, 1 atm matches NIST value.

        NIST reference: rho ~ 1.177 kg/m^3 at 300K, 101325 Pa
        """
        nist_rho = 1.177  # kg/m^3
        calculated_rho = air.density(self.T_REF, self.P_REF)

        assert calculated_rho == pytest.approx(nist_rho, rel=self.DENSITY_TOL), (
            f"Air density at 300K, 1 atm: NIST={nist_rho} kg/m^3, "
            f"calculated={calculated_rho:.4f} kg/m^3"
        )

    def test_air_sound_speed_nist(self, air: Air) -> None:
        """
        Verify air sound speed at 300K matches NIST value.

        NIST reference: a ~ 347 m/s at 300K
        """
        nist_a = 347.0  # m/s
        calculated_a = air.sound_speed(self.T_REF)

        assert calculated_a == pytest.approx(nist_a, rel=self.SOUND_SPEED_TOL), (
            f"Air sound speed at 300K: NIST={nist_a} m/s, "
            f"calculated={calculated_a:.1f} m/s"
        )

    def test_air_viscosity_nist(self, air: Air) -> None:
        """
        Verify air viscosity at 300K matches NIST value.

        NIST reference: mu ~ 1.85e-5 Pa.s at 300K
        """
        nist_mu = 1.85e-5  # Pa.s
        calculated_mu = air.viscosity(self.T_REF)

        assert calculated_mu == pytest.approx(nist_mu, rel=self.TRANSPORT_TOL), (
            f"Air viscosity at 300K: NIST={nist_mu:.3e} Pa.s, "
            f"calculated={calculated_mu:.3e} Pa.s"
        )


# ============================================================================
# Test: Additional Property Methods
# ============================================================================


class TestAdditionalProperties:
    """Tests for additional gas property methods and edge cases."""

    def test_specific_gas_constant_method(self, helium: Helium) -> None:
        """
        Verify the specific_gas_constant() method returns R_universal / M.

        The specific gas constant is a fundamental property used in many
        thermodynamic calculations.
        """
        expected_R = R_UNIVERSAL / MOLAR_MASS["helium"]
        calculated_R = helium.specific_gas_constant()

        assert calculated_R == pytest.approx(expected_R, rel=1e-10), (
            f"Helium specific gas constant: expected {expected_R:.3f}, "
            f"got {calculated_R:.3f} J/(kg.K)"
        )

    def test_gas_name_property(
        self,
        helium: Helium,
        argon: Argon,
        nitrogen: Nitrogen,
        air: Air,
    ) -> None:
        """Verify that each gas class returns the correct name."""
        assert helium.name == "Helium"
        assert argon.name == "Argon"
        assert nitrogen.name == "Nitrogen"
        assert air.name == "Air"

    def test_molar_mass_property(
        self,
        helium: Helium,
        argon: Argon,
        nitrogen: Nitrogen,
        air: Air,
    ) -> None:
        """Verify that molar mass properties return expected values."""
        assert helium.molar_mass == pytest.approx(MOLAR_MASS["helium"], rel=1e-10)
        assert argon.molar_mass == pytest.approx(MOLAR_MASS["argon"], rel=1e-10)
        assert nitrogen.molar_mass == pytest.approx(MOLAR_MASS["nitrogen"], rel=1e-10)
        assert air.molar_mass == pytest.approx(MOLAR_MASS["air"], rel=1e-10)

    def test_mean_pressure_property(self, helium: Helium) -> None:
        """Verify that mean_pressure property returns the initialized value."""
        assert helium.mean_pressure == P_ATM

    def test_different_mean_pressures(self) -> None:
        """Verify that gases can be initialized with different mean pressures."""
        pressures = [50000.0, 101325.0, 200000.0, 500000.0]

        for P in pressures:
            gas = Helium(mean_pressure=P)
            assert gas.mean_pressure == P, (
                f"Mean pressure should be {P}, got {gas.mean_pressure}"
            )


# ============================================================================
# Test: Cross-gas Comparisons
# ============================================================================


class TestCrossGasComparisons:
    """
    Tests comparing properties across different gases.

    These tests verify physical relationships that should hold between
    different gas types.
    """

    T = 300.0  # Test temperature

    def test_helium_higher_sound_speed_than_air(
        self, helium: Helium, air: Air
    ) -> None:
        """
        Verify helium has higher sound speed than air at same temperature.

        Sound speed a = sqrt(gamma * R_specific * T) is higher for lighter
        gases because R_specific = R_universal / M is larger.
        """
        a_helium = helium.sound_speed(self.T)
        a_air = air.sound_speed(self.T)

        assert a_helium > a_air, (
            f"Helium sound speed ({a_helium:.1f} m/s) should be > "
            f"air sound speed ({a_air:.1f} m/s)"
        )

    def test_helium_lower_density_than_air(
        self, helium: Helium, air: Air
    ) -> None:
        """
        Verify helium has lower density than air at same T and P.

        From ideal gas law: rho = P*M / (R_universal * T), lighter gases
        have lower density.
        """
        rho_helium = helium.density(self.T)
        rho_air = air.density(self.T)

        assert rho_helium < rho_air, (
            f"Helium density ({rho_helium:.4f} kg/m^3) should be < "
            f"air density ({rho_air:.4f} kg/m^3)"
        )

    def test_monatomic_higher_gamma_than_diatomic(
        self, helium: Helium, air: Air
    ) -> None:
        """
        Verify monatomic gases have higher gamma than diatomic gases.

        Monatomic: gamma = 5/3 ~ 1.667 (3 translational DOF)
        Diatomic: gamma = 7/5 = 1.4 (3 translational + 2 rotational DOF)
        """
        gamma_helium = helium.gamma(self.T)
        gamma_air = air.gamma(self.T)

        assert gamma_helium > gamma_air, (
            f"Monatomic gamma ({gamma_helium:.4f}) should be > "
            f"diatomic gamma ({gamma_air:.4f})"
        )

    def test_helium_higher_cp_per_mass_than_air(
        self, helium: Helium, air: Air
    ) -> None:
        """
        Verify helium has higher cp per unit mass than air.

        cp = gamma / (gamma - 1) * R_specific

        While helium has higher gamma, it also has much higher R_specific
        (due to lower molar mass), resulting in higher cp per unit mass.
        """
        cp_helium = helium.specific_heat_cp(self.T)
        cp_air = air.specific_heat_cp(self.T)

        assert cp_helium > cp_air, (
            f"Helium cp ({cp_helium:.1f} J/(kg.K)) should be > "
            f"air cp ({cp_air:.1f} J/(kg.K))"
        )


# ============================================================================
# Test: Numerical Stability
# ============================================================================


class TestNumericalStability:
    """
    Tests verifying numerical stability at extreme but valid conditions.

    These tests ensure the gas property calculations remain stable and
    physically reasonable at the edges of the expected operating range.
    """

    def test_low_temperature_stability(self, helium: Helium) -> None:
        """
        Verify properties are stable at low temperatures (100K).

        While real helium would be liquid below ~4K, the gas models should
        remain numerically stable at low temperatures within the gas phase.
        """
        T = 100.0  # K

        # All properties should be positive and finite
        assert helium.density(T) > 0
        assert helium.sound_speed(T) > 0
        assert helium.viscosity(T) > 0
        assert helium.thermal_conductivity(T) > 0
        assert helium.prandtl(T) > 0
        assert helium.specific_heat_cp(T) > 0

    def test_high_temperature_stability(self, helium: Helium) -> None:
        """
        Verify properties are stable at high temperatures (1000K).

        At very high temperatures, some gas behaviors deviate from ideal,
        but the models should remain numerically stable.
        """
        T = 1000.0  # K

        # All properties should be positive and finite
        assert helium.density(T) > 0
        assert helium.sound_speed(T) > 0
        assert helium.viscosity(T) > 0
        assert helium.thermal_conductivity(T) > 0
        assert helium.prandtl(T) > 0
        assert helium.specific_heat_cp(T) > 0

        # Verify no NaN or inf values
        assert math.isfinite(helium.density(T))
        assert math.isfinite(helium.sound_speed(T))
        assert math.isfinite(helium.viscosity(T))
        assert math.isfinite(helium.thermal_conductivity(T))

    def test_high_pressure_stability(self, helium: Helium) -> None:
        """
        Verify properties are stable at high pressures (10 atm).

        High pressure operation is common in thermoacoustic devices.
        """
        T = 300.0
        P = 10 * P_ATM  # 10 atmospheres

        rho = helium.density(T, P)
        assert rho > 0
        assert math.isfinite(rho)

        # Density should scale linearly with pressure
        rho_1atm = helium.density(T, P_ATM)
        assert rho == pytest.approx(10 * rho_1atm, rel=1e-10)

    def test_low_pressure_stability(self, helium: Helium) -> None:
        """
        Verify properties are stable at low pressures (0.1 atm).
        """
        T = 300.0
        P = 0.1 * P_ATM  # 0.1 atmospheres

        rho = helium.density(T, P)
        assert rho > 0
        assert math.isfinite(rho)


# ============================================================================
# Test: Air-specific Sutherland's Law
# ============================================================================


class TestAirSutherlandsLaw:
    """
    Tests specific to Air's Sutherland's law viscosity implementation.

    Sutherland's law: mu = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)

    This provides more accurate viscosity values than a simple power law
    across a wider temperature range.
    """

    # Sutherland's law constants for air
    MU_REF = 1.716e-5  # Pa.s at T_ref
    T_REF = 273.15  # K
    S = 110.4  # K (Sutherland constant)

    def test_sutherland_at_reference_temperature(self, air: Air) -> None:
        """
        Verify Sutherland's law returns reference viscosity at reference temperature.
        """
        mu_at_ref = air.viscosity(self.T_REF)

        # At T = T_ref, the formula simplifies to mu_ref
        assert mu_at_ref == pytest.approx(self.MU_REF, rel=1e-10), (
            f"Air viscosity at T_ref={self.T_REF}K: expected {self.MU_REF:.3e}, "
            f"got {mu_at_ref:.3e} Pa.s"
        )

    def test_sutherland_formula_explicit(self, air: Air) -> None:
        """
        Verify Sutherland's law formula is correctly implemented.
        """
        T = 400.0  # Test temperature

        expected_mu = (
            self.MU_REF
            * (T / self.T_REF) ** 1.5
            * (self.T_REF + self.S)
            / (T + self.S)
        )
        calculated_mu = air.viscosity(T)

        assert calculated_mu == pytest.approx(expected_mu, rel=1e-10), (
            f"Sutherland's law mismatch at T={T}K: "
            f"expected {expected_mu:.3e}, got {calculated_mu:.3e} Pa.s"
        )

    @pytest.mark.parametrize("T", [200.0, 300.0, 400.0, 500.0, 600.0])
    def test_sutherland_temperature_scaling(self, air: Air, T: float) -> None:
        """
        Verify Sutherland's law produces expected temperature scaling.

        The exponent in Sutherland's law varies between 0.5 and 1.5 depending
        on temperature, but viscosity should always increase with T.
        """
        expected_mu = (
            self.MU_REF
            * (T / self.T_REF) ** 1.5
            * (self.T_REF + self.S)
            / (T + self.S)
        )
        calculated_mu = air.viscosity(T)

        assert calculated_mu == pytest.approx(expected_mu, rel=1e-10)

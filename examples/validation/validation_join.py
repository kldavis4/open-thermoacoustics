#!/usr/bin/env python3
"""
Validation of JOIN segment physics.

JOIN handles the adiabatic-isothermal interface at pulse tube ends,
accounting for jet pump losses and temperature overshoot effects.

Since no Reference baseline example files with JOIN segments were found in:
- <external proprietary source>
- <external proprietary source>

This script validates the JOIN implementation against the physics
described in the published literature and Swift's "Thermoacoustics".

References:
- published literature, governing relations
- Swift, G.W. (2002), "Thermoacoustics: A Unifying Perspective"
- Kittel, P. (1992), Cryogenics 32, 843
- Storch et al. (1990), NIST Tech Note 1343
"""

import numpy as np
from openthermoacoustics import gas, segments


def test_pressure_preservation():
    """Test that JOIN preserves pressure (p1_out = p1_in)."""
    print("Test 1: Pressure Preservation")
    print("-" * 50)

    helium = gas.Helium(mean_pressure=3.1e6)
    omega = 2 * np.pi * 40.0  # 40 Hz

    join = segments.Join(area=7.0e-3, dT_dx=-3800.0)

    # Test with various input conditions
    test_cases = [
        (1.35e5, -17.66, 2.92e-4, 138.5, 300.0),  # Typical pulse tube conditions
        (2.5e5, -45.0, 1.0e-4, 90.0, 350.0),  # Different phasing
        (1.0e5, 0.0, 5.0e-4, 0.0, 250.0),  # In-phase condition
    ]

    all_passed = True
    for i, (p1_mag, p1_ph, U1_mag, U1_ph, T_m) in enumerate(test_cases):
        p1_in = p1_mag * np.exp(1j * np.radians(p1_ph))
        U1_in = U1_mag * np.exp(1j * np.radians(U1_ph))

        p1_out, U1_out, T_out = join.propagate(p1_in, U1_in, T_m, omega, helium)

        # Pressure should be unchanged
        p1_err = abs(p1_out - p1_in) / abs(p1_in) * 100

        status = "PASS" if p1_err < 1e-10 else "FAIL"
        if p1_err >= 1e-10:
            all_passed = False

        print(f"  Case {i+1}: |p1_in| = {abs(p1_in):.1f} Pa")
        print(f"           |p1_out| = {abs(p1_out):.1f} Pa")
        print(f"           Error: {p1_err:.2e}% -> {status}")

    print()
    return all_passed


def test_phase_preservation():
    """Test that JOIN preserves the phase of U1."""
    print("Test 2: Volume Flow Phase Preservation")
    print("-" * 50)

    helium = gas.Helium(mean_pressure=3.1e6)
    omega = 2 * np.pi * 40.0

    join = segments.Join(area=7.0e-3, dT_dx=-3800.0)

    test_cases = [
        (1.35e5, -17.66, 2.92e-4, 138.5, 300.0),
        (2.5e5, -45.0, 1.0e-4, 90.0, 350.0),
        (1.0e5, 0.0, 5.0e-4, -45.0, 250.0),
    ]

    all_passed = True
    for i, (p1_mag, p1_ph, U1_mag, U1_ph, T_m) in enumerate(test_cases):
        p1_in = p1_mag * np.exp(1j * np.radians(p1_ph))
        U1_in = U1_mag * np.exp(1j * np.radians(U1_ph))

        p1_out, U1_out, T_out = join.propagate(p1_in, U1_in, T_m, omega, helium)

        phase_in = np.degrees(np.angle(U1_in))
        phase_out = np.degrees(np.angle(U1_out))
        phase_err = abs(phase_out - phase_in)

        # Handle phase wrap-around
        if phase_err > 180:
            phase_err = 360 - phase_err

        status = "PASS" if phase_err < 1e-10 else "FAIL"
        if phase_err >= 1e-10:
            all_passed = False

        print(f"  Case {i+1}: Ph(U1)_in = {phase_in:.3f} deg")
        print(f"           Ph(U1)_out = {phase_out:.3f} deg")
        print(f"           Error: {phase_err:.2e} deg -> {status}")

    print()
    return all_passed


def test_volume_flow_reduction():
    """
    Test that JOIN reduces |U1| according to governing relations.

    The magnitude reduction is:
    |U1|_out = |U1|_in - (16/(3*pi)) * ((gamma-1)/(rho*a^2)) * E_dot

    where E_dot = (1/2) * Re[p1 * conj(U1)] is the acoustic power.
    """
    print("Test 3: Volume Flow Magnitude Reduction")
    print("-" * 50)

    helium = gas.Helium(mean_pressure=3.1e6)
    omega = 2 * np.pi * 40.0
    T_m = 300.0

    # Gas properties
    rho_m = helium.density(T_m)
    a = helium.sound_speed(T_m)
    gamma = helium.gamma(T_m)

    join = segments.Join(area=7.0e-3, dT_dx=0.0)  # No temperature gradient

    # Test with positive acoustic power (typical engine/refrigerator condition)
    # Acoustic power is positive when p1 and U1 are in phase
    p1_mag = 1.35e5  # Pa
    p1_ph = 0.0  # deg (reference)
    U1_mag = 2.92e-4  # m^3/s
    U1_ph = 0.0  # deg (in phase - positive power)

    p1_in = p1_mag * np.exp(1j * np.radians(p1_ph))
    U1_in = U1_mag * np.exp(1j * np.radians(U1_ph))

    # Calculate expected reduction
    E_dot = 0.5 * np.real(p1_in * np.conj(U1_in))
    expected_delta_U1 = (16.0 / (3.0 * np.pi)) * ((gamma - 1) / (rho_m * a**2)) * E_dot
    expected_U1_out_mag = U1_mag - expected_delta_U1

    p1_out, U1_out, T_out = join.propagate(p1_in, U1_in, T_m, omega, helium)

    actual_U1_out_mag = abs(U1_out)

    print(f"  Input conditions:")
    print(f"    |p1| = {p1_mag:.2e} Pa")
    print(f"    |U1| = {U1_mag:.6f} m^3/s")
    print(f"    E_dot = {E_dot:.2f} W")
    print()
    print(f"  Expected |U1|_out = {expected_U1_out_mag:.6f} m^3/s")
    print(f"  Actual |U1|_out   = {actual_U1_out_mag:.6f} m^3/s")

    rel_err = abs(actual_U1_out_mag - expected_U1_out_mag) / expected_U1_out_mag * 100
    status = "PASS" if rel_err < 0.01 else "FAIL"
    passed = rel_err < 0.01

    print(f"  Relative error: {rel_err:.4f}% -> {status}")
    print()

    # Also verify the reduction is in the correct direction
    print(f"  Magnitude reduction: {(U1_mag - actual_U1_out_mag)*1e6:.3f} mm^3/s")
    print(f"  (Should be positive for positive acoustic power)")

    if U1_mag - actual_U1_out_mag <= 0 and E_dot > 0:
        print("  WARNING: Reduction has wrong sign!")
        passed = False

    print()
    return passed


def test_temperature_discontinuity():
    """
    Test temperature discontinuity with temperature gradient (governing relation).

    delta_T = -(T_m * beta / (rho_m * c_p)) *
              (|p1| * sin(theta) - |U1|_in * dT_m/dx / (omega * A_gas)) * F

    where theta is the phase by which p1 leads U1.
    """
    print("Test 4: Temperature Discontinuity with Gradient")
    print("-" * 50)

    helium = gas.Helium(mean_pressure=3.1e6)
    freq = 40.0
    omega = 2 * np.pi * freq
    T_m = 300.0
    area = 7.0e-3  # m^2
    dT_dx = -3800.0  # K/m (temperature decreases along x for pulse tube)

    # Gas properties
    rho_m = helium.density(T_m)
    cp = helium.specific_heat_cp(T_m)
    beta = 1.0 / T_m  # Ideal gas thermal expansion coefficient

    join = segments.Join(area=area, dT_dx=dT_dx)

    # Test with p1 leading U1 by 90 degrees (maximum temperature effect)
    p1_mag = 1.35e5  # Pa
    p1_ph = 0.0  # deg
    U1_mag = 2.92e-4  # m^3/s
    U1_ph = -90.0  # deg (p1 leads U1 by 90 deg)

    p1_in = p1_mag * np.exp(1j * np.radians(p1_ph))
    U1_in = U1_mag * np.exp(1j * np.radians(U1_ph))

    # Phase angle by which p1 leads U1
    theta = np.angle(p1_in) - np.angle(U1_in)

    # Temperature overshoot term (F = 1 approximation)
    temp_term = p1_mag * np.sin(theta) - U1_mag * dT_dx / (omega * area)

    # Expected temperature discontinuity
    F = 1.0  # Simplified factor
    expected_delta_T = -(T_m * beta / (rho_m * cp)) * temp_term * F
    expected_T_out = T_m + expected_delta_T

    p1_out, U1_out, T_out = join.propagate(p1_in, U1_in, T_m, omega, helium)

    print(f"  Input conditions:")
    print(f"    T_m = {T_m:.1f} K")
    print(f"    dT/dx = {dT_dx:.1f} K/m")
    print(f"    theta (p1 leads U1) = {np.degrees(theta):.1f} deg")
    print()
    print(f"  Temperature discontinuity:")
    print(f"    Expected T_out = {expected_T_out:.4f} K")
    print(f"    Actual T_out   = {T_out:.4f} K")
    print(f"    Delta T        = {T_out - T_m:.4f} K")

    T_err = abs(T_out - expected_T_out)
    status = "PASS" if T_err < 0.01 else "FAIL"
    passed = T_err < 0.01

    print(f"  Absolute error: {T_err:.6f} K -> {status}")
    print()
    return passed


def test_no_temperature_change_without_gradient():
    """Test that JOIN produces no temperature change when dT_dx = 0."""
    print("Test 5: No Temperature Change Without Gradient")
    print("-" * 50)

    helium = gas.Helium(mean_pressure=3.1e6)
    omega = 2 * np.pi * 40.0
    T_m = 300.0

    # JOIN with no temperature gradient
    join = segments.Join(area=7.0e-3, dT_dx=0.0)

    test_cases = [
        (1.35e5, -17.66, 2.92e-4, 138.5),
        (2.5e5, 0.0, 1.0e-4, 90.0),
        (1.0e5, 45.0, 5.0e-4, -45.0),
    ]

    all_passed = True
    for i, (p1_mag, p1_ph, U1_mag, U1_ph) in enumerate(test_cases):
        p1_in = p1_mag * np.exp(1j * np.radians(p1_ph))
        U1_in = U1_mag * np.exp(1j * np.radians(U1_ph))

        p1_out, U1_out, T_out = join.propagate(p1_in, U1_in, T_m, omega, helium)

        T_err = abs(T_out - T_m)
        status = "PASS" if T_err < 1e-10 else "FAIL"
        if T_err >= 1e-10:
            all_passed = False

        print(f"  Case {i+1}: T_in = {T_m:.1f} K, T_out = {T_out:.6f} K")
        print(f"           Error: {T_err:.2e} K -> {status}")

    print()
    return all_passed


def test_acoustic_power_dissipation():
    """Test that acoustic power dissipation is calculated correctly."""
    print("Test 6: Acoustic Power Dissipation")
    print("-" * 50)

    helium = gas.Helium(mean_pressure=3.1e6)
    omega = 2 * np.pi * 40.0
    T_m = 300.0

    join = segments.Join(area=7.0e-3, dT_dx=-3800.0)

    # Condition with positive acoustic power
    p1_mag = 1.35e5  # Pa
    p1_ph = 0.0  # deg
    U1_mag = 2.92e-4  # m^3/s
    U1_ph = 0.0  # deg (in phase)

    p1_in = p1_mag * np.exp(1j * np.radians(p1_ph))
    U1_in = U1_mag * np.exp(1j * np.radians(U1_ph))

    # Calculate acoustic power in and reference case
    E_dot_in = 0.5 * np.real(p1_in * np.conj(U1_in))

    p1_out, U1_out, T_out = join.propagate(p1_in, U1_in, T_m, omega, helium)

    E_dot_out = 0.5 * np.real(p1_out * np.conj(U1_out))

    # Dissipation via dedicated method
    dissipation = join.acoustic_power_dissipation(p1_in, U1_in, T_m, omega, helium)

    print(f"  Acoustic power in:   {E_dot_in:.4f} W")
    print(f"  Acoustic power reference case:  {E_dot_out:.4f} W")
    print(f"  Dissipation (calc):  {E_dot_in - E_dot_out:.4f} W")
    print(f"  Dissipation (method):{dissipation:.4f} W")

    # Verify dissipation is non-negative
    passed = dissipation >= 0
    if dissipation < 0:
        print("  WARNING: Negative dissipation (non-physical)!")

    # Verify consistency
    calc_dissipation = E_dot_in - E_dot_out
    consistency_err = abs(dissipation - calc_dissipation)
    if consistency_err > 1e-10:
        print(f"  WARNING: Dissipation calculation inconsistent: {consistency_err:.2e}")
        passed = False

    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")
    print()
    return passed


def test_zero_velocity():
    """Test behavior with zero input velocity."""
    print("Test 7: Zero Velocity Input")
    print("-" * 50)

    helium = gas.Helium(mean_pressure=3.1e6)
    omega = 2 * np.pi * 40.0
    T_m = 300.0

    join = segments.Join(area=7.0e-3, dT_dx=0.0)

    p1_in = 1.0e5 + 0j  # Pressure only
    U1_in = 0.0 + 0j  # Zero velocity

    p1_out, U1_out, T_out = join.propagate(p1_in, U1_in, T_m, omega, helium)

    # With zero velocity, acoustic power is zero, so no U1 reduction
    # Temperature should also be unchanged (no gradient effect with zero U1)
    passed = True

    print(f"  Input:  p1 = {p1_in}, U1 = {U1_in}")
    print(f"  Output: p1 = {p1_out}, U1 = {U1_out}")

    if abs(U1_out) > 1e-20:
        print(f"  WARNING: Non-zero output velocity from zero input")
        passed = False

    if abs(p1_out - p1_in) > 1e-10:
        print(f"  WARNING: Pressure changed unexpectedly")
        passed = False

    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")
    print()
    return passed


def test_negative_acoustic_power():
    """Test behavior with negative acoustic power (power flowing backward)."""
    print("Test 8: Negative Acoustic Power")
    print("-" * 50)

    helium = gas.Helium(mean_pressure=3.1e6)
    omega = 2 * np.pi * 40.0
    T_m = 300.0

    # Gas properties
    rho_m = helium.density(T_m)
    a = helium.sound_speed(T_m)
    gamma = helium.gamma(T_m)

    join = segments.Join(area=7.0e-3, dT_dx=0.0)

    # Negative acoustic power: p1 and U1 are 180 deg reference case of phase
    p1_mag = 1.35e5  # Pa
    p1_ph = 0.0  # deg
    U1_mag = 2.92e-4  # m^3/s
    U1_ph = 180.0  # deg (reference case of phase - negative power)

    p1_in = p1_mag * np.exp(1j * np.radians(p1_ph))
    U1_in = U1_mag * np.exp(1j * np.radians(U1_ph))

    E_dot = 0.5 * np.real(p1_in * np.conj(U1_in))
    print(f"  Acoustic power: {E_dot:.4f} W (negative)")

    # Expected: |U1| should INCREASE (negative delta)
    expected_delta_U1 = (16.0 / (3.0 * np.pi)) * ((gamma - 1) / (rho_m * a**2)) * E_dot
    expected_U1_out_mag = U1_mag - expected_delta_U1  # Subtracting negative = adding

    p1_out, U1_out, T_out = join.propagate(p1_in, U1_in, T_m, omega, helium)

    actual_U1_out_mag = abs(U1_out)

    print(f"  |U1|_in  = {U1_mag:.6f} m^3/s")
    print(f"  |U1|_out = {actual_U1_out_mag:.6f} m^3/s")
    print(f"  Expected |U1|_out = {expected_U1_out_mag:.6f} m^3/s")

    # With negative acoustic power, |U1| should increase
    increased = actual_U1_out_mag > U1_mag
    print(f"  |U1| increased: {increased}")

    rel_err = abs(actual_U1_out_mag - expected_U1_out_mag) / expected_U1_out_mag * 100
    passed = rel_err < 0.01 and increased

    status = "PASS" if passed else "FAIL"
    print(f"  Relative error: {rel_err:.4f}% -> {status}")
    print()
    return passed


def test_repr():
    """Test string representation of JOIN segment."""
    print("Test 9: String Representation")
    print("-" * 50)

    join = segments.Join(area=7.0e-3, dT_dx=-3800.0, name="PT_hot_end")
    repr_str = repr(join)

    print(f"  repr: {repr_str}")

    passed = "Join" in repr_str and "area" in repr_str and "dT_dx" in repr_str

    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status}")
    print()
    return passed


def main():
    print("=" * 70)
    print("VALIDATION: JOIN Segment (Adiabatic-Isothermal Interface)")
    print("=" * 70)
    print()
    print("JOIN models end effects at pulse tube interfaces, accounting for:")
    print("  1. Temperature overshoot from pressure-induced oscillations")
    print("  2. Volume flow rate reduction from irreversible mixing")
    print()
    print("No Reference baseline example files with JOIN were found, so this validation")
    print("tests the implementation against the physics equations directly.")
    print()

    results = []

    results.append(("Pressure Preservation", test_pressure_preservation()))
    results.append(("Phase Preservation", test_phase_preservation()))
    results.append(("Volume Flow Reduction", test_volume_flow_reduction()))
    results.append(("Temperature Discontinuity", test_temperature_discontinuity()))
    results.append(("No Temp Change w/o Gradient", test_no_temperature_change_without_gradient()))
    results.append(("Acoustic Power Dissipation", test_acoustic_power_dissipation()))
    results.append(("Zero Velocity Input", test_zero_velocity()))
    results.append(("Negative Acoustic Power", test_negative_acoustic_power()))
    results.append(("String Representation", test_repr()))

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Test':<35} {'Status':<10}")
    print("-" * 45)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{name:<35} {status:<10}")

    print("-" * 45)

    if all_passed:
        print("\nALL TESTS PASSED")
        print("\nThe JOIN segment implementation correctly models:")
        print("  - Pressure preservation across the interface")
        print("  - Phase preservation of volume flow rate")
        print("  - Volume flow magnitude reduction (governing relation)")
        print("  - Temperature discontinuity with gradient (governing relation)")
        print("  - Acoustic power dissipation from mixing losses")
    else:
        print("\nSOME TESTS FAILED")
        print("Review the output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

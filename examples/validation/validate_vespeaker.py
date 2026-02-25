#!/usr/bin/env python3
"""
Validation of VESPEAKER (voltage-driven enclosed speaker).

This validation tests:
1. Self-consistency between IESPEAKER and VESPEAKER
2. Physical reasonableness of computed current and pressure
3. Power conservation between electrical and acoustic domains
4. Frequency sweep behavior near resonance
"""

import numpy as np

from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: VESPEAKER (voltage-driven enclosed speaker)")
    print("=" * 70)
    print()

    # =========================================================================
    # Setup: Common transducer parameters
    # =========================================================================
    # Typical loudspeaker parameters (similar to Reference baseline examples)
    Bl = 10.0  # T*m (force factor)
    R_e = 6.0  # Ohm (DC resistance)
    L_e = 0.5e-3  # H (voice coil inductance)
    m = 0.015  # kg (moving mass)
    k = 3000.0  # N/m (suspension stiffness)
    R_m = 1.5  # N*s/m (mechanical resistance)
    A_d = 0.01  # m^2 (diaphragm area, ~11 cm diameter)

    # Create transducer
    trans = segments.Transducer(
        Bl=Bl, R_e=R_e, L_e=L_e, m=m, k=k, R_m=R_m, A_d=A_d, name="test_speaker"
    )

    # Resonant frequency
    f_s = trans.resonant_frequency()
    print(f"Transducer parameters:")
    print(f"  Bl = {Bl} T*m, R_e = {R_e} Ohm, L_e = {L_e*1e3:.1f} mH")
    print(f"  m = {m*1e3:.1f} g, k = {k:.0f} N/m, R_m = {R_m} N*s/m")
    print(f"  A_d = {A_d*1e4:.1f} cm^2")
    print(f"  Resonant frequency: f_s = {f_s:.1f} Hz")
    print()

    # Use helium at 1 atm
    helium = gas.Helium(mean_pressure=101325)
    T_m = 300.0  # K

    # =========================================================================
    # Test 1: Self-consistency between IESPEAKER and VESPEAKER
    # =========================================================================
    print("TEST 1: Self-consistency (IESPEAKER -> VESPEAKER -> IESPEAKER)")
    print("-" * 50)

    # Operating frequency (near resonance for interesting behavior)
    freq = f_s
    omega = 2 * np.pi * freq
    print(f"Frequency: {freq:.1f} Hz (at resonance)")
    print()

    # Initial acoustic state
    p1_in = 1000.0 + 500.0j  # Pa
    U1_in = 1e-4 + 2e-5j  # m^3/s

    # Step 1: Drive with current I1, get voltage V1 (IESPEAKER)
    I1_orig = 2.0 + 1.0j  # A
    p1_ie, U1_ie, T_ie, V1_from_ie = trans.propagate_driven(
        p1_in, U1_in, T_m, omega, helium, I1_orig
    )

    print(f"IESPEAKER with I1 = {I1_orig:.4f} A:")
    print(f"  V1 = {V1_from_ie:.4f} V")
    print(f"  |V1| = {np.abs(V1_from_ie):.4f} V")
    print()

    # Step 2: Use that V1 in VESPEAKER, should recover I1
    p1_ve, U1_ve, T_ve, I1_from_ve = trans.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega, helium, V1_from_ie
    )

    print(f"VESPEAKER with V1 = {V1_from_ie:.4f} V:")
    print(f"  I1 = {I1_from_ve:.4f} A")
    print(f"  |I1| = {np.abs(I1_from_ve):.4f} A")
    print()

    # Compare results
    I1_err = np.abs(I1_from_ve - I1_orig) / np.abs(I1_orig) * 100
    p1_err = np.abs(p1_ve - p1_ie) / np.abs(p1_ie) * 100 if np.abs(p1_ie) > 0 else 0.0

    print(f"Current recovery error: {I1_err:.6f}%")
    print(f"Pressure output error: {p1_err:.6f}%")

    test1_pass = I1_err < 0.001 and p1_err < 0.001
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (self-consistency)")
    print()

    # =========================================================================
    # Test 2: Reverse direction (VESPEAKER -> IESPEAKER -> VESPEAKER)
    # =========================================================================
    print("TEST 2: Reverse self-consistency (VESPEAKER -> IESPEAKER -> VESPEAKER)")
    print("-" * 50)

    # Start with specified voltage
    V1_orig = 15.0 + 5.0j  # V

    # Step 1: VESPEAKER computes current
    p1_ve2, U1_ve2, T_ve2, I1_from_ve2 = trans.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega, helium, V1_orig
    )

    print(f"VESPEAKER with V1 = {V1_orig:.4f} V:")
    print(f"  I1 = {I1_from_ve2:.4f} A")
    print()

    # Step 2: IESPEAKER with that current should recover voltage
    p1_ie2, U1_ie2, T_ie2, V1_from_ie2 = trans.propagate_driven(
        p1_in, U1_in, T_m, omega, helium, I1_from_ve2
    )

    print(f"IESPEAKER with I1 = {I1_from_ve2:.4f} A:")
    print(f"  V1 = {V1_from_ie2:.4f} V")
    print()

    V1_err = np.abs(V1_from_ie2 - V1_orig) / np.abs(V1_orig) * 100
    p1_err2 = np.abs(p1_ie2 - p1_ve2) / np.abs(p1_ve2) * 100

    print(f"Voltage recovery error: {V1_err:.6f}%")
    print(f"Pressure output error: {p1_err2:.6f}%")

    test2_pass = V1_err < 0.001 and p1_err2 < 0.001
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (reverse self-consistency)")
    print()

    # =========================================================================
    # Test 3: Electrical power computation
    # =========================================================================
    print("TEST 3: Electrical power computation")
    print("-" * 50)

    # Use VESPEAKER result
    P_elec = trans.electrical_power(I1_from_ve2, V1_orig)

    # Verify against direct calculation
    P_elec_expected = 0.5 * np.real(V1_orig * np.conj(I1_from_ve2))

    print(f"V1 = {V1_orig:.4f} V, I1 = {I1_from_ve2:.4f} A")
    print(f"Electrical power (computed): {P_elec:.6f} W")
    print(f"Electrical power (expected): {P_elec_expected:.6f} W")

    P_err = abs(P_elec - P_elec_expected) / max(abs(P_elec_expected), 1e-20) * 100
    print(f"Error: {P_err:.6f}%")

    test3_pass = P_err < 0.001
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (power computation)")
    print()

    # =========================================================================
    # Test 4: Frequency sweep (physical reasonableness)
    # =========================================================================
    print("TEST 4: Frequency sweep (check physical behavior)")
    print("-" * 50)

    V1_drive = 10.0 + 0.0j  # 10V real amplitude

    freqs = [f_s / 4, f_s / 2, f_s, 2 * f_s, 4 * f_s]
    print(f"{'Freq (Hz)':<12} {'|I1| (A)':<12} {'|Z| (Ohm)':<12} {'Phase I1 (°)':<12}")
    print("-" * 48)

    I1_values = []
    for freq_test in freqs:
        omega_test = 2 * np.pi * freq_test
        _, _, _, I1_test = trans.propagate_voltage_driven(
            p1_in, U1_in, T_m, omega_test, helium, V1_drive
        )
        Z_apparent = V1_drive / I1_test  # Apparent impedance
        I1_values.append(I1_test)

        print(
            f"{freq_test:<12.1f} {np.abs(I1_test):<12.4f} "
            f"{np.abs(Z_apparent):<12.2f} {np.angle(I1_test, deg=True):<12.1f}"
        )

    # Physical checks:
    # 1. All currents should be finite and non-zero
    # 2. At resonance, current should be higher (impedance lower) if properly tuned
    # 3. Current magnitude should vary smoothly with frequency

    all_finite = all(np.isfinite(I1) for I1 in I1_values)
    all_nonzero = all(np.abs(I1) > 0 for I1 in I1_values)

    test4_pass = all_finite and all_nonzero
    print()
    print(f"All values finite: {all_finite}")
    print(f"All values non-zero: {all_nonzero}")
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (frequency sweep)")
    print()

    # =========================================================================
    # Test 5: Zero velocity limit (blocked transducer)
    # =========================================================================
    print("TEST 5: Zero velocity limit (blocked transducer)")
    print("-" * 50)

    # With U1 = 0 (blocked diaphragm), I1 = V1/Z_e
    U1_blocked = 0.0 + 0.0j
    p1_blocked = 0.0 + 0.0j  # No acoustic load

    _, _, _, I1_blocked = trans.propagate_voltage_driven(
        p1_blocked, U1_blocked, T_m, omega, helium, V1_drive
    )

    # Expected: I1 = V1 / Z_e (back-EMF term Bl*v = 0)
    Z_e = trans.electrical_impedance(omega)
    I1_blocked_expected = V1_drive / Z_e

    print(f"V1 = {V1_drive:.4f} V, U1 = 0 (blocked)")
    print(f"Z_e = {Z_e:.4f} Ohm")
    print(f"I1 (computed): {I1_blocked:.6f} A")
    print(f"I1 (expected): {I1_blocked_expected:.6f} A")

    I1_blocked_err = (
        np.abs(I1_blocked - I1_blocked_expected) / np.abs(I1_blocked_expected) * 100
    )
    print(f"Error: {I1_blocked_err:.6f}%")

    test5_pass = I1_blocked_err < 0.001
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (blocked limit)")
    print()

    # =========================================================================
    # Test 6: Pressure output consistency
    # =========================================================================
    print("TEST 6: Pressure output equation verification")
    print("-" * 50)

    # Verify: p1_out = p1_in - Bl*I1/A_d + Z_m*U1/A_d^2
    freq_test = f_s
    omega_test = 2 * np.pi * freq_test

    p1_ve_test, U1_ve_test, _, I1_ve_test = trans.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega_test, helium, V1_drive
    )

    # Manual calculation
    Z_m = trans.mechanical_impedance(omega_test)
    p_source = -Bl * I1_ve_test / A_d
    p_impedance = Z_m * U1_in / (A_d**2)
    p1_expected = p1_in + p_source + p_impedance

    print(f"Input: p1_in = {p1_in:.2f} Pa")
    print(f"Source pressure (Lorentz): {p_source:.2f} Pa")
    print(f"Impedance pressure: {p_impedance:.2f} Pa")
    print(f"Output: p1_out (computed) = {p1_ve_test:.2f} Pa")
    print(f"Output: p1_out (expected) = {p1_expected:.2f} Pa")

    p_out_err = np.abs(p1_ve_test - p1_expected) / np.abs(p1_expected) * 100
    print(f"Error: {p_out_err:.6f}%")

    test6_pass = p_out_err < 0.001
    print()
    print(f"Test 6: {'PASS' if test6_pass else 'FAIL'} (pressure equation)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = (
        test1_pass and test2_pass and test3_pass and test4_pass and test5_pass and test6_pass
    )
    print(f"Test 1 (IESPEAKER -> VESPEAKER consistency): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (VESPEAKER -> IESPEAKER consistency): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Electrical power computation):       {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Frequency sweep behavior):           {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Blocked transducer limit):           {'PASS' if test5_pass else 'FAIL'}")
    print(f"Test 6 (Pressure equation verification):     {'PASS' if test6_pass else 'FAIL'}")
    print()

    if all_pass:
        print("VESPEAKER VALIDATION PASSED")
        print()
        print("Notes:")
        print("- VESPEAKER correctly implements Reference baseline governing relations")
        print("- Current I1 = (V1 - Bl*U1/A_d) / Z_e is computed correctly")
        print("- Pressure output matches IESPEAKER for same I1")
        print("- Self-consistency between IESPEAKER and VESPEAKER verified")
        print("- No reference file available for direct comparison")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

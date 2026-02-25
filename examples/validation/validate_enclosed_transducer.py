#!/usr/bin/env python3
"""
Validation of enclosed (series) transducers with direct coefficients (IEDUCER, VEDUCER).

This validation tests:
1. Self-consistency between current-driven and voltage-driven modes
2. Comparison with IESPEAKER/VESPEAKER (same physics, different parameterization)
3. Flow continuity (series behavior)
4. Equation verification against Reference baseline formulas
"""

import numpy as np

from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: Enclosed transducers (IEDUCER, VEDUCER)")
    print("=" * 70)
    print()

    # =========================================================================
    # Test 1: IEDUCER/VEDUCER Self-consistency
    # =========================================================================
    print("TEST 1: IEDUCER/VEDUCER self-consistency")
    print("-" * 50)

    # Create an enclosed transducer with direct coefficients
    Ze = 6.0 + 0.5j  # Ohm
    tau = 1000.0 + 100.0j  # V·s/m³
    tau_prime = -1000.0 - 100.0j  # Pa/A (typically τ' = -τ for speakers)
    Z = 5e6 + 1e6j  # Pa·s/m³

    educer = segments.EnclosedTransducer(
        Ze=Ze, tau=tau, tau_prime=tau_prime, Z=Z, name="test_educer"
    )

    print(f"Transducer parameters:")
    print(f"  Ze = {Ze} Ohm")
    print(f"  τ  = {tau} V·s/m³")
    print(f"  τ' = {tau_prime} Pa/A")
    print(f"  Z  = {Z} Pa·s/m³")
    print()

    # Use helium at 1 atm
    helium = gas.Helium(mean_pressure=101325)
    T_m = 300.0
    omega = 2 * np.pi * 100  # 100 Hz

    # Initial acoustic state
    p1_in = 1000.0 + 500.0j  # Pa
    U1_in = 1e-4 + 2e-5j  # m³/s

    # Step 1: IEDUCER - specify current, get voltage
    I1_orig = 2.0 + 1.0j  # A
    p1_i, U1_i, T_i, V1_from_i = educer.propagate_current_driven(
        p1_in, U1_in, T_m, omega, helium, I1_orig
    )

    print(f"IEDUCER with I1 = {I1_orig:.4f} A:")
    print(f"  p1_out = {p1_i:.4f} Pa")
    print(f"  V1 = {V1_from_i:.4f} V")
    print()

    # Step 2: VEDUCER - use that voltage, should recover current
    p1_v, U1_v, T_v, I1_from_v = educer.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega, helium, V1_from_i
    )

    print(f"VEDUCER with V1 = {V1_from_i:.4f} V:")
    print(f"  p1_out = {p1_v:.4f} Pa")
    print(f"  I1 = {I1_from_v:.4f} A")
    print()

    # Compare
    I1_err = np.abs(I1_from_v - I1_orig) / np.abs(I1_orig) * 100
    p1_err = np.abs(p1_v - p1_i) / np.abs(p1_i) * 100

    print(f"Current recovery error: {I1_err:.6f}%")
    print(f"Pressure output error: {p1_err:.6f}%")

    test1_pass = I1_err < 0.001 and p1_err < 0.001
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (IEDUCER/VEDUCER self-consistency)")
    print()

    # =========================================================================
    # Test 2: Flow continuity (series behavior)
    # =========================================================================
    print("TEST 2: Flow continuity (series behavior)")
    print("-" * 50)

    # For enclosed transducers, U1_out should equal U1_in
    U1_err_i = np.abs(U1_i - U1_in) / np.abs(U1_in) * 100
    U1_err_v = np.abs(U1_v - U1_in) / np.abs(U1_in) * 100

    print(f"IEDUCER: U1_in = {U1_in:.6e}, U1_out = {U1_i:.6e}")
    print(f"  Error: {U1_err_i:.6f}%")
    print(f"VEDUCER: U1_in = {U1_in:.6e}, U1_out = {U1_v:.6e}")
    print(f"  Error: {U1_err_v:.6f}%")

    test2_pass = U1_err_i < 0.001 and U1_err_v < 0.001
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (flow continuity)")
    print()

    # =========================================================================
    # Test 3: Direct equation verification for IEDUCER
    # =========================================================================
    print("TEST 3: Direct equation verification for IEDUCER")
    print("-" * 50)

    # For IEDUCER: p1_out = p1_in + τ'*I1 + Z*U1, V1 = Ze*I1 + τ*U1
    p1_expected = p1_in + tau_prime * I1_orig + Z * U1_in
    V1_expected = Ze * I1_orig + tau * U1_in

    p1_eq_err = np.abs(p1_i - p1_expected) / np.abs(p1_expected) * 100
    V1_eq_err = np.abs(V1_from_i - V1_expected) / np.abs(V1_expected) * 100

    print(f"p1_out (computed): {p1_i:.4f} Pa")
    print(f"p1_out (expected): {p1_expected:.4f} Pa")
    print(f"p1 error: {p1_eq_err:.6f}%")
    print()
    print(f"V1 (computed): {V1_from_i:.4f} V")
    print(f"V1 (expected): {V1_expected:.4f} V")
    print(f"V1 error: {V1_eq_err:.6f}%")

    test3_pass = p1_eq_err < 0.001 and V1_eq_err < 0.001
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (IEDUCER equations)")
    print()

    # =========================================================================
    # Test 4: Direct equation verification for VEDUCER
    # =========================================================================
    print("TEST 4: Direct equation verification for VEDUCER")
    print("-" * 50)

    V1_test = 10.0 + 5.0j  # V
    p1_vt, U1_vt, T_vt, I1_vt = educer.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega, helium, V1_test
    )

    # I1 = (V1 - τ*U1) / Ze
    I1_expected = (V1_test - tau * U1_in) / Ze

    # p1_out = p1_in + τ'*I1 + Z*U1
    p1_expected_v = p1_in + tau_prime * I1_expected + Z * U1_in

    I1_eq_err = np.abs(I1_vt - I1_expected) / np.abs(I1_expected) * 100
    p1_eq_err_v = np.abs(p1_vt - p1_expected_v) / np.abs(p1_expected_v) * 100

    print(f"V1 = {V1_test} V")
    print(f"I1 (computed): {I1_vt:.6f} A")
    print(f"I1 (expected): {I1_expected:.6f} A")
    print(f"I1 error: {I1_eq_err:.6f}%")
    print()
    print(f"p1_out (computed): {p1_vt:.4f} Pa")
    print(f"p1_out (expected): {p1_expected_v:.4f} Pa")
    print(f"p1 error: {p1_eq_err_v:.6f}%")

    test4_pass = I1_eq_err < 0.001 and p1_eq_err_v < 0.001
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (VEDUCER equations)")
    print()

    # =========================================================================
    # Test 5: Comparison with IESPEAKER/VESPEAKER using equivalent coefficients
    # =========================================================================
    print("TEST 5: Comparison with IESPEAKER/VESPEAKER")
    print("-" * 50)

    # Create an IESPEAKER with known parameters
    area = 0.01  # m²
    Bl = 10.0  # T·m
    R_e = 6.0  # Ohm
    L_e = 0.5e-3  # H
    m = 0.015  # kg
    k = 3000.0  # N/m
    R_m = 1.5  # N·s/m

    speaker = segments.Transducer(
        Bl=Bl, R_e=R_e, L_e=L_e, m=m, k=k, R_m=R_m, A_d=area, name="test_speaker"
    )

    # Compute the equivalent *EDUCER coefficients at this frequency
    freq = 100.0
    omega_test = 2 * np.pi * freq

    Ze_spk = speaker.electrical_impedance(omega_test)
    Z_spk = speaker.mechanical_impedance(omega_test) / area**2
    tau_spk = Bl / area
    tau_prime_spk = -Bl / area

    print(f"IESPEAKER parameters at f = {freq:.0f} Hz:")
    print(f"  Ze = {Ze_spk:.4f} Ohm")
    print(f"  τ  = {tau_spk:.4f} V·s/m³")
    print(f"  τ' = {tau_prime_spk:.4f} Pa/A")
    print(f"  Z  = {Z_spk:.4e} Pa·s/m³")
    print()

    # Create equivalent IEDUCER
    educer_equiv = segments.EnclosedTransducer(
        Ze=Ze_spk, tau=tau_spk, tau_prime=tau_prime_spk, Z=Z_spk
    )

    # Compare outputs for same input
    I1_test = 1.0 + 0.5j  # A

    # IESPEAKER output
    p1_spk, U1_spk, T_spk, V1_spk = speaker.propagate_driven(
        p1_in, U1_in, T_m, omega_test, helium, I1_test
    )

    # IEDUCER output
    p1_edu, U1_edu, T_edu, V1_edu = educer_equiv.propagate_current_driven(
        p1_in, U1_in, T_m, omega_test, helium, I1_test
    )

    # Compare
    p1_diff = np.abs(p1_edu - p1_spk) / np.abs(p1_spk) * 100
    V1_diff = np.abs(V1_edu - V1_spk) / np.abs(V1_spk) * 100

    print(f"For I1 = {I1_test} A:")
    print(f"  IESPEAKER: p1_out = {p1_spk:.4f} Pa, V1 = {V1_spk:.4f} V")
    print(f"  IEDUCER:   p1_out = {p1_edu:.4f} Pa, V1 = {V1_edu:.4f} V")
    print(f"  Pressure difference: {p1_diff:.6f}%")
    print(f"  Voltage difference: {V1_diff:.6f}%")

    test5_pass = p1_diff < 0.01 and V1_diff < 0.01
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (comparison with IESPEAKER)")
    print()

    # =========================================================================
    # Test 6: Electrical power computation
    # =========================================================================
    print("TEST 6: Electrical power computation")
    print("-" * 50)

    P_elec = educer.electrical_power(I1_orig, V1_from_i)
    P_expected = 0.5 * np.real(V1_from_i * np.conj(I1_orig))

    print(f"V1 = {V1_from_i:.4f} V, I1 = {I1_orig:.4f} A")
    print(f"Power (computed): {P_elec:.6f} W")
    print(f"Power (expected): {P_expected:.6f} W")

    P_err = abs(P_elec - P_expected) / max(abs(P_expected), 1e-20) * 100
    print(f"Error: {P_err:.6f}%")

    test6_pass = P_err < 0.001
    print()
    print(f"Test 6: {'PASS' if test6_pass else 'FAIL'} (electrical power)")
    print()

    # =========================================================================
    # Test 7: Reverse self-consistency (VEDUCER -> IEDUCER -> VEDUCER)
    # =========================================================================
    print("TEST 7: Reverse self-consistency (VEDUCER -> IEDUCER -> VEDUCER)")
    print("-" * 50)

    # Start with voltage
    V1_orig = 15.0 + 5.0j  # V

    # VEDUCER computes current
    p1_ve, U1_ve, T_ve, I1_from_ve = educer.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega, helium, V1_orig
    )

    print(f"VEDUCER with V1 = {V1_orig:.4f} V:")
    print(f"  I1 = {I1_from_ve:.4f} A")
    print()

    # IEDUCER with that current should recover voltage
    p1_ie, U1_ie, T_ie, V1_from_ie = educer.propagate_current_driven(
        p1_in, U1_in, T_m, omega, helium, I1_from_ve
    )

    print(f"IEDUCER with I1 = {I1_from_ve:.4f} A:")
    print(f"  V1 = {V1_from_ie:.4f} V")
    print()

    V1_err = np.abs(V1_from_ie - V1_orig) / np.abs(V1_orig) * 100
    p1_err_rev = np.abs(p1_ie - p1_ve) / np.abs(p1_ve) * 100

    print(f"Voltage recovery error: {V1_err:.6f}%")
    print(f"Pressure output error: {p1_err_rev:.6f}%")

    test7_pass = V1_err < 0.001 and p1_err_rev < 0.001
    print()
    print(f"Test 7: {'PASS' if test7_pass else 'FAIL'} (reverse self-consistency)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = (
        test1_pass and test2_pass and test3_pass and test4_pass
        and test5_pass and test6_pass and test7_pass
    )
    print(f"Test 1 (IEDUCER/VEDUCER self-consistency): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Flow continuity):                  {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (IEDUCER equations):                {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (VEDUCER equations):                {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Comparison with IESPEAKER):        {'PASS' if test5_pass else 'FAIL'}")
    print(f"Test 6 (Electrical power):                 {'PASS' if test6_pass else 'FAIL'}")
    print(f"Test 7 (Reverse self-consistency):         {'PASS' if test7_pass else 'FAIL'}")
    print()

    if all_pass:
        print("ENCLOSED TRANSDUCER (IEDUCER/VEDUCER) VALIDATION PASSED")
        print()
        print("Notes:")
        print("- IEDUCER/VEDUCER correctly implement Reference baseline governing relations")
        print("- Flow is continuous (U1_out = U1_in) for series transducers")
        print("- Pressure changes according to τ'*I1 - Z*U1")
        print("- Self-consistency between current- and voltage-driven modes verified")
        print("- Equivalent coefficients produce same results as IESPEAKER")
        print("- No reference file available for direct comparison")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

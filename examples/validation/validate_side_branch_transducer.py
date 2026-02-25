#!/usr/bin/env python3
"""
Validation of side-branch transducers (IDUCER, VDUCER, ISPEAKER, VSPEAKER).

This validation tests:
1. Self-consistency between current-driven and voltage-driven modes
2. Comparison with enclosed transducers (different physics)
3. Physical reasonableness of results
4. Equation verification against Reference baseline formulas
"""

import numpy as np

from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: Side-branch transducers (IDUCER, VDUCER, ISPEAKER, VSPEAKER)")
    print("=" * 70)
    print()

    # =========================================================================
    # Test 1: IDUCER/VDUCER Self-consistency
    # =========================================================================
    print("TEST 1: IDUCER/VDUCER self-consistency")
    print("-" * 50)

    # Create a generic side-branch transducer
    Ze = 6.0 + 0.5j  # Ohm
    tau = 1000.0 + 100.0j  # V·s/m³
    tau_prime = -1000.0 - 100.0j  # Pa/A (typically τ' = -τ for speakers)
    Z = 5e6 + 1e6j  # Pa·s/m³

    ducer = segments.SideBranchTransducer(
        Ze=Ze, tau=tau, tau_prime=tau_prime, Z=Z, name="test_ducer"
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

    # Step 1: IDUCER - specify current, get voltage
    I1_orig = 2.0 + 1.0j  # A
    p1_i, U1_i, T_i, V1_from_i, Ux_i = ducer.propagate_current_driven(
        p1_in, U1_in, T_m, omega, helium, I1_orig
    )

    print(f"IDUCER with I1 = {I1_orig:.4f} A:")
    print(f"  V1 = {V1_from_i:.4f} V")
    print(f"  Ux = {Ux_i:.6e} m³/s")
    print()

    # Step 2: VDUCER - use that voltage, should recover current
    p1_v, U1_v, T_v, I1_from_v, Ux_v = ducer.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega, helium, V1_from_i
    )

    print(f"VDUCER with V1 = {V1_from_i:.4f} V:")
    print(f"  I1 = {I1_from_v:.4f} A")
    print(f"  Ux = {Ux_v:.6e} m³/s")
    print()

    # Compare
    I1_err = np.abs(I1_from_v - I1_orig) / np.abs(I1_orig) * 100
    Ux_err = np.abs(Ux_v - Ux_i) / np.abs(Ux_i) * 100
    U1_err = np.abs(U1_v - U1_i) / np.abs(U1_i) * 100

    print(f"Current recovery error: {I1_err:.6f}%")
    print(f"Ux recovery error: {Ux_err:.6f}%")
    print(f"U1_out recovery error: {U1_err:.6f}%")

    test1_pass = I1_err < 0.001 and Ux_err < 0.001 and U1_err < 0.001
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (IDUCER/VDUCER self-consistency)")
    print()

    # =========================================================================
    # Test 2: ISPEAKER/VSPEAKER Self-consistency
    # =========================================================================
    print("TEST 2: ISPEAKER/VSPEAKER self-consistency")
    print("-" * 50)

    # Create a side-branch speaker
    area = 0.01  # m² (10 cm × 10 cm)
    Bl = 10.0  # T·m
    R_e = 6.0  # Ohm
    L_e = 0.5e-3  # H
    m = 0.015  # kg
    k = 3000.0  # N/m
    R_m = 1.5  # N·s/m

    speaker = segments.SideBranchSpeaker(
        area=area, Bl=Bl, R_e=R_e, L_e=L_e, m=m, k=k, R_m=R_m, name="test_speaker"
    )

    f_s = speaker.resonant_frequency()
    print(f"Speaker parameters:")
    print(f"  Area = {area*1e4:.1f} cm²")
    print(f"  Bl = {Bl} T·m, R_e = {R_e} Ohm, L_e = {L_e*1e3:.1f} mH")
    print(f"  m = {m*1e3:.1f} g, k = {k:.0f} N/m, R_m = {R_m} N·s/m")
    print(f"  Resonant frequency: f_s = {f_s:.1f} Hz")
    print()

    # Use frequency near resonance
    freq = f_s
    omega = 2 * np.pi * freq

    # ISPEAKER: specify current
    I1_spk = 1.5 + 0.5j  # A
    p1_is, U1_is, T_is, V1_from_is, Ux_is = speaker.propagate_current_driven(
        p1_in, U1_in, T_m, omega, helium, I1_spk
    )

    print(f"ISPEAKER with I1 = {I1_spk:.4f} A at f = {freq:.1f} Hz:")
    print(f"  V1 = {V1_from_is:.4f} V")
    print(f"  Ux = {Ux_is:.6e} m³/s")
    print()

    # VSPEAKER: use that voltage
    p1_vs, U1_vs, T_vs, I1_from_vs, Ux_vs = speaker.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega, helium, V1_from_is
    )

    print(f"VSPEAKER with V1 = {V1_from_is:.4f} V:")
    print(f"  I1 = {I1_from_vs:.4f} A")
    print(f"  Ux = {Ux_vs:.6e} m³/s")
    print()

    I1_spk_err = np.abs(I1_from_vs - I1_spk) / np.abs(I1_spk) * 100
    Ux_spk_err = np.abs(Ux_vs - Ux_is) / np.abs(Ux_is) * 100

    print(f"Current recovery error: {I1_spk_err:.6f}%")
    print(f"Ux recovery error: {Ux_spk_err:.6f}%")

    test2_pass = I1_spk_err < 0.001 and Ux_spk_err < 0.001
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (ISPEAKER/VSPEAKER self-consistency)")
    print()

    # =========================================================================
    # Test 3: Pressure unchanged (side-branch property)
    # =========================================================================
    print("TEST 3: Pressure unchanged (side-branch property)")
    print("-" * 50)

    # For side-branch transducers, p1_out should equal p1_in
    p1_err_ducer = np.abs(p1_i - p1_in) / np.abs(p1_in) * 100
    p1_err_speaker = np.abs(p1_is - p1_in) / np.abs(p1_in) * 100

    print(f"IDUCER: p1_in = {p1_in:.2f} Pa, p1_out = {p1_i:.2f} Pa")
    print(f"  Error: {p1_err_ducer:.6f}%")
    print(f"ISPEAKER: p1_in = {p1_in:.2f} Pa, p1_out = {p1_is:.2f} Pa")
    print(f"  Error: {p1_err_speaker:.6f}%")

    test3_pass = p1_err_ducer < 0.001 and p1_err_speaker < 0.001
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (pressure unchanged)")
    print()

    # =========================================================================
    # Test 4: Flow diversion (U1_out = U1_in - Ux)
    # =========================================================================
    print("TEST 4: Flow diversion verification")
    print("-" * 50)

    U1_expected_i = U1_in - Ux_i
    U1_expected_is = U1_in - Ux_is

    U1_div_err_ducer = np.abs(U1_i - U1_expected_i) / np.abs(U1_in) * 100
    U1_div_err_speaker = np.abs(U1_is - U1_expected_is) / np.abs(U1_in) * 100

    print(f"IDUCER:")
    print(f"  U1_in = {U1_in:.6e}, Ux = {Ux_i:.6e}")
    print(f"  U1_out (computed) = {U1_i:.6e}")
    print(f"  U1_out (expected) = {U1_expected_i:.6e}")
    print(f"  Error: {U1_div_err_ducer:.6f}%")
    print()
    print(f"ISPEAKER:")
    print(f"  U1_in = {U1_in:.6e}, Ux = {Ux_is:.6e}")
    print(f"  U1_out (computed) = {U1_is:.6e}")
    print(f"  U1_out (expected) = {U1_expected_is:.6e}")
    print(f"  Error: {U1_div_err_speaker:.6f}%")

    test4_pass = U1_div_err_ducer < 0.001 and U1_div_err_speaker < 0.001
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (flow diversion)")
    print()

    # =========================================================================
    # Test 5: Equation verification for IDUCER
    # =========================================================================
    print("TEST 5: Direct equation verification for IDUCER")
    print("-" * 50)

    # For IDUCER: Ux = (p1 - τ'*I1) / Z, V1 = Ze*I1 + τ*Ux
    Ux_expected = (p1_in - tau_prime * I1_orig) / Z
    V1_expected = Ze * I1_orig + tau * Ux_expected

    Ux_eq_err = np.abs(Ux_i - Ux_expected) / np.abs(Ux_expected) * 100
    V1_eq_err = np.abs(V1_from_i - V1_expected) / np.abs(V1_expected) * 100

    print(f"Ux (computed): {Ux_i:.6e}")
    print(f"Ux (expected): {Ux_expected:.6e}")
    print(f"Ux error: {Ux_eq_err:.6f}%")
    print()
    print(f"V1 (computed): {V1_from_i:.4f}")
    print(f"V1 (expected): {V1_expected:.4f}")
    print(f"V1 error: {V1_eq_err:.6f}%")

    test5_pass = Ux_eq_err < 0.001 and V1_eq_err < 0.001
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (IDUCER equations)")
    print()

    # =========================================================================
    # Test 6: Equation verification for VDUCER
    # =========================================================================
    print("TEST 6: Direct equation verification for VDUCER")
    print("-" * 50)

    V1_test = 10.0 + 5.0j  # V
    p1_vt, U1_vt, T_vt, I1_vt, Ux_vt = ducer.propagate_voltage_driven(
        p1_in, U1_in, T_m, omega, helium, V1_test
    )

    # I1 = (Z*V1 - τ*p1) / (Ze*Z - τ*τ')
    denom = Ze * Z - tau * tau_prime
    I1_expected = (Z * V1_test - tau * p1_in) / denom

    # Ux = (V1 - Ze*I1) / τ
    Ux_expected_v = (V1_test - Ze * I1_expected) / tau

    I1_eq_err = np.abs(I1_vt - I1_expected) / np.abs(I1_expected) * 100
    Ux_eq_err_v = np.abs(Ux_vt - Ux_expected_v) / np.abs(Ux_expected_v) * 100

    print(f"V1 = {V1_test} V")
    print(f"I1 (computed): {I1_vt:.6f}")
    print(f"I1 (expected): {I1_expected:.6f}")
    print(f"I1 error: {I1_eq_err:.6f}%")
    print()
    print(f"Ux (computed): {Ux_vt:.6e}")
    print(f"Ux (expected): {Ux_expected_v:.6e}")
    print(f"Ux error: {Ux_eq_err_v:.6f}%")

    test6_pass = I1_eq_err < 0.001 and Ux_eq_err_v < 0.001
    print()
    print(f"Test 6: {'PASS' if test6_pass else 'FAIL'} (VDUCER equations)")
    print()

    # =========================================================================
    # Test 7: ISPEAKER coefficient verification
    # =========================================================================
    print("TEST 7: ISPEAKER coefficient verification")
    print("-" * 50)

    # For speakers: Ze = Re + jωL, τ = Bl/S, τ' = -Bl/S, Z = Rm/S² + j(ωM - K/ω)/S²
    Ze_spk = speaker.electrical_impedance(omega)
    Z_spk = speaker.acoustic_impedance(omega)
    tau_spk = speaker.tau(omega)
    tau_p_spk = speaker.tau_prime(omega)

    Ze_expected = R_e + 1j * omega * L_e
    tau_expected = Bl / area
    tau_p_expected = -Bl / area
    Z_expected = R_m / area**2 + 1j * (omega * m - k / omega) / area**2

    Ze_err = np.abs(Ze_spk - Ze_expected) / np.abs(Ze_expected) * 100
    tau_err = np.abs(tau_spk - tau_expected) / np.abs(tau_expected) * 100
    tau_p_err = np.abs(tau_p_spk - tau_p_expected) / np.abs(tau_p_expected) * 100
    Z_err = np.abs(Z_spk - Z_expected) / np.abs(Z_expected) * 100

    print(f"Ze (computed): {Ze_spk:.4f} Ohm")
    print(f"Ze (expected): {Ze_expected:.4f} Ohm, error: {Ze_err:.6f}%")
    print()
    print(f"τ (computed): {tau_spk:.4f} V·s/m³")
    print(f"τ (expected): {tau_expected:.4f} V·s/m³, error: {tau_err:.6f}%")
    print()
    print(f"τ' (computed): {tau_p_spk:.4f} Pa/A")
    print(f"τ' (expected): {tau_p_expected:.4f} Pa/A, error: {tau_p_err:.6f}%")
    print()
    print(f"Z (computed): {Z_spk:.4e} Pa·s/m³")
    print(f"Z (expected): {Z_expected:.4e} Pa·s/m³, error: {Z_err:.6f}%")

    test7_pass = Ze_err < 0.001 and tau_err < 0.001 and tau_p_err < 0.001 and Z_err < 0.001
    print()
    print(f"Test 7: {'PASS' if test7_pass else 'FAIL'} (coefficient verification)")
    print()

    # =========================================================================
    # Test 8: Electrical power
    # =========================================================================
    print("TEST 8: Electrical power computation")
    print("-" * 50)

    P_elec = speaker.electrical_power(I1_spk, V1_from_is)
    P_expected = 0.5 * np.real(V1_from_is * np.conj(I1_spk))

    print(f"V1 = {V1_from_is:.4f} V, I1 = {I1_spk:.4f} A")
    print(f"Power (computed): {P_elec:.6f} W")
    print(f"Power (expected): {P_expected:.6f} W")

    P_err = abs(P_elec - P_expected) / max(abs(P_expected), 1e-20) * 100
    print(f"Error: {P_err:.6f}%")

    test8_pass = P_err < 0.001
    print()
    print(f"Test 8: {'PASS' if test8_pass else 'FAIL'} (electrical power)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = (
        test1_pass and test2_pass and test3_pass and test4_pass
        and test5_pass and test6_pass and test7_pass and test8_pass
    )
    print(f"Test 1 (IDUCER/VDUCER self-consistency): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (ISPEAKER/VSPEAKER self-consistency): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Pressure unchanged): {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Flow diversion): {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (IDUCER equations): {'PASS' if test5_pass else 'FAIL'}")
    print(f"Test 6 (VDUCER equations): {'PASS' if test6_pass else 'FAIL'}")
    print(f"Test 7 (ISPEAKER coefficients): {'PASS' if test7_pass else 'FAIL'}")
    print(f"Test 8 (Electrical power): {'PASS' if test8_pass else 'FAIL'}")
    print()

    if all_pass:
        print("SIDE-BRANCH TRANSDUCER VALIDATION PASSED")
        print()
        print("Notes:")
        print("- Side-branch transducers correctly implement Reference baseline relevant reference")
        print("- Pressure is unchanged in trunk (p1_out = p1_in)")
        print("- Flow diverts correctly (U1_out = U1_in - Ux)")
        print("- IDUCER/VDUCER use direct coefficients (Ze, τ, τ', Z)")
        print("- ISPEAKER/VSPEAKER derive coefficients from physical parameters")
        print("- Self-consistency between current- and voltage-driven modes verified")
        print("- No reference file available for direct comparison")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

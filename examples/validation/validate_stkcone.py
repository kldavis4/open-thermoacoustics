#!/usr/bin/env python3
"""
Validation of STKCONE (tapered boundary-layer pulse tube).

This validation tests:
1. STKCONE with uniform geometry should match STKDUCT
2. Tapering behavior produces physically reasonable results
3. Temperature gradient handling works correctly

Since no specific Reference baseline STKCONE example is available, we validate by:
- Comparing uniform STKCONE to STKDUCT (should match exactly)
- Verifying tapered STKCONE produces reasonable results
"""

import numpy as np
from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: STKCONE (tapered boundary-layer pulse tube)")
    print("=" * 70)
    print()

    # =========================================================================
    # Test 1: STKCONE with uniform geometry should match STKDUCT
    # =========================================================================
    print("TEST 1: STKCONE with uniform geometry vs STKDUCT")
    print("-" * 50)

    # System parameters
    mean_P = 3.0e6  # Pa (3 MPa helium)
    freq = 50.0  # Hz
    omega = 2 * np.pi * freq

    # Geometry: circular tube, D = 0.03 m
    radius = 0.015  # m
    area = np.pi * radius**2
    perimeter = 2 * np.pi * radius
    length = 0.1  # m

    # Temperature gradient
    T_cold = 300.0  # K
    T_hot = 80.0  # K

    helium = gas.Helium(mean_pressure=mean_P)

    # Create STKDUCT
    stkduct = segments.StackDuct(
        length=length,
        area=area,
        perimeter=perimeter,
        T_cold=T_cold,
        T_hot=T_hot,
        name="stkduct",
    )

    # Create STKCONE with uniform geometry (same area/perimeter at both ends)
    stkcone_uniform = segments.StackCone(
        length=length,
        area_in=area,
        area_out=area,  # same as inlet
        perimeter_in=perimeter,
        perimeter_out=perimeter,  # same as inlet
        T_cold=T_cold,
        T_hot=T_hot,
        name="stkcone_uniform",
    )

    # Input conditions
    p1_in = 50000.0 + 10000.0j  # Pa
    U1_in = 0.001 + 0.0002j  # m³/s
    T_in = T_cold

    print(f"Input: |p1| = {np.abs(p1_in):.0f} Pa, |U1| = {np.abs(U1_in)*1000:.3f} L/s")
    print(f"       T_in = {T_in:.0f} K, T_out = {T_hot:.0f} K")
    print()

    # Propagate through both
    p1_duct, U1_duct, T_duct = stkduct.propagate(p1_in, U1_in, T_in, omega, helium)
    p1_cone, U1_cone, T_cone = stkcone_uniform.propagate(p1_in, U1_in, T_in, omega, helium)

    # Compare results
    p1_diff = np.abs(p1_cone - p1_duct) / np.abs(p1_duct) * 100
    U1_diff = np.abs(U1_cone - U1_duct) / np.abs(U1_duct) * 100
    T_diff = abs(T_cone - T_duct)

    print(f"STKDUCT:  |p1| = {np.abs(p1_duct):.1f} Pa, |U1| = {np.abs(U1_duct)*1000:.4f} L/s, T = {T_duct:.2f} K")
    print(f"STKCONE:  |p1| = {np.abs(p1_cone):.1f} Pa, |U1| = {np.abs(U1_cone)*1000:.4f} L/s, T = {T_cone:.2f} K")
    print(f"Diff:     |p1| = {p1_diff:.4f}%, |U1| = {U1_diff:.4f}%, T = {T_diff:.4f} K")
    print()

    test1_pass = p1_diff < 0.01 and U1_diff < 0.01 and T_diff < 0.001
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (uniform STKCONE matches STKDUCT)")
    print()

    # =========================================================================
    # Test 2: Tapered geometry produces reasonable results
    # =========================================================================
    print("TEST 2: Tapered STKCONE behavior")
    print("-" * 50)

    # Expanding tube: radius goes from 0.01 m to 0.02 m
    r_in = 0.01
    r_out = 0.02
    area_in = np.pi * r_in**2
    area_out = np.pi * r_out**2
    perim_in = 2 * np.pi * r_in
    perim_out = 2 * np.pi * r_out

    stkcone_expand = segments.StackCone(
        length=length,
        area_in=area_in,
        area_out=area_out,
        perimeter_in=perim_in,
        perimeter_out=perim_out,
        T_cold=T_cold,
        T_hot=T_hot,
        name="stkcone_expand",
    )

    # Contracting tube
    stkcone_contract = segments.StackCone(
        length=length,
        area_in=area_out,  # larger end first
        area_out=area_in,  # smaller end at outlet
        perimeter_in=perim_out,
        perimeter_out=perim_in,
        T_cold=T_cold,
        T_hot=T_hot,
        name="stkcone_contract",
    )

    print(f"Expanding:   A_in = {area_in*1e4:.2f} cm², A_out = {area_out*1e4:.2f} cm²")
    print(f"Contracting: A_in = {area_out*1e4:.2f} cm², A_out = {area_in*1e4:.2f} cm²")
    print()

    # Propagate
    p1_exp, U1_exp, T_exp = stkcone_expand.propagate(p1_in, U1_in, T_in, omega, helium)
    p1_con, U1_con, T_con = stkcone_contract.propagate(p1_in, U1_in, T_in, omega, helium)

    print(f"Expanding:   |p1| = {np.abs(p1_exp):.1f} Pa, |U1| = {np.abs(U1_exp)*1000:.4f} L/s")
    print(f"Contracting: |p1| = {np.abs(p1_con):.1f} Pa, |U1| = {np.abs(U1_con)*1000:.4f} L/s")
    print()

    # Physical expectation: expanding area should have different impedance characteristics
    # than contracting area, so results should differ
    test2_pass = True
    # Check that results are different (tapering has an effect)
    p1_diff_taper = np.abs(p1_exp - p1_con) / np.abs(p1_in) * 100
    print(f"Difference: |p1_exp - p1_con| / |p1_in| = {p1_diff_taper:.3f}%")
    if p1_diff_taper < 0.001:  # Less than 0.001% would indicate no tapering effect
        print("WARNING: Expanding and contracting gave nearly identical pressure")
        test2_pass = False

    # Check that results are physically reasonable (no NaN, finite values)
    if not (np.isfinite(p1_exp) and np.isfinite(U1_exp)):
        print("ERROR: Non-finite values in expanding case")
        test2_pass = False
    if not (np.isfinite(p1_con) and np.isfinite(U1_con)):
        print("ERROR: Non-finite values in contracting case")
        test2_pass = False

    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (tapering produces distinct, finite results)")
    print()

    # =========================================================================
    # Test 3: Position-dependent functions work correctly
    # =========================================================================
    print("TEST 3: Position-dependent area and temperature")
    print("-" * 50)

    # Check area at various positions
    for x_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = x_frac * length
        A = stkcone_expand.area_at(x)
        T = stkcone_expand.temperature_at(x, T_cold)
        A_expected = area_in + (area_out - area_in) * x_frac
        T_expected = T_cold + (T_hot - T_cold) * x_frac
        print(f"  x = {x_frac:.2f}L: A = {A*1e4:.3f} cm² (expected {A_expected*1e4:.3f}), T = {T:.1f} K (expected {T_expected:.1f})")

    test3_pass = True
    # Verify at x=0 and x=L
    if abs(stkcone_expand.area_at(0) - area_in) > 1e-10:
        test3_pass = False
    if abs(stkcone_expand.area_at(length) - area_out) > 1e-10:
        test3_pass = False
    if abs(stkcone_expand.temperature_at(0, T_cold) - T_cold) > 1e-10:
        test3_pass = False
    if abs(stkcone_expand.temperature_at(length, T_cold) - T_hot) > 1e-10:
        test3_pass = False

    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (position-dependent functions correct)")
    print()

    # =========================================================================
    # Test 4: Compare to STKDUCT reference from optr reference case
    # =========================================================================
    print("TEST 4: STKCONE (uniform) vs STKDUCT reference from optr reference case")
    print("-" * 50)

    # Use optr reference case parameters for STKDUCT validation
    # From validate_optr_stkduct.py or similar
    optr_mean_P = 2.5e6  # Pa
    optr_freq = 40.0  # Hz
    optr_omega = 2 * np.pi * optr_freq

    # STKDUCT segment 4 parameters (approximate)
    optr_area = 7.669e-5  # m²
    optr_perimeter = np.sqrt(4 * np.pi * optr_area)  # circular assumption
    optr_length = 0.15  # m

    optr_helium = gas.Helium(mean_pressure=optr_mean_P)

    # Create both segments with same parameters
    optr_stkduct = segments.StackDuct(
        length=optr_length,
        area=optr_area,
        perimeter=optr_perimeter,
        name="optr_stkduct",
    )

    optr_stkcone = segments.StackCone(
        length=optr_length,
        area_in=optr_area,
        area_out=optr_area,
        perimeter_in=optr_perimeter,
        perimeter_out=optr_perimeter,
        name="optr_stkcone",
    )

    # Input from typical optr conditions
    optr_p1_in = 1.5e5 + 0.5e5j
    optr_U1_in = 5e-5 + 2e-5j
    optr_T_in = 300.0

    p1_sd, U1_sd, T_sd = optr_stkduct.propagate(optr_p1_in, optr_U1_in, optr_T_in, optr_omega, optr_helium)
    p1_sc, U1_sc, T_sc = optr_stkcone.propagate(optr_p1_in, optr_U1_in, optr_T_in, optr_omega, optr_helium)

    p1_err = np.abs(p1_sc - p1_sd) / np.abs(p1_sd) * 100
    U1_err = np.abs(U1_sc - U1_sd) / np.abs(U1_sd) * 100

    print(f"STKDUCT: |p1| = {np.abs(p1_sd):.1f} Pa, |U1| = {np.abs(U1_sd)*1e6:.2f} mm³/s")
    print(f"STKCONE: |p1| = {np.abs(p1_sc):.1f} Pa, |U1| = {np.abs(U1_sc)*1e6:.2f} mm³/s")
    print(f"Error:   |p1| = {p1_err:.6f}%, |U1| = {U1_err:.6f}%")

    test4_pass = p1_err < 0.001 and U1_err < 0.001
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (STKCONE matches STKDUCT at optr conditions)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"Test 1 (Uniform STKCONE = STKDUCT):      {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Tapering produces distinct results): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Position-dependent functions):   {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (OPTR conditions match):          {'PASS' if test4_pass else 'FAIL'}")
    print()

    if all_pass:
        print("STKCONE VALIDATION PASSED")
        print()
        print("Notes:")
        print("- STKCONE correctly reduces to STKDUCT for uniform geometry")
        print("- Tapering produces physically reasonable, distinct results")
        print("- Position-dependent area and temperature work correctly")
        print("- No reference file available for tapered case")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

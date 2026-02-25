#!/usr/bin/env python3
"""
Validation of VXQ2 (two-pass variable heat flux heat exchanger).

This validation tests:
1. Basic instantiation and propagation
2. Two-pass heating with different heat inputs
3. Mixed heating/cooling passes
4. Comparison with two sequential VXQ1 segments
5. Parameter validation
"""

import numpy as np

from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: VXQ2 (Two-Pass Variable Heat Flux Heat Exchanger)")
    print("=" * 70)
    print()

    # Use helium at 3 MPa
    helium = gas.Helium(mean_pressure=3e6)
    T_m = 300.0  # Start at 300 K
    omega = 2 * np.pi * 50  # 50 Hz

    # =========================================================================
    # Test 1: Basic instantiation and propagation
    # =========================================================================
    print("TEST 1: Basic instantiation and propagation")
    print("-" * 50)

    # Create VXQ2 with two heating passes
    vxq2 = segments.VXQ2(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.005,
        length_pass1=0.03,  # First pass
        length_pass2=0.03,  # Second pass
        length_tubesheet2=0.005,
        heat_power_1=50.0,  # 50 W in pass 1
        heat_power_2=100.0,  # 100 W in pass 2
        name="vxq2_test",
    )

    print(f"VXQ2 parameters:")
    print(f"  Total length = {vxq2.length*1000:.1f} mm")
    print(f"  Tubesheet 1 = 5 mm")
    print(f"  Pass 1 = 30 mm (heat = 50 W)")
    print(f"  Pass 2 = 30 mm (heat = 100 W)")
    print(f"  Tubesheet 2 = 5 mm")
    print()

    # Input acoustic state
    p1_in = 50000.0 + 10000.0j  # Pa
    U1_in = 5e-5 + 1e-5j  # m³/s

    # Propagate
    p1_out, U1_out, T_out = vxq2.propagate(p1_in, U1_in, T_m, omega, helium)

    print(f"Input:  T_m = {T_m:.2f} K")
    print(f"Output: T_m = {T_out:.2f} K")
    print(f"Total heat = {vxq2.heat_power_1 + vxq2.heat_power_2:.1f} W")
    print()
    print(f"Acoustic propagation:")
    print(f"  |p1_in|  = {np.abs(p1_in)/1000:.4f} kPa")
    print(f"  |p1_out| = {np.abs(p1_out)/1000:.4f} kPa")

    # Temperature should increase (positive total heat)
    test1_pass = T_out > T_m
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (temperature increased)")
    print()

    # =========================================================================
    # Test 2: Mixed heating and cooling passes
    # =========================================================================
    print("TEST 2: Mixed heating and cooling passes")
    print("-" * 50)

    vxq2_mixed = segments.VXQ2(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_pass1=0.05,
        length_pass2=0.05,
        length_tubesheet2=0.002,
        heat_power_1=100.0,  # Heating in pass 1
        heat_power_2=-50.0,  # Cooling in pass 2
        name="mixed_passes",
    )

    p1_test = 30000.0 + 0j
    U1_test = 3e-5 + 5e-6j

    _, _, T_mixed = vxq2_mixed.propagate(p1_test, U1_test, T_m, omega, helium)

    print(f"  T_in = {T_m:.2f} K")
    print(f"  Pass 1: +100 W (heating)")
    print(f"  Pass 2: -50 W (cooling)")
    print(f"  Net heat: +50 W")
    print(f"  T_out = {T_mixed:.2f} K")

    # Net positive heat, so should still heat up
    test2_pass = T_mixed > T_m
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (net heating correct)")
    print()

    # =========================================================================
    # Test 3: Comparison with sequential VXQ1 segments
    # =========================================================================
    print("TEST 3: Comparison with sequential VXQ1 segments")
    print("-" * 50)

    T_start = 300.0

    # VXQ2 with two passes
    vxq2_compare = segments.VXQ2(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_pass1=0.04,
        length_pass2=0.04,
        length_tubesheet2=0.002,
        heat_power_1=60.0,
        heat_power_2=40.0,
        name="vxq2",
    )

    # Two sequential VXQ1 segments
    vxq1_pass1 = segments.VXQ1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_heat_transfer=0.04,
        length_tubesheet2=0.0,  # No tubesheet between passes
        heat_power=60.0,
        name="vxq1_pass1",
    )

    vxq1_pass2 = segments.VXQ1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.0,  # No tubesheet between passes
        length_heat_transfer=0.04,
        length_tubesheet2=0.002,
        heat_power=40.0,
        name="vxq1_pass2",
    )

    # VXQ2 propagation
    p1_vxq2, U1_vxq2, T_vxq2 = vxq2_compare.propagate(p1_test, U1_test, T_start, omega, helium)

    # Sequential VXQ1 propagation
    p1_mid, U1_mid, T_mid = vxq1_pass1.propagate(p1_test, U1_test, T_start, omega, helium)
    p1_seq, U1_seq, T_seq = vxq1_pass2.propagate(p1_mid, U1_mid, T_mid, omega, helium)

    print(f"VXQ2 result:")
    print(f"  T_out = {T_vxq2:.2f} K")
    print(f"  |p1_out| = {np.abs(p1_vxq2)/1000:.4f} kPa")

    print(f"Sequential VXQ1 result:")
    print(f"  T_mid = {T_mid:.2f} K (after pass 1)")
    print(f"  T_out = {T_seq:.2f} K (after pass 2)")
    print(f"  |p1_out| = {np.abs(p1_seq)/1000:.4f} kPa")

    # Results should be similar
    T_diff = abs(T_vxq2 - T_seq)
    p_diff = abs(np.abs(p1_vxq2) - np.abs(p1_seq)) / np.abs(p1_vxq2) * 100

    print(f"  Temperature difference: {T_diff:.2f} K")
    print(f"  Pressure difference: {p_diff:.2f}%")

    # Allow reasonable tolerance for tubesheet differences
    test3_pass = T_diff < 10.0 and p_diff < 5.0
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (similar to sequential VXQ1)")
    print()

    # =========================================================================
    # Test 4: Parameter validation
    # =========================================================================
    print("TEST 4: Parameter validation")
    print("-" * 50)

    errors_caught = 0

    try:
        segments.VXQ2(
            area=1e-3,
            gas_area_fraction=0.4,
            solid_area_fraction=0.3,
            hydraulic_radius=0.5e-3,
            length_tubesheet1=0.002,
            length_pass1=-0.04,  # Invalid: negative
            length_pass2=0.04,
            length_tubesheet2=0.002,
            heat_power_1=50.0,
            heat_power_2=50.0,
        )
    except ValueError:
        errors_caught += 1
        print("  Caught invalid length_pass1 < 0")

    try:
        segments.VXQ2(
            area=1e-3,
            gas_area_fraction=0.4,
            solid_area_fraction=0.3,
            hydraulic_radius=0.5e-3,
            length_tubesheet1=0.002,
            length_pass1=0.04,
            length_pass2=-0.04,  # Invalid: negative
            length_tubesheet2=0.002,
            heat_power_1=50.0,
            heat_power_2=50.0,
        )
    except ValueError:
        errors_caught += 1
        print("  Caught invalid length_pass2 < 0")

    test4_pass = errors_caught == 2
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (parameter validation)")
    print()

    # =========================================================================
    # Test 5: Equal heat in both passes
    # =========================================================================
    print("TEST 5: Equal heat in both passes")
    print("-" * 50)

    heat_same = 75.0
    T_in_same = 300.0

    vxq2_same = segments.VXQ2(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_pass1=0.04,
        length_pass2=0.04,
        length_tubesheet2=0.002,
        heat_power_1=heat_same,
        heat_power_2=heat_same,
        name="same_heat",
    )

    _, _, T_same_out = vxq2_same.propagate(p1_test, U1_test, T_in_same, omega, helium)

    print(f"  T_in = {T_in_same:.2f} K")
    print(f"  Heat_1 = Heat_2 = {heat_same:.1f} W")
    print(f"  Total heat = {2*heat_same:.1f} W")
    print(f"  T_out = {T_same_out:.2f} K")

    # Should heat up with total 150 W
    test5_pass = T_same_out > T_in_same
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (equal heats - temperature increased)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass

    print(f"Test 1 (Basic propagation):       {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Mixed heating/cooling):   {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Compare to VXQ1):         {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Parameter validation):    {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Equal heats):             {'PASS' if test5_pass else 'FAIL'}")
    print()

    if all_pass:
        print("VXQ2 VALIDATION PASSED")
        print()
        print("Notes:")
        print("- VXQ2 models two-pass heat exchangers with fixed heat per pass")
        print("- Each pass can have different heat power (positive or negative)")
        print("- Results similar to sequential VXQ1 segments")
        print("- No reference file available for direct comparison")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

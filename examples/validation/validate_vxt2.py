#!/usr/bin/env python3
"""
Validation of VXT2 (two-pass variable temperature heat exchanger).

This validation tests:
1. Basic instantiation and propagation
2. Two-pass temperature evolution (different solid temperatures)
3. Comparison with two sequential VXT1 segments
4. Parameter validation
"""

import numpy as np

from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: VXT2 (Two-Pass Variable Temperature Heat Exchanger)")
    print("=" * 70)
    print()

    # Use helium at 3 MPa
    helium = gas.Helium(mean_pressure=3e6)
    T_m = 400.0  # Start at 400 K
    omega = 2 * np.pi * 50  # 50 Hz

    # =========================================================================
    # Test 1: Basic instantiation and propagation
    # =========================================================================
    print("TEST 1: Basic instantiation and propagation")
    print("-" * 50)

    # Create VXT2 with two cooling stages
    vxt2 = segments.VXT2(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.005,
        length_pass1=0.03,  # First pass: cool to 350 K
        length_pass2=0.03,  # Second pass: cool to 300 K
        length_tubesheet2=0.005,
        solid_temperature_1=350.0,
        solid_temperature_2=300.0,
        name="vxt2_test",
    )

    print(f"VXT2 parameters:")
    print(f"  Total length = {vxt2.length*1000:.1f} mm")
    print(f"  Tubesheet 1 = 5 mm")
    print(f"  Pass 1 = 30 mm (T_solid = 350 K)")
    print(f"  Pass 2 = 30 mm (T_solid = 300 K)")
    print(f"  Tubesheet 2 = 5 mm")
    print()

    # Input acoustic state
    p1_in = 50000.0 + 10000.0j  # Pa
    U1_in = 5e-5 + 1e-5j  # m³/s

    # Propagate
    p1_out, U1_out, T_out = vxt2.propagate(p1_in, U1_in, T_m, omega, helium)

    print(f"Input:  T_m = {T_m:.2f} K")
    print(f"Output: T_m = {T_out:.2f} K")
    print(f"T_solid_1 = {vxt2.solid_temperature_1:.2f} K")
    print(f"T_solid_2 = {vxt2.solid_temperature_2:.2f} K")
    print()
    print(f"Acoustic propagation:")
    print(f"  |p1_in|  = {np.abs(p1_in)/1000:.4f} kPa")
    print(f"  |p1_out| = {np.abs(p1_out)/1000:.4f} kPa")

    # Temperature should move toward T_solid_2 (final pass)
    temp_direction_correct = T_out < T_m  # Should cool down

    test1_pass = temp_direction_correct
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (temperature decreased)")
    print()

    # =========================================================================
    # Test 2: Two passes with different temperatures
    # =========================================================================
    print("TEST 2: Two-pass heating (cold to hot)")
    print("-" * 50)

    T_cold = 250.0

    vxt2_heat = segments.VXT2(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_pass1=0.05,
        length_pass2=0.05,
        length_tubesheet2=0.002,
        solid_temperature_1=300.0,  # First pass: warm to 300 K
        solid_temperature_2=350.0,  # Second pass: warm to 350 K
        name="heating_stages",
    )

    p1_test = 30000.0 + 0j
    U1_test = 3e-5 + 5e-6j

    _, _, T_heated = vxt2_heat.propagate(p1_test, U1_test, T_cold, omega, helium)

    print(f"  T_in = {T_cold:.2f} K")
    print(f"  T_solid_1 = 300 K (first pass)")
    print(f"  T_solid_2 = 350 K (second pass)")
    print(f"  T_out = {T_heated:.2f} K")

    # Temperature should increase
    test2_pass = T_heated > T_cold
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (heating direction correct)")
    print()

    # =========================================================================
    # Test 3: Comparison with two sequential VXT1 segments
    # =========================================================================
    print("TEST 3: Comparison with sequential VXT1 segments")
    print("-" * 50)

    T_start = 400.0

    # VXT2 with two passes
    vxt2_compare = segments.VXT2(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_pass1=0.04,
        length_pass2=0.04,
        length_tubesheet2=0.002,
        solid_temperature_1=350.0,
        solid_temperature_2=300.0,
        name="vxt2",
    )

    # Two sequential VXT1 segments
    vxt1_pass1 = segments.VXT1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_heat_transfer=0.04,
        length_tubesheet2=0.0,  # No tubesheet between passes
        solid_temperature=350.0,
        name="vxt1_pass1",
    )

    vxt1_pass2 = segments.VXT1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.0,  # No tubesheet between passes
        length_heat_transfer=0.04,
        length_tubesheet2=0.002,
        solid_temperature=300.0,
        name="vxt1_pass2",
    )

    # VXT2 propagation
    p1_vxt2, U1_vxt2, T_vxt2 = vxt2_compare.propagate(p1_test, U1_test, T_start, omega, helium)

    # Sequential VXT1 propagation
    p1_mid, U1_mid, T_mid = vxt1_pass1.propagate(p1_test, U1_test, T_start, omega, helium)
    p1_seq, U1_seq, T_seq = vxt1_pass2.propagate(p1_mid, U1_mid, T_mid, omega, helium)

    print(f"VXT2 result:")
    print(f"  T_out = {T_vxt2:.2f} K")
    print(f"  |p1_out| = {np.abs(p1_vxt2)/1000:.4f} kPa")

    print(f"Sequential VXT1 result:")
    print(f"  T_mid = {T_mid:.2f} K (after pass 1)")
    print(f"  T_out = {T_seq:.2f} K (after pass 2)")
    print(f"  |p1_out| = {np.abs(p1_seq)/1000:.4f} kPa")

    # Results should be similar (not exact due to tubesheets)
    T_diff = abs(T_vxt2 - T_seq)
    p_diff = abs(np.abs(p1_vxt2) - np.abs(p1_seq)) / np.abs(p1_vxt2) * 100

    print(f"  Temperature difference: {T_diff:.2f} K")
    print(f"  Pressure difference: {p_diff:.2f}%")

    # Allow reasonable tolerance for tubesheet differences
    test3_pass = T_diff < 10.0 and p_diff < 5.0
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (similar to sequential VXT1)")
    print()

    # =========================================================================
    # Test 4: Parameter validation
    # =========================================================================
    print("TEST 4: Parameter validation")
    print("-" * 50)

    errors_caught = 0

    try:
        segments.VXT2(
            area=1e-3,
            gas_area_fraction=0.4,
            solid_area_fraction=0.3,
            hydraulic_radius=0.5e-3,
            length_tubesheet1=0.002,
            length_pass1=0.04,
            length_pass2=0.04,
            length_tubesheet2=0.002,
            solid_temperature_1=-100.0,  # Invalid
            solid_temperature_2=300.0,
        )
    except ValueError:
        errors_caught += 1
        print("  Caught invalid solid_temperature_1 < 0")

    try:
        segments.VXT2(
            area=1e-3,
            gas_area_fraction=0.4,
            solid_area_fraction=0.3,
            hydraulic_radius=0.5e-3,
            length_tubesheet1=0.002,
            length_pass1=0.04,
            length_pass2=0.04,
            length_tubesheet2=0.002,
            solid_temperature_1=300.0,
            solid_temperature_2=-100.0,  # Invalid
        )
    except ValueError:
        errors_caught += 1
        print("  Caught invalid solid_temperature_2 < 0")

    test4_pass = errors_caught == 2
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (parameter validation)")
    print()

    # =========================================================================
    # Test 5: Same solid temperatures (should behave like VXT1)
    # =========================================================================
    print("TEST 5: Same solid temperatures in both passes")
    print("-" * 50)

    T_solid_same = 300.0
    T_in_same = 400.0

    vxt2_same = segments.VXT2(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_pass1=0.04,
        length_pass2=0.04,
        length_tubesheet2=0.002,
        solid_temperature_1=T_solid_same,
        solid_temperature_2=T_solid_same,
        name="same_temps",
    )

    _, _, T_same_out = vxt2_same.propagate(p1_test, U1_test, T_in_same, omega, helium)

    print(f"  T_in = {T_in_same:.2f} K")
    print(f"  T_solid_1 = T_solid_2 = {T_solid_same:.2f} K")
    print(f"  T_out = {T_same_out:.2f} K")

    # Should cool toward T_solid
    test5_pass = T_same_out < T_in_same
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (same temps - monotonic cooling)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass

    print(f"Test 1 (Basic propagation):       {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Two-pass heating):        {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Compare to VXT1):         {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Parameter validation):    {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Same solid temps):        {'PASS' if test5_pass else 'FAIL'}")
    print()

    if all_pass:
        print("VXT2 VALIDATION PASSED")
        print()
        print("Notes:")
        print("- VXT2 models two-pass heat exchangers with different solid temperatures")
        print("- Each pass drives gas temperature toward its solid temperature")
        print("- Results similar to sequential VXT1 segments")
        print("- No reference file available for direct comparison")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

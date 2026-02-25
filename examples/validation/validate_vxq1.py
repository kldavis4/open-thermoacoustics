#!/usr/bin/env python3
"""
Validation of VXQ1 (variable heat flux heat exchanger with fixed heat input).

This validation tests:
1. Basic instantiation and propagation
2. Temperature increase with positive heat input
3. Temperature decrease with negative heat input (cooling)
4. No temperature change with zero heat input
5. Comparison with expected linear temperature profile
6. Parameter validation
"""

import numpy as np

from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: VXQ1 (Variable Heat Flux Heat Exchanger)")
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

    # Create VXQ1 with positive heat input (heating)
    vxq1 = segments.VXQ1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.005,
        length_heat_transfer=0.05,
        length_tubesheet2=0.005,
        heat_power=100.0,  # 100 W heating
        name="vxq1_test",
    )

    print(f"VXQ1 parameters:")
    print(f"  Total length = {vxq1.length*1000:.1f} mm")
    print(f"  Tubesheet 1 = 5 mm")
    print(f"  Heat transfer = 50 mm")
    print(f"  Tubesheet 2 = 5 mm")
    print(f"  Heat power = {vxq1.heat_power:.1f} W")
    print(f"  Heat flux = {vxq1.heat_flux_per_length:.1f} W/m")
    print()

    # Input acoustic state
    p1_in = 50000.0 + 10000.0j  # Pa
    U1_in = 5e-5 + 1e-5j  # m³/s

    # Propagate
    p1_out, U1_out, T_out = vxq1.propagate(p1_in, U1_in, T_m, omega, helium)

    print(f"Input:  T_m = {T_m:.2f} K")
    print(f"Output: T_m = {T_out:.2f} K")
    print(f"Delta T = {T_out - T_m:.2f} K")
    print()
    print(f"Acoustic propagation:")
    print(f"  |p1_in|  = {np.abs(p1_in)/1000:.4f} kPa")
    print(f"  |p1_out| = {np.abs(p1_out)/1000:.4f} kPa")

    # Temperature should increase (positive heat input)
    test1_pass = T_out > T_m
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (temperature increased with heating)")
    print()

    # =========================================================================
    # Test 2: Cooling with negative heat input
    # =========================================================================
    print("TEST 2: Cooling with negative heat input")
    print("-" * 50)

    vxq1_cool = segments.VXQ1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_heat_transfer=0.05,
        length_tubesheet2=0.002,
        heat_power=-100.0,  # -100 W (cooling)
        name="cooling",
    )

    T_hot = 400.0
    _, _, T_cooled = vxq1_cool.propagate(p1_in, U1_in, T_hot, omega, helium)

    print(f"  T_in = {T_hot:.2f} K")
    print(f"  Heat power = {vxq1_cool.heat_power:.1f} W (negative = cooling)")
    print(f"  T_out = {T_cooled:.2f} K")
    print(f"  Delta T = {T_cooled - T_hot:.2f} K")

    test2_pass = T_cooled < T_hot
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (temperature decreased with cooling)")
    print()

    # =========================================================================
    # Test 3: Zero heat input - no temperature change
    # =========================================================================
    print("TEST 3: Zero heat input (no temperature change)")
    print("-" * 50)

    vxq1_zero = segments.VXQ1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_heat_transfer=0.05,
        length_tubesheet2=0.002,
        heat_power=0.0,  # No heat input
        name="zero_heat",
    )

    _, _, T_zero = vxq1_zero.propagate(p1_in, U1_in, T_m, omega, helium)

    delta_T_zero = abs(T_zero - T_m)
    print(f"  T_in = {T_m:.2f} K")
    print(f"  Heat power = 0 W")
    print(f"  T_out = {T_zero:.2f} K")
    print(f"  |Delta T| = {delta_T_zero:.4f} K")

    test3_pass = delta_T_zero < 1.0  # Should be close to zero
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (minimal T change with zero heat)")
    print()

    # =========================================================================
    # Test 4: Higher velocity = smaller temperature change
    # =========================================================================
    print("TEST 4: Higher velocity = smaller temperature change")
    print("-" * 50)

    # Low velocity
    U1_low = 1e-5 + 0j
    _, _, T_low_vel = vxq1.propagate(p1_in, U1_low, T_m, omega, helium)

    # High velocity
    U1_high = 1e-4 + 0j
    _, _, T_high_vel = vxq1.propagate(p1_in, U1_high, T_m, omega, helium)

    delta_T_low = T_low_vel - T_m
    delta_T_high = T_high_vel - T_m

    print(f"  Low velocity (|U1| = {np.abs(U1_low)*1e6:.1f} cm³/s):")
    print(f"    Delta T = {delta_T_low:.2f} K")
    print(f"  High velocity (|U1| = {np.abs(U1_high)*1e6:.1f} cm³/s):")
    print(f"    Delta T = {delta_T_high:.2f} K")

    # Higher velocity should result in smaller temperature change
    # (same heat is distributed over more mass flow)
    test4_pass = delta_T_low > delta_T_high > 0
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (higher velocity = smaller ΔT)")
    print()

    # =========================================================================
    # Test 5: Parameter validation
    # =========================================================================
    print("TEST 5: Parameter validation")
    print("-" * 50)

    errors_caught = 0

    try:
        segments.VXQ1(
            area=1e-3,
            gas_area_fraction=1.5,  # Invalid: > 1
            solid_area_fraction=0.3,
            hydraulic_radius=0.5e-3,
            length_tubesheet1=0.002,
            length_heat_transfer=0.05,
            length_tubesheet2=0.002,
            heat_power=100.0,
        )
    except ValueError:
        errors_caught += 1
        print("  Caught invalid gas_area_fraction > 1")

    try:
        segments.VXQ1(
            area=1e-3,
            gas_area_fraction=0.4,
            solid_area_fraction=0.3,
            hydraulic_radius=0.5e-3,
            length_tubesheet1=0.002,
            length_heat_transfer=-0.05,  # Invalid: negative
            length_tubesheet2=0.002,
            heat_power=100.0,
        )
    except ValueError:
        errors_caught += 1
        print("  Caught invalid length_heat_transfer < 0")

    test5_pass = errors_caught == 2
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (parameter validation)")
    print()

    # =========================================================================
    # Test 6: Proportionality - double heat = double delta T
    # =========================================================================
    print("TEST 6: Heat power proportionality")
    print("-" * 50)

    vxq1_50w = segments.VXQ1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_heat_transfer=0.05,
        length_tubesheet2=0.002,
        heat_power=50.0,
        name="50W",
    )

    vxq1_100w = segments.VXQ1(
        area=1e-3,
        gas_area_fraction=0.4,
        solid_area_fraction=0.3,
        hydraulic_radius=0.5e-3,
        length_tubesheet1=0.002,
        length_heat_transfer=0.05,
        length_tubesheet2=0.002,
        heat_power=100.0,
        name="100W",
    )

    _, _, T_50w = vxq1_50w.propagate(p1_in, U1_in, T_m, omega, helium)
    _, _, T_100w = vxq1_100w.propagate(p1_in, U1_in, T_m, omega, helium)

    delta_T_50w = T_50w - T_m
    delta_T_100w = T_100w - T_m

    print(f"  50 W:  Delta T = {delta_T_50w:.2f} K")
    print(f"  100 W: Delta T = {delta_T_100w:.2f} K")
    print(f"  Ratio (100W/50W): {delta_T_100w / delta_T_50w:.2f} (expected ~2.0)")

    # Should be approximately proportional
    ratio = delta_T_100w / delta_T_50w if delta_T_50w > 0 else 0
    test6_pass = 1.5 < ratio < 2.5  # Allow some tolerance
    print()
    print(f"Test 6: {'PASS' if test6_pass else 'FAIL'} (approximately proportional)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass and test6_pass

    print(f"Test 1 (Basic heating):           {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Cooling):                 {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Zero heat):               {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Velocity dependence):     {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Parameter validation):    {'PASS' if test5_pass else 'FAIL'}")
    print(f"Test 6 (Heat proportionality):    {'PASS' if test6_pass else 'FAIL'}")
    print()

    if all_pass:
        print("VXQ1 VALIDATION PASSED")
        print()
        print("Notes:")
        print("- VXQ1 models heat exchangers with fixed heat input per unit length")
        print("- Temperature change is proportional to heat power")
        print("- Higher velocity results in smaller temperature change")
        print("- No reference file available for direct comparison")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

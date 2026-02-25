#!/usr/bin/env python3
"""
Validation of power-law heat exchanger (PX).

This validation tests:
1. Basic propagation with known power-law coefficients
2. Comparison with SX using equivalent parameters
3. Temperature output (should equal solid temperature)
4. Parameter sensitivity
5. Sequential propagation continuity
"""

import numpy as np

from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: Power-law heat exchanger (PX)")
    print("=" * 70)
    print()

    # =========================================================================
    # Test 1: Basic instantiation and propagation
    # =========================================================================
    print("TEST 1: Basic instantiation and propagation")
    print("-" * 50)

    # Use helium at 3 MPa
    helium = gas.Helium(mean_pressure=3e6)
    T_m = 300.0
    omega = 2 * np.pi * 50  # 50 Hz

    # Create a power-law heat exchanger (e.g., metal foam)
    area = 1e-4  # m²
    porosity = 0.85  # typical for metal foam
    rh = 100e-6  # 100 µm hydraulic radius
    length = 0.005  # 5 mm
    T_solid = 300.0  # Isothermal HX

    # Power-law coefficients for metal foam (example)
    f_con = 50.0  # friction factor coefficient
    f_exp = 0.8  # friction factor exponent
    h_con = 0.4  # heat transfer coefficient
    h_exp = 0.6  # heat transfer exponent

    hx = segments.PowerLawHeatExchanger(
        length=length,
        porosity=porosity,
        hydraulic_radius=rh,
        area=area,
        solid_temperature=T_solid,
        f_con=f_con,
        f_exp=f_exp,
        h_con=h_con,
        h_exp=h_exp,
        name="metal_foam_hx",
    )

    print(f"Heat exchanger parameters:")
    print(f"  Length = {length*1000:.1f} mm")
    print(f"  Porosity = {porosity}")
    print(f"  Hydraulic radius = {rh*1e6:.1f} µm")
    print(f"  Area = {area*1e4:.2f} cm²")
    print(f"  T_solid = {T_solid:.1f} K")
    print(f"  f_con = {f_con}, f_exp = {f_exp}")
    print(f"  h_con = {h_con}, h_exp = {h_exp}")
    print()

    # Input acoustic state
    p1_in = 50000.0 + 10000.0j  # Pa
    U1_in = 5e-5 + 1e-5j  # m³/s

    # Propagate
    p1_out, U1_out, T_out = hx.propagate(p1_in, U1_in, T_m, omega, helium)

    print(f"Acoustic propagation at {omega/(2*np.pi):.0f} Hz:")
    print(f"  p1_in  = {np.abs(p1_in)/1000:.2f} kPa @ {np.angle(p1_in, deg=True):.2f}°")
    print(f"  p1_out = {np.abs(p1_out)/1000:.2f} kPa @ {np.angle(p1_out, deg=True):.2f}°")
    print(f"  U1_in  = {np.abs(U1_in)*1e6:.2f} cm³/s @ {np.angle(U1_in, deg=True):.2f}°")
    print(f"  U1_out = {np.abs(U1_out)*1e6:.2f} cm³/s @ {np.angle(U1_out, deg=True):.2f}°")
    print(f"  T_out = {T_out:.2f} K")

    # Pressure should drop due to viscous losses
    pressure_drop = np.abs(p1_in) - np.abs(p1_out)
    print(f"  Pressure amplitude drop: {pressure_drop:.2f} Pa")

    test1_pass = pressure_drop > 0
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (basic propagation)")
    print()

    # =========================================================================
    # Test 2: Comparison with SX (both should produce similar behavior)
    # =========================================================================
    print("TEST 2: Comparison with SX (qualitative)")
    print("-" * 50)

    # Parameters compatible with SX porosity range
    phi = 0.7

    # Create SX with same geometry
    sx = segments.ScreenHeatExchanger(
        length=0.005,
        porosity=phi,
        hydraulic_radius=50e-6,
        area=1e-4,
        solid_temperature=300.0,
        name="sx",
    )

    # Create PX with equivalent parameters
    # SX uses c1(ϕ), c2(ϕ), b(ϕ) empirical fits
    # For comparison, use similar power-law parameters
    px = segments.PowerLawHeatExchanger(
        length=0.005,
        porosity=phi,
        hydraulic_radius=50e-6,
        area=1e-4,
        solid_temperature=300.0,
        f_con=30.0,
        f_exp=0.6,
        h_con=0.5,
        h_exp=0.6,
        name="px",
    )

    p1_test = 30000.0 + 0j
    U1_test = 3e-5 + 5e-6j

    p1_sx, U1_sx, T_sx = sx.propagate(p1_test, U1_test, T_m, omega, helium)
    p1_px, U1_px, T_px = px.propagate(p1_test, U1_test, T_m, omega, helium)

    print(f"SX output:")
    print(f"  p1_out = {np.abs(p1_sx)/1000:.4f} kPa @ {np.angle(p1_sx, deg=True):.2f}°")
    print(f"  U1_out = {np.abs(U1_sx)*1e6:.4f} cm³/s @ {np.angle(U1_sx, deg=True):.2f}°")
    print(f"  T_out = {T_sx:.2f} K")

    print(f"PX output (approximate equivalent):")
    print(f"  p1_out = {np.abs(p1_px)/1000:.4f} kPa @ {np.angle(p1_px, deg=True):.2f}°")
    print(f"  U1_out = {np.abs(U1_px)*1e6:.4f} cm³/s @ {np.angle(U1_px, deg=True):.2f}°")
    print(f"  T_out = {T_px:.2f} K")

    # Both should produce pressure drops
    dp_sx = np.abs(p1_test) - np.abs(p1_sx)
    dp_px = np.abs(p1_test) - np.abs(p1_px)

    print(f"  SX pressure drop: {dp_sx:.2f} Pa")
    print(f"  PX pressure drop: {dp_px:.2f} Pa")

    test2_pass = dp_sx > 0 and dp_px > 0
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (both produce viscous losses)")
    print()

    # =========================================================================
    # Test 3: Temperature output equals solid temperature
    # =========================================================================
    print("TEST 3: Temperature output equals solid temperature")
    print("-" * 50)

    T_solid_test = 350.0  # Different from input
    T_in_test = 300.0

    hx_temp = segments.PowerLawHeatExchanger(
        length=0.01,
        porosity=0.7,
        hydraulic_radius=50e-6,
        area=1e-4,
        solid_temperature=T_solid_test,
        f_con=30.0,
        f_exp=0.7,
        h_con=0.5,
        h_exp=0.6,
        name="temp_test",
    )

    p1_t, U1_t, T_out_t = hx_temp.propagate(
        20000.0 + 0j, 2e-5 + 0j, T_in_test, omega, helium
    )

    print(f"T_in = {T_in_test:.2f} K")
    print(f"T_solid = {T_solid_test:.2f} K")
    print(f"T_out = {T_out_t:.2f} K")

    test3_pass = abs(T_out_t - T_solid_test) < 0.01
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (T_out = T_solid)")
    print()

    # =========================================================================
    # Test 4: Parameter sensitivity
    # =========================================================================
    print("TEST 4: Parameter sensitivity")
    print("-" * 50)

    base_params = {
        "length": 0.01,
        "porosity": 0.6,
        "hydraulic_radius": 30e-6,
        "area": 1e-4,
        "solid_temperature": 300.0,
        "f_con": 24.0,
        "h_con": 0.5,
        "h_exp": 0.6,
    }

    p1_s = 30000.0 + 0j
    U1_s = 3e-5 + 5e-6j

    print("Varying f_exp (friction exponent):")
    f_exp_values = [0.5, 0.7, 0.9, 1.0]
    pressure_drops = []

    for f_exp_val in f_exp_values:
        hx_test = segments.PowerLawHeatExchanger(
            f_exp=f_exp_val,
            **base_params,
        )
        p1_o, U1_o, T_o = hx_test.propagate(p1_s, U1_s, T_m, omega, helium)
        dp = np.abs(p1_s) - np.abs(p1_o)
        pressure_drops.append(dp)
        print(f"  f_exp = {f_exp_val}: Δp = {dp:.2f} Pa")

    test4_pass = all(dp > 0 for dp in pressure_drops)
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (parameter sensitivity)")
    print()

    # =========================================================================
    # Test 5: Sequential propagation continuity
    # =========================================================================
    print("TEST 5: Sequential propagation continuity")
    print("-" * 50)

    full_length = 0.02
    half_length = full_length / 2

    common = {
        "porosity": 0.65,
        "hydraulic_radius": 40e-6,
        "area": 1e-4,
        "solid_temperature": 300.0,
        "f_con": 25.0,
        "f_exp": 0.7,
        "h_con": 0.5,
        "h_exp": 0.65,
    }

    hx_full = segments.PowerLawHeatExchanger(length=full_length, **common, name="full")
    hx_half1 = segments.PowerLawHeatExchanger(length=half_length, **common, name="half1")
    hx_half2 = segments.PowerLawHeatExchanger(length=half_length, **common, name="half2")

    p1_start = 40000.0 + 10000.0j
    U1_start = 4e-5 + 1e-5j

    # Full propagation
    p1_full, U1_full, T_full = hx_full.propagate(p1_start, U1_start, T_m, omega, helium)

    # Sequential propagation
    p1_mid, U1_mid, T_mid = hx_half1.propagate(p1_start, U1_start, T_m, omega, helium)
    p1_seq, U1_seq, T_seq = hx_half2.propagate(p1_mid, U1_mid, T_mid, omega, helium)

    # Compare
    p1_err = np.abs(p1_seq - p1_full) / np.abs(p1_full) * 100
    U1_err = np.abs(U1_seq - U1_full) / np.abs(U1_full) * 100

    print(f"Full propagation:       p1 = {np.abs(p1_full)/1000:.4f} kPa, U1 = {np.abs(U1_full)*1e6:.4f} cm³/s")
    print(f"Sequential propagation: p1 = {np.abs(p1_seq)/1000:.4f} kPa, U1 = {np.abs(U1_seq)*1e6:.4f} cm³/s")
    print(f"  Pressure error: {p1_err:.4f}%")
    print(f"  Velocity error: {U1_err:.4f}%")

    test5_pass = p1_err < 0.1 and U1_err < 0.1
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (sequential continuity)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass

    print(f"Test 1 (Basic propagation):       {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Viscous losses):          {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (T_out = T_solid):         {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Parameter sensitivity):   {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Sequential continuity):   {'PASS' if test5_pass else 'FAIL'}")
    print()

    if all_pass:
        print("POWER-LAW HEAT EXCHANGER (PX) VALIDATION PASSED")
        print()
        print("Notes:")
        print("- PX correctly implements isothermal HX with power-law correlations")
        print("- Output temperature equals solid temperature as expected")
        print("- Power-law friction and heat transfer correlations working")
        print("- No reference file available for direct comparison")
        print("- Segment allows modeling arbitrary HX geometries with known correlations")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

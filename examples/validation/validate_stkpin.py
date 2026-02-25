#!/usr/bin/env python3
"""
Validation of STKPIN (pin array stack/regenerator geometry).

This validation tests:
1. PinArray geometry computes correct thermoviscous functions
2. Limiting behaviors are correct (small and large |z|)
3. PinArray can be used with Stack segment
4. Physical reasonableness of results

Since no specific Reference baseline STKPIN example is available, we validate by:
- Verifying the Bessel function formula implementation
- Checking limiting behaviors match theory
- Comparing to other geometries in appropriate limits
"""

import numpy as np
from scipy.special import jv, yv  # Bessel functions

from openthermoacoustics import gas
from openthermoacoustics.geometry import PinArray, CircularPore
from openthermoacoustics.segments import Stack


def main():
    print("=" * 70)
    print("VALIDATION: STKPIN (pin array geometry)")
    print("=" * 70)
    print()

    # =========================================================================
    # Test 1: Geometry parameters
    # =========================================================================
    print("TEST 1: Geometry parameter calculation")
    print("-" * 50)

    # Example: pins with 40 µm radius, 320 µm center-to-center spacing
    ri = 40e-6  # pin radius
    spacing = 320e-6  # 2*y0

    pin = PinArray(pin_radius=ri, pin_spacing=spacing)

    # Expected outer radius: ro = spacing * sqrt(sqrt(3)/(2*pi))
    ro_expected = spacing * np.sqrt(np.sqrt(3) / (2 * np.pi))

    # Expected porosity: 1 - pi*ri² / (sqrt(3)/2 * spacing²)
    area_per_pin = (np.sqrt(3) / 2) * spacing**2
    porosity_expected = (area_per_pin - np.pi * ri**2) / area_per_pin

    print(f"Pin radius: {ri*1e6:.1f} µm")
    print(f"Pin spacing: {spacing*1e6:.1f} µm")
    print(f"Outer radius (computed): {pin.outer_radius*1e6:.3f} µm")
    print(f"Outer radius (expected): {ro_expected*1e6:.3f} µm")
    print(f"Porosity (computed): {pin.porosity:.6f}")
    print(f"Porosity (expected): {porosity_expected:.6f}")
    print(f"Hydraulic radius: {pin.hydraulic_radius*1e6:.2f} µm")
    print()

    ro_err = abs(pin.outer_radius - ro_expected) / ro_expected * 100
    por_err = abs(pin.porosity - porosity_expected) / porosity_expected * 100

    test1_pass = ro_err < 0.001 and por_err < 0.001
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (geometry parameters)")
    print()

    # =========================================================================
    # Test 2: Direct formula verification
    # =========================================================================
    print("TEST 2: Direct Bessel function formula verification")
    print("-" * 50)

    # Test at a specific delta value
    delta = 50e-6  # 50 µm penetration depth
    ri = pin.pin_radius
    ro = pin.outer_radius

    # Compute z arguments: z = (-1+1j)*r/delta (Reference baseline convention)
    z_i = (-1 + 1j) * ri / delta
    z_o = (-1 + 1j) * ro / delta

    # Compute Bessel functions manually
    J0_zi = jv(0, z_i)
    J1_zi = jv(1, z_i)
    J1_zo = jv(1, z_o)
    Y0_zi = yv(0, z_i)
    Y1_zi = yv(1, z_i)
    Y1_zo = yv(1, z_o)

    # Reference baseline governing relations:
    # f = -delta/(i-1) * 2*ri/(ro² - ri²) * [Y1(zo)*J1(zi) - J1(zo)*Y1(zi)]
    #                                      / [Y1(zo)*J0(zi) - J1(zo)*Y0(zi)]
    numer = Y1_zo * J1_zi - J1_zo * Y1_zi
    denom = Y1_zo * J0_zi - J1_zo * Y0_zi
    prefactor = -delta / (-1 + 1j) * 2 * ri / (ro**2 - ri**2)
    f_expected = prefactor * numer / denom

    # Get from PinArray
    f_computed = pin._compute_f(delta, pin.hydraulic_radius)

    print(f"Delta: {delta*1e6:.1f} µm")
    print(f"|z_i| = {np.abs(z_i):.4f}, |z_o| = {np.abs(z_o):.4f}")
    print(f"f (computed): {f_computed:.6f}")
    print(f"f (expected): {f_expected:.6f}")

    f_err = np.abs(f_computed - f_expected) / np.abs(f_expected) * 100
    print(f"Error: {f_err:.6f}%")
    print()

    test2_pass = f_err < 0.01
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (Bessel formula verification)")
    print()

    # =========================================================================
    # Test 3: Small |z| limit (inviscid limit)
    # =========================================================================
    print("TEST 3: Small |z| limit (f -> 1 as delta -> infinity)")
    print("-" * 50)

    # Very large delta means very small |z|
    delta_large = 1.0  # 1 meter (huge penetration depth)
    f_small_z = pin._compute_f(delta_large, pin.hydraulic_radius)

    print(f"Delta = {delta_large*1e3:.0f} mm (very large)")
    print(f"f = {f_small_z:.6f}")
    print(f"Expected: ~1.0")

    # Should be very close to 1
    test3_pass = np.abs(f_small_z - 1.0) < 0.01
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (inviscid limit)")
    print()

    # =========================================================================
    # Test 4: Large |z| asymptotic behavior
    # =========================================================================
    print("TEST 4: Large |z| limit (boundary layer regime)")
    print("-" * 50)

    # Very small delta means very large |z|
    delta_small = 1e-9  # 1 nm (extremely small)
    f_large_z = pin._compute_f(delta_small, pin.hydraulic_radius)

    # Asymptotic form: f -> (1-j)*delta/r_h
    f_asymptotic = (1 - 1j) * delta_small / pin.hydraulic_radius

    print(f"Delta = {delta_small*1e9:.1f} nm (very small)")
    print(f"f (computed): {f_large_z:.6e}")
    print(f"f (asymptotic): {f_asymptotic:.6e}")

    # Check that the asymptotic form is approximately correct
    ratio = np.abs(f_large_z / f_asymptotic)
    print(f"|f/f_asymptotic| = {ratio:.4f} (should be ~1)")

    test4_pass = 0.5 < ratio < 2.0  # Allow some tolerance
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (boundary layer limit)")
    print()

    # =========================================================================
    # Test 5: f_nu and f_kappa with different penetration depths
    # =========================================================================
    print("TEST 5: f_nu and f_kappa with Prandtl number effect")
    print("-" * 50)

    # For helium, Prandtl number ~ 0.67, so delta_kappa > delta_nu
    delta_nu = 50e-6
    delta_kappa = delta_nu / np.sqrt(0.67)  # ~61 µm

    f_nu = pin.f_nu(omega=1000.0, delta_nu=delta_nu, hydraulic_radius=pin.hydraulic_radius)
    f_kappa = pin.f_kappa(
        omega=1000.0, delta_kappa=delta_kappa, hydraulic_radius=pin.hydraulic_radius
    )

    print(f"Delta_nu = {delta_nu*1e6:.1f} µm")
    print(f"Delta_kappa = {delta_kappa*1e6:.1f} µm (Pr=0.67)")
    print(f"f_nu = {f_nu:.6f}")
    print(f"f_kappa = {f_kappa:.6f}")

    # f_kappa should be larger since delta_kappa > delta_nu
    # (closer to inviscid limit)
    print(f"|f_kappa| / |f_nu| = {np.abs(f_kappa)/np.abs(f_nu):.4f} (should be > 1)")

    test5_pass = np.abs(f_kappa) > np.abs(f_nu) and np.isfinite(f_nu) and np.isfinite(f_kappa)
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (Prandtl number effect)")
    print()

    # =========================================================================
    # Test 6: Use with Stack segment
    # =========================================================================
    print("TEST 6: Integration with Stack segment")
    print("-" * 50)

    # Create a stack with pin array geometry
    helium = gas.Helium(mean_pressure=3e6)
    T_m = 300.0

    # Pin array stack parameters
    total_area = 1e-3  # 10 cm²
    porosity = pin.porosity
    length = 0.05  # 5 cm

    stack = Stack(
        length=length,
        area=total_area,
        porosity=porosity,
        hydraulic_radius=pin.hydraulic_radius,
        geometry=pin,
        name="pin_stack",
    )

    # Propagate through the stack
    omega = 2 * np.pi * 100  # 100 Hz
    p1_in = 50000.0 + 10000.0j
    U1_in = 1e-4 + 2e-5j

    p1_out, U1_out, T_out = stack.propagate(p1_in, U1_in, T_m, omega, helium)

    print(f"Stack: {length*100:.1f} cm pin array, porosity={porosity:.3f}")
    print(f"Input:  |p1| = {np.abs(p1_in):.0f} Pa, |U1| = {np.abs(U1_in)*1e6:.2f} mm³/s")
    print(f"Output: |p1| = {np.abs(p1_out):.0f} Pa, |U1| = {np.abs(U1_out)*1e6:.2f} mm³/s")

    # Check that results are finite and reasonable
    test6_pass = (
        np.isfinite(p1_out)
        and np.isfinite(U1_out)
        and np.abs(p1_out) > 0
        and np.abs(U1_out) > 0
    )
    print()
    print(f"Test 6: {'PASS' if test6_pass else 'FAIL'} (Stack integration)")
    print()

    # =========================================================================
    # Test 7: Frequency sweep
    # =========================================================================
    print("TEST 7: Frequency sweep (f should vary smoothly)")
    print("-" * 50)

    freqs = [10, 50, 100, 200, 500, 1000]
    print(f"{'Freq (Hz)':<12} {'|f_nu|':<12} {'phase(f_nu)':<12}")
    print("-" * 36)

    f_values = []
    for freq in freqs:
        omega = 2 * np.pi * freq
        # Compute penetration depths for helium at 300 K, 3 MPa
        rho_m = helium.density(T_m)
        mu = helium.viscosity(T_m)
        delta_nu = np.sqrt(2 * mu / (rho_m * omega))

        f_nu = pin.f_nu(omega, delta_nu, pin.hydraulic_radius)
        f_values.append(f_nu)
        print(f"{freq:<12} {np.abs(f_nu):<12.6f} {np.angle(f_nu, deg=True):<12.2f}°")

    # Check monotonicity of |f| (should decrease with frequency)
    magnitudes = [np.abs(f) for f in f_values]
    is_monotonic = all(magnitudes[i] >= magnitudes[i + 1] for i in range(len(magnitudes) - 1))

    test7_pass = is_monotonic and all(np.isfinite(f) for f in f_values)
    print()
    print(f"|f| monotonically decreasing: {is_monotonic}")
    print(f"Test 7: {'PASS' if test7_pass else 'FAIL'} (frequency sweep)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = (
        test1_pass
        and test2_pass
        and test3_pass
        and test4_pass
        and test5_pass
        and test6_pass
        and test7_pass
    )
    print(f"Test 1 (Geometry parameters):       {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Bessel formula):            {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Inviscid limit):            {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Boundary layer limit):      {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Prandtl number effect):     {'PASS' if test5_pass else 'FAIL'}")
    print(f"Test 6 (Stack integration):         {'PASS' if test6_pass else 'FAIL'}")
    print(f"Test 7 (Frequency sweep):           {'PASS' if test7_pass else 'FAIL'}")
    print()

    if all_pass:
        print("STKPIN (PinArray) VALIDATION PASSED")
        print()
        print("Notes:")
        print("- PinArray geometry correctly implements Reference baseline governing relations")
        print("- Bessel function formula matches direct calculation")
        print("- Limiting behaviors are physically correct")
        print("- Successfully integrates with Stack segment")
        print("- No reference file available for direct comparison")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

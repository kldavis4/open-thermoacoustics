#!/usr/bin/env python3
"""
Validation of AcousticState (Python equivalent of the baseline's RPN segment).

This demonstrates how AcousticState provides access to all the variables
and derived quantities that would be available through RPN in Reference baseline.

Examples include:
1. Basic gas properties
2. Penetration depths
3. Acoustic power and impedance
4. Displacement and velocity calculations
5. Thermoviscous functions
"""

import numpy as np

from openthermoacoustics import AcousticState, gas, geometry


def main():
    print("=" * 70)
    print("VALIDATION: AcousticState (Python RPN Equivalent)")
    print("=" * 70)
    print()

    # =========================================================================
    # Setup: Helium at 3 MPa, 300 K, 100 Hz
    # =========================================================================
    helium = gas.Helium(mean_pressure=3e6)
    T_m = 300.0
    omega = 2 * np.pi * 100  # 100 Hz

    # Acoustic state
    p1 = 50000.0 + 10000.0j  # Pa
    U1 = 1e-4 + 2e-5j  # m³/s
    area = 1e-3  # 10 cm²

    # Create AcousticState
    state = AcousticState(
        p1=p1,
        U1=U1,
        T_m=T_m,
        omega=omega,
        gas=helium,
        area=area,
    )

    # =========================================================================
    # Test 1: Gas Properties
    # =========================================================================
    print("TEST 1: Gas Properties (equivalent to RPN: rho cp a mu k gamma Pr)")
    print("-" * 50)

    # Compare with direct gas calls
    rho_direct = helium.density(T_m)
    cp_direct = helium.specific_heat_cp(T_m)
    a_direct = helium.sound_speed(T_m)
    mu_direct = helium.viscosity(T_m)
    k_direct = helium.thermal_conductivity(T_m)
    gamma_direct = helium.gamma(T_m)
    Pr_direct = helium.prandtl(T_m)

    print(f"  Property        State        Direct       Match")
    print(f"  rho (kg/m³)     {state.rho:12.6f} {rho_direct:12.6f} {'✓' if np.isclose(state.rho, rho_direct) else '✗'}")
    print(f"  cp (J/kg·K)     {state.cp:12.2f} {cp_direct:12.2f} {'✓' if np.isclose(state.cp, cp_direct) else '✗'}")
    print(f"  a (m/s)         {state.a:12.4f} {a_direct:12.4f} {'✓' if np.isclose(state.a, a_direct) else '✗'}")
    print(f"  mu (Pa·s)       {state.mu:12.3e} {mu_direct:12.3e} {'✓' if np.isclose(state.mu, mu_direct) else '✗'}")
    print(f"  k (W/m·K)       {state.k:12.6f} {k_direct:12.6f} {'✓' if np.isclose(state.k, k_direct) else '✗'}")
    print(f"  gamma           {state.gamma:12.6f} {gamma_direct:12.6f} {'✓' if np.isclose(state.gamma, gamma_direct) else '✗'}")
    print(f"  Pr              {state.Pr:12.6f} {Pr_direct:12.6f} {'✓' if np.isclose(state.Pr, Pr_direct) else '✗'}")

    test1_pass = all([
        np.isclose(state.rho, rho_direct),
        np.isclose(state.cp, cp_direct),
        np.isclose(state.a, a_direct),
        np.isclose(state.mu, mu_direct),
        np.isclose(state.k, k_direct),
        np.isclose(state.gamma, gamma_direct),
        np.isclose(state.Pr, Pr_direct),
    ])
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Test 2: Penetration Depths
    # =========================================================================
    print("TEST 2: Penetration Depths (equivalent to RPN: dn dk)")
    print("-" * 50)

    # Manual calculation
    delta_nu_manual = np.sqrt(2 * mu_direct / (omega * rho_direct))
    alpha = k_direct / (rho_direct * cp_direct)
    delta_kappa_manual = np.sqrt(2 * alpha / omega)

    print(f"  δ_ν (viscous):  {state.delta_nu*1e6:.4f} μm (manual: {delta_nu_manual*1e6:.4f} μm)")
    print(f"  δ_κ (thermal):  {state.delta_kappa*1e6:.4f} μm (manual: {delta_kappa_manual*1e6:.4f} μm)")
    print(f"  Ratio δ_κ/δ_ν:  {state.delta_kappa/state.delta_nu:.4f} (should be ~1/√Pr = {1/np.sqrt(state.Pr):.4f})")

    test2_pass = (
        np.isclose(state.delta_nu, delta_nu_manual) and
        np.isclose(state.delta_kappa, delta_kappa_manual)
    )
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Test 3: Pressure and Volume Velocity
    # =========================================================================
    print("TEST 3: Pressure & Volume Velocity (equivalent to RPN: p1 U1 mag ph)")
    print("-" * 50)

    p1_mag_manual = np.abs(p1)
    p1_phase_manual = np.angle(p1, deg=True)
    U1_mag_manual = np.abs(U1)
    U1_phase_manual = np.angle(U1, deg=True)

    print(f"  |p1|:    {state.p1_mag:.2f} Pa (manual: {p1_mag_manual:.2f} Pa)")
    print(f"  ∠p1:     {state.p1_phase:.2f}° (manual: {p1_phase_manual:.2f}°)")
    print(f"  |U1|:    {state.U1_mag:.4e} m³/s (manual: {U1_mag_manual:.4e} m³/s)")
    print(f"  ∠U1:     {state.U1_phase:.2f}° (manual: {U1_phase_manual:.2f}°)")
    print(f"  Drive ratio: {state.drive_ratio*100:.4f}%")

    test3_pass = (
        np.isclose(state.p1_mag, p1_mag_manual) and
        np.isclose(state.p1_phase, p1_phase_manual) and
        np.isclose(state.U1_mag, U1_mag_manual) and
        np.isclose(state.U1_phase, U1_phase_manual)
    )
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Test 4: Acoustic Power
    # =========================================================================
    print("TEST 4: Acoustic Power (equivalent to RPN: Edot)")
    print("-" * 50)

    # Manual: Edot = (1/2) * Re[p1 * conj(U1)]
    Edot_manual = 0.5 * np.real(p1 * np.conj(U1))

    print(f"  Edot (acoustic power): {state.acoustic_power:.6f} W")
    print(f"  Manual calculation:    {Edot_manual:.6f} W")
    print(f"  Reactive power:        {state.reactive_power:.6f} W")
    print(f"  Power factor:          {state.power_factor:.4f}")
    print(f"  Phase difference:      {state.phase_difference:.2f}°")

    test4_pass = np.isclose(state.acoustic_power, Edot_manual)
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Test 5: Velocity and Displacement (require area)
    # =========================================================================
    print("TEST 5: Velocity & Displacement (equivalent to RPN: U1 A / mag)")
    print("-" * 50)

    # Manual calculations
    u1_manual = U1 / area
    u1_mag_manual = np.abs(u1_manual)
    xi1_mag_manual = np.abs(U1) / (omega * area)

    print(f"  |u1| (velocity):     {state.u1_mag:.6f} m/s (manual: {u1_mag_manual:.6f} m/s)")
    print(f"  |ξ1| (displacement): {state.xi1_mag*1000:.6f} mm (manual: {xi1_mag_manual*1000:.6f} mm)")
    print(f"  Intensity:           {state.intensity:.4f} W/m²")

    test5_pass = (
        np.isclose(state.u1_mag, u1_mag_manual) and
        np.isclose(state.xi1_mag, xi1_mag_manual)
    )
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Test 6: Impedance
    # =========================================================================
    print("TEST 6: Impedance (equivalent to RPN: p1 U1 /)")
    print("-" * 50)

    Z_manual = p1 / U1

    print(f"  Z = p1/U1:  {state.Z:.4e}")
    print(f"  |Z|:        {state.Z_mag:.4e} Pa·s/m³")
    print(f"  ∠Z:         {state.Z_phase:.2f}°")
    print(f"  Manual |Z|: {np.abs(Z_manual):.4e} Pa·s/m³")

    test6_pass = np.isclose(state.Z, Z_manual)
    print()
    print(f"Test 6: {'PASS' if test6_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Test 7: Thermoviscous Functions
    # =========================================================================
    print("TEST 7: Thermoviscous Functions (boundary layer approximation)")
    print("-" * 50)

    r_h = 1e-3  # 1 mm hydraulic radius

    f_nu_bl = state.f_nu_boundary_layer(r_h)
    f_kappa_bl = state.f_kappa_boundary_layer(r_h)

    # Manual: f = (1-j) * δ / r_h
    f_nu_manual = (1 - 1j) * state.delta_nu / r_h
    f_kappa_manual = (1 - 1j) * state.delta_kappa / r_h

    print(f"  Hydraulic radius: {r_h*1000:.1f} mm")
    print(f"  f_ν (boundary layer): {f_nu_bl:.6f}")
    print(f"  f_κ (boundary layer): {f_kappa_bl:.6f}")
    print(f"  Manual f_ν:           {f_nu_manual:.6f}")
    print(f"  Manual f_κ:           {f_kappa_manual:.6f}")

    test7_pass = (
        np.isclose(f_nu_bl, f_nu_manual) and
        np.isclose(f_kappa_bl, f_kappa_manual)
    )
    print()
    print(f"Test 7: {'PASS' if test7_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Test 8: Full Thermoviscous Functions with Geometry
    # =========================================================================
    print("TEST 8: Full Thermoviscous Functions with Geometry")
    print("-" * 50)

    pp_geom = geometry.ParallelPlate()
    y0 = 0.5e-3  # half-spacing (hydraulic radius for parallel plate)

    f_nu_full = state.f_nu(pp_geom, hydraulic_radius=y0)
    f_kappa_full = state.f_kappa(pp_geom, hydraulic_radius=y0)

    print(f"  Parallel plate (y0 = 0.5 mm):")
    print(f"  f_ν: {f_nu_full:.6f}")
    print(f"  f_κ: {f_kappa_full:.6f}")
    print(f"  |f_ν|: {np.abs(f_nu_full):.6f}")
    print(f"  |f_κ|: {np.abs(f_kappa_full):.6f}")

    # Should be valid complex numbers
    test8_pass = (
        np.isfinite(f_nu_full) and
        np.isfinite(f_kappa_full) and
        0 < np.abs(f_nu_full) < 1 and
        0 < np.abs(f_kappa_full) < 1
    )
    print()
    print(f"Test 8: {'PASS' if test8_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Test 9: Summary Output
    # =========================================================================
    print("TEST 9: Summary Output")
    print("-" * 50)
    print()
    print(state.summary())
    print()
    test9_pass = True  # Just verify it runs without error
    print(f"Test 9: {'PASS' if test9_pass else 'FAIL'}")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = all([test1_pass, test2_pass, test3_pass, test4_pass,
                    test5_pass, test6_pass, test7_pass, test8_pass, test9_pass])

    print(f"Test 1 (Gas properties):          {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Penetration depths):      {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Pressure/velocity):       {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Acoustic power):          {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Velocity/displacement):   {'PASS' if test5_pass else 'FAIL'}")
    print(f"Test 6 (Impedance):               {'PASS' if test6_pass else 'FAIL'}")
    print(f"Test 7 (Boundary layer f):        {'PASS' if test7_pass else 'FAIL'}")
    print(f"Test 8 (Full f with geometry):    {'PASS' if test8_pass else 'FAIL'}")
    print(f"Test 9 (Summary output):          {'PASS' if test9_pass else 'FAIL'}")
    print()

    if all_pass:
        print("AcousticState VALIDATION PASSED")
        print()
        print("AcousticState provides Python-native access to:")
        print("- Gas properties: rho, cp, cv, a, mu, k, gamma, Pr")
        print("- Penetration depths: delta_nu, delta_kappa")
        print("- Pressure: p1_mag, p1_phase, drive_ratio")
        print("- Volume velocity: U1_mag, U1_phase")
        print("- Velocity: u1, u1_mag (with area)")
        print("- Displacement: xi1, xi1_mag (with area)")
        print("- Impedance: Z, Z_mag, Z_phase, z (normalized)")
        print("- Power: acoustic_power, reactive_power, power_factor")
        print("- Thermoviscous functions: f_nu(), f_kappa()")
        print()
        print("This replaces the baseline's RPN segment with a more Pythonic API.")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

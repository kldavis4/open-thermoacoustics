#!/usr/bin/env python3
"""
Validation Against embedded reference baseline: bottle2 reference case

Tests basic DUCT + CONE + SURFACE propagation for a simple acoustic resonator
(the 1982 Penn State Championship Bottle).

Segments tested:
- Segment 1: DUCT (the neck)
- Segment 2: CONE (transition from neck to volume)
- Segment 3: DUCT (bottle volume)
- Segment 4: SURFACE (bottom end - thermal hysteresis)

Embedded reference: bottle2 reference case from embedded reference baseline
Located at: <external proprietary source>

Target accuracy: <2% amplitude error, <2 degrees phase error
Note: CONE tolerance is 2% due to circular cross-section approximation
(Reference baseline specifies Area + Perimeter which can be non-circular)
"""

import numpy as np

from openthermoacoustics import gas, geometry, segments
from openthermoacoustics.segments import Surface

# Tolerances
AMPLITUDE_TOLERANCE_PCT = 1.0  # 1% for amplitudes (default)
CONE_AMPLITUDE_TOLERANCE_PCT = 2.0  # 2% for CONE (geometry approximation)
PHASE_TOLERANCE_DEG = 2.0  # 2 degrees for phases


def compare_results(name, reference_baseline, tolerance_pct=AMPLITUDE_TOLERANCE_PCT):
    """Compare results and return pass/fail status."""
    results = []
    max_err = 0.0

    for param, (dec_val, ota_val) in reference_baseline.items():
        if "ph" in param.lower() or "phase" in param.lower():
            # Phase comparison (absolute degrees)
            err = abs(ota_val - dec_val)
            # Handle wraparound at +/-180 degrees
            if err > 180:
                err = 360 - err
            err_str = f"{err:+.3f} deg"
            passed = err < PHASE_TOLERANCE_DEG
        else:
            # Magnitude comparison (percentage)
            if abs(dec_val) > 1e-10:
                err = 100 * (ota_val - dec_val) / dec_val
                err_str = f"{err:+.2f}%"
                passed = abs(err) < tolerance_pct
                max_err = max(max_err, abs(err))
            else:
                err = ota_val - dec_val
                err_str = f"{err:+.4f}"
                passed = abs(err) < 0.01

        status = "PASS" if passed else "FAIL"
        results.append((param, dec_val, ota_val, err_str, status))

    return results, max_err


def print_comparison(name, results, max_err):
    """Print formatted comparison results."""
    print(f"\n{'='*70}")
    print(f"SEGMENT: {name}")
    print(f"{'='*70}")
    print(f"{'Parameter':<20} {'Reference baseline':<15} {'OpenThermo':<15} {'Error':<12} {'Status'}")
    print("-" * 70)

    all_passed = True
    for param, dec_val, ota_val, err_str, status in results:
        if status == "FAIL":
            all_passed = False
        print(f"{param:<20} {dec_val:<15.4g} {ota_val:<15.4g} {err_str:<12} {status}")

    print("-" * 70)
    if all_passed:
        print(f"PASSED (max amplitude error: {max_err:.2f}%)")
    else:
        print(f"NEEDS REVIEW (max amplitude error: {max_err:.2f}%)")

    return all_passed


def main():
    print("=" * 70)
    print("VALIDATION: OpenThermoacoustics vs embedded reference baseline")
    print("Model: 1982 Penn State Championship Bottle (bottle2 reference case)")
    print("=" * 70)
    print(f"\nTolerances: amplitude <{AMPLITUDE_TOLERANCE_PCT}%, phase <{PHASE_TOLERANCE_DEG} deg")

    # Common setup from BEGIN segment
    air_gas = gas.Air(mean_pressure=1.0e5)
    freq = 300.0  # Hz
    omega = 2 * np.pi * freq
    T_mean = 300.0  # K

    all_passed = []

    # =========================================================================
    # Initial conditions from BEGIN (segment 0)
    # =========================================================================
    p1_in = 1.0 * np.exp(1j * np.radians(0.0))  # Pa
    U1_in = 1.0e-4 * np.exp(1j * np.radians(0.0))  # m^3/s

    print("\nInitial conditions (BEGIN):")
    print(f"  |p1| = {np.abs(p1_in):.4g} Pa, ph(p1) = {np.degrees(np.angle(p1_in)):.2f} deg")
    print(f"  |U1| = {np.abs(U1_in):.4e} m^3/s, ph(U1) = {np.degrees(np.angle(U1_in)):.2f} deg")
    print(f"  T_m = {T_mean:.1f} K")
    print(f"  omega = {omega:.2f} rad/s (f = {freq:.0f} Hz)")

    # Print gas properties for reference
    print(f"\nGas properties at T = {T_mean} K:")
    print(f"  rho = {air_gas.density(T_mean):.4f} kg/m^3")
    print(f"  a = {air_gas.sound_speed(T_mean):.2f} m/s")
    print(f"  gamma = {air_gas.gamma(T_mean):.4f}")
    print(f"  mu = {air_gas.viscosity(T_mean):.4e} Pa.s")
    print(f"  kappa = {air_gas.thermal_conductivity(T_mean):.4f} W/(m.K)")

    # =========================================================================
    # Segment 1: DUCT (the neck)
    # =========================================================================
    # Reference baseline parameters:
    #   Area = 2.1410E-4 m^2
    #   Perim = 5.1870E-2 m
    #   Length = 1.7780E-2 m
    # Reference baseline output:
    #   |p| = 18.450 Pa, Ph(p) = -87.822 deg
    #   |U| = 9.9523E-5 m^3/s, Ph(U) = -2.3264E-2 deg

    duct1_area = 2.1410e-4  # m^2
    duct1_length = 1.7780e-2  # m
    # Compute radius from area (assuming circular cross-section)
    duct1_radius = np.sqrt(duct1_area / np.pi)

    dec_out = {
        "|p1|": (18.450, None),
        "ph(p1)": (-87.822, None),
        "|U1|": (9.9523e-5, None),
        "ph(U1)": (-2.3264e-2, None),
    }

    duct1 = segments.Duct(
        length=duct1_length,
        radius=duct1_radius,
        geometry=geometry.CircularPore(),
    )
    p1_out, U1_out, T_out = duct1.propagate(p1_in, U1_in, T_mean, omega, air_gas)

    dec_out["|p1|"] = (18.450, np.abs(p1_out))
    dec_out["ph(p1)"] = (-87.822, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (9.9523e-5, np.abs(U1_out))
    dec_out["ph(U1)"] = (-2.3264e-2, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("DUCT (neck)", dec_out)
    passed = print_comparison("1: DUCT (the neck)", results, max_err)
    all_passed.append(("DUCT (neck)", passed, max_err))

    # Use Reference baseline output as input for next segment (to isolate errors)
    p1_in = 18.450 * np.exp(1j * np.radians(-87.822))
    U1_in = 9.9523e-5 * np.exp(1j * np.radians(-2.3264e-2))

    # =========================================================================
    # Segment 2: CONE (transition from neck to volume)
    # =========================================================================
    # Reference baseline parameters:
    #   AreaI = 2.1410E-4 m^2 (inlet area)
    #   PerimI = 5.1870E-2 m (inlet perimeter)
    #   Length = 0.1003 m
    #   AreaF = 1.8680E-3 m^2 (final area)
    #   PerimF = 0.1532 m (final perimeter)
    # Reference baseline output:
    #   |p| = 50.122 Pa, Ph(p) = -89.64 deg
    #   |U| = 4.4376E-5 m^3/s, Ph(U) = -0.39626 deg

    cone_area_in = 2.1410e-4  # m^2
    cone_area_out = 1.8680e-3  # m^2
    cone_length = 0.1003  # m
    cone_radius_in = np.sqrt(cone_area_in / np.pi)
    cone_radius_out = np.sqrt(cone_area_out / np.pi)

    dec_out = {
        "|p1|": (50.122, None),
        "ph(p1)": (-89.64, None),
        "|U1|": (4.4376e-5, None),
        "ph(U1)": (-0.39626, None),
    }

    cone = segments.Cone(
        length=cone_length,
        radius_in=cone_radius_in,
        radius_out=cone_radius_out,
        geometry=geometry.CircularPore(),
    )
    p1_out, U1_out, T_out = cone.propagate(p1_in, U1_in, T_mean, omega, air_gas)

    dec_out["|p1|"] = (50.122, np.abs(p1_out))
    dec_out["ph(p1)"] = (-89.64, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (4.4376e-5, np.abs(U1_out))
    dec_out["ph(U1)"] = (-0.39626, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("CONE", dec_out, tolerance_pct=CONE_AMPLITUDE_TOLERANCE_PCT)
    passed = print_comparison("2: CONE (transition)", results, max_err)
    all_passed.append(("CONE", passed, max_err))

    # Use Reference baseline output as input for next segment
    p1_in = 50.122 * np.exp(1j * np.radians(-89.64))
    U1_in = 4.4376e-5 * np.exp(1j * np.radians(-0.39626))

    # =========================================================================
    # Segment 3: DUCT (bottle volume)
    # =========================================================================
    # Reference baseline parameters:
    #   Area = 1.8680E-3 m^2
    #   Perim = 0.1532 m
    #   Length = 0.1270 m
    # Reference baseline output:
    #   |p| = 44.708 Pa, Ph(p) = -89.668 deg
    #   |U| = 1.1383E-4 m^3/s, Ph(U) = -179.59 deg

    duct2_area = 1.8680e-3  # m^2
    duct2_length = 0.1270  # m
    duct2_radius = np.sqrt(duct2_area / np.pi)

    dec_out = {
        "|p1|": (44.708, None),
        "ph(p1)": (-89.668, None),
        "|U1|": (1.1383e-4, None),
        "ph(U1)": (-179.59, None),
    }

    duct2 = segments.Duct(
        length=duct2_length,
        radius=duct2_radius,
        geometry=geometry.CircularPore(),
    )
    p1_out, U1_out, T_out = duct2.propagate(p1_in, U1_in, T_mean, omega, air_gas)

    # Handle phase wraparound for comparison
    ph_U1_out = np.degrees(np.angle(U1_out))

    dec_out["|p1|"] = (44.708, np.abs(p1_out))
    dec_out["ph(p1)"] = (-89.668, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (1.1383e-4, np.abs(U1_out))
    dec_out["ph(U1)"] = (-179.59, ph_U1_out)

    results, max_err = compare_results("DUCT (volume)", dec_out)
    passed = print_comparison("3: DUCT (bottle volume)", results, max_err)
    all_passed.append(("DUCT (volume)", passed, max_err))

    # Use Reference baseline output as input for next segment
    p1_in = 44.708 * np.exp(1j * np.radians(-89.668))
    U1_in = 1.1383e-4 * np.exp(1j * np.radians(-179.59))

    # =========================================================================
    # Segment 4: SURFACE (bottom end)
    # =========================================================================
    # Reference baseline parameters:
    #   Area = 1.8680E-3 m^2
    # Reference baseline output:
    #   |p| = 44.708 Pa, Ph(p) = -89.668 deg (unchanged)
    #   |U| = 1.1386E-4 m^3/s, Ph(U) = -179.61 deg

    surface = Surface(
        area=duct2_area,
        epsilon_s=0.0,  # ideal solid
        name="bottom",
    )

    dec_out = {
        "|p1|": (44.708, None),
        "ph(p1)": (-89.668, None),
        "|U1|": (1.1386e-4, None),
        "ph(U1)": (-179.61, None),
    }

    p1_out, U1_out, T_out = surface.propagate(p1_in, U1_in, T_mean, omega, air_gas)

    ph_U1_out = np.degrees(np.angle(U1_out))

    dec_out["|p1|"] = (44.708, np.abs(p1_out))
    dec_out["ph(p1)"] = (-89.668, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (1.1386e-4, np.abs(U1_out))
    dec_out["ph(U1)"] = (-179.61, ph_U1_out)

    results, max_err = compare_results("SURFACE (bottom)", dec_out)
    passed = print_comparison("4: SURFACE (bottom end)", results, max_err)
    all_passed.append(("SURFACE", passed, max_err))

    # =========================================================================
    # Chained propagation test (full system using our computed values)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHAINED PROPAGATION (full system)")
    print("=" * 70)

    # Re-run from initial conditions, chaining through our calculations
    p1_chain = 1.0 * np.exp(1j * np.radians(0.0))
    U1_chain = 1.0e-4 * np.exp(1j * np.radians(0.0))
    T_chain = T_mean

    # Segment 1: DUCT (neck)
    p1_chain, U1_chain, T_chain = duct1.propagate(
        p1_chain, U1_chain, T_chain, omega, air_gas
    )
    print("\nAfter DUCT (neck):")
    print(f"  |p1| = {np.abs(p1_chain):.4g} Pa, ph(p1) = {np.degrees(np.angle(p1_chain)):.2f} deg")
    print(f"  |U1| = {np.abs(U1_chain):.4e} m^3/s, ph(U1) = {np.degrees(np.angle(U1_chain)):.2f} deg")

    # Segment 2: CONE
    p1_chain, U1_chain, T_chain = cone.propagate(
        p1_chain, U1_chain, T_chain, omega, air_gas
    )
    print("\nAfter CONE:")
    print(f"  |p1| = {np.abs(p1_chain):.4g} Pa, ph(p1) = {np.degrees(np.angle(p1_chain)):.2f} deg")
    print(f"  |U1| = {np.abs(U1_chain):.4e} m^3/s, ph(U1) = {np.degrees(np.angle(U1_chain)):.2f} deg")

    # Segment 3: DUCT (volume)
    p1_chain, U1_chain, T_chain = duct2.propagate(
        p1_chain, U1_chain, T_chain, omega, air_gas
    )
    print("\nAfter DUCT (volume):")
    print(f"  |p1| = {np.abs(p1_chain):.4g} Pa, ph(p1) = {np.degrees(np.angle(p1_chain)):.2f} deg")
    print(f"  |U1| = {np.abs(U1_chain):.4e} m^3/s, ph(U1) = {np.degrees(np.angle(U1_chain)):.2f} deg")

    # Segment 4: SURFACE (bottom)
    p1_chain, U1_chain, T_chain = surface.propagate(
        p1_chain, U1_chain, T_chain, omega, air_gas
    )
    print("\nAfter SURFACE (bottom):")
    print(f"  |p1| = {np.abs(p1_chain):.4g} Pa, ph(p1) = {np.degrees(np.angle(p1_chain)):.2f} deg")
    print(f"  |U1| = {np.abs(U1_chain):.4e} m^3/s, ph(U1) = {np.degrees(np.angle(U1_chain)):.2f} deg")

    # Compare to Reference baseline final output (after SURFACE)
    print("\nReference baseline final output (after SURFACE):")
    print("  |p1| = 44.708 Pa, ph(p1) = -89.668 deg")
    print("  |U1| = 1.1386e-4 m^3/s, ph(U1) = -179.61 deg")

    p1_err = 100 * (np.abs(p1_chain) - 44.708) / 44.708
    u1_err = 100 * (np.abs(U1_chain) - 1.1386e-4) / 1.1386e-4
    print("\nChained propagation error (vs embedded baseline final):")
    print(f"  |p1| error: {p1_err:+.2f}%")
    print(f"  |U1| error: {u1_err:+.2f}%")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Segment':<25} {'Status':<10} {'Max Error'}")
    print("-" * 70)

    total_passed = 0
    for name, passed, max_err in all_passed:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:<25} {status:<10} {max_err:.2f}%")
        if passed:
            total_passed += 1

    print("-" * 70)
    print(f"Total: {total_passed}/{len(all_passed)} segments passed validation")
    print()

    if total_passed == len(all_passed):
        print("ALL SEGMENTS VALIDATED SUCCESSFULLY")
        return 0
    else:
        print(f"{len(all_passed) - total_passed} segment(s) need review")
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Validation of SX -> STKSCREEN -> SX chain against gamma reference case.

gamma reference case is a Stirling cooler example from Reference baseline. This validation tests
the sequence: aftercooler (SX) -> regenerator (STKSCREEN) -> cold HX (SX).

Reference: <external proprietary source>
"""

import numpy as np
from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: SX -> STKSCREEN -> SX chain (gamma reference case)")
    print("=" * 70)
    print()

    # =========================================================================
    # System parameters from gamma reference case
    # =========================================================================
    mean_P = 2.0e6  # Pa (2 MPa)
    freq = 55.0  # Hz

    # =========================================================================
    # Segment parameters from gamma reference case
    # =========================================================================

    # Segment 9: SX aftercooler
    sx_aftercooler_params = {
        "area": 1.167e-4,  # m² (sameas 10a)
        "porosity": 0.60,
        "length": 1.0e-3,  # m (1 mm)
        "hydraulic_radius": 1.39e-5,  # m (sameas 10d)
        "solid_temperature": 300.0,  # K
    }

    # Segment 10: STKSCREEN regenerator
    stkscreen_params = {
        "area": 1.167e-4,  # m²
        "porosity": 0.6860,
        "length": 5.0e-2,  # m (5 cm)
        "hydraulic_radius": 1.39e-5,  # m
        "ks_frac": 0.3,
    }

    # Segment 11: SX cold heat exchanger
    sx_cold_params = {
        "area": 1.167e-4,  # m² (sameas 10a)
        "porosity": 0.60,
        "length": 1.0e-3,  # m (1 mm)
        "hydraulic_radius": 1.39e-5,  # m (sameas 10d)
        "solid_temperature": 80.0,  # K
    }

    # =========================================================================
    # Reference values from gamma reference case
    # =========================================================================

    # Input to segment 9 (TRUNK flow from TBRANCH segment 4)
    # TBRANCH: U1_trunk = U1_in - U1_branch
    # U1_in = 2.7596e-4 at 18.368° (from segment 3)
    # U1_branch = 6.5763e-5 at 93.004° (Reference baseline segment 4 output)
    # U1_trunk ≈ 2.66e-4 at 4.6°
    input_ref = {
        "p1_mag": 2.9241e5,  # Pa
        "p1_ph": -39.288,  # deg
        "U1_mag": 2.66e-4,  # m³/s (trunk flow, calculated from TBRANCH)
        "U1_ph": 4.6,  # deg (trunk flow)
    }

    # Output of segment 9 (SX aftercooler)
    sx_aftercooler_ref = {
        "p1_mag": 2.8990e5,  # Pa
        "p1_ph": -39.765,  # deg
        "U1_mag": 2.6375e-4,  # m³/s
        "U1_ph": 4.0412,  # deg
        "T_m": 300.21,  # K (GasT)
    }

    # Output of segment 10 (STKSCREEN regenerator)
    stkscreen_ref = {
        "p1_mag": 2.5401e5,  # Pa
        "p1_ph": -44.172,  # deg
        "U1_mag": 5.6191e-5,  # m³/s
        "U1_ph": -74.427,  # deg
        "T_m": 79.964,  # K (TEnd)
    }

    # Output of segment 11 (SX cold HX)
    sx_cold_ref = {
        "p1_mag": 2.5369e5,  # Pa
        "p1_ph": -44.129,  # deg
        "U1_mag": 5.7679e-5,  # m³/s
        "U1_ph": -77.062,  # deg
        "T_m": 79.964,  # K (GasT)
    }

    # =========================================================================
    # Setup
    # =========================================================================
    helium = gas.Helium(mean_pressure=mean_P)
    omega = 2 * np.pi * freq

    print(f"Gas: Helium at {mean_P/1e6:.1f} MPa")
    print(f"Frequency: {freq} Hz (omega = {omega:.1f} rad/s)")
    print()

    # =========================================================================
    # Create segments
    # =========================================================================

    # Solid heat capacity for copper: rho_s * c_s
    # ~8900 kg/m³ * 390 J/(kg·K) = 3.5e6 J/(m³·K)
    solid_heat_capacity_copper = 3.5e6

    # Solid heat capacity for stainless steel at cryogenic temperatures
    solid_heat_capacity_ss = 2.8e6

    aftercooler = segments.SX(
        length=sx_aftercooler_params["length"],
        porosity=sx_aftercooler_params["porosity"],
        hydraulic_radius=sx_aftercooler_params["hydraulic_radius"],
        area=sx_aftercooler_params["area"],
        solid_temperature=sx_aftercooler_params["solid_temperature"],
        solid_heat_capacity=solid_heat_capacity_copper,
        name="aftercooler",
    )

    regenerator = segments.StackScreen(
        length=stkscreen_params["length"],
        porosity=stkscreen_params["porosity"],
        hydraulic_radius=stkscreen_params["hydraulic_radius"],
        area=stkscreen_params["area"],
        ks_frac=stkscreen_params["ks_frac"],
        solid_heat_capacity=solid_heat_capacity_ss,
        T_cold=300.21,  # Input temperature
        T_hot=79.964,   # Output temperature
        name="regenerator",
    )

    cold_hx = segments.SX(
        length=sx_cold_params["length"],
        porosity=sx_cold_params["porosity"],
        hydraulic_radius=sx_cold_params["hydraulic_radius"],
        area=sx_cold_params["area"],
        solid_temperature=sx_cold_params["solid_temperature"],
        solid_heat_capacity=solid_heat_capacity_copper,
        name="cold_hx",
    )

    print("Segments created:")
    print(f"  {aftercooler}")
    print(f"  {regenerator}")
    print(f"  {cold_hx}")
    print()

    # =========================================================================
    # Propagate through the chain
    # =========================================================================
    print("=" * 70)
    print("PROPAGATION")
    print("=" * 70)
    print()

    # Input state
    p1_in = input_ref["p1_mag"] * np.exp(1j * np.radians(input_ref["p1_ph"]))
    U1_in = input_ref["U1_mag"] * np.exp(1j * np.radians(input_ref["U1_ph"]))
    T_m_in = 300.21  # K

    print(f"INPUT (from TBRANCH trunk):")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, Ph = {np.degrees(np.angle(p1_in)):.3f}°")
    print(f"  |U1| = {np.abs(U1_in):.6f} m³/s, Ph = {np.degrees(np.angle(U1_in)):.3f}°")
    print()

    # Segment 9: SX aftercooler
    p1_1, U1_1, T_1 = aftercooler.propagate(p1_in, U1_in, T_m_in, omega, helium)
    print(f"After AFTERCOOLER (SX):")
    print(f"  |p1| = {np.abs(p1_1):.1f} Pa (ref: {sx_aftercooler_ref['p1_mag']:.1f})")
    print(f"  Ph = {np.degrees(np.angle(p1_1)):.3f}° (ref: {sx_aftercooler_ref['p1_ph']:.3f}°)")
    print(f"  |U1| = {np.abs(U1_1):.6f} m³/s (ref: {sx_aftercooler_ref['U1_mag']:.6f})")
    print(f"  Ph = {np.degrees(np.angle(U1_1)):.3f}° (ref: {sx_aftercooler_ref['U1_ph']:.3f}°)")
    print(f"  T_m = {T_1:.2f} K")
    print()

    # Segment 10: STKSCREEN regenerator
    p1_2, U1_2, T_2 = regenerator.propagate(p1_1, U1_1, T_1, omega, helium)
    print(f"After REGENERATOR (STKSCREEN):")
    print(f"  |p1| = {np.abs(p1_2):.1f} Pa (ref: {stkscreen_ref['p1_mag']:.1f})")
    print(f"  Ph = {np.degrees(np.angle(p1_2)):.3f}° (ref: {stkscreen_ref['p1_ph']:.3f}°)")
    print(f"  |U1| = {np.abs(U1_2):.6f} m³/s (ref: {stkscreen_ref['U1_mag']:.6f})")
    print(f"  Ph = {np.degrees(np.angle(U1_2)):.3f}° (ref: {stkscreen_ref['U1_ph']:.3f}°)")
    print(f"  T_m = {T_2:.2f} K")
    print()

    # Segment 11: SX cold HX
    p1_3, U1_3, T_3 = cold_hx.propagate(p1_2, U1_2, T_2, omega, helium)
    print(f"After COLD HX (SX):")
    print(f"  |p1| = {np.abs(p1_3):.1f} Pa (ref: {sx_cold_ref['p1_mag']:.1f})")
    print(f"  Ph = {np.degrees(np.angle(p1_3)):.3f}° (ref: {sx_cold_ref['p1_ph']:.3f}°)")
    print(f"  |U1| = {np.abs(U1_3):.6f} m³/s (ref: {sx_cold_ref['U1_mag']:.6f})")
    print(f"  Ph = {np.degrees(np.angle(U1_3)):.3f}° (ref: {sx_cold_ref['U1_ph']:.3f}°)")
    print(f"  T_m = {T_3:.2f} K")
    print()

    # =========================================================================
    # Validation summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    def phase_error(ours, ref):
        err = ours - ref
        while err > 180:
            err -= 360
        while err < -180:
            err += 360
        return err

    checks = [
        # Aftercooler output
        ("SX1 |p1|", np.abs(p1_1), sx_aftercooler_ref["p1_mag"], "Pa", 5.0),
        ("SX1 Ph(p1)", np.degrees(np.angle(p1_1)), sx_aftercooler_ref["p1_ph"], "deg", 2.0),
        ("SX1 |U1|", np.abs(U1_1), sx_aftercooler_ref["U1_mag"], "m³/s", 10.0),
        ("SX1 Ph(U1)", np.degrees(np.angle(U1_1)), sx_aftercooler_ref["U1_ph"], "deg", 5.0),
        # Regenerator output
        ("STKSCR |p1|", np.abs(p1_2), stkscreen_ref["p1_mag"], "Pa", 5.0),
        ("STKSCR Ph(p1)", np.degrees(np.angle(p1_2)), stkscreen_ref["p1_ph"], "deg", 2.0),
        ("STKSCR |U1|", np.abs(U1_2), stkscreen_ref["U1_mag"], "m³/s", 10.0),
        ("STKSCR Ph(U1)", np.degrees(np.angle(U1_2)), stkscreen_ref["U1_ph"], "deg", 10.0),
        # Cold HX output
        ("SX2 |p1|", np.abs(p1_3), sx_cold_ref["p1_mag"], "Pa", 5.0),
        ("SX2 Ph(p1)", np.degrees(np.angle(p1_3)), sx_cold_ref["p1_ph"], "deg", 2.0),
        ("SX2 |U1|", np.abs(U1_3), sx_cold_ref["U1_mag"], "m³/s", 10.0),
        ("SX2 Ph(U1)", np.degrees(np.angle(U1_3)), sx_cold_ref["U1_ph"], "deg", 10.0),
    ]

    print(f"{'Parameter':<15} {'Ours':<15} {'Reference baseline':<15} {'Error':<12} {'Status'}")
    print("-" * 70)

    all_pass = True
    for name, ours, ref, unit, threshold in checks:
        if "Ph" in name:
            err = phase_error(ours, ref)
            err_str = f"{err:+.2f}°"
            status = "✓ PASS" if abs(err) < threshold else "✗ FAIL"
            if abs(err) >= threshold:
                all_pass = False
            print(f"{name:<15} {ours:<15.3f} {ref:<15.3f} {err_str:<12} {status}")
        else:
            if ref != 0:
                err = 100 * (ours - ref) / ref
                err_str = f"{err:+.1f}%"
                status = "✓ PASS" if abs(err) < threshold else "✗ FAIL"
                if abs(err) >= threshold:
                    all_pass = False
            else:
                err_str = "N/A"
                status = "?"
            if "U1" in name and "Ph" not in name:
                print(f"{name:<15} {ours:<15.6e} {ref:<15.6e} {err_str:<12} {status}")
            else:
                print(f"{name:<15} {ours:<15.1f} {ref:<15.1f} {err_str:<12} {status}")

    print("-" * 70)

    if all_pass:
        print("\n✓ ALL CHECKS PASSED")
    else:
        print("\n✗ Some checks failed")
        print("  Note: Phase errors in the regenerator are expected due to")
        print("  approximations in gc/gv integrals and theta_T calculation.")


if __name__ == "__main__":
    main()

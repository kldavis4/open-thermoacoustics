#!/usr/bin/env python3
"""
Validation of STKDUCT against optr reference case (Orifice Pulse Tube Refrigerator).

optr reference case is a simple OPTR example from Reference baseline. This validation tests
the STKDUCT (pulse tube) segment which connects the cold end to the hot end.

Reference: <external proprietary source>
"""

import numpy as np
from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: STKDUCT against optr reference case (Pulse Tube)")
    print("=" * 70)
    print()

    # =========================================================================
    # System parameters from optr reference case
    # =========================================================================
    mean_P = 3.0e6  # Pa (3 MPa)
    freq = 300.0  # Hz

    # =========================================================================
    # STKDUCT parameters from segment 4 of optr reference case
    # =========================================================================
    stkduct_params = {
        "area": 5.687e-5,  # m²
        "perimeter": 2.674e-2,  # m
        "length": 0.2,  # m
        "wall_area": 1.0e-5,  # m²
        "T_begin": 149.90,  # K (from segment 3 output)
        "T_end": 300.20,  # K
    }

    # Input to STKDUCT (output of segment 3: SX cold HX)
    input_ref = {
        "p1_mag": 1.1527e5,  # Pa
        "p1_ph": -15.367,  # deg
        "U1_mag": 1.2192e-3,  # m³/s
        "U1_ph": -29.702,  # deg
        "T_m": 149.90,  # K
    }

    # Output of STKDUCT (segment 4)
    stkduct_ref = {
        "p1_mag": 1.0482e5,  # Pa
        "p1_ph": -44.00,  # deg
        "U1_mag": 1.3018e-3,  # m³/s
        "U1_ph": -51.089,  # deg
        "T_m": 300.20,  # K
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
    # Gas properties at different temperatures
    # =========================================================================
    T_cold = stkduct_params["T_begin"]
    T_hot = stkduct_params["T_end"]

    print("Gas properties:")
    for T, label in [(T_cold, "cold"), (T_hot, "hot")]:
        rho = helium.density(T)
        a = helium.sound_speed(T)
        mu = helium.viscosity(T)
        k = helium.thermal_conductivity(T)
        cp = helium.specific_heat_cp(T)
        delta_nu = np.sqrt(2 * mu / (omega * rho))
        delta_kappa = np.sqrt(2 * k / (omega * rho * cp))
        print(f"  At {T:.1f} K ({label}):")
        print(f"    rho = {rho:.3f} kg/m³, a = {a:.1f} m/s")
        print(f"    delta_nu = {delta_nu*1e6:.1f} µm, delta_kappa = {delta_kappa*1e6:.1f} µm")
    print()

    # Check boundary layer validity
    A = stkduct_params["area"]
    Pi = stkduct_params["perimeter"]
    rho_cold = helium.density(T_cold)
    mu_cold = helium.viscosity(T_cold)
    delta_nu_cold = np.sqrt(2 * mu_cold / (omega * rho_cold))
    ratio = 2 * A / (Pi * delta_nu_cold)
    print(f"Boundary layer check: 2A/(Π*δ_ν) = {ratio:.1f} (valid if > 30)")
    print()

    # =========================================================================
    # Create STKDUCT
    # =========================================================================
    # Solid heat capacity for stainless steel
    solid_heat_capacity = 3.9e6  # J/(m³·K)
    solid_thermal_conductivity = 15.0  # W/(m·K)

    pulse_tube = segments.StackDuct(
        length=stkduct_params["length"],
        area=stkduct_params["area"],
        perimeter=stkduct_params["perimeter"],
        wall_area=stkduct_params["wall_area"],
        solid_thermal_conductivity=solid_thermal_conductivity,
        solid_heat_capacity=solid_heat_capacity,
        T_cold=stkduct_params["T_begin"],
        T_hot=stkduct_params["T_end"],
        name="pulse_tube",
    )

    print("STKDUCT parameters:")
    print(f"  Area = {stkduct_params['area']*1e4:.4f} cm²")
    print(f"  Perimeter = {stkduct_params['perimeter']*100:.2f} cm")
    print(f"  Length = {stkduct_params['length']*100:.1f} cm")
    print(f"  Wall area = {stkduct_params['wall_area']*1e4:.4f} cm²")
    print(f"  T_begin = {stkduct_params['T_begin']:.2f} K")
    print(f"  T_end = {stkduct_params['T_end']:.2f} K")
    print()
    print(f"Created: {pulse_tube}")
    print(f"  Temperature gradient: {pulse_tube.temperature_gradient():.1f} K/m")
    print()

    # =========================================================================
    # Propagation
    # =========================================================================
    print("=" * 70)
    print("PROPAGATION")
    print("=" * 70)
    print()

    # Input state
    p1_in = input_ref["p1_mag"] * np.exp(1j * np.radians(input_ref["p1_ph"]))
    U1_in = input_ref["U1_mag"] * np.exp(1j * np.radians(input_ref["U1_ph"]))
    T_m_in = input_ref["T_m"]

    print(f"INPUT (from cold HX):")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, Ph = {np.degrees(np.angle(p1_in)):.3f}°")
    print(f"  |U1| = {np.abs(U1_in):.6f} m³/s, Ph = {np.degrees(np.angle(U1_in)):.3f}°")
    print(f"  T_m = {T_m_in:.2f} K")
    print()

    # Propagate
    p1_out, U1_out, T_out = pulse_tube.propagate(p1_in, U1_in, T_m_in, omega, helium)

    print(f"OUTPUT (our calculation):")
    print(f"  |p1| = {np.abs(p1_out):.1f} Pa, Ph = {np.degrees(np.angle(p1_out)):.3f}°")
    print(f"  |U1| = {np.abs(U1_out):.6f} m³/s, Ph = {np.degrees(np.angle(U1_out)):.3f}°")
    print(f"  T_m = {T_out:.2f} K")
    print()

    print(f"Embedded reference:")
    print(f"  |p1| = {stkduct_ref['p1_mag']:.1f} Pa, Ph = {stkduct_ref['p1_ph']:.3f}°")
    print(f"  |U1| = {stkduct_ref['U1_mag']:.6f} m³/s, Ph = {stkduct_ref['U1_ph']:.3f}°")
    print(f"  T_m = {stkduct_ref['T_m']:.2f} K")
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
        ("Pressure |p1|", np.abs(p1_out), stkduct_ref["p1_mag"], "Pa", 5.0),
        ("Pressure Ph(p1)", np.degrees(np.angle(p1_out)), stkduct_ref["p1_ph"], "deg", 5.0),
        ("Velocity |U1|", np.abs(U1_out), stkduct_ref["U1_mag"], "m³/s", 10.0),
        ("Velocity Ph(U1)", np.degrees(np.angle(U1_out)), stkduct_ref["U1_ph"], "deg", 10.0),
        ("Temperature", T_out, stkduct_ref["T_m"], "K", 1.0),
    ]

    print(f"{'Parameter':<20} {'Ours':<15} {'Reference baseline':<15} {'Error':<12} {'Status'}")
    print("-" * 75)

    all_pass = True
    for name, ours, ref, unit, threshold in checks:
        if "Ph" in name:
            err = phase_error(ours, ref)
            err_str = f"{err:+.2f}°"
            status = "✓ PASS" if abs(err) < threshold else "✗ FAIL"
            if abs(err) >= threshold:
                all_pass = False
            print(f"{name:<20} {ours:<15.3f} {ref:<15.3f} {err_str:<12} {status}")
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
                print(f"{name:<20} {ours:<15.6e} {ref:<15.6e} {err_str:<12} {status}")
            else:
                print(f"{name:<20} {ours:<15.1f} {ref:<15.1f} {err_str:<12} {status}")

    print("-" * 75)

    if all_pass:
        print("\n✓ ALL CHECKS PASSED")
    else:
        print("\n✗ Some checks failed")
        print("  Note: Discrepancies may arise from:")
        print("  - Boundary layer vs exact functions")
        print("  - Solid wall thermal effects (ε_s)")
        print("  - Temperature-dependent property variations")


if __name__ == "__main__":
    main()

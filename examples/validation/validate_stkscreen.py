#!/usr/bin/env python3
"""
Validation of STKSCREEN (wire mesh regenerator) using gamma embedded reference values.

The gamma reference case is a Stirling cooler baseline. The STKSCREEN segment
models the regenerator, which has a temperature gradient from hot (300.21 K)
to cold (79.964 K).

Reference: <external proprietary source>
"""

import numpy as np
from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: STKSCREEN against gamma reference case (Stirling cooler)")
    print("=" * 70)
    print()

    # =========================================================================
    # Embedded reference values from gamma reference case
    # =========================================================================

    # System parameters
    mean_P = 2.0e6  # Pa (2 MPa)
    freq = 55.0  # Hz

    # STKSCREEN parameters (segment 10)
    stkscreen_params = {
        "area": 1.167e-4,  # m²
        "porosity": 0.6860,
        "length": 5.0e-2,  # m (5 cm)
        "hydraulic_radius": 1.39e-5,  # m
        "ks_frac": 0.3,
        "solid_thermal_conductivity": 15.0,  # stainless steel
        "T_hot": 300.21,  # K (at x=length)
        "T_cold": 79.964,  # K (at x=0)
    }

    # Note on temperature convention:
    # In gamma reference case, TBeg=300.21 K (input side) and TEnd=79.964 K (output side)
    # The aftercooler (segment 9) is at ~300 K, so flow goes hot->cold
    # Our convention: T_cold at x=0, T_hot at x=length
    # But gamma flow direction is: aftercooler (300K) -> regenerator -> cold HX (80K)
    # So x=0 is at aftercooler (hot), x=length is at cold end
    # We need to swap: T_cold=300.21 at input (x=0), T_hot=79.964 at output

    # Actually, looking at gamma reference case more carefully:
    # - Segment 9 (SX aftercooler) output: GasT = 300.21 K
    # - Segment 10 (STKSCREEN): TBeg = 300.21 K, TEnd = 79.964 K
    # So input is at 300.21 K, output at 79.964 K
    # Our StackScreen: T_cold at x=0, T_hot at x=length
    # For gamma flow: input (x=0) at 300K, output (x=length) at 80K
    # So we need T_cold at input? No, that's backwards.
    #
    # Let me re-read: temperature_at() does T_cold at x=0, T_hot at x=length
    # If input is hot (300K) and output is cold (80K), then:
    # - x=0 (input) should be 300K
    # - x=length (output) should be 80K
    # So T_cold=80K and T_hot=300K gives: T(0)=80K, T(length)=300K - wrong!
    #
    # The issue is our convention: we have T_cold at x=0 always.
    # For gamma reference case, the flow is hot->cold, so:
    # - T(x=0) = 300.21 K (hot input)
    # - T(x=length) = 79.964 K (cold output)
    # This means we need to swap our T_hot and T_cold interpretation.

    # For this validation, use T_cold and T_hot as encoded in the embedded baseline:
    # T_hot = temperature at input (x=0) = 300.21 K
    # T_cold = temperature at output (x=length) = 79.964 K
    # We need to modify the StackScreen or interpret correctly.

    # Looking at StackScreen.temperature_at():
    # return self._T_cold + (self._T_hot - self._T_cold) * x / self._length
    # At x=0: returns T_cold
    # At x=length: returns T_hot
    #
    # For gamma reference case:
    # x=0 (input) = 300.21 K --> this should be T_cold in our code
    # x=length (output) = 79.964 K --> this should be T_hot in our code
    #
    # So: T_cold = 300.21 K, T_hot = 79.964 K (counterintuitive naming!)

    stkscreen_params["T_cold"] = 300.21  # K (at x=0, the INPUT)
    stkscreen_params["T_hot"] = 79.964  # K (at x=length, the OUTPUT)

    # Input state (from segment 9 - SX aftercooler output)
    input_state = {
        "p1_mag": 2.8990e5,  # Pa
        "p1_ph": -39.765,  # deg
        "U1_mag": 2.6375e-4,  # m³/s
        "U1_ph": 4.0412,  # deg
        "T_m": 300.21,  # K (GasT at aftercooler output)
    }

    # Output state (STKSCREEN segment 10 output)
    output_ref = {
        "p1_mag": 2.5401e5,  # Pa
        "p1_ph": -44.172,  # deg
        "U1_mag": 5.6191e-5,  # m³/s
        "U1_ph": -74.427,  # deg
        "T_m": 79.964,  # K (TEnd)
    }

    # =========================================================================
    # Setup
    # =========================================================================
    helium = gas.Helium(mean_pressure=mean_P)
    omega = 2 * np.pi * freq

    print(f"Gas: Helium at {mean_P/1e6:.1f} MPa")
    print(f"Frequency: {freq} Hz (omega = {omega:.1f} rad/s)")
    print()

    # Print gas properties at both temperatures
    print("Gas properties:")
    for T, label in [(300.21, "hot end"), (79.964, "cold end")]:
        rho = helium.density(T)
        a = helium.sound_speed(T)
        mu = helium.viscosity(T)
        k = helium.thermal_conductivity(T)
        cp = helium.specific_heat_cp(T)
        gamma = helium.gamma(T)
        sigma = helium.prandtl(T)
        print(f"  At {T:.1f} K ({label}):")
        print(f"    rho = {rho:.3f} kg/m³, a = {a:.1f} m/s")
        print(f"    gamma = {gamma:.3f}, Pr = {sigma:.3f}")
    print()

    # =========================================================================
    # Create StackScreen segment
    # =========================================================================
    print("STKSCREEN parameters:")
    print(f"  Area = {stkscreen_params['area']*1e4:.4f} cm²")
    print(f"  Porosity = {stkscreen_params['porosity']:.4f}")
    print(f"  Length = {stkscreen_params['length']*1e2:.1f} cm")
    print(f"  Hydraulic radius = {stkscreen_params['hydraulic_radius']*1e6:.1f} µm")
    print(f"  ks_frac = {stkscreen_params['ks_frac']}")
    print(f"  T (x=0) = {stkscreen_params['T_cold']:.2f} K")
    print(f"  T (x=length) = {stkscreen_params['T_hot']:.2f} K")
    print()

    # Solid heat capacity for stainless steel: rho_s * c_s
    # At 300K: ~8000 kg/m³ * 500 J/(kg·K) = 4.0e6 J/(m³·K)
    # At 80K: ~8000 kg/m³ * 200 J/(kg·K) = 1.6e6 J/(m³·K)
    # Use average: ~2.8e6 J/(m³·K)
    solid_heat_capacity = 2.8e6  # J/(m³·K)

    regen = segments.StackScreen(
        length=stkscreen_params["length"],
        porosity=stkscreen_params["porosity"],
        hydraulic_radius=stkscreen_params["hydraulic_radius"],
        area=stkscreen_params["area"],
        ks_frac=stkscreen_params["ks_frac"],
        solid_heat_capacity=solid_heat_capacity,
        solid_thermal_conductivity=stkscreen_params["solid_thermal_conductivity"],
        T_hot=stkscreen_params["T_hot"],
        T_cold=stkscreen_params["T_cold"],
        name="regenerator",
    )

    print(f"Created: {regen}")
    print(f"  Temperature gradient: {regen.temperature_gradient():.1f} K/m")
    print()

    # =========================================================================
    # Propagate through regenerator
    # =========================================================================
    print("=" * 70)
    print("PROPAGATION")
    print("=" * 70)
    print()

    # Convert input to complex
    p1_in = input_state["p1_mag"] * np.exp(1j * np.radians(input_state["p1_ph"]))
    U1_in = input_state["U1_mag"] * np.exp(1j * np.radians(input_state["U1_ph"]))
    T_m_in = input_state["T_m"]

    print(f"INPUT (from aftercooler):")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, Ph = {np.degrees(np.angle(p1_in)):.3f}°")
    print(f"  |U1| = {np.abs(U1_in):.6f} m³/s, Ph = {np.degrees(np.angle(U1_in)):.3f}°")
    print(f"  T_m = {T_m_in:.2f} K")
    print()

    # Propagate
    p1_out, U1_out, T_m_out = regen.propagate(p1_in, U1_in, T_m_in, omega, helium)

    print(f"OUTPUT (our calculation):")
    print(f"  |p1| = {np.abs(p1_out):.1f} Pa, Ph = {np.degrees(np.angle(p1_out)):.3f}°")
    print(f"  |U1| = {np.abs(U1_out):.6f} m³/s, Ph = {np.degrees(np.angle(U1_out)):.3f}°")
    print(f"  T_m = {T_m_out:.2f} K")
    print()

    print(f"Embedded reference:")
    print(f"  |p1| = {output_ref['p1_mag']:.1f} Pa, Ph = {output_ref['p1_ph']:.3f}°")
    print(f"  |U1| = {output_ref['U1_mag']:.6f} m³/s, Ph = {output_ref['U1_ph']:.3f}°")
    print(f"  T_m = {output_ref['T_m']:.2f} K")
    print()

    # =========================================================================
    # Validation summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    checks = [
        ("Pressure |p1|", np.abs(p1_out), output_ref["p1_mag"], "Pa"),
        ("Pressure Ph(p1)", np.degrees(np.angle(p1_out)), output_ref["p1_ph"], "deg"),
        ("Velocity |U1|", np.abs(U1_out), output_ref["U1_mag"], "m³/s"),
        ("Velocity Ph(U1)", np.degrees(np.angle(U1_out)), output_ref["U1_ph"], "deg"),
        ("Temperature", T_m_out, output_ref["T_m"], "K"),
    ]

    print(f"{'Parameter':<20} {'Ours':<15} {'Reference baseline':<15} {'Error':<10} {'Status'}")
    print("-" * 75)

    all_pass = True
    for name, ours, ref, unit in checks:
        if "Ph" in name:
            # Phase error should be checked differently
            err = ours - ref
            # Normalize to [-180, 180]
            while err > 180:
                err -= 360
            while err < -180:
                err += 360
            err_str = f"{err:+.2f}°"
            # 7° tolerance accounts for gc/gv integral approximations (documented in CLAUDE.md)
            status = "✓ PASS" if abs(err) < 7.0 else "✗ FAIL"
            if abs(err) >= 7.0:
                all_pass = False
        else:
            if ref != 0:
                err = 100 * (ours - ref) / ref
                err_str = f"{err:+.1f}%"
                status = "✓ PASS" if abs(err) < 5.0 else "✗ FAIL"
                if abs(err) >= 5.0:
                    all_pass = False
            else:
                err_str = "N/A"
                status = "?"

        if "U1" in name and "Ph" not in name:
            print(f"{name:<20} {ours:<15.6e} {ref:<15.6e} {err_str:<10} {status}")
        elif "Ph" in name:
            print(f"{name:<20} {ours:<15.3f} {ref:<15.3f} {err_str:<10} {status}")
        else:
            print(f"{name:<20} {ours:<15.1f} {ref:<15.1f} {err_str:<10} {status}")

    print("-" * 75)

    if all_pass:
        print("\n✓ STKSCREEN VALIDATION PASSED (<5% error on magnitudes, <7° on phases)")
    else:
        print("\n✗ Some checks failed")
        print()
        print("ANALYSIS:")
        print("Large errors may indicate:")
        print("1. Different thermoviscous function formulation (boundary layer vs exact)")
        print("2. Reference baseline may use different correlations for wire mesh")
        print("3. Our boundary layer approximation may not be appropriate for this rh")
        print()

        # Calculate some diagnostic values
        rh = stkscreen_params["hydraulic_radius"]
        T_avg = (300.21 + 79.964) / 2
        rho = helium.density(T_avg)
        mu = helium.viscosity(T_avg)
        k = helium.thermal_conductivity(T_avg)
        cp = helium.specific_heat_cp(T_avg)

        from openthermoacoustics.utils import (
            penetration_depth_thermal,
            penetration_depth_viscous,
        )

        delta_nu = penetration_depth_viscous(omega, rho, mu)
        delta_kappa = penetration_depth_thermal(omega, rho, k, cp)

        print(f"Diagnostic parameters at T_avg = {T_avg:.1f} K:")
        print(f"  delta_nu = {delta_nu*1e6:.1f} µm")
        print(f"  delta_kappa = {delta_kappa*1e6:.1f} µm")
        print(f"  rh = {rh*1e6:.1f} µm")
        print(f"  rh/delta_nu = {rh/delta_nu:.2f}")
        print(f"  rh/delta_kappa = {rh/delta_kappa:.2f}")


if __name__ == "__main__":
    main()

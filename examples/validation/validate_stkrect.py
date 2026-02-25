#!/usr/bin/env python3
"""
Validation of STKRECT (rectangular pore stack) against reid4 reference case from published literature.

reid4 reference case is a thermoacoustic refrigerator example with a rectangular-pore stack
made of kapton. The stack has different half-widths in the two directions,
making it a true rectangular (not square) pore geometry.

Reference: published literature Version 7.0.2, page 113 (reid4 reference case example)
"""

import numpy as np
from openthermoacoustics import gas, segments
from openthermoacoustics.geometry import RectangularPore


def main():
    print("=" * 70)
    print("VALIDATION: STKRECT against reid4 reference case (rectangular pore stack)")
    print("=" * 70)
    print()

    # =========================================================================
    # Embedded reference values from reid4 reference case (published literature page 113)
    # =========================================================================

    # System parameters (from BEGIN segment 0)
    mean_P = 3.2388e5  # Pa
    freq = 92.0  # Hz

    # Gas: HeAr mixture with nL=0.92 (92% Helium, 8% Argon by mole fraction)
    he_fraction = 0.92

    # STKRECT parameters (segment 6)
    stkrect_params = {
        "area": 1.8824e-2,  # m²
        "porosity": 0.7050,  # GasA/A
        "length": 0.1524,  # m
        "half_width_a": 4.0640e-4,  # m (aa parameter)
        "half_width_b": 6.3500e-3,  # m (bb parameter)
        "T_beg": 302.34,  # K (at x=0, input)
        "T_end": 295.21,  # K (at x=length, output)
    }

    # Input state (from segment 5 HX output)
    input_state = {
        "p1_mag": 6169.6,  # Pa
        "p1_ph": 89.798,  # deg
        "U1_mag": 7.4562e-2,  # m³/s
        "U1_ph": 6.2380,  # deg
        "T_m": 302.34,  # K (GasT)
    }

    # Output state (STKRECT segment 6 output)
    output_ref = {
        "p1_mag": 5548.2,  # Pa
        "p1_ph": 93.318,  # deg
        "U1_mag": 9.1545e-2,  # m³/s
        "U1_ph": 3.8444,  # deg
        "T_m": 295.21,  # K (TEnd)
        "Htot": -66.673,  # W
        "Edot": 2.3326,  # W
    }

    # =========================================================================
    # Setup
    # =========================================================================
    omega = 2 * np.pi * freq

    # Create HeAr gas mixture
    he_ar = gas.helium_argon(he_fraction=he_fraction, mean_pressure=mean_P)

    print(f"Gas: Helium-Argon mixture ({he_fraction*100:.0f}% He) at {mean_P/1e5:.3f} bar")
    print(f"Frequency: {freq} Hz (omega = {omega:.2f} rad/s)")
    print()

    # Print gas properties at input temperature
    T = input_state["T_m"]
    print("Gas properties at input temperature:")
    print(f"  T = {T:.2f} K")
    print(f"  rho = {he_ar.density(T):.4f} kg/m³")
    print(f"  a = {he_ar.sound_speed(T):.1f} m/s")
    print(f"  gamma = {he_ar.gamma(T):.4f}")
    print(f"  Pr = {he_ar.prandtl(T):.4f}")
    print(f"  mu = {he_ar.viscosity(T):.4e} Pa·s")
    print(f"  k = {he_ar.thermal_conductivity(T):.4f} W/(m·K)")
    print()

    # Calculate penetration depths
    from openthermoacoustics.utils import (
        penetration_depth_thermal,
        penetration_depth_viscous,
    )

    rho = he_ar.density(T)
    mu = he_ar.viscosity(T)
    k = he_ar.thermal_conductivity(T)
    cp = he_ar.specific_heat_cp(T)

    delta_nu = penetration_depth_viscous(omega, rho, mu)
    delta_kappa = penetration_depth_thermal(omega, rho, k, cp)

    print("Penetration depths:")
    print(f"  delta_nu = {delta_nu*1e6:.1f} µm")
    print(f"  delta_kappa = {delta_kappa*1e6:.1f} µm")
    print()

    # =========================================================================
    # Create RectangularPore geometry and Stack segment
    # =========================================================================
    rect_geom = RectangularPore(
        half_width_a=stkrect_params["half_width_a"],
        half_width_b=stkrect_params["half_width_b"],
    )

    print("RectangularPore geometry:")
    print(f"  half_width_a (aa) = {rect_geom.half_width_a*1e6:.1f} µm")
    print(f"  half_width_b (bb) = {rect_geom.half_width_b*1e6:.1f} µm")
    print(f"  hydraulic radius = {rect_geom.hydraulic_radius*1e6:.1f} µm")
    print(f"  aspect ratio (b/a) = {rect_geom.aspect_ratio:.2f}")
    print(f"  aa/delta_nu = {stkrect_params['half_width_a']/delta_nu:.2f}")
    print(f"  bb/delta_nu = {stkrect_params['half_width_b']/delta_nu:.2f}")
    print(f"  r_h/delta_nu = {rect_geom.hydraulic_radius/delta_nu:.2f}")
    print()

    # Temperature convention:
    # - Our Stack uses T_cold at x=0, T_hot at x=length
    # - reid4 has TBeg=302.34 K at x=0, TEnd=295.21 K at x=length
    # - So T_cold (at x=0) = 302.34 K, T_hot (at x=length) = 295.21 K
    stack = segments.Stack(
        length=stkrect_params["length"],
        porosity=stkrect_params["porosity"],
        hydraulic_radius=rect_geom.hydraulic_radius,
        area=stkrect_params["area"],
        geometry=rect_geom,
        T_cold=stkrect_params["T_beg"],  # T at x=0 (input)
        T_hot=stkrect_params["T_end"],  # T at x=length (output)
        name="rectangular_stack",
    )

    print("Stack parameters:")
    print(f"  Area = {stkrect_params['area']*1e4:.3f} cm²")
    print(f"  Porosity = {stkrect_params['porosity']:.4f}")
    print(f"  Length = {stkrect_params['length']*1e2:.2f} cm")
    print(f"  Hydraulic radius = {rect_geom.hydraulic_radius*1e6:.1f} µm")
    print(f"  T (x=0, input) = {stkrect_params['T_beg']:.2f} K")
    print(f"  T (x=length, output) = {stkrect_params['T_end']:.2f} K")
    print(f"  Temperature gradient = {(stkrect_params['T_end'] - stkrect_params['T_beg'])/stkrect_params['length']:.1f} K/m")
    print()

    # =========================================================================
    # Propagate through stack
    # =========================================================================
    print("=" * 70)
    print("PROPAGATION")
    print("=" * 70)
    print()

    # Convert input to complex
    p1_in = input_state["p1_mag"] * np.exp(1j * np.radians(input_state["p1_ph"]))
    U1_in = input_state["U1_mag"] * np.exp(1j * np.radians(input_state["U1_ph"]))
    T_m_in = input_state["T_m"]

    print("INPUT (from HX segment 5):")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, Ph = {np.degrees(np.angle(p1_in)):.3f}°")
    print(f"  |U1| = {np.abs(U1_in):.6f} m³/s, Ph = {np.degrees(np.angle(U1_in)):.3f}°")
    print(f"  T_m = {T_m_in:.2f} K")
    print()

    # Propagate
    p1_out, U1_out, T_m_out = stack.propagate(p1_in, U1_in, T_m_in, omega, he_ar)

    print("OUTPUT (our calculation):")
    print(f"  |p1| = {np.abs(p1_out):.1f} Pa, Ph = {np.degrees(np.angle(p1_out)):.3f}°")
    print(f"  |U1| = {np.abs(U1_out):.6f} m³/s, Ph = {np.degrees(np.angle(U1_out)):.3f}°")
    print(f"  T_m = {T_m_out:.2f} K")
    print()

    print("Embedded reference:")
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
            # Phase error
            err = ours - ref
            # Normalize to [-180, 180]
            while err > 180:
                err -= 360
            while err < -180:
                err += 360
            err_str = f"{err:+.2f}°"
            # Allow up to 5° phase error for stacks
            status = "PASS" if abs(err) < 5.0 else "FAIL"
            if abs(err) >= 5.0:
                all_pass = False
        else:
            if ref != 0:
                err = 100 * (ours - ref) / ref
                err_str = f"{err:+.1f}%"
                # Allow up to 5% amplitude error
                status = "PASS" if abs(err) < 5.0 else "FAIL"
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
        print("\nSTKRECT VALIDATION PASSED (<5% amplitude error, <5 degree phase error)")
    else:
        print("\nSome checks failed")
        print()
        print("Notes:")
        print("- STKRECT uses rectangular pore geometry with independent aa and bb")
        print("- The thermoviscous function uses a double series expansion")
        print("- Errors may be due to different series truncation or gas property models")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

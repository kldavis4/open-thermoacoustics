#!/usr/bin/env python3
"""
Reference Comparison: Hofler Refrigerator Stack Segment

Compares OpenThermoacoustics stack propagation against embedded reference baseline.

Embedded Hofler1 reference values for stack segment (segment 4):
- Input from HX (segment 3): |p1|=29570 Pa, Ph=-0.13°, |U1|=3.057e-3 m³/s, Ph=-81.9°
- Output: |p1|=26103 Pa, Ph=1.43°, |U1|=6.800e-3 m³/s, Ph=-88.0°
- Temperature: 300 K -> 217 K
- Acoustic power: 6.49 W -> 0.94 W (power absorbed by refrigeration)
"""

import numpy as np
from openthermoacoustics import gas, segments, geometry


def main():
    print("=" * 70)
    print("COMPARISON: OpenThermoacoustics vs embedded reference baseline")
    print("Model: Hofler 1986 Stack Segment (parallel plate stack)")
    print("=" * 70)
    print()

    # =========================================================================
    # Embedded reference values
    # =========================================================================
    # Input to stack (from segment 3 output)
    reference_baseline_in = {
        "p1": 29570.0,  # Pa
        "ph_p": -0.12971,  # deg
        "U1": 3.0568e-3,  # m³/s
        "ph_U": -81.875,  # deg
        "Edot": 6.4892,  # W
        "T": 300.0,  # K
    }

    # Output from stack (segment 4)
    reference_baseline_out = {
        "p1": 26103.0,  # Pa
        "ph_p": 1.4277,  # deg
        "U1": 6.8004e-3,  # m³/s
        "ph_U": -87.965,  # deg
        "Edot": 0.94042,  # W
        "T": 217.03,  # K
    }

    # =========================================================================
    # Setup
    # =========================================================================
    helium = gas.Helium(mean_pressure=1.0e6)
    T_m = 300.0  # K (mean for property evaluation)
    freq = 500.0  # Hz
    omega = 2 * np.pi * freq

    # Stack geometry from embedded reference baseline
    # Area = 1.134e-3 m² (sameas 1a)
    # GasA/A (porosity) = 0.7240
    # Length = 7.85e-2 m
    # y0 (half-gap) = 1.8e-4 m
    # Lplate = 4.0e-5 m
    area = 1.134e-3  # m²
    porosity = 0.7240
    length = 7.85e-2  # m
    y0 = 1.8e-4  # m (half-gap for parallel plates)
    hydraulic_radius = y0  # For parallel plates, r_h = y0

    print("Stack Geometry (Parallel Plate):")
    print(f"  Cross-sectional area: {area*1e4:.4f} cm²")
    print(f"  Porosity: {porosity:.4f}")
    print(f"  Length: {length*100:.2f} cm")
    print(f"  Half-gap y0: {y0*1e6:.1f} µm")
    print(f"  Hydraulic radius: {hydraulic_radius*1e6:.1f} µm")
    print()

    print("Temperature Profile:")
    print(f"  T_hot (input): {reference_baseline_in['T']:.1f} K")
    print(f"  T_cold (output): {reference_baseline_out['T']:.2f} K")
    print(f"  Temperature gradient: {(reference_baseline_out['T']-reference_baseline_in['T'])/length:.1f} K/m")
    print()

    # =========================================================================
    # Create Stack Segment
    # =========================================================================
    # Our Stack uses T_cold at x=0 and T_hot at x=length
    # Reference baseline has 300K at input (x=0), 217K at output (x=length)
    # So: T_cold (at x=0) = 300K, T_hot (at x=L) = 217K
    # (confusing naming, but matches the math)
    stack = segments.Stack(
        length=length,
        porosity=porosity,
        hydraulic_radius=hydraulic_radius,
        area=area,  # Must specify total area for correct impedance
        geometry=geometry.ParallelPlate(),
        T_cold=reference_baseline_in["T"],  # 300 K at x=0 (input)
        T_hot=reference_baseline_out["T"],  # 217 K at x=L (output)
    )

    # Initial conditions
    p1_in = reference_baseline_in["p1"] * np.exp(1j * np.radians(reference_baseline_in["ph_p"]))
    U1_in = reference_baseline_in["U1"] * np.exp(1j * np.radians(reference_baseline_in["ph_U"]))

    print("Input Conditions:")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, phase = {np.degrees(np.angle(p1_in)):.3f}°")
    print(f"  |U1| = {np.abs(U1_in)*1e3:.4f} m³/s, phase = {np.degrees(np.angle(U1_in)):.3f}°")
    print(f"  Acoustic power: {reference_baseline_in['Edot']:.4f} W")
    print()

    # =========================================================================
    # Propagate through stack
    # =========================================================================
    p1_out, U1_out, T_out = stack.propagate(p1_in, U1_in, reference_baseline_in["T"], omega, helium)

    # Calculate acoustic power
    power_in = 0.5 * np.real(p1_in * np.conj(U1_in))
    power_out = 0.5 * np.real(p1_out * np.conj(U1_out))

    # =========================================================================
    # Compare Results
    # =========================================================================
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Parameter':<25} {'Reference baseline':<18} {'OpenThermo':<18} {'Error':<12}")
    print("-" * 70)

    # |p1| at output
    ota_p1 = np.abs(p1_out)
    dec_p1 = reference_baseline_out["p1"]
    err_p1 = 100 * (ota_p1 - dec_p1) / dec_p1
    print(f"|p1| at output (Pa)      {dec_p1:<18.1f} {ota_p1:<18.1f} {err_p1:<+12.2f}%")

    # Phase of p1
    ota_ph_p = np.degrees(np.angle(p1_out))
    dec_ph_p = reference_baseline_out["ph_p"]
    err_ph_p = ota_ph_p - dec_ph_p
    print(f"Phase(p1) (deg)         {dec_ph_p:<18.3f} {ota_ph_p:<18.3f} {err_ph_p:<+12.3f}°")

    # |U1| at output
    ota_U1 = np.abs(U1_out)
    dec_U1 = reference_baseline_out["U1"]
    err_U1 = 100 * (ota_U1 - dec_U1) / dec_U1
    print(f"|U1| at output (m³/s)   {dec_U1:<18.6f} {ota_U1:<18.6f} {err_U1:<+12.2f}%")

    # Phase of U1
    ota_ph_U = np.degrees(np.angle(U1_out))
    dec_ph_U = reference_baseline_out["ph_U"]
    err_ph_U = ota_ph_U - dec_ph_U
    print(f"Phase(U1) (deg)         {dec_ph_U:<18.3f} {ota_ph_U:<18.3f} {err_ph_U:<+12.3f}°")

    # Output temperature
    err_T = T_out - reference_baseline_out["T"]
    print(f"Temperature (K)          {reference_baseline_out['T']:<18.2f} {T_out:<18.2f} {err_T:<+12.2f} K")

    # Acoustic power
    ota_Edot = power_out
    dec_Edot = reference_baseline_out["Edot"]
    if abs(dec_Edot) > 0.01:
        err_Edot = 100 * (ota_Edot - dec_Edot) / dec_Edot
        print(f"Acoustic power (W)       {dec_Edot:<18.4f} {ota_Edot:<18.4f} {err_Edot:<+12.2f}%")
    else:
        print(f"Acoustic power (W)       {dec_Edot:<18.4f} {ota_Edot:<18.4f} (small)")

    print("-" * 70)
    print()

    # Power absorption
    power_absorbed = power_in - power_out
    dec_absorbed = reference_baseline_in["Edot"] - reference_baseline_out["Edot"]
    print(f"Power absorbed by stack: Reference baseline={dec_absorbed:.3f} W, OpenThermo={power_absorbed:.3f} W")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    max_err = max(abs(err_p1), abs(err_U1))
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Maximum amplitude error: {max_err:.1f}%")
    print()

    if max_err < 10.0:
        print("✓ VALIDATION PASSED: Results agree within 10%")
    elif max_err < 20.0:
        print("~ VALIDATION MARGINAL: Results agree within 20%")
    else:
        print("✗ VALIDATION NEEDS REVIEW: Results differ significantly")
        print()
        print("Note: Stack physics is complex. Differences may be due to:")
        print("  - Different treatment of temperature-dependent properties")
        print("  - Different thermoviscous function implementations")
        print("  - the baseline's STKSLAB uses specific parallel-plate formulas")


if __name__ == "__main__":
    main()

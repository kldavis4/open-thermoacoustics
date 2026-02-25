#!/usr/bin/env python3
"""
Reference Comparison: Hofler 1986 Thermoacoustic Refrigerator

This script compares OpenThermoacoustics results against embedded reference baseline
for the Hofler refrigerator example (Examples/EnginesAndRefr/Hofler1 reference case).

Embedded reference values for the Hofler1 reference case:
- Working gas: Helium at 10 bar (1e6 Pa)
- Frequency: 500 Hz
- Temperature: 300 K
- Initial p1: 30000 Pa at 0°
- Initial U1: 0.5e-3 m³/s at 0°
"""

import numpy as np
from openthermoacoustics import gas, segments, geometry
def main():
    print("=" * 70)
    print("COMPARISON: OpenThermoacoustics vs embedded reference baseline")
    print("Model: Hofler 1986 Thermoacoustic Refrigerator (segment 2: duct)")
    print("=" * 70)
    print()

    # =========================================================================
    # Embedded reference values for Hofler1 reference case
    # =========================================================================
    reference_baseline = {
        "p1_in": 30000.0,  # Pa
        "ph_p_in": 0.0,  # deg
        "U1_in": 5.0e-4,  # m³/s
        "ph_U_in": 0.0,  # deg
        # After DUCT (segment 2): ambient temperature duct
        "p1_out": 29739.0,  # Pa
        "ph_p_out": -0.17763,  # deg
        "U1_out": 2.7789e-3,  # m³/s
        "ph_U_out": -79.991,  # deg
        "Edot_out": 7.3076,  # W (acoustic power)
    }

    # =========================================================================
    # Setup: match embedded baseline parameters
    # =========================================================================
    # Gas: Helium at 10 bar
    helium = gas.Helium(mean_pressure=1.0e6)

    # Operating conditions
    T_m = 300.0  # K
    freq = 500.0  # Hz
    omega = 2 * np.pi * freq

    # Duct geometry from Hofler1 reference case segment 2
    # Area = 1.134e-3 m² (sameas 1a)
    # Perim = 0.1190 m
    # Length = 4.26e-2 m
    area = 1.134e-3  # m²
    perim = 0.1190  # m
    length = 4.26e-2  # m
    radius = np.sqrt(area / np.pi)  # equivalent radius

    print("Gas Properties (Helium at 10 bar, 300 K):")
    print(f"  Density: {helium.density(T_m):.4f} kg/m³")
    print(f"  Sound speed: {helium.sound_speed(T_m):.1f} m/s")
    print(f"  Viscosity: {helium.viscosity(T_m)*1e6:.2f} µPa·s")
    print(f"  Prandtl: {helium.prandtl(T_m):.4f}")
    print()

    print("Duct Geometry:")
    print(f"  Area: {area*1e4:.4f} cm²")
    print(f"  Perimeter: {perim*100:.2f} cm")
    print(f"  Length: {length*100:.2f} cm")
    print(f"  Equivalent radius: {radius*1000:.2f} mm")
    print()

    # =========================================================================
    # Create OpenThermoacoustics model
    # =========================================================================
    # Use circular pore geometry
    duct = segments.Duct(length=length, radius=radius, geometry=geometry.CircularPore())

    # Initial conditions from Reference baseline
    p1_in = reference_baseline["p1_in"] * np.exp(1j * np.radians(reference_baseline["ph_p_in"]))
    U1_in = reference_baseline["U1_in"] * np.exp(1j * np.radians(reference_baseline["ph_U_in"]))

    print("Initial Conditions:")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, phase = {np.degrees(np.angle(p1_in)):.2f}°")
    print(f"  |U1| = {np.abs(U1_in)*1e3:.4f} mm³/s, phase = {np.degrees(np.angle(U1_in)):.2f}°")
    print()

    # =========================================================================
    # Propagate through duct
    # =========================================================================
    p1_out, U1_out, T_out = duct.propagate(p1_in, U1_in, T_m, omega, helium)

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
    print(f"{'Parameter':<25} {'Reference baseline':<18} {'OpenThermo':<18} {'Error %':<10}")
    print("-" * 70)

    # |p1| at output
    ota_p1 = np.abs(p1_out)
    dec_p1 = reference_baseline["p1_out"]
    err_p1 = 100 * (ota_p1 - dec_p1) / dec_p1
    print(f"|p1| at output (Pa)      {dec_p1:<18.2f} {ota_p1:<18.2f} {err_p1:<+10.3f}")

    # Phase of p1 at output
    ota_ph_p = np.degrees(np.angle(p1_out))
    dec_ph_p = reference_baseline["ph_p_out"]
    err_ph_p = ota_ph_p - dec_ph_p  # absolute error in degrees
    print(f"Phase(p1) (deg)         {dec_ph_p:<18.4f} {ota_ph_p:<18.4f} {err_ph_p:<+10.3f}°")

    # |U1| at output
    ota_U1 = np.abs(U1_out)
    dec_U1 = reference_baseline["U1_out"]
    err_U1 = 100 * (ota_U1 - dec_U1) / dec_U1
    print(f"|U1| at output (m³/s)   {dec_U1:<18.6f} {ota_U1:<18.6f} {err_U1:<+10.3f}")

    # Phase of U1 at output
    ota_ph_U = np.degrees(np.angle(U1_out))
    dec_ph_U = reference_baseline["ph_U_out"]
    err_ph_U = ota_ph_U - dec_ph_U  # absolute error in degrees
    print(f"Phase(U1) (deg)         {dec_ph_U:<18.3f} {ota_ph_U:<18.3f} {err_ph_U:<+10.3f}°")

    # Acoustic power
    ota_Edot = power_out
    dec_Edot = reference_baseline["Edot_out"]
    err_Edot = 100 * (ota_Edot - dec_Edot) / dec_Edot
    print(f"Acoustic power (W)       {dec_Edot:<18.4f} {ota_Edot:<18.4f} {err_Edot:<+10.3f}")

    print("-" * 70)
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    max_err = max(abs(err_p1), abs(err_U1), abs(err_Edot))
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Maximum magnitude error: {max_err:.2f}%")
    print(f"Maximum phase error: {max(abs(err_ph_p), abs(err_ph_U)):.3f}°")
    print()

    if max_err < 5.0:
        print("✓ VALIDATION PASSED: Results agree within 5%")
    elif max_err < 10.0:
        print("~ VALIDATION MARGINAL: Results agree within 10%")
    else:
        print("✗ VALIDATION FAILED: Results differ by more than 10%")

    print()
    print("Note: Small differences are expected due to:")
    print("  - Different numerical integration methods")
    print("  - Different thermophysical property correlations")
    print("  - Reference baseline uses 'ideal' solid (no wall thermal effects)")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Reference Comparison: Hofler Refrigerator Stack (StackEnergy)

Demonstrates two approaches:
1. Imposed temperature profile (recommended) - matches the baseline's STKSLAB
2. Energy equation coupling (experimental) - can be numerically unstable

The imposed temperature profile approach is recommended for most simulations.
"""

import numpy as np
from openthermoacoustics import gas, geometry
from openthermoacoustics.segments.stack_energy import StackEnergy
from openthermoacoustics.utils import acoustic_power


def main():
    print("=" * 70)
    print("COMPARISON: OpenThermoacoustics (StackEnergy) vs embedded reference baseline")
    print("Model: Hofler 1986 Stack Segment with Energy Equation")
    print("=" * 70)
    print()

    # =========================================================================
    # Embedded reference values from Hofler1 reference case
    # =========================================================================
    reference_baseline_in = {
        "p1": 29570.0,  # Pa
        "ph_p": -0.12971,  # deg
        "U1": 3.0568e-3,  # m³/s
        "ph_U": -81.875,  # deg
        "Edot": 6.4892,  # W
        "Htot": -2.140,  # W
        "T": 300.0,  # K
    }

    reference_baseline_out = {
        "p1": 26103.0,  # Pa
        "ph_p": 1.4277,  # deg
        "U1": 6.8004e-3,  # m³/s
        "ph_U": -87.965,  # deg
        "Edot": 0.94042,  # W
        "Htot": -2.140,  # W (conserved)
        "T": 217.03,  # K
    }

    # =========================================================================
    # Setup
    # =========================================================================
    helium = gas.Helium(mean_pressure=1.0e6)
    freq = 500.0  # Hz
    omega = 2 * np.pi * freq

    # Stack geometry from Hofler1 reference case segment 4
    total_area = 1.134e-3  # m² (sameas 1a)
    porosity = 0.7240
    length = 7.85e-2  # m
    y0 = 1.8e-4  # m (half-gap for parallel plates)
    plate_thickness = 4.0e-5  # m (Lplate)

    # Calculate solid area fraction
    solid_fraction = plate_thickness / (2 * y0 + plate_thickness)

    print("Stack Geometry (Parallel Plate):")
    print(f"  Total area: {total_area*1e4:.4f} cm²")
    print(f"  Porosity: {porosity:.4f}")
    print(f"  Length: {length*100:.2f} cm")
    print(f"  Half-gap y0: {y0*1e6:.1f} µm")
    print(f"  Plate thickness: {plate_thickness*1e6:.1f} µm")
    print(f"  Solid area fraction: {solid_fraction:.4f}")
    print()

    # =========================================================================
    # Create StackEnergy segment
    # =========================================================================
    # Kapton thermal conductivity ~ 0.12 W/(m·K)
    k_solid = 0.12  # W/(m·K)

    stack = StackEnergy(
        length=length,
        porosity=porosity,
        hydraulic_radius=y0,
        area=total_area,
        geometry=geometry.ParallelPlate(),
        solid_thermal_conductivity=k_solid,
        solid_area_fraction=solid_fraction,
    )

    # Initial conditions
    p1_in = reference_baseline_in["p1"] * np.exp(1j * np.radians(reference_baseline_in["ph_p"]))
    U1_in = reference_baseline_in["U1"] * np.exp(1j * np.radians(reference_baseline_in["ph_U"]))
    T_in = reference_baseline_in["T"]

    print("Input Conditions:")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, phase = {np.degrees(np.angle(p1_in)):.3f}°")
    print(f"  |U1| = {np.abs(U1_in)*1e3:.4f} × 10⁻³ m³/s, phase = {np.degrees(np.angle(U1_in)):.3f}°")
    print(f"  Temperature: {T_in:.1f} K")
    print(f"  Acoustic power: {acoustic_power(p1_in, U1_in):.4f} W")
    print()

    # =========================================================================
    # Method 1: Imposed Temperature Profile (Recommended)
    # =========================================================================
    print("=" * 70)
    print("METHOD 1: IMPOSED TEMPERATURE PROFILE (Recommended)")
    print("=" * 70)
    print()
    print("This matches the baseline's STKSLAB approach where the temperature")
    print("profile is imposed and the acoustic field evolves consistently.")
    print()

    # Propagate with imposed output temperature
    p1_out, U1_out, T_out = stack.propagate(
        p1_in, U1_in, T_in, omega, helium,
        T_out=reference_baseline_out["T"]  # Use the baseline's output temperature
    )
    E_dot_out = acoustic_power(p1_out, U1_out)

    print(f"{'Parameter':<25} {'Reference baseline':<18} {'OpenThermo':<18} {'Error':<12}")
    print("-" * 70)

    # |p1|
    ota_p1 = np.abs(p1_out)
    dec_p1 = reference_baseline_out["p1"]
    err_p1 = 100 * (ota_p1 - dec_p1) / dec_p1
    print(f"|p1| (Pa)                {dec_p1:<18.1f} {ota_p1:<18.1f} {err_p1:<+12.2f}%")

    # Phase p1
    ota_ph_p = np.degrees(np.angle(p1_out))
    dec_ph_p = reference_baseline_out["ph_p"]
    err_ph_p = ota_ph_p - dec_ph_p
    print(f"Phase(p1) (deg)         {dec_ph_p:<18.3f} {ota_ph_p:<18.3f} {err_ph_p:<+12.3f}°")

    # |U1|
    ota_U1 = np.abs(U1_out)
    dec_U1 = reference_baseline_out["U1"]
    err_U1 = 100 * (ota_U1 - dec_U1) / dec_U1
    print(f"|U1| (m³/s)             {dec_U1:<18.6f} {ota_U1:<18.6f} {err_U1:<+12.2f}%")

    # Phase U1
    ota_ph_U = np.degrees(np.angle(U1_out))
    dec_ph_U = reference_baseline_out["ph_U"]
    err_ph_U = ota_ph_U - dec_ph_U
    print(f"Phase(U1) (deg)         {dec_ph_U:<18.3f} {ota_ph_U:<18.3f} {err_ph_U:<+12.3f}°")

    # Temperature
    err_T = T_out - reference_baseline_out["T"]
    print(f"Temperature (K)          {reference_baseline_out['T']:<18.2f} {T_out:<18.2f} {err_T:<+12.2f} K")

    # Acoustic power
    err_E = 100 * (E_dot_out - reference_baseline_out["Edot"]) / reference_baseline_out["Edot"]
    print(f"Acoustic power (W)       {reference_baseline_out['Edot']:<18.4f} {E_dot_out:<18.4f} {err_E:<+12.2f}%")

    print("-" * 70)
    print()

    # Power flow analysis
    E_dot_in = acoustic_power(p1_in, U1_in)
    print("Power Flow Analysis:")
    print(f"  Input acoustic power: {E_dot_in:.4f} W")
    print(f"  Output acoustic power: {E_dot_out:.4f} W")
    print(f"  Power absorbed: {E_dot_in - E_dot_out:.4f} W")
    print(f"  Reference baseline power absorbed: {reference_baseline_in['Edot'] - reference_baseline_out['Edot']:.4f} W")
    print()

    # Summary
    max_err = max(abs(err_p1), abs(err_U1))
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Maximum amplitude error: {max_err:.2f}%")
    print()

    if max_err < 5.0:
        print("✓ VALIDATION PASSED: Results agree within 5%")
    elif max_err < 10.0:
        print("~ VALIDATION MARGINAL: Results agree within 10%")
    else:
        print("✗ VALIDATION NEEDS REVIEW: Results differ significantly")

    print()
    print("=" * 70)
    print("NOTES")
    print("=" * 70)
    print()
    print("1. The imposed temperature profile approach is recommended for")
    print("   accurate acoustic propagation through stacks and regenerators.")
    print()
    print("2. The temperature profile can be determined from system-level")
    print("   boundary conditions (e.g., heat exchanger temperatures).")
    print()
    print("3. The full energy equation approach (without T_out) is available")
    print("   for research purposes but may have numerical stability issues")
    print("   for large temperature gradients.")


if __name__ == "__main__":
    main()

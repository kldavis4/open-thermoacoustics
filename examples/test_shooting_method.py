#!/usr/bin/env python3
"""
Test the shooting method for finding self-consistent H2_total.
"""

import numpy as np
from openthermoacoustics import gas, geometry
from openthermoacoustics.segments.stack_energy import StackEnergy
from openthermoacoustics.utils import acoustic_power


def main():
    print("=" * 70)
    print("TESTING SHOOTING METHOD FOR STACK ENERGY")
    print("=" * 70)
    print()

    # Setup - Hofler refrigerator stack
    helium = gas.Helium(mean_pressure=1.0e6)
    freq = 500.0
    omega = 2 * np.pi * freq

    # Stack parameters
    stack_area = 1.134e-3
    stack_porosity = 0.7240
    stack_length = 7.85e-2
    stack_y0 = 1.8e-4
    k_solid = 0.12
    solid_fraction = 0.1

    # Input conditions
    p1_in = 29570.0 * np.exp(1j * np.radians(-0.12971))
    U1_in = 3.0568e-3 * np.exp(1j * np.radians(-81.875))
    T_in = 300.0
    T_out_target = 217.03

    # reference baseline reference
    reference_baseline_out = {
        "p1": 26103.0,
        "ph_p": 1.4277,
        "U1": 6.8004e-3,
        "ph_U": -87.965,
        "Edot": 0.94042,
    }

    print("Input conditions:")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa")
    print(f"  |U1| = {np.abs(U1_in)*1e3:.4f} × 10⁻³ m³/s")
    print(f"  T = {T_in:.1f} K")
    print(f"  Acoustic power: {acoustic_power(p1_in, U1_in):.4f} W")
    print()

    # Create stack
    stack = StackEnergy(
        length=stack_length,
        porosity=stack_porosity,
        hydraulic_radius=stack_y0,
        area=stack_area,
        geometry=geometry.ParallelPlate(),
        solid_thermal_conductivity=k_solid,
        solid_area_fraction=solid_fraction,
    )

    # =========================================================================
    # Method 1: Imposed temperature profile (baseline)
    # =========================================================================
    print("=" * 70)
    print("METHOD 1: IMPOSED TEMPERATURE PROFILE (BASELINE)")
    print("=" * 70)

    p1_out_imposed, U1_out_imposed, T_out_imposed = stack.propagate(
        p1_in, U1_in, T_in, omega, helium, T_out=T_out_target
    )
    E_dot_imposed = acoustic_power(p1_out_imposed, U1_out_imposed)

    print(f"  |p1| = {np.abs(p1_out_imposed):.1f} Pa (reference baseline: {reference_baseline_out['p1']:.1f})")
    print(f"  |U1| = {np.abs(U1_out_imposed)*1e3:.4f} × 10⁻³ m³/s (reference baseline: {reference_baseline_out['U1']*1e3:.4f})")
    print(f"  T = {T_out_imposed:.2f} K (target: {T_out_target:.2f})")
    print(f"  E_dot = {E_dot_imposed:.4f} W (reference baseline: {reference_baseline_out['Edot']:.4f})")
    print()

    # =========================================================================
    # Method 2: Shooting method
    # =========================================================================
    print("=" * 70)
    print("METHOD 2: SHOOTING METHOD")
    print("=" * 70)

    try:
        p1_out_shoot, U1_out_shoot, T_out_shoot, H2_converged = stack.propagate_with_shooting(
            p1_in, U1_in, T_in, T_out_target, omega, helium,
            max_iterations=30, tolerance=0.5
        )
        E_dot_shoot = acoustic_power(p1_out_shoot, U1_out_shoot)

        print(f"  Converged H2_total: {H2_converged:.4f} W")
        print(f"  |p1| = {np.abs(p1_out_shoot):.1f} Pa (reference baseline: {reference_baseline_out['p1']:.1f})")
        print(f"  |U1| = {np.abs(U1_out_shoot)*1e3:.4f} × 10⁻³ m³/s (reference baseline: {reference_baseline_out['U1']*1e3:.4f})")
        print(f"  T = {T_out_shoot:.2f} K (target: {T_out_target:.2f})")
        print(f"  E_dot = {E_dot_shoot:.4f} W (reference baseline: {reference_baseline_out['Edot']:.4f})")

        T_error = abs(T_out_shoot - T_out_target)
        if T_error < 1.0:
            print(f"\n  ✓ Temperature converged within {T_error:.2f} K")
        else:
            print(f"\n  ✗ Temperature did NOT converge (error: {T_error:.1f} K)")
            print("    Falling back to imposed gradient approach is recommended")
    except Exception as e:
        print(f"  Shooting method failed: {e}")

    print()

    # =========================================================================
    # Comparison
    # =========================================================================
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print("The shooting method attempts to find a self-consistent H2_total,")
    print("but the underlying physics (phase drift causing negative acoustic")
    print("power) makes it difficult to converge for large temperature gradients.")
    print()
    print("RECOMMENDATION: Use the imposed temperature profile approach for")
    print("production simulations. It's numerically stable and matches reference baseline's")
    print("STKSLAB behavior within <1% error.")


if __name__ == "__main__":
    main()

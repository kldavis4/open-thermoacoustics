#!/usr/bin/env python3
"""
Diagnose the energy equation approach to understand instability.

This script compares:
1. Imposed temperature profile approach (stable, matches reference baseline)
2. Full energy equation coupling (potentially unstable)

It shows how the acoustic field and temperature evolve and identifies
where the instability arises.
"""

import numpy as np
from scipy.integrate import solve_ivp

from openthermoacoustics import gas, geometry
from openthermoacoustics.segments.stack_energy import StackEnergy
from openthermoacoustics.utils import acoustic_power


def main():
    print("=" * 70)
    print("DIAGNOSING ENERGY EQUATION INSTABILITY")
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
    T_out_target = 217.03  # reference baseline's output temperature

    print("Input conditions:")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, phase = {np.degrees(np.angle(p1_in)):.3f}°")
    print(f"  |U1| = {np.abs(U1_in)*1e3:.4f} × 10⁻³ m³/s, phase = {np.degrees(np.angle(U1_in)):.3f}°")
    print(f"  T = {T_in:.1f} K")
    print(f"  Acoustic power: {acoustic_power(p1_in, U1_in):.4f} W")
    print(f"  Phase(p1) - Phase(U1) = {np.degrees(np.angle(p1_in) - np.angle(U1_in)):.2f}°")
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
    # Method 1: Imposed temperature profile
    # =========================================================================
    print("=" * 70)
    print("METHOD 1: IMPOSED TEMPERATURE PROFILE")
    print("=" * 70)

    p1_out_imposed, U1_out_imposed, T_out_imposed = stack.propagate(
        p1_in, U1_in, T_in, omega, helium, T_out=T_out_target
    )

    print(f"Output: |p1| = {np.abs(p1_out_imposed):.1f} Pa, phase = {np.degrees(np.angle(p1_out_imposed)):.3f}°")
    print(f"Output: |U1| = {np.abs(U1_out_imposed)*1e3:.4f} × 10⁻³ m³/s, phase = {np.degrees(np.angle(U1_out_imposed)):.3f}°")
    print(f"Output: T = {T_out_imposed:.2f} K")
    print(f"Output: Acoustic power = {acoustic_power(p1_out_imposed, U1_out_imposed):.4f} W")
    print()

    # =========================================================================
    # Method 2: Energy equation coupling - step by step diagnosis
    # =========================================================================
    print("=" * 70)
    print("METHOD 2: ENERGY EQUATION COUPLING (DIAGNOSTIC)")
    print("=" * 70)

    # Compute H2_total from inlet
    H2_inlet = stack.compute_H2_total(p1_in, U1_in, T_in, omega, helium)
    print(f"H2_total from inlet (acoustic power only): {H2_inlet:.4f} W")

    # What H2 would give us the target temperature?
    H2_estimated = stack.estimate_H2_for_temperature_change(
        p1_in, U1_in, T_in, T_out_target, omega, helium
    )
    print(f"H2_total estimated for T_out = {T_out_target} K: {H2_estimated:.4f} W")
    print()

    # Analyze the dT/dx at inlet for different H2 values
    print("Analysis of dT/dx at inlet for different H2 values:")
    print("-" * 50)

    for H2_test in [H2_inlet, H2_estimated, -2.0, -5.0, -10.0]:
        dT_dx = stack._compute_dT_dx(p1_in, U1_in, T_in, omega, helium, H2_test)
        print(f"  H2 = {H2_test:+8.3f} W  ->  dT/dx = {dT_dx:+8.1f} K/m")

    # Expected gradient
    expected_gradient = (T_out_target - T_in) / stack_length
    print(f"\n  Expected gradient for T: {T_in} -> {T_out_target} K: {expected_gradient:.1f} K/m")
    print()

    # =========================================================================
    # Detailed integration with energy equation
    # =========================================================================
    print("=" * 70)
    print("INTEGRATION WITH ENERGY EQUATION")
    print("=" * 70)

    # Try integration with different H2 values
    H2_values_to_try = [H2_inlet, H2_estimated, -2.0]

    for H2_test in H2_values_to_try:
        print(f"\n--- H2_total = {H2_test:.3f} W ---")

        # Custom integration to track evolution
        y0 = np.array([
            p1_in.real, p1_in.imag,
            U1_in.real, U1_in.imag,
            T_in
        ])

        def ode_func(x, y):
            return stack.get_derivatives(x, y, omega, helium, T_in, H2_test)

        # Evaluate at multiple points
        x_eval = np.linspace(0, stack_length, 20)

        try:
            sol = solve_ivp(
                ode_func,
                (0, stack_length),
                y0,
                method="RK45",
                t_eval=x_eval,
                rtol=1e-6,
                atol=1e-8,
            )

            if sol.success:
                # Extract results
                x = sol.t * 100  # Convert to cm
                p1_mag = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
                U1_mag = np.sqrt(sol.y[2]**2 + sol.y[3]**2)
                T = sol.y[4]

                # Phase difference
                p1_phase = np.degrees(np.arctan2(sol.y[1], sol.y[0]))
                U1_phase = np.degrees(np.arctan2(sol.y[3], sol.y[2]))
                phase_diff = p1_phase - U1_phase

                # Acoustic power
                E_dot = np.array([
                    acoustic_power(complex(sol.y[0, i], sol.y[1, i]),
                                 complex(sol.y[2, i], sol.y[3, i]))
                    for i in range(len(x))
                ])

                print(f"  x=0 cm:    T={T[0]:.1f} K, |p1|={p1_mag[0]:.0f} Pa, |U1|={U1_mag[0]*1e3:.4f} mm³/s, E_dot={E_dot[0]:.3f} W")
                print(f"  x={x[-1]:.1f} cm: T={T[-1]:.1f} K, |p1|={p1_mag[-1]:.0f} Pa, |U1|={U1_mag[-1]*1e3:.4f} mm³/s, E_dot={E_dot[-1]:.3f} W")
                print(f"  Temperature drop: {T[0] - T[-1]:.1f} K (target: {T_in - T_out_target:.1f} K)")

                # Check if phase went past 90 degrees
                if any(abs(phase_diff) > 90):
                    print(f"  WARNING: Phase difference exceeded 90° - acoustic power went negative!")
            else:
                print(f"  Integration failed: {sol.message}")

        except Exception as e:
            print(f"  Integration error: {e}")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("Key observations:")
    print()
    print("1. The energy equation approach requires knowing H2_total, but at the")
    print("   inlet we only have acoustic power. The streaming and conduction terms")
    print("   depend on dT/dx which is what we're trying to find.")
    print()
    print("2. For a refrigerator, heat flows from cold to hot (against the gradient),")
    print("   so H2_total should be NEGATIVE (heat pumped from cold reservoir).")
    print("   But acoustic power at inlet is positive.")
    print()
    print("3. The full self-consistent solution requires iteration or a shooting")
    print("   method to find the H2 that gives the correct boundary conditions.")
    print()
    print("Recommendation: Use the imposed temperature profile approach for stable")
    print("results, or implement a shooting method to find consistent H2 values.")


if __name__ == "__main__":
    main()

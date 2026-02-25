#!/usr/bin/env python3
"""
Example: Standing Wave Resonator Analysis

This example demonstrates how to use OpenThermoacoustics to analyze a simple
closed-closed acoustic resonator filled with helium. It calculates the resonant
frequencies and compares them to the analytical predictions.

The analytical resonant frequencies for a closed-closed tube of length L are:
    f_n = n * a / (2L)  for n = 1, 2, 3, ...

where a is the sound speed.
"""

import numpy as np

from openthermoacoustics import gas, segments
from openthermoacoustics.solver import NetworkTopology, ShootingSolver


def main() -> None:
    """Analyze a simple closed-closed helium resonator."""
    # Define the working gas: helium at 1 atm
    helium = gas.Helium(mean_pressure=101325.0)  # Pa (1 atm)

    # Operating conditions
    T_m = 300.0  # K (room temperature)

    # Calculate sound speed for reference
    sound_speed = helium.sound_speed(T_m)
    print(f"Helium at {T_m} K, 1 atm:")
    print(f"  Sound speed: {sound_speed:.1f} m/s")
    print(f"  Density: {helium.density(T_m):.4f} kg/m³")
    print(f"  Prandtl number: {helium.prandtl(T_m):.3f}")
    print()

    # Create the resonator: a 1-meter long tube with 5 cm radius
    length = 1.0  # m
    radius = 0.05  # m

    # Build the network
    network = NetworkTopology()
    network.add_segment(segments.Duct(length=length, radius=radius))

    # Calculate expected resonant frequencies
    print(f"Resonator: L = {length} m, r = {radius*100:.1f} cm")
    print()
    print("Expected resonant frequencies (lossless):")
    for n in range(1, 4):
        f_expected = n * sound_speed / (2 * length)
        print(f"  n={n}: f_{n} = {f_expected:.1f} Hz")
    print()

    # Solve for each harmonic
    print("Computed resonant frequencies (with thermoviscous losses):")
    solver = ShootingSolver(network, helium)

    for n in range(1, 4):
        # Initial guess based on analytical solution
        f_guess = n * sound_speed / (2 * length)

        # Solve for resonance
        # We solve for frequency and phase to satisfy U1=0 at the end
        # p1_amplitude defaults to 1000 Pa (just a scaling factor)
        result = solver.solve(
            guesses={
                "p1_phase": 0.0,
                "frequency": f_guess,
            },
            targets={
                "U1_end_real": 0.0,  # Closed end: U1 = 0
                "U1_end_imag": 0.0,
            },
            options={
                "T_m_start": T_m,
                "tol": 1e-10,
            },
        )

        if result.converged:
            f_expected = n * sound_speed / (2 * length)
            error_percent = 100 * (result.frequency - f_expected) / f_expected
            print(
                f"  n={n}: f_{n} = {result.frequency:.2f} Hz "
                f"(error: {error_percent:+.2f}%)"
            )
        else:
            print(f"  n={n}: Failed to converge")

    # Detailed analysis of fundamental mode
    print()
    print("=" * 60)
    print("Detailed analysis of fundamental mode:")
    print("=" * 60)

    f_guess = sound_speed / (2 * length)
    result = solver.solve(
        guesses={"p1_phase": 0.0, "frequency": f_guess},
        targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
        options={"T_m_start": T_m, "n_points_per_segment": 200},
    )

    if result.converged:
        print(f"Frequency: {result.frequency:.4f} Hz")
        print(f"Angular frequency: {result.omega:.4f} rad/s")
        print(f"Converged in {result.n_iterations} iterations")
        print(f"Residual norm: {result.residual_norm:.2e}")
        print()

        # Calculate wavelength and wavenumber
        wavelength = sound_speed / result.frequency
        k = 2 * np.pi * result.frequency / sound_speed
        print(f"Wavelength: {wavelength:.4f} m")
        print(f"Wavenumber: {k:.4f} rad/m")
        print(f"L/λ = {length/wavelength:.4f} (should be ≈ 0.5 for n=1)")
        print()

        # Analyze pressure and velocity profiles
        x = result.x_profile
        p1 = result.p1_profile
        U1 = result.U1_profile
        power = result.acoustic_power

        print("Profile statistics:")
        print(f"  |p1| range: {np.min(np.abs(p1)):.2f} - {np.max(np.abs(p1)):.2f} Pa")
        print(
            f"  |U1| range: {np.min(np.abs(U1)):.2e} - {np.max(np.abs(U1)):.2e} m³/s"
        )
        print(f"  Power range: {np.min(power):.4f} - {np.max(power):.4f} W")
        print()

        # Check boundary conditions
        print("Boundary conditions:")
        print(f"  U1 at x=0: {U1[0]:.4e} m³/s (should be ≈ 0)")
        print(f"  U1 at x=L: {U1[-1]:.4e} m³/s (should be ≈ 0)")
        print(f"  |p1| at x=0: {np.abs(p1[0]):.2f} Pa (antinode)")
        print(f"  |p1| at x=L: {np.abs(p1[-1]):.2f} Pa (antinode)")

    print()
    print("Done!")


if __name__ == "__main__":
    main()

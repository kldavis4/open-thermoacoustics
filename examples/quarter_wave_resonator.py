#!/usr/bin/env python3
"""
Example: Quarter-Wave (Closed-Open) Resonator Validation

This example validates OpenThermoacoustics against the analytical solution
for a closed-open (quarter-wave) tube resonator.

Physics Background:
-------------------
A closed-open tube supports standing wave modes where the closed end has a
pressure antinode (velocity node) and the open end has a pressure node
(velocity antinode). The wavelength must satisfy:

    L = (2n-1) * lambda/4  for n = 1, 2, 3, ...

This gives resonant frequencies:

    f_n = (2n-1) * a / (4L)

where:
    n = mode number (1, 2, 3, ...)
    a = sound speed (m/s)
    L = tube length (m)

The fundamental mode (n=1) has f_1 = a/(4L), which is why it's called
a "quarter-wave" resonator.

Mode shapes:
  n=1: Fundamental - 1/4 wavelength fits in tube
  n=2: First overtone - 3/4 wavelengths fit in tube
  n=3: Second overtone - 5/4 wavelengths fit in tube

Unlike a closed-closed tube (which has only even harmonics relative to
the half-wave fundamental), a closed-open tube has only odd harmonics
of the quarter-wave fundamental.

References:
-----------
Kinsler et al., "Fundamentals of Acoustics", 4th ed., Chapter 9
Swift, "Thermoacoustics: A Unifying Perspective", Chapter 4
"""

import numpy as np

from openthermoacoustics import gas, segments
from openthermoacoustics.solver import NetworkTopology, ShootingSolver


def analytical_quarter_wave_frequency(
    n: int, sound_speed: float, length: float
) -> float:
    """
    Calculate the analytical resonant frequency for mode n.

    Parameters
    ----------
    n : int
        Mode number (1, 2, 3, ...).
    sound_speed : float
        Sound speed in the gas (m/s).
    length : float
        Length of the tube (m).

    Returns
    -------
    float
        Resonant frequency in Hz.
    """
    return (2 * n - 1) * sound_speed / (4 * length)


def main() -> None:
    """Validate quarter-wave resonator against analytical solution."""
    print("=" * 70)
    print("Quarter-Wave (Closed-Open) Resonator Validation")
    print("=" * 70)
    print()

    # Working gas: Helium at 1 atm, 300 K
    helium = gas.Helium(mean_pressure=101325.0)  # Pa (1 atm)
    T_m = 300.0  # K

    # Gas properties
    sound_speed = helium.sound_speed(T_m)
    rho_m = helium.density(T_m)

    print("Working gas: Helium at 1 atm, 300 K")
    print(f"  Sound speed: {sound_speed:.2f} m/s")
    print(f"  Density: {rho_m:.4f} kg/m^3")
    print()

    # Resonator geometry: tube
    length = 0.5  # m
    radius = 0.025  # m = 2.5 cm

    print("Resonator geometry:")
    print(f"  Length: {length*100:.1f} cm")
    print(f"  Radius: {radius*100:.2f} cm")
    print(f"  Boundary conditions: Closed at x=0, Open at x=L")
    print()

    # Build the numerical model
    network = NetworkTopology()
    duct = segments.Duct(length=length, radius=radius)
    network.add_segment(duct)

    print(f"Numerical model: {duct}")
    print()

    # Calculate and compare first 3 modes
    print("Analytical resonant frequencies (lossless):")
    for n in range(1, 4):
        f_n = analytical_quarter_wave_frequency(n, sound_speed, length)
        print(f"  n={n}: f_{n} = {f_n:.2f} Hz  (L/lambda = {length/(sound_speed/f_n):.4f})")
    print()

    # Solve for each mode
    solver = ShootingSolver(network, helium)

    print("Numerical solutions (with thermoviscous losses):")
    print("-" * 60)
    print(f"{'Mode':^6} {'f_analytical':^14} {'f_numerical':^14} {'Error (%)':^12}")
    print("-" * 60)

    results = []
    for n in range(1, 4):
        # Analytical frequency for initial guess
        f_analytical = analytical_quarter_wave_frequency(n, sound_speed, length)

        # Solve for resonance with open end boundary condition (p1 = 0)
        # At closed end: U1 = 0 (enforced by starting from closed end with U1=0)
        # At open end: p1 = 0 (our target)
        #
        # We need to match number of guesses to number of targets
        # Guesses: frequency, p1_phase
        # Targets: p1_end_real = 0, p1_end_imag = 0

        result = solver.solve(
            guesses={
                "frequency": f_analytical,
                "p1_phase": 0.0,
            },
            targets={
                "p1_end_real": 0.0,
                "p1_end_imag": 0.0,
            },
            options={
                "T_m_start": T_m,
                "tol": 1e-10,
            },
        )

        if result.converged:
            f_numerical = result.frequency
            error_percent = 100 * (f_numerical - f_analytical) / f_analytical
            results.append((n, f_analytical, f_numerical, error_percent, result))
            print(f"{n:^6} {f_analytical:^14.2f} {f_numerical:^14.2f} {error_percent:^+12.4f}")
        else:
            results.append((n, f_analytical, None, None, result))
            print(f"{n:^6} {f_analytical:^14.2f} {'FAILED':^14} {'N/A':^12}")

    print("-" * 60)
    print()

    # Detailed analysis of fundamental mode
    if results[0][2] is not None:
        n, f_analytical, f_numerical, error_percent, result = results[0]

        print("=" * 60)
        print("Detailed Analysis: Fundamental Mode (n=1)")
        print("=" * 60)

        print(f"Frequency: {f_numerical:.4f} Hz")
        print(f"Angular frequency: {result.omega:.4f} rad/s")
        print(f"Iterations: {result.n_iterations}")
        print(f"Residual norm: {result.residual_norm:.2e}")
        print()

        # Mode shape analysis
        wavelength = sound_speed / f_numerical
        k = 2 * np.pi / wavelength

        print("Mode shape analysis:")
        print(f"  Wavelength: {wavelength:.4f} m")
        print(f"  L/lambda: {length/wavelength:.4f} (should be ~ 0.25 for n=1)")
        print(f"  Wavenumber: {k:.4f} rad/m")
        print()

        # Check boundary conditions
        print("Boundary conditions verification:")
        print(f"  |p1| at closed end (x=0): {np.abs(result.p1_profile[0]):.2f} Pa (antinode)")
        print(f"  |p1| at open end (x=L):   {np.abs(result.p1_profile[-1]):.4f} Pa (should be ~ 0)")
        print(f"  |U1| at closed end (x=0): {np.abs(result.U1_profile[0]):.4e} m^3/s (should be ~ 0)")
        print(f"  |U1| at open end (x=L):   {np.abs(result.U1_profile[-1]):.4e} m^3/s (antinode)")
        print()

        # Effect of losses
        print("Effect of thermoviscous losses:")
        # The lossy frequency is typically slightly lower than lossless
        # due to effective length increase from boundary layers
        delta_f = f_numerical - f_analytical
        print(f"  Frequency shift: {delta_f:+.4f} Hz")
        print(f"  Relative shift: {100*delta_f/f_analytical:+.4f}%")

        # Calculate Q factor from acoustic power loss
        power_in = result.acoustic_power[0]
        power_out = result.acoustic_power[-1]
        power_loss = power_in - power_out
        print(f"  Acoustic power at input:  {power_in:.6f} W")
        print(f"  Acoustic power at output: {power_out:.6f} W")
        print(f"  Power dissipated: {power_loss:.6f} W")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for n, f_analytical, f_numerical, error_percent, result in results:
        if f_numerical is not None:
            status = "PASS" if abs(error_percent) < 1.0 else "WARN"
            if abs(error_percent) >= 1.0:
                all_passed = False
            print(f"Mode n={n}: Analytical={f_analytical:.2f} Hz, "
                  f"Numerical={f_numerical:.2f} Hz, Error={error_percent:+.4f}% [{status}]")
        else:
            all_passed = False
            print(f"Mode n={n}: FAILED to converge")

    print()
    if all_passed:
        print("VALIDATION PASSED: All modes have error < 1%")
    else:
        print("VALIDATION WARNING: Some modes have error >= 1% or failed to converge")

    print()


if __name__ == "__main__":
    main()

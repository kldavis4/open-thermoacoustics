#!/usr/bin/env python3
"""
Example: Helmholtz Resonator Validation

This example validates OpenThermoacoustics against the analytical Helmholtz
resonator solution. A Helmholtz resonator consists of a neck (inertance)
connected to a cavity (compliance).

Physics Background:
-------------------
A Helmholtz resonator is an acoustic system where a volume of gas oscillates
in a short neck connecting a larger cavity to the environment. It is analogous
to a mass-spring system:
  - The gas in the neck provides the inertia (mass)
  - The compressibility of the gas in the cavity provides the stiffness (spring)

The analytical resonant frequency is:
    f = (a / 2*pi) * sqrt(A / (V * L_eff))

where:
    a = sound speed (m/s)
    A = neck cross-sectional area (m^2)
    V = cavity volume (m^3)
    L_eff = L + 1.7*a_neck  (effective neck length with end corrections)
    a_neck = neck radius (m)

The factor 1.7*a_neck accounts for the "end corrections" - the acoustic mass
of gas participating in the oscillation just outside the neck openings
(flanged ends: 0.85*a per end).

References:
-----------
Kinsler et al., "Fundamentals of Acoustics", 4th ed., Chapter 10
Swift, "Thermoacoustics: A Unifying Perspective", relevant reference
"""

import numpy as np
from scipy.optimize import minimize_scalar

from openthermoacoustics import gas, segments


def analytical_helmholtz_frequency(
    sound_speed: float,
    cavity_volume: float,
    neck_length: float,
    neck_area: float,
    neck_radius: float,
) -> float:
    """
    Calculate the analytical Helmholtz resonator frequency.

    Parameters
    ----------
    sound_speed : float
        Sound speed in the gas (m/s).
    cavity_volume : float
        Volume of the cavity (m^3).
    neck_length : float
        Physical length of the neck (m).
    neck_area : float
        Cross-sectional area of the neck (m^2).
    neck_radius : float
        Radius of the neck (m).

    Returns
    -------
    float
        Resonant frequency in Hz.
    """
    # End correction: 0.85*a for each flanged end
    # Total end correction for two flanged ends: 2 * 0.85 * a = 1.7 * a
    L_eff = neck_length + 1.7 * neck_radius

    # Helmholtz resonance frequency
    f = (sound_speed / (2 * np.pi)) * np.sqrt(neck_area / (cavity_volume * L_eff))

    return f


def main() -> None:
    """Validate Helmholtz resonator against analytical solution."""
    print("=" * 70)
    print("Helmholtz Resonator Validation")
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

    # Helmholtz resonator geometry
    # Cavity (compliance)
    cavity_volume = 1e-4  # m^3 = 100 cm^3

    # Neck (inertance)
    neck_length = 0.02  # m = 2 cm
    neck_radius = 0.005  # m = 5 mm
    neck_area = np.pi * neck_radius**2

    print("Resonator geometry:")
    print(f"  Cavity volume: {cavity_volume * 1e6:.1f} cm^3")
    print(f"  Neck length: {neck_length * 100:.1f} cm")
    print(f"  Neck radius: {neck_radius * 1000:.1f} mm")
    print(f"  Neck area: {neck_area * 1e6:.2f} mm^2")
    print()

    # Calculate analytical resonant frequency
    f_analytical = analytical_helmholtz_frequency(
        sound_speed=sound_speed,
        cavity_volume=cavity_volume,
        neck_length=neck_length,
        neck_area=neck_area,
        neck_radius=neck_radius,
    )

    # Effective length for reference
    L_eff = neck_length + 1.7 * neck_radius
    print(f"Effective neck length (with end corrections): {L_eff * 100:.2f} cm")
    print(f"Analytical resonant frequency: {f_analytical:.2f} Hz")
    print()

    # Build the numerical model using lumped elements directly
    # Since the integrator has an issue with lumped elements (length=0),
    # we use the propagate method directly to compute the resonance.
    #
    # Model: Compliance (cavity) + Inertance (neck)
    # At resonance, the combined impedance is minimum (imaginary part = 0).
    #
    # For Helmholtz resonator with open end:
    #   - Start with pressure p1 in cavity
    #   - Propagate through compliance: U1 changes (gas compression)
    #   - Propagate through inertance: p1 drops (gas acceleration)
    #   - At open end: p1 = 0 (pressure release boundary condition)

    compliance = segments.Compliance(volume=cavity_volume)

    # The inertance length should include end corrections to match the analytical model
    # End correction = 1.7 * radius for flanged ends (0.85 per end)
    # This accounts for the acoustic mass of gas just outside the neck openings
    neck_length_effective = neck_length + 1.7 * neck_radius

    inertance = segments.Inertance(
        length=neck_length_effective, radius=neck_radius, include_resistance=False
    )

    print("Numerical model (lumped elements):")
    print(f"  Segment 1: {compliance}")
    print(f"  Segment 2: {inertance}")
    print()

    # Solve for resonance using direct propagation
    # At resonance with p1_end = 0, we need to find the frequency where
    # the imaginary part of the input impedance crosses zero.

    def compute_p1_end_magnitude(f: float) -> float:
        """
        Compute |p1_end| for a given frequency.

        At resonance, |p1_end| should be zero (minimum).
        """
        omega = 2 * np.pi * f

        # Start with unit pressure in the cavity
        p1_in = 1000.0 + 0j  # Pa

        # At the cavity (compliance) input, assume we're at a pressure antinode
        # with some initial velocity. For Helmholtz resonator driven from inside:
        U1_in = 0.0 + 0j  # m^3/s (start from rest)

        # Propagate through compliance
        p1_mid, U1_mid, T_mid = compliance.propagate(p1_in, U1_in, T_m, omega, helium)

        # Propagate through inertance
        p1_out, U1_out, T_out = inertance.propagate(p1_mid, U1_mid, T_mid, omega, helium)

        # For an open end, p1_out should be zero at resonance
        return np.abs(p1_out)

    # Use minimization to find the resonance (where |p1_end| is minimum)
    # Search in a range around the analytical frequency
    f_min = f_analytical * 0.5
    f_max = f_analytical * 2.0

    result = minimize_scalar(
        compute_p1_end_magnitude,
        bounds=(f_min, f_max),
        method="bounded",
        options={"xatol": 0.01}  # Tolerance on frequency
    )

    converged = result.success
    f_numerical = result.x
    residual = result.fun

    print("Numerical solution:")
    if converged:
        error_percent = 100 * (f_numerical - f_analytical) / f_analytical

        print(f"  Converged: Yes")
        print(f"  Numerical frequency: {f_numerical:.2f} Hz")
        print(f"  Analytical frequency: {f_analytical:.2f} Hz")
        print(f"  Error: {error_percent:+.3f}%")
        print(f"  |p1_end| at resonance: {residual:.4f} Pa (should be ~0)")
        print()

        # Check lumped element approximation validity
        wavelength = sound_speed / f_numerical
        print("Lumped element approximation check:")
        print(f"  Wavelength at resonance: {wavelength:.2f} m")
        print(f"  Cavity size / wavelength: {(cavity_volume**(1/3)):.4f} / {wavelength:.2f} = {(cavity_volume**(1/3))/wavelength:.4f}")
        print(f"  Neck length / wavelength: {neck_length:.4f} / {wavelength:.2f} = {neck_length/wavelength:.4f}")
        print("  (Lumped approximation valid when these ratios << 1)")
    else:
        print(f"  FAILED to converge")
        print(f"  Best frequency found: {f_numerical:.2f} Hz")
        print(f"  Residual: {residual:.4f}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if converged:
        print(f"Analytical frequency:  {f_analytical:.4f} Hz")
        print(f"Numerical frequency:   {f_numerical:.4f} Hz")
        print(f"Absolute error:        {abs(f_numerical - f_analytical):.4f} Hz")
        print(f"Relative error:        {abs(error_percent):.4f}%")

        if abs(error_percent) < 1.0:
            print("\nVALIDATION PASSED: Error < 1%")
        else:
            print(f"\nVALIDATION WARNING: Error = {abs(error_percent):.2f}%")
    else:
        print("VALIDATION FAILED: Solver did not converge")

    print()


if __name__ == "__main__":
    main()

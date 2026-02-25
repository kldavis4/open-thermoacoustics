#!/usr/bin/env python3
"""
Example: Lossy Duct Attenuation Validation

This example validates OpenThermoacoustics against the analytical Kirchhoff-Helmholtz
formula for acoustic attenuation in a circular duct with thermoviscous losses.

Physics Background:
-------------------
When an acoustic wave propagates through a tube, energy is dissipated due to:
1. Viscous losses - friction at the wall due to the oscillating boundary layer
2. Thermal losses - heat transfer between the oscillating gas and the wall

For a circular tube with isothermal walls, the attenuation coefficient is given
by the Kirchhoff-Helmholtz formula:

    alpha = (omega / (2 * a)) * (delta_nu + (gamma - 1) * delta_kappa) / r

where:
    omega = angular frequency (rad/s)
    a = sound speed (m/s)
    delta_nu = viscous penetration depth = sqrt(2*nu/omega)
    delta_kappa = thermal penetration depth = sqrt(2*alpha_th/omega)
    gamma = ratio of specific heats
    r = tube radius (m)
    nu = kinematic viscosity (m^2/s)
    alpha_th = thermal diffusivity (m^2/s)

This can also be written as:

    alpha = (1/r) * sqrt(omega/2) * [sqrt(nu) + (gamma-1)*sqrt(alpha_th)] / a

The pressure amplitude decays as:
    |p1(x)| = |p1(0)| * exp(-alpha * x)

This formula is valid in the "wide tube" limit where r >> delta_nu, delta_kappa.

References:
-----------
Swift, "Thermoacoustics: A Unifying Perspective", relevant reference
Tijdeman, "On the propagation of sound waves in cylindrical tubes", J. Sound Vib. 39 (1975)
"""

import numpy as np

from openthermoacoustics import gas, segments
from openthermoacoustics.solver import NetworkTopology, ShootingSolver
from openthermoacoustics.utils import penetration_depth_thermal, penetration_depth_viscous


def analytical_attenuation_coefficient(
    omega: float,
    sound_speed: float,
    radius: float,
    delta_nu: float,
    delta_kappa: float,
    gamma: float,
) -> float:
    """
    Calculate the Kirchhoff-Helmholtz attenuation coefficient.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    sound_speed : float
        Sound speed (m/s).
    radius : float
        Tube radius (m).
    delta_nu : float
        Viscous penetration depth (m).
    delta_kappa : float
        Thermal penetration depth (m).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        Attenuation coefficient alpha (1/m).
    """
    # Kirchhoff-Helmholtz formula
    alpha = (omega / (2 * sound_speed)) * (delta_nu + (gamma - 1) * delta_kappa) / radius
    return alpha


def main() -> None:
    """Validate lossy duct attenuation against Kirchhoff-Helmholtz formula."""
    print("=" * 70)
    print("Lossy Duct Attenuation Validation")
    print("=" * 70)
    print()

    # Working gas: Helium at 1 atm, 300 K
    helium = gas.Helium(mean_pressure=101325.0)  # Pa (1 atm)
    T_m = 300.0  # K

    # Gas properties
    sound_speed = helium.sound_speed(T_m)
    rho_m = helium.density(T_m)
    mu = helium.viscosity(T_m)
    kappa = helium.thermal_conductivity(T_m)
    cp = helium.specific_heat_cp(T_m)
    gamma = helium.gamma(T_m)
    nu = mu / rho_m  # Kinematic viscosity
    alpha_th = kappa / (rho_m * cp)  # Thermal diffusivity

    print("Working gas: Helium at 1 atm, 300 K")
    print(f"  Sound speed: {sound_speed:.2f} m/s")
    print(f"  Density: {rho_m:.4f} kg/m^3")
    print(f"  Dynamic viscosity: {mu:.3e} Pa*s")
    print(f"  Kinematic viscosity: {nu:.3e} m^2/s")
    print(f"  Thermal conductivity: {kappa:.4f} W/(m*K)")
    print(f"  Thermal diffusivity: {alpha_th:.3e} m^2/s")
    print(f"  Gamma: {gamma:.4f}")
    print()

    # Duct geometry
    length = 2.0  # m - long duct to see significant attenuation
    radius = 0.01  # m = 1 cm - small radius to enhance losses

    # Frequency
    frequency = 200.0  # Hz
    omega = 2 * np.pi * frequency

    # Calculate penetration depths
    delta_nu = penetration_depth_viscous(omega, rho_m, mu)
    delta_kappa = penetration_depth_thermal(omega, rho_m, kappa, cp)

    print("Duct geometry:")
    print(f"  Length: {length*100:.1f} cm")
    print(f"  Radius: {radius*100:.2f} cm ({radius*1000:.1f} mm)")
    print()

    print(f"Operating frequency: {frequency} Hz (omega = {omega:.2f} rad/s)")
    print()

    print("Penetration depths:")
    print(f"  Viscous: delta_nu = {delta_nu*1000:.4f} mm")
    print(f"  Thermal: delta_kappa = {delta_kappa*1000:.4f} mm")
    print(f"  r / delta_nu = {radius/delta_nu:.1f} (should be >> 1 for wide tube limit)")
    print(f"  r / delta_kappa = {radius/delta_kappa:.1f}")
    print()

    # Analytical attenuation coefficient
    alpha_analytical = analytical_attenuation_coefficient(
        omega=omega,
        sound_speed=sound_speed,
        radius=radius,
        delta_nu=delta_nu,
        delta_kappa=delta_kappa,
        gamma=gamma,
    )

    print("Analytical predictions (Kirchhoff-Helmholtz):")
    print(f"  Attenuation coefficient: alpha = {alpha_analytical:.6f} 1/m")
    print(f"  Attenuation over duct length: exp(-alpha*L) = {np.exp(-alpha_analytical*length):.4f}")
    print(f"  Attenuation in dB/m: {20*np.log10(np.e)*alpha_analytical:.4f} dB/m")
    print()

    # Numerical calculation
    # Propagate a wave through the duct and measure the pressure decay
    #
    # We use the Duct's propagate method directly with scipy integration
    # to get the pressure profile with thermoviscous losses.
    # The Duct's get_derivatives method includes boundary layer losses.

    duct = segments.Duct(length=length, radius=radius)

    print(f"Numerical model: {duct}")
    print()

    # We'll propagate at the specified frequency (not solve for resonance)
    # Just integrate through the duct
    p1_start = 1000.0 + 0j  # Pa - starting pressure amplitude

    # For a traveling wave: U1 = p1 * A / (rho * a)
    A = np.pi * radius**2
    Z_acoustic = rho_m * sound_speed / A  # Plane wave impedance
    U1_start = p1_start / Z_acoustic  # Traveling wave relation

    print(f"Initial conditions (traveling wave):")
    print(f"  p1(0) = {np.abs(p1_start):.2f} Pa")
    print(f"  U1(0) = {np.abs(U1_start):.4e} m^3/s")
    print(f"  Z = p1/U1 = {np.abs(p1_start/U1_start):.2f} Pa*s/m^3")
    print()

    # Propagate through the duct using scipy's ODE integrator
    # We integrate the Duct's get_derivatives method directly
    from scipy.integrate import solve_ivp
    from openthermoacoustics.utils import complex_to_state, state_to_complex, acoustic_power

    y0 = complex_to_state(p1_start, U1_start)

    def ode_func(x_pos: float, y: np.ndarray) -> np.ndarray:
        return duct.get_derivatives(x_pos, y, omega, helium, T_m)

    # Number of output points
    n_points = 500
    x_eval = np.linspace(0, length, n_points)

    sol = solve_ivp(
        ode_func,
        (0, length),
        y0,
        method="RK45",
        t_eval=x_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    x = sol.t
    p1 = np.array([state_to_complex(sol.y[:, i])[0] for i in range(len(x))])
    U1 = np.array([state_to_complex(sol.y[:, i])[1] for i in range(len(x))])
    power = np.array([acoustic_power(p1[i], U1[i]) for i in range(len(x))])

    # Calculate numerical attenuation
    p1_magnitude = np.abs(p1)
    p1_start_mag = p1_magnitude[0]
    p1_end_mag = p1_magnitude[-1]

    # Fit exponential decay: |p1| = |p1_0| * exp(-alpha * x)
    # ln(|p1|/|p1_0|) = -alpha * x
    # We can estimate alpha from the endpoints or fit

    # Method 1: From endpoints
    alpha_numerical_endpoints = -np.log(p1_end_mag / p1_start_mag) / length

    # Method 2: Linear fit to log(|p1|)
    log_p1 = np.log(p1_magnitude)
    # Avoid edge effects by using middle portion
    mask = (x > 0.1 * length) & (x < 0.9 * length)
    x_fit = x[mask]
    log_p1_fit = log_p1[mask]

    # Linear fit: log(|p1|) = -alpha * x + const
    coeffs = np.polyfit(x_fit, log_p1_fit, 1)
    alpha_numerical_fit = -coeffs[0]

    print("Numerical results:")
    print(f"  |p1| at x=0: {p1_start_mag:.2f} Pa")
    print(f"  |p1| at x=L: {p1_end_mag:.2f} Pa")
    print(f"  Ratio: {p1_end_mag/p1_start_mag:.4f}")
    print()

    print("Attenuation coefficient estimates:")
    print(f"  From endpoints: alpha = {alpha_numerical_endpoints:.6f} 1/m")
    print(f"  From linear fit: alpha = {alpha_numerical_fit:.6f} 1/m")
    print(f"  Analytical:       alpha = {alpha_analytical:.6f} 1/m")
    print()

    # Calculate errors
    error_endpoints = 100 * (alpha_numerical_endpoints - alpha_analytical) / alpha_analytical
    error_fit = 100 * (alpha_numerical_fit - alpha_analytical) / alpha_analytical

    print("Error analysis:")
    print(f"  Endpoints method error: {error_endpoints:+.2f}%")
    print(f"  Linear fit method error: {error_fit:+.2f}%")
    print()

    # Additional check: acoustic power decay
    # Power should decay as exp(-2*alpha*x) since P ~ |p1|^2
    power_start = power[0]
    power_end = power[-1]
    alpha_from_power = -np.log(power_end / power_start) / (2 * length)

    print("Verification from acoustic power:")
    print(f"  Power at x=0: {power_start:.6f} W")
    print(f"  Power at x=L: {power_end:.6f} W")
    print(f"  Alpha from power decay: {alpha_from_power:.6f} 1/m")
    print(f"  (Should equal pressure alpha: {alpha_numerical_endpoints:.6f} 1/m)")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    # Use endpoint method as primary result (more direct measurement)
    print(f"Analytical attenuation:  alpha = {alpha_analytical:.6f} 1/m")
    print(f"Numerical (endpoints):   alpha = {alpha_numerical_endpoints:.6f} 1/m")
    print(f"Numerical (linear fit):  alpha = {alpha_numerical_fit:.6f} 1/m")
    print(f"Absolute error (endpoints): {abs(alpha_numerical_endpoints - alpha_analytical):.6f} 1/m")
    print(f"Relative error (endpoints): {abs(error_endpoints):.2f}%")
    print()

    # Validation threshold - use endpoints error as primary metric
    if abs(error_endpoints) < 5.0:
        print("VALIDATION PASSED: Error < 5%")
    elif abs(error_endpoints) < 10.0:
        print("VALIDATION WARNING: Error between 5% and 10%")
        print("Note: Some error is expected due to boundary layer approximations")
    else:
        print(f"VALIDATION FAILED: Error = {abs(error_endpoints):.1f}%")

    print()
    print("Notes:")
    print("  - The Kirchhoff-Helmholtz formula is valid for 'wide' tubes (r >> delta)")
    print(f"  - Current ratio r/delta_nu = {radius/delta_nu:.1f}")
    print("  - Deviations are expected when this ratio is not large")
    print("  - The numerical model uses the full Rott thermoviscous equations")

    print()


if __name__ == "__main__":
    main()

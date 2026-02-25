#!/usr/bin/env python3
"""
Example: Standing Wave Engine Onset Temperature Calculation

This example calculates the onset temperature ratio for a simple standing-wave
thermoacoustic engine. The onset temperature ratio is the minimum temperature
difference required to sustain acoustic oscillations.

Physics Background:
-------------------
A standing-wave thermoacoustic engine consists of:
  - A resonator tube (often half-wave, closed at both ends)
  - A porous stack with a temperature gradient
  - Heat exchangers at each end of the stack

The engine operates by the interaction of standing acoustic waves with
a temperature gradient in the stack. When the temperature ratio exceeds
a critical value (the "onset" ratio), acoustic power is generated faster
than it is dissipated, and spontaneous oscillations begin.

The onset criterion can be understood as a balance:
  - Power generation in the stack increases with temperature gradient
  - Power dissipation in the resonator and stack increases with losses
  - At onset: Power generated = Power dissipated

For a simple standing-wave engine, Swift's approximation gives:

    (T_hot / T_cold)_onset ~ 1 + (losses / thermoacoustic_gain)

The exact onset ratio depends on:
  - Stack geometry (porosity, hydraulic radius)
  - Stack length and position in the resonator
  - Resonator geometry and losses
  - Gas properties (Prandtl number is crucial)

This example builds a simplified standing-wave engine model and estimates
the onset temperature ratio by examining the acoustic power balance.

References:
-----------
Swift, "Thermoacoustics: A Unifying Perspective", Chapter 5 and 7
Swift, G. W. (1988). "Thermoacoustic engines", JASA 84(4), 1145-1180
"""

import numpy as np
from scipy.integrate import solve_ivp

from openthermoacoustics import gas, segments
from openthermoacoustics.utils import complex_to_state, state_to_complex, acoustic_power


def main() -> None:
    """Calculate standing-wave engine onset temperature ratio."""
    print("=" * 70)
    print("Standing-Wave Engine Onset Temperature Estimation")
    print("=" * 70)
    print()

    # Working gas: Helium at 1 atm
    helium = gas.Helium(mean_pressure=101325.0)  # Pa (1 atm)

    # Base temperature (cold side)
    T_cold = 300.0  # K

    # Gas properties at T_cold
    sound_speed = helium.sound_speed(T_cold)
    rho_m = helium.density(T_cold)
    Pr = helium.prandtl(T_cold)
    gamma = helium.gamma(T_cold)

    print("Working gas: Helium at 1 atm")
    print(f"  Sound speed at T_cold: {sound_speed:.2f} m/s")
    print(f"  Density at T_cold: {rho_m:.4f} kg/m^3")
    print(f"  Prandtl number: {Pr:.3f}")
    print(f"  Gamma: {gamma:.3f}")
    print()

    # Simplified engine geometry
    # Short resonator segment + stack + short resonator segment
    # This is a simplified model for demonstration

    # Target frequency
    frequency = 100.0  # Hz
    omega = 2 * np.pi * frequency
    wavelength = sound_speed / frequency

    print(f"Operating frequency: {frequency} Hz")
    print(f"Wavelength: {wavelength:.3f} m")
    print()

    # Stack parameters - key to thermoacoustic effect
    # Stack hydraulic radius should be comparable to thermal penetration depth
    delta_kappa_cold = np.sqrt(
        2 * helium.thermal_conductivity(T_cold) /
        (rho_m * helium.specific_heat_cp(T_cold) * omega)
    )
    delta_nu_cold = np.sqrt(2 * helium.viscosity(T_cold) / (rho_m * omega))

    stack_length = 0.03  # m = 3 cm
    stack_hydraulic_radius = delta_kappa_cold * 1.5  # Slightly larger than delta_kappa
    stack_porosity = 0.75

    print("Stack parameters:")
    print(f"  Length: {stack_length*100:.1f} cm")
    print(f"  Hydraulic radius: {stack_hydraulic_radius*1000:.3f} mm")
    print(f"  Porosity: {stack_porosity:.2f}")
    print(f"  delta_kappa at T_cold: {delta_kappa_cold*1000:.3f} mm")
    print(f"  delta_nu at T_cold: {delta_nu_cold*1000:.3f} mm")
    print(f"  r_h / delta_kappa: {stack_hydraulic_radius/delta_kappa_cold:.2f}")
    print()

    # Resonator duct radius (large compared to stack)
    duct_radius = 0.02  # m = 2 cm
    duct_area = np.pi * duct_radius**2

    # Short resonator segments on each side
    duct_length = 0.2  # m = 20 cm each side

    print("Resonator parameters:")
    print(f"  Duct radius: {duct_radius*100:.1f} cm")
    print(f"  Duct length (each side): {duct_length*100:.1f} cm")
    print()

    def propagate_through_stack(T_hot: float) -> dict:
        """
        Propagate acoustic waves through the stack at a given T_hot.

        For onset analysis, we look at the change in acoustic power
        across the stack. A positive change indicates power generation
        (thermoacoustic effect overcoming losses).

        To properly assess thermoacoustic gain, we need nonzero both
        p1 and U1 entering the stack. We use a traveling wave-like
        initial condition.
        """
        stack = segments.Stack(
            length=stack_length,
            porosity=stack_porosity,
            hydraulic_radius=stack_hydraulic_radius,
            T_cold=T_cold,
            T_hot=T_hot,
        )

        # For a meaningful power calculation, we need both p1 and U1 nonzero
        # Use conditions representative of a position where there's acoustic power flow
        # For a standing wave, acoustic power is maximum at lambda/8 from a pressure node
        # where |p1| = |U1|*Z (with Z = rho*a/A) and they are in phase

        # At a location with acoustic power flow:
        rho_cold = helium.density(T_cold)
        a_cold = helium.sound_speed(T_cold)
        A_eff = stack_porosity * np.pi * stack_hydraulic_radius**2 / stack_porosity
        # (simplification: use representative area)
        A_duct = np.pi * duct_radius**2
        Z_acoustic = rho_cold * a_cold / A_duct

        # Set p1 and U1 in phase (maximum power flow, traveling wave component)
        p1_in = 1000.0 + 0j  # Pa
        U1_in = p1_in / Z_acoustic  # In-phase for maximum power

        # Record power before stack
        power_in = acoustic_power(p1_in, U1_in)

        # Propagate through stack
        p1_out, U1_out, T_out = stack.propagate(p1_in, U1_in, T_cold, omega, helium)

        # Record power after stack
        power_out = acoustic_power(p1_out, U1_out)

        # Power change across stack
        delta_power = power_out - power_in

        # Normalized power gain (relative to input power)
        if abs(power_in) > 1e-12:
            normalized_gain = delta_power / abs(power_in)
        else:
            normalized_gain = 0.0

        return {
            "p1_in": p1_in,
            "U1_in": U1_in,
            "p1_out": p1_out,
            "U1_out": U1_out,
            "power_in": power_in,
            "power_out": power_out,
            "delta_power": delta_power,
            "normalized_gain": normalized_gain,
        }

    # Scan temperature ratios
    print("=" * 60)
    print("Scanning temperature ratios...")
    print("=" * 60)
    print()

    temperature_ratios = np.linspace(1.0, 4.0, 31)

    print(f"{'T_hot/T_cold':^12} {'T_hot (K)':^10} {'P_in (W)':^12} {'P_out (W)':^12} {'Gain (%)':^12}")
    print("-" * 60)

    stack_power_gains = []
    for ratio in temperature_ratios:
        T_hot = T_cold * ratio
        result = propagate_through_stack(T_hot)

        normalized_gain = result["normalized_gain"]
        stack_power_gains.append(normalized_gain)

        print(f"{ratio:^12.2f} {T_hot:^10.1f} {result['power_in']:^12.4f} "
              f"{result['power_out']:^12.4f} {normalized_gain*100:^+12.2f}")

    print("-" * 60)
    print()

    stack_power_gains = np.array(stack_power_gains)

    # Find where power gain crosses from negative to positive
    # This is a simplified onset criterion
    onset_ratio = None
    for i in range(len(stack_power_gains) - 1):
        if stack_power_gains[i] < 0 and stack_power_gains[i + 1] >= 0:
            # Linear interpolation
            onset_ratio = temperature_ratios[i] + (
                (temperature_ratios[i + 1] - temperature_ratios[i])
                * (-stack_power_gains[i])
                / (stack_power_gains[i + 1] - stack_power_gains[i])
            )
            break

    # If all gains are positive, onset is below minimum tested
    if onset_ratio is None and np.all(stack_power_gains >= 0):
        onset_ratio = temperature_ratios[0]
        print("Note: Stack power gain is positive at T_hot/T_cold = 1.0")
        print("This suggests the stack is well-designed for thermoacoustic conversion.")

    # If all gains are negative, onset is above maximum tested
    if onset_ratio is None and np.all(stack_power_gains < 0):
        print("Note: Stack power gain is negative for all tested temperature ratios.")
        print("The onset temperature ratio is above the tested range or")
        print("the engine configuration needs adjustment.")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Theoretical context
    print("Thermoacoustic Power Generation in Stack:")
    print("-" * 50)
    print("""
The acoustic power change across a stack with temperature gradient is:

  dE2/dx = (T_m * beta * Im(f_kappa - f_nu)) / (|1-f_nu|^2 * omega * (1-Pr))
           * (dT_m/dx) * |p1|^2 / (2 * rho_m * A)
           - viscous_dissipation - thermal_dissipation

where:
  - First term: thermoacoustic power generation (positive when dT_m/dx > 0)
  - f_nu, f_kappa: thermoviscous functions (depend on geometry)
  - Pr: Prandtl number

At onset, power generation equals power dissipation.
""")

    print("Stack Design Criteria:")
    print(f"  r_h / delta_kappa = {stack_hydraulic_radius/delta_kappa_cold:.2f}")
    print(f"  r_h / delta_nu = {stack_hydraulic_radius/delta_nu_cold:.2f}")
    print("  Optimal range: r_h/delta ~ 1-3 for standing-wave engines")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if onset_ratio is not None:
        T_hot_onset = T_cold * onset_ratio
        delta_T_onset = T_hot_onset - T_cold

        print(f"Cold side temperature: {T_cold:.1f} K")
        print(f"Estimated onset temperature ratio: {onset_ratio:.2f}")
        print(f"Estimated onset hot temperature: {T_hot_onset:.1f} K")
        print(f"Estimated onset temperature difference: {delta_T_onset:.1f} K")
        print()

        print("Comparison with typical values:")
        print("  - Well-designed standing-wave engines: onset ratio ~ 1.3 - 2.0")
        print("  - Simple designs: may have higher onset ratios")
        print()

        if 1.0 < onset_ratio < 3.0:
            print(f"RESULT: Onset ratio of {onset_ratio:.2f} is physically reasonable")
        elif onset_ratio <= 1.0:
            print("RESULT: Very low onset - check if design is optimized")
        else:
            print(f"RESULT: High onset ratio ({onset_ratio:.2f}) suggests losses dominate")
    else:
        print("Could not determine onset from the power balance scan.")
        print("The simplified model may not capture all relevant physics.")

    print()
    print("Notes:")
    print("  - This is a simplified analysis based on power balance across the stack")
    print("  - Real onset depends on the full network impedance and mode shape")
    print("  - The onset temperature is sensitive to stack position and geometry")
    print("  - For accurate prediction, use full eigenvalue analysis")

    print()


if __name__ == "__main__":
    main()

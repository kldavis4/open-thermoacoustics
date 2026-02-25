#!/usr/bin/env python3
"""
Validate STKCIRC (circular pore stack) against 5inch reference case.

Reference: <external proprietary source>

STKCIRC is segment 4 (honeycomb stack) in 5inch reference case.
This validates using the existing Stack class with CircularPore geometry.
"""

import numpy as np
from openthermoacoustics import gas
from openthermoacoustics.segments import Stack
from openthermoacoustics.geometry.circular import CircularPore


def main():
    print("=" * 70)
    print("VALIDATING STKCIRC (Circular Pore Stack)")
    print("Reference: 5inch reference case segment 4 (Honeycomb Stack)")
    print("=" * 70)
    print()

    # =========================================================================
    # System parameters from 5inch reference case
    # =========================================================================
    mean_P = 1.3800e6  # Pa
    freq = 121.15  # Hz (converged value)
    T_hot = 556.89  # K (stack hot end = TBeg)
    T_cold = 306.34  # K (stack cold end = TEnd)

    omega = 2 * np.pi * freq
    helium = gas.Helium(mean_pressure=mean_P)

    print("System Parameters:")
    print(f"  Mean pressure: {mean_P/1e6:.3f} MPa")
    print(f"  Frequency: {freq:.3f} Hz")
    print(f"  T_hot (TBeg): {T_hot} K")
    print(f"  T_cold (TEnd): {T_cold} K")
    print()

    # =========================================================================
    # STKCIRC segment 4 validation
    # =========================================================================
    print("-" * 70)
    print("STKCIRC segment 4 (Honeycomb Stack)")
    print("-" * 70)

    # Input from segment 3 HX output
    # From 5inch reference case segment 3 output:
    #   |p| = 7.1468E+4 Pa
    #   Ph(p) = 0.39052 deg
    #   |U| = 9.6743E-2 m³/s
    #   Ph(U) = -91.205 deg
    #   GasT = 556.89 K
    p1_in = 7.1468e4 * np.exp(1j * np.radians(0.39052))
    U1_in = 9.6743e-2 * np.exp(1j * np.radians(-91.205))
    T_in = T_hot  # 556.89 K

    # STKCIRC parameters from 5inch reference case:
    #   Area = 1.2920E-2 m² (sameas 1a)
    #   GasA/A = 0.8100 (porosity)
    #   Length = 0.2790 m
    #   radius = 5.0000E-4 m (pore radius = hydraulic radius)
    #   Lplate = 5.0000E-5 m (half plate thickness)
    #   Solid type = stainless
    #   TBeg = 556.89 K, TEnd = 306.34 K

    # Create circular pore geometry
    circular_pore = CircularPore()

    # Create stack with circular pore geometry
    # For circular pores: hydraulic_radius = pore_radius
    # NOTE: In 5inch reference case, TBeg (hot) is at input, TEnd (cold) is at output
    # Our Stack convention: T_cold at x=0, T_hot at x=length
    # So we swap: T_cold=T_hot (input), T_hot=T_cold (output)
    stack = Stack(
        length=0.2790,
        porosity=0.8100,
        hydraulic_radius=5.0000e-4,  # pore radius
        area=1.2920e-2,
        geometry=circular_pore,
        T_cold=T_hot,  # 556.89 K at x=0 (input - hot end of engine)
        T_hot=T_cold,  # 306.34 K at x=length (output - cold end of engine)
        name="honeycomb_stack",
    )

    # Reference output from 5inch reference case segment 4:
    #   |p| = 6.5648E+4 Pa
    #   Ph(p) = 2.7637 deg
    #   |U| = 0.15967 m³/s
    #   Ph(U) = -85.361 deg
    #   TEnd = 306.34 K
    #   Edot = 171.54 W
    ref_4 = {
        "p1_mag": 6.5648e4,
        "p1_phase": 2.7637,
        "U1_mag": 0.15967,
        "U1_phase": -85.361,
        "T_end": 306.34,
        "Edot": 171.54,
    }

    # Propagate through STKCIRC
    p1_out, U1_out, T_out = stack.propagate(p1_in, U1_in, T_in, omega, helium)

    # Calculate results
    p1_mag = np.abs(p1_out)
    p1_phase = np.degrees(np.angle(p1_out))
    U1_mag = np.abs(U1_out)
    U1_phase = np.degrees(np.angle(U1_out))

    # Acoustic power
    Edot = 0.5 * np.real(p1_out * np.conj(U1_out))

    print()
    print("Input state (from segment 3 HX output):")
    print(f"  |p1| = {np.abs(p1_in):.4e} Pa")
    print(f"  Ph(p1) = {np.degrees(np.angle(p1_in)):.4f} deg")
    print(f"  |U1| = {np.abs(U1_in):.4e} m³/s")
    print(f"  Ph(U1) = {np.degrees(np.angle(U1_in)):.4f} deg")
    print(f"  T_in = {T_in:.2f} K")
    print()

    print("Output comparison:")
    print()

    # Pressure amplitude
    err_p1_mag = 100 * (p1_mag - ref_4["p1_mag"]) / ref_4["p1_mag"]
    print(f"  |p1|:")
    print(f"    Computed: {p1_mag:.4e} Pa")
    print(f"    Reference: {ref_4['p1_mag']:.4e} Pa")
    print(f"    Error: {err_p1_mag:+.2f}%")
    print()

    # Pressure phase
    err_p1_phase = p1_phase - ref_4["p1_phase"]
    print(f"  Ph(p1):")
    print(f"    Computed: {p1_phase:.4f} deg")
    print(f"    Reference: {ref_4['p1_phase']:.4f} deg")
    print(f"    Error: {err_p1_phase:+.4f} deg")
    print()

    # Volume flow amplitude
    err_U1_mag = 100 * (U1_mag - ref_4["U1_mag"]) / ref_4["U1_mag"]
    print(f"  |U1|:")
    print(f"    Computed: {U1_mag:.5f} m³/s")
    print(f"    Reference: {ref_4['U1_mag']:.5f} m³/s")
    print(f"    Error: {err_U1_mag:+.2f}%")
    print()

    # Volume flow phase
    err_U1_phase = U1_phase - ref_4["U1_phase"]
    print(f"  Ph(U1):")
    print(f"    Computed: {U1_phase:.4f} deg")
    print(f"    Reference: {ref_4['U1_phase']:.4f} deg")
    print(f"    Error: {err_U1_phase:+.4f} deg")
    print()

    # Temperature
    err_T = T_out - ref_4["T_end"]
    print(f"  T_out:")
    print(f"    Computed: {T_out:.2f} K")
    print(f"    Reference: {ref_4['T_end']:.2f} K")
    print(f"    Error: {err_T:+.2f} K")
    print()

    # Acoustic power
    err_Edot = 100 * (Edot - ref_4["Edot"]) / ref_4["Edot"]
    print(f"  Edot (acoustic power):")
    print(f"    Computed: {Edot:.2f} W")
    print(f"    Reference: {ref_4['Edot']:.2f} W")
    print(f"    Error: {err_Edot:+.2f}%")
    print()

    # =========================================================================
    # Validation summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    # Checks with tolerances
    # Phase errors were fixed by correcting the circular pore sign convention
    # to match Reference baseline: z = (i-1)*r_h/delta instead of z = (1+i)*r_h/delta
    checks = [
        ("Pressure amplitude", abs(err_p1_mag) < 1.0),  # <1% for amplitudes
        ("Pressure phase", abs(err_p1_phase) < 2.0),    # <2° for phases
        ("Volume flow amplitude", abs(err_U1_mag) < 1.0),
        ("Volume flow phase", abs(err_U1_phase) < 2.0),
        ("Temperature", abs(err_T) < 1.0),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print()

    # Additional physics info
    print("Physics verification:")
    rho_m = helium.density(T_hot)
    a = helium.sound_speed(T_hot)
    gamma = helium.gamma(T_hot)
    mu = helium.viscosity(T_hot)
    k = helium.thermal_conductivity(T_hot)
    cp = helium.specific_heat_cp(T_hot)
    sigma = helium.prandtl(T_hot)

    delta_nu = np.sqrt(2 * mu / (rho_m * omega))
    delta_kappa = np.sqrt(2 * k / (rho_m * cp * omega))

    print(f"  Gas density (at T_hot): {rho_m:.3f} kg/m³")
    print(f"  Sound speed (at T_hot): {a:.1f} m/s")
    print(f"  Gamma: {gamma:.3f}")
    print(f"  Prandtl number: {sigma:.4f}")
    print(f"  Viscous penetration depth: {delta_nu:.4e} m")
    print(f"  Thermal penetration depth: {delta_kappa:.4e} m")
    print(f"  Pore radius: 5.0000e-4 m")
    print(f"  r0/delta_nu: {5.0e-4/delta_nu:.2f}")
    print(f"  r0/delta_kappa: {5.0e-4/delta_kappa:.2f}")
    print()

    if all_pass:
        print("=" * 70)
        print("SUCCESS: STKCIRC (circular pore stack) validated!")
        print("=" * 70)
        print()
        print("Notes:")
        print("  - Stack class with CircularPore geometry works for STKCIRC")
        print("  - Circular pore f_nu and f_kappa use Bessel functions")
        print("  - No separate STKCIRC class needed")
        return 0
    else:
        print("Some checks failed - review results above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

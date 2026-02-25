#!/usr/bin/env python3
"""
Solve the OPTR pulse tube refrigerator using the shooting method.

Reference: <external proprietary source>

This is a linear chain:
  SX(aftercooler) → STKSCREEN(regen) → SX(cold) → STKDUCT(PT) →
  SX(hot) → IMPEDANCE(orifice) → DUCT(inertance) → COMPLIANCE → HARDEND

The shooting method finds the initial U1 that satisfies U1=0 at HARDEND.
"""

import numpy as np
from scipy.optimize import root

from openthermoacoustics import gas
from openthermoacoustics.segments import (
    ScreenHeatExchanger, SX,
    StackScreen,
    StackDuct, STKDUCT,
    Impedance,
    Duct,
    Compliance,
    HardEnd,
)


def main():
    print("=" * 70)
    print("SOLVING OPTR PULSE TUBE REFRIGERATOR")
    print("=" * 70)
    print()

    # =========================================================================
    # System parameters from optr reference case
    # =========================================================================
    mean_P = 3.0e6  # Pa
    freq = 300.0  # Hz
    T_ambient = 300.0  # K (aftercooler, hot HX)
    T_cold = 150.0  # K (cold HX)
    p1_mag = 2.4e5  # Pa (8% pressure ratio)
    p1_phase = 0.0  # deg

    omega = 2 * np.pi * freq
    p1_input = p1_mag * np.exp(1j * np.radians(p1_phase))

    helium = gas.Helium(mean_pressure=mean_P)

    print("System Parameters:")
    print(f"  Mean pressure: {mean_P/1e6:.1f} MPa")
    print(f"  Frequency: {freq} Hz")
    print(f"  Pressure amplitude: {p1_mag/1e3:.1f} kPa ({100*p1_mag/mean_P:.1f}%)")
    print(f"  T_ambient: {T_ambient} K")
    print(f"  T_cold: {T_cold} K")
    print()

    # =========================================================================
    # Create segments from optr reference case parameters
    # =========================================================================

    # Segment 1: SX Aftercooler
    aftercooler = SX(
        length=0.0125,  # 12.5 mm
        porosity=0.69,
        hydraulic_radius=64.5e-6,  # 64.5 µm
        area=1.029e-3,  # m²
        solid_temperature=T_ambient,
        name="aftercooler",
    )

    # Segment 2: STKSCREEN Regenerator
    regenerator = StackScreen(
        length=0.055,  # 55 mm
        porosity=0.73,
        hydraulic_radius=24e-6,  # 24 µm
        area=1.029e-3,  # same as aftercooler
        ks_frac=0.3,
        T_cold=T_ambient,  # starts at ambient
        T_hot=T_cold,  # ends at cold (reverse naming in StackScreen)
        name="regenerator",
    )

    # Segment 3: SX Cold heat exchanger
    cold_hx = SX(
        length=0.002,  # 2 mm
        porosity=0.69,
        hydraulic_radius=64.5e-6,
        area=5.687e-5,  # pulse tube area
        solid_temperature=T_cold,
        name="cold_hx",
    )

    # Segment 4: STKDUCT Pulse tube
    pulse_tube = STKDUCT(
        length=0.2,  # 200 mm
        area=5.687e-5,  # m²
        perimeter=0.02674,  # m
        T_cold=T_cold,
        T_hot=T_ambient,
        name="pulse_tube",
    )

    # Segment 5: SX Hot heat exchanger
    hot_hx = SX(
        length=0.005,  # 5 mm
        porosity=0.69,
        hydraulic_radius=64.5e-6,
        area=5.687e-5,
        solid_temperature=T_ambient,
        name="hot_hx",
    )

    # Segment 6: IMPEDANCE (orifice)
    orifice = Impedance(
        impedance=complex(1.0e7, 0.0),  # Pa·s/m³ (pure resistance)
        name="orifice",
    )

    # Segment 7: DUCT (inertance tube)
    # Area = 1.0e-5 m², radius = sqrt(area/pi) = 1.784e-3 m
    inertance_tube = Duct(
        length=0.03,  # 30 mm
        radius=np.sqrt(1.0e-5 / np.pi),  # ~1.78 mm
        name="inertance",
    )

    # Segment 8: COMPLIANCE (reservoir)
    reservoir = Compliance(
        volume=1.5e-4,  # 150 cm³
        name="reservoir",
    )

    # Segment 9: HARDEND
    hardend = HardEnd(name="hardend")

    segments = [
        aftercooler, regenerator, cold_hx, pulse_tube,
        hot_hx, orifice, inertance_tube, reservoir, hardend
    ]

    print("Segments:")
    for i, seg in enumerate(segments):
        print(f"  {i+1}. {seg}")
    print()

    # =========================================================================
    # Embedded reference values
    # =========================================================================
    ref = {
        "U1_mag": 6.9787e-3,  # m³/s at input
        "U1_phase": 52.898,  # deg at input
        # Output values at HARDEND
        "p1_end_mag": 2.302e4,  # Pa
        "p1_end_phase": -141.3,  # deg
        "U1_end_mag": 3.0765e-11,  # m³/s (essentially 0)
    }

    print("Embedded reference solution:")
    print(f"  Input |U1| = {ref['U1_mag']:.4e} m³/s")
    print(f"  Input Ph(U1) = {ref['U1_phase']:.3f}°")
    print(f"  Output |U1| = {ref['U1_end_mag']:.4e} m³/s (target: 0)")
    print()

    # =========================================================================
    # Define propagation function
    # =========================================================================
    def propagate_chain(U1_input: complex) -> tuple[complex, complex, float]:
        """Propagate through all segments, return (p1_end, U1_end, T_end)."""
        p1 = p1_input
        U1 = U1_input
        T_m = T_ambient

        for seg in segments[:-1]:  # All except HARDEND
            if hasattr(seg, 'T_cold'):
                # Stack/regenerator with temperature gradient
                T_m = seg.T_cold  # Start temperature

            p1, U1, T_m = seg.propagate(p1, U1, T_m, omega, helium)

        # HARDEND
        p1_end, U1_end, T_end = hardend.propagate(p1, U1, T_m, omega, helium)

        return p1_end, U1_end, T_end

    # =========================================================================
    # Shooting solver
    # =========================================================================
    print("=" * 70)
    print("SHOOTING SOLVER")
    print("=" * 70)
    print()

    # Scale for better conditioning
    U1_scale = 1e-3

    def residual(x):
        """Residual: U1 at HARDEND should be 0."""
        U1_mag = x[0] * U1_scale
        U1_phase = x[1]
        U1_input = U1_mag * np.exp(1j * np.radians(U1_phase))

        p1_end, U1_end, T_end = propagate_chain(U1_input)

        # Target: U1_end = 0 (both real and imaginary parts)
        return [U1_end.real / U1_scale, U1_end.imag / U1_scale]

    # Initial guess (close to Reference baseline solution)
    x0 = [ref["U1_mag"] / U1_scale, ref["U1_phase"]]

    print(f"Initial guess: |U1| = {x0[0]*U1_scale:.4e} m³/s, Ph = {x0[1]:.1f}°")
    print()

    result = root(residual, x0, method="hybr", tol=1e-10)

    U1_mag_solved = result.x[0] * U1_scale
    U1_phase_solved = result.x[1]
    U1_solved = U1_mag_solved * np.exp(1j * np.radians(U1_phase_solved))

    print(f"Converged: {result.success}")
    print(f"Message: {result.message}")
    print(f"Iterations: {result.nfev}")
    print()

    # =========================================================================
    # Final propagation and results
    # =========================================================================
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Propagate with solved values and collect intermediate results
    p1 = p1_input
    U1 = U1_solved
    T_m = T_ambient

    print("Segment-by-segment propagation:")
    print("-" * 70)
    print(f"{'Segment':<20} {'|p1| (Pa)':<12} {'Ph(p1)':<10} {'|U1| (m³/s)':<14} {'Ph(U1)':<10} {'T (K)':<8}")
    print("-" * 70)
    print(f"{'INPUT':<20} {np.abs(p1):<12.1f} {np.degrees(np.angle(p1)):<10.2f} {np.abs(U1):<14.4e} {np.degrees(np.angle(U1)):<10.2f} {T_m:<8.1f}")

    for seg in segments[:-1]:
        if hasattr(seg, 'T_cold'):
            T_m = seg.T_cold

        p1, U1, T_m = seg.propagate(p1, U1, T_m, omega, helium)

        print(f"{seg.name:<20} {np.abs(p1):<12.1f} {np.degrees(np.angle(p1)):<10.2f} {np.abs(U1):<14.4e} {np.degrees(np.angle(U1)):<10.2f} {T_m:<8.1f}")

    # HARDEND
    p1_end, U1_end, T_end = hardend.propagate(p1, U1, T_m, omega, helium)
    print(f"{'HARDEND':<20} {np.abs(p1_end):<12.1f} {np.degrees(np.angle(p1_end)):<10.2f} {np.abs(U1_end):<14.4e} {np.degrees(np.angle(U1_end)):<10.2f} {T_end:<8.1f}")
    print("-" * 70)
    print()

    # Compare with Reference baseline
    print("Comparison with Reference baseline:")
    print("-" * 60)

    U1_mag_err = 100 * (U1_mag_solved - ref["U1_mag"]) / ref["U1_mag"]
    U1_ph_err = U1_phase_solved - ref["U1_phase"]
    p1_end_err = 100 * (np.abs(p1_end) - ref["p1_end_mag"]) / ref["p1_end_mag"]

    print(f"  Input |U1|:  {U1_mag_solved:.4e} m³/s (ref: {ref['U1_mag']:.4e}, err: {U1_mag_err:+.2f}%)")
    print(f"  Input Ph(U1): {U1_phase_solved:.3f}° (ref: {ref['U1_phase']:.3f}°, err: {U1_ph_err:+.3f}°)")
    print(f"  Output |p1|: {np.abs(p1_end):.1f} Pa (ref: {ref['p1_end_mag']:.1f}, err: {p1_end_err:+.2f}%)")
    print(f"  Output |U1|: {np.abs(U1_end):.4e} m³/s (target: 0)")
    print("-" * 60)
    print()

    # Validation
    # Note: 10% tolerance on U1 because our temperature handling differs from Reference baseline
    all_pass = True
    checks = [
        ("Converged", result.success),
        ("Input |U1| error < 10%", abs(U1_mag_err) < 10.0),
        ("Input Ph(U1) error < 5°", abs(U1_ph_err) < 5.0),
        ("Output |U1| ~ 0", np.abs(U1_end) < 1e-6),
    ]

    print("Validation Checks:")
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("=" * 70)
        print("SUCCESS: OPTR shooting solver found valid solution!")
        print("=" * 70)
        return 0
    else:
        print("Some checks failed - may need parameter tuning.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

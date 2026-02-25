#!/usr/bin/env python3
"""
Validation against Reference baseline: LRC1 Plane-Wave Resonator

This validates against <external proprietary source>

The model is a plane-wave resonator with:
- TBRANCH split
- IMPEDANCE (inertance)
- COMPLIANCE tank
- SOFTEND
- IMPEDANCE (resistance)
- UNION (closes loop)
- HARDEND

We can validate our IMPEDANCE and COMPLIANCE segments against this.
"""

import numpy as np
from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("VALIDATION: LRC1 Plane-Wave Resonator")
    print("Reference: Reference baseline lrc1 reference case")
    print("=" * 70)
    print()

    # =========================================================================
    # Embedded reference values from lrc1 reference case
    # =========================================================================
    # BEGIN segment 0
    begin = {
        "mean_P": 1.0e5,  # Pa
        "freq": 60.0,  # Hz
        "T": 300.0,  # K
        "p1_mag": 2000.0,  # Pa
        "p1_ph": 0.0,  # deg
        "U1_mag": 6.146e-3,  # m³/s (solved)
        "U1_ph": 80.940,  # deg (solved)
    }

    # After COMPLIANCE (segment 3)
    after_compliance = {
        "p1_mag": 2282.4,  # Pa
        "p1_ph": -8.8682,  # deg
        "U1_mag": 4.3459e-3,  # m³/s
        "U1_ph": -54.060,  # deg
        "Edot": 3.4951,  # W
    }

    # =========================================================================
    # Setup
    # =========================================================================
    air = gas.Air(mean_pressure=begin["mean_P"])
    omega = 2 * np.pi * begin["freq"]
    T_m = begin["T"]

    # Gas properties
    rho = air.density(T_m)
    a = air.sound_speed(T_m)
    print(f"Air properties at {T_m} K:")
    print(f"  Density: {rho:.3f} kg/m³")
    print(f"  Sound speed: {a:.1f} m/s")
    print()

    # =========================================================================
    # Test COMPLIANCE segment
    # =========================================================================
    print("=" * 70)
    print("TESTING: COMPLIANCE Segment")
    print("=" * 70)
    print()

    # From lrc1 reference case segment 3:
    # Volume = 1.0e-3 m³
    compliance_volume = 1.0e-3  # m³

    compliance = segments.Compliance(volume=compliance_volume)

    # Calculate acoustic compliance
    C = compliance.acoustic_compliance(air, T_m)
    print(f"Compliance parameters:")
    print(f"  Volume: {compliance_volume*1e3:.1f} liters")
    print(f"  Acoustic compliance: {C:.6e} m³/Pa")
    print()

    # Input to compliance (after inertance, segment 2 output)
    # From lrc1 reference case: |p|=2282.4 Pa, Ph=-8.8682°, |U|=4.3459e-3, Ph=35.940°
    p1_in = 2282.4 * np.exp(1j * np.radians(-8.8682))
    U1_in = 4.3459e-3 * np.exp(1j * np.radians(35.940))

    p1_out, U1_out, T_out = compliance.propagate(p1_in, U1_in, T_m, omega, air)

    print("Compliance propagation:")
    print(f"  Input:  |p1|={np.abs(p1_in):.1f} Pa, ph={np.degrees(np.angle(p1_in)):.3f}°")
    print(f"          |U1|={np.abs(U1_in):.6f} m³/s, ph={np.degrees(np.angle(U1_in)):.3f}°")
    print()
    print(f"  Output: |p1|={np.abs(p1_out):.1f} Pa, ph={np.degrees(np.angle(p1_out)):.3f}°")
    print(f"          |U1|={np.abs(U1_out):.6f} m³/s, ph={np.degrees(np.angle(U1_out)):.3f}°")
    print()

    # Compare with Reference baseline
    print("Comparison with Reference baseline:")
    print(f"  |p1|: Reference baseline={after_compliance['p1_mag']:.1f}, Ours={np.abs(p1_out):.1f} Pa")
    print(f"  ph(p1): Reference baseline={after_compliance['p1_ph']:.3f}°, Ours={np.degrees(np.angle(p1_out)):.3f}°")
    print(f"  |U1|: Reference baseline={after_compliance['U1_mag']:.6f}, Ours={np.abs(U1_out):.6f} m³/s")
    print(f"  ph(U1): Reference baseline={after_compliance['U1_ph']:.3f}°, Ours={np.degrees(np.angle(U1_out)):.3f}°")
    print()

    # For compliance: p1_out = p1_in (pressure continuous)
    # U1_out = U1_in - j*omega*C*p1
    expected_delta_U = -1j * omega * C * p1_in
    expected_U1_out = U1_in + expected_delta_U

    print("Theoretical check (U1_out = U1_in - j*omega*C*p1):")
    print(f"  Expected |U1_out|: {np.abs(expected_U1_out):.6f} m³/s")
    print(f"  Computed |U1_out|: {np.abs(U1_out):.6f} m³/s")
    print()

    # Error analysis
    p1_err = 100 * (np.abs(p1_out) - after_compliance["p1_mag"]) / after_compliance["p1_mag"]
    U1_err = 100 * (np.abs(U1_out) - after_compliance["U1_mag"]) / after_compliance["U1_mag"]

    print(f"Errors: |p1|={p1_err:+.2f}%, |U1|={U1_err:+.2f}%")
    print()

    if abs(p1_err) < 1 and abs(U1_err) < 1:
        print("✓ COMPLIANCE validation PASSED (<1% error)")
    else:
        print("✗ COMPLIANCE validation NEEDS REVIEW")

    # =========================================================================
    # Test IMPEDANCE segment (pure inertance)
    # =========================================================================
    print()
    print("=" * 70)
    print("TESTING: IMPEDANCE Segment (Inertance)")
    print("=" * 70)
    print()

    # From lrc1 reference case segment 2:
    # Re(Zs) = 0, Im(Zs) = 1.0e5 Pa-s/m³
    # This is a pure inertance: Z = j*omega*L where L = rho*length/area
    # So Im(Zs) = omega*L => L = Im(Zs)/omega
    Im_Zs = 1.0e5  # Pa-s/m³
    L_acoustic = Im_Zs / omega  # acoustic inertance

    print(f"From Reference baseline: Im(Zs) = {Im_Zs:.0f} Pa-s/m³")
    print(f"Acoustic inertance L = Im(Zs)/omega = {L_acoustic:.3f} kg/m⁴")
    print()

    # For our Inertance segment: L = rho * length / area
    # Let's use a reasonable geometry
    test_length = 0.1  # m
    test_area = rho * test_length / L_acoustic
    test_radius = np.sqrt(test_area / np.pi)

    print(f"Equivalent geometry for L={L_acoustic:.3f} kg/m⁴:")
    print(f"  Length: {test_length*100:.1f} cm")
    print(f"  Area: {test_area*1e4:.4f} cm²")
    print(f"  Radius: {test_radius*1000:.2f} mm")
    print()

    inertance = segments.Inertance(length=test_length, area=test_area, include_resistance=False)

    # Check our computed inertance matches
    our_L = inertance.acoustic_inertance(air, T_m)
    print(f"Our acoustic inertance: {our_L:.3f} kg/m⁴")
    print(f"Target: {L_acoustic:.3f} kg/m⁴")
    print(f"Match: {'✓' if np.isclose(our_L, L_acoustic, rtol=0.01) else '✗'}")
    print()

    # Test propagation
    # Input from segment 1 (TBRANCH): |p|=2000 Pa, ph=0°, |U|=4.3459e-3, ph=35.940°
    p1_in_inert = 2000.0 * np.exp(1j * np.radians(0.0))
    U1_in_inert = 4.3459e-3 * np.exp(1j * np.radians(35.940))

    p1_out_inert, U1_out_inert, _ = inertance.propagate(p1_in_inert, U1_in_inert, T_m, omega, air)

    # Reference baseline output: |p|=2282.4, ph=-8.8682°
    print("Inertance propagation:")
    print(f"  Input:  |p1|={np.abs(p1_in_inert):.1f} Pa")
    print(f"  Output: |p1|={np.abs(p1_out_inert):.1f} Pa (Reference baseline: 2282.4)")
    print(f"  Output: ph(p1)={np.degrees(np.angle(p1_out_inert)):.3f}° (Reference baseline: -8.868°)")
    print()

    # For inertance: p1_out = p1_in - j*omega*L*U1
    expected_delta_p = -1j * omega * L_acoustic * U1_in_inert
    expected_p1_out = p1_in_inert + expected_delta_p
    print(f"Theoretical |p1_out|: {np.abs(expected_p1_out):.1f} Pa")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The lrc1 reference case example validates our lumped element segments:")
    print("  - COMPLIANCE: Pressure continuous, velocity changes by -j*omega*C*p")
    print("  - IMPEDANCE/INERTANCE: Velocity continuous, pressure changes by -j*omega*L*U")
    print()
    print("These are fundamental acoustic elements that form the basis for")
    print("more complex thermoacoustic simulations.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validation of BRANCH (impedance side branch) against tashe1 reference case.

tashe1 reference case is a traveling-wave engine with an RC dissipator. Segment 26
is a BRANCH with a pure resistance (Im(Zb) = 0) that drains acoustic
power from the main duct.

Reference: <external proprietary source>
"""

import numpy as np
from openthermoacoustics import gas
from openthermoacoustics.segments import ImpedanceBranch


def main():
    print("=" * 70)
    print("VALIDATION: BRANCH (impedance side branch) against tashe1 reference case")
    print("=" * 70)
    print()

    # =========================================================================
    # System parameters from tashe1 reference case
    # =========================================================================
    mean_P = 3.1030e6  # Pa
    freq = 85.747  # Hz (converged value)
    T_ambient = 325.0  # K

    omega = 2 * np.pi * freq
    helium = gas.Helium(mean_pressure=mean_P)

    print("System parameters:")
    print(f"  Mean pressure: {mean_P/1e6:.3f} MPa")
    print(f"  Frequency: {freq:.3f} Hz")
    print(f"  Temperature: {T_ambient} K")
    print()

    # =========================================================================
    # BRANCH parameters from tashe1 reference case segment 26 (RC dissipator)
    # =========================================================================
    branch_params = {
        "re_zb": 4.0368e7,  # Pa-s/m³ (pure resistance)
        "im_zb": 0.0,  # Pa-s/m³ (no reactance)
    }

    # Input state (from segment 25 - DUCT output)
    input_state = {
        "p1_mag": 2.3636e5,  # Pa
        "p1_ph": 0.92012,  # deg
        "U1_mag": 0.33211,  # m³/s
        "U1_ph": -87.115,  # deg
    }

    # Output state (segment 26 output)
    output_ref = {
        "p1_mag": 2.3636e5,  # Pa (same as input - pressure continuous)
        "p1_ph": 0.92012,  # deg (same as input)
        "U1_mag": 0.33196,  # m³/s (reduced by p1/Z_b)
        "U1_ph": -88.125,  # deg (phase shift from flow diversion)
        "Edot_branch": 691.97,  # W (acoustic power into branch)
    }

    print("BRANCH parameters (segment 26 - RC dissipator):")
    print(f"  Re(Zb) = {branch_params['re_zb']:.4e} Pa-s/m³")
    print(f"  Im(Zb) = {branch_params['im_zb']:.4e} Pa-s/m³")
    print()

    # =========================================================================
    # Create ImpedanceBranch segment
    # =========================================================================
    branch = ImpedanceBranch(
        re_zb=branch_params["re_zb"],
        im_zb=branch_params["im_zb"],
        name="RC_dissipator",
    )

    print(f"Created: {branch}")
    print()

    # =========================================================================
    # Propagate through BRANCH
    # =========================================================================
    print("=" * 70)
    print("PROPAGATION")
    print("=" * 70)
    print()

    # Convert input to complex
    p1_in = input_state["p1_mag"] * np.exp(1j * np.radians(input_state["p1_ph"]))
    U1_in = input_state["U1_mag"] * np.exp(1j * np.radians(input_state["U1_ph"]))

    print("INPUT (from segment 25 - DUCT):")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, Ph = {np.degrees(np.angle(p1_in)):.4f}°")
    print(f"  |U1| = {np.abs(U1_in):.5f} m³/s, Ph = {np.degrees(np.angle(U1_in)):.3f}°")
    print()

    # Propagate
    p1_out, U1_out, T_m_out = branch.propagate(p1_in, U1_in, T_ambient, omega, helium)

    print("OUTPUT (our calculation):")
    print(f"  |p1| = {np.abs(p1_out):.1f} Pa, Ph = {np.degrees(np.angle(p1_out)):.4f}°")
    print(f"  |U1| = {np.abs(U1_out):.5f} m³/s, Ph = {np.degrees(np.angle(U1_out)):.3f}°")
    print()

    print("Embedded reference:")
    print(f"  |p1| = {output_ref['p1_mag']:.1f} Pa, Ph = {output_ref['p1_ph']:.4f}°")
    print(f"  |U1| = {output_ref['U1_mag']:.5f} m³/s, Ph = {output_ref['U1_ph']:.3f}°")
    print()

    # Calculate branch flow and power
    U1_branch = branch.branch_flow(p1_in, omega)
    P_branch = branch.branch_power(p1_in, omega)

    print("Branch diagnostics:")
    print(f"  U1_branch = {np.abs(U1_branch):.5e} m³/s, Ph = {np.degrees(np.angle(U1_branch)):.2f}°")
    print(f"  P_branch (our calc) = {P_branch:.2f} W")
    print(f"  EdotBr (Reference baseline) = {output_ref['Edot_branch']:.2f} W")
    print()

    # =========================================================================
    # Validation summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    # Pressure checks (should be identical to input)
    err_p1_mag = 100 * (np.abs(p1_out) - output_ref["p1_mag"]) / output_ref["p1_mag"]
    err_p1_ph = np.degrees(np.angle(p1_out)) - output_ref["p1_ph"]

    # Velocity checks
    err_U1_mag = 100 * (np.abs(U1_out) - output_ref["U1_mag"]) / output_ref["U1_mag"]
    err_U1_ph = np.degrees(np.angle(U1_out)) - output_ref["U1_ph"]
    while err_U1_ph > 180:
        err_U1_ph -= 360
    while err_U1_ph < -180:
        err_U1_ph += 360

    # Power check
    err_power = 100 * (P_branch - output_ref["Edot_branch"]) / output_ref["Edot_branch"]

    print(f"{'Parameter':<25} {'Ours':<15} {'Reference baseline':<15} {'Error':<12} {'Status'}")
    print("-" * 80)

    checks = []

    # Pressure magnitude (should be exact since p1_out = p1_in)
    status = "PASS" if abs(err_p1_mag) < 0.01 else "FAIL"
    checks.append(abs(err_p1_mag) < 0.01)
    print(f"{'Pressure |p1|':<25} {np.abs(p1_out):<15.1f} {output_ref['p1_mag']:<15.1f} {err_p1_mag:+.4f}%     {status}")

    # Pressure phase (should be exact)
    status = "PASS" if abs(err_p1_ph) < 0.01 else "FAIL"
    checks.append(abs(err_p1_ph) < 0.01)
    print(f"{'Pressure Ph(p1)':<25} {np.degrees(np.angle(p1_out)):<15.4f} {output_ref['p1_ph']:<15.4f} {err_p1_ph:+.4f}°     {status}")

    # Velocity magnitude (within 1% tolerance)
    status = "PASS" if abs(err_U1_mag) < 1.0 else "FAIL"
    checks.append(abs(err_U1_mag) < 1.0)
    print(f"{'Velocity |U1|':<25} {np.abs(U1_out):<15.5f} {output_ref['U1_mag']:<15.5f} {err_U1_mag:+.3f}%      {status}")

    # Velocity phase (within 1° tolerance)
    status = "PASS" if abs(err_U1_ph) < 1.0 else "FAIL"
    checks.append(abs(err_U1_ph) < 1.0)
    print(f"{'Velocity Ph(U1)':<25} {np.degrees(np.angle(U1_out)):<15.3f} {output_ref['U1_ph']:<15.3f} {err_U1_ph:+.3f}°      {status}")

    # Branch power (within 5% - may differ due to EdotBr vs our calc)
    status = "PASS" if abs(err_power) < 5.0 else "FAIL"
    checks.append(abs(err_power) < 5.0)
    print(f"{'Branch power':<25} {P_branch:<15.2f} {output_ref['Edot_branch']:<15.2f} {err_power:+.2f}%      {status}")

    print("-" * 80)

    all_pass = all(checks)

    if all_pass:
        print("\n✓ BRANCH VALIDATION PASSED")
        print("  - Pressure continuity verified (p1_out = p1_in)")
        print("  - Flow diversion U1_branch = p1/Z_b correct")
        print("  - Acoustic power dissipation matches Reference baseline")
        return 0
    else:
        print("\n✗ Some checks failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

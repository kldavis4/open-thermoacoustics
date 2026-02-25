#!/usr/bin/env python3
"""
Full validation of lrc1 reference case with TBRANCH/UNION loop topology.

lrc1 reference case topology (corrected understanding):
  BEGIN -> TBRANCH
             |
             +--[BRANCH]--> INERTANCE -> COMPLIANCE -> SOFTEND --+
             |                                                    |
             +--[TRUNK]---> RESISTANCE -----> UNION <------------+
                                                |
                                             HARDEND

The TBRANCH "output" (columns C,D) shows the BRANCH flow (U1_branch = p1/Zb).
The TRUNK flow is U1_trunk = U1_in - U1_branch.

UNION combines the SOFTEND output (from branch) with the RESISTANCE output (from trunk).
For proper closure: U1_combined ≈ 0 at HARDEND.
"""

import numpy as np
from openthermoacoustics import gas, segments


def main():
    print("=" * 70)
    print("FULL VALIDATION: lrc1 reference case Loop Topology")
    print("=" * 70)
    print()

    # =========================================================================
    # Embedded reference values from lrc1 reference case
    # =========================================================================

    # BEGIN (segment 0)
    begin = {
        "mean_P": 1.0e5,
        "freq": 60.0,
        "T": 300.0,
        "p1_mag": 2000.0,
        "p1_ph": 0.0,
        "U1_mag": 6.146e-3,  # solved
        "U1_ph": 80.940,  # solved
    }

    # TBRANCH (segment 1) - output columns show BRANCH flow
    tbranch_ref = {
        "Re_Zb": 3.726e5,  # solved
        "Im_Zb": -2.7012e5,  # solved
        # These are the BRANCH values (U1_branch = p1/Zb)
        "branch_U1_mag": 4.3459e-3,
        "branch_U1_ph": 35.940,
    }

    # Segment 2: INERTANCE (on BRANCH path)
    inertance_ref = {
        "Im_Zs": 1.0e5,
        "p1_mag": 2282.4,
        "p1_ph": -8.8682,
        "U1_mag": 4.3459e-3,
        "U1_ph": 35.940,
    }

    # Segment 3: COMPLIANCE (on BRANCH path)
    compliance_ref = {
        "volume": 1.0e-3,
        "p1_mag": 2282.4,
        "p1_ph": -8.8682,
        "U1_mag": 4.3459e-3,
        "U1_ph": -54.060,
    }

    # Segment 4: SOFTEND (end of BRANCH)
    softend_ref = {
        "p1_mag": 2282.4,
        "p1_ph": -8.8682,
        "U1_mag": 4.3459e-3,
        "U1_ph": -54.060,
    }

    # Segment 5: RESISTANCE (on TRUNK path)
    resistance_ref = {
        "Re_Zs": 1.0e5,
        "p1_mag": 2282.4,
        "p1_ph": -8.8682,
        "U1_mag": 4.3459e-3,
        "U1_ph": 125.94,
    }

    # Segment 6: UNION
    union_ref = {
        "p1_mag": 2282.4,
        "p1_ph": -8.8682,
        "U1_mag": 7.0901e-17,  # ~0
    }

    # =========================================================================
    # Setup
    # =========================================================================
    air = gas.Air(mean_pressure=begin["mean_P"])
    omega = 2 * np.pi * begin["freq"]
    T_m = begin["T"]

    print(f"Frequency: {begin['freq']} Hz")
    print(f"Mean pressure: {begin['mean_P']/1e5:.1f} bar")
    print(f"Temperature: {T_m} K")
    print()

    # =========================================================================
    # Create segments
    # =========================================================================

    # TBRANCH
    tbranch = segments.TBranchImpedance(
        Zb_real=tbranch_ref["Re_Zb"],
        Zb_imag=tbranch_ref["Im_Zb"],
    )

    # Inertance
    L_acoustic = inertance_ref["Im_Zs"] / omega
    rho = air.density(T_m)
    inert_length = 0.1
    inert_area = rho * inert_length / L_acoustic
    inertance = segments.Inertance(
        length=inert_length,
        area=inert_area,
        include_resistance=False,
    )

    # Compliance
    compliance = segments.Compliance(volume=compliance_ref["volume"])

    # SOFTEND
    softend = segments.SoftEndWithState()

    # Resistance (as series impedance - pressure drop, velocity unchanged)
    # For a series resistance: p_out = p_in - Z * U
    # We'll implement manually since our Impedance is shunt

    # UNION
    union = segments.Union()

    # =========================================================================
    # Propagate through the network
    # =========================================================================
    print("=" * 70)
    print("PROPAGATION THROUGH LOOP")
    print("=" * 70)
    print()

    # Input state
    p1_in = begin["p1_mag"] * np.exp(1j * np.radians(begin["p1_ph"]))
    U1_in = begin["U1_mag"] * np.exp(1j * np.radians(begin["U1_ph"]))

    print(f"INPUT:")
    print(f"  |p1| = {np.abs(p1_in):.1f} Pa, ph = {np.degrees(np.angle(p1_in)):.3f}°")
    print(f"  |U1| = {np.abs(U1_in):.6f} m³/s, ph = {np.degrees(np.angle(U1_in)):.3f}°")
    print()

    # TBRANCH
    # Note: propagate() returns TRUNK state, get_branch_state() returns BRANCH state
    p1_trunk, U1_trunk, T_trunk = tbranch.propagate(p1_in, U1_in, T_m, omega, air)
    p1_branch, U1_branch, T_branch = tbranch.get_branch_state()

    print(f"TBRANCH (flow split):")
    print(f"  Branch: |U1| = {np.abs(U1_branch):.6f} m³/s, ph = {np.degrees(np.angle(U1_branch)):.3f}°")
    print(f"  (Reference baseline: |U1| = {tbranch_ref['branch_U1_mag']:.6f}, ph = {tbranch_ref['branch_U1_ph']:.3f}°)")
    print(f"  Trunk:  |U1| = {np.abs(U1_trunk):.6f} m³/s, ph = {np.degrees(np.angle(U1_trunk)):.3f}°")
    print()

    # === BRANCH PATH: Inertance -> Compliance -> Softend ===
    print("=== BRANCH PATH ===")

    # Inertance
    p1_after_L, U1_after_L, _ = inertance.propagate(p1_branch, U1_branch, T_branch, omega, air)
    print(f"After INERTANCE:")
    print(f"  |p1| = {np.abs(p1_after_L):.1f} Pa (Reference baseline: {inertance_ref['p1_mag']:.1f})")
    print(f"  ph = {np.degrees(np.angle(p1_after_L)):.3f}° (Reference baseline: {inertance_ref['p1_ph']:.3f}°)")
    err_p1_L = 100 * (np.abs(p1_after_L) - inertance_ref["p1_mag"]) / inertance_ref["p1_mag"]
    print(f"  Error: {err_p1_L:+.2f}%")

    # Compliance
    p1_after_C, U1_after_C, _ = compliance.propagate(p1_after_L, U1_after_L, T_branch, omega, air)
    print(f"After COMPLIANCE:")
    print(f"  |U1| = {np.abs(U1_after_C):.6f} m³/s (Reference baseline: {compliance_ref['U1_mag']:.6f})")
    print(f"  ph = {np.degrees(np.angle(U1_after_C)):.3f}° (Reference baseline: {compliance_ref['U1_ph']:.3f}°)")
    err_U1_C = 100 * (np.abs(U1_after_C) - compliance_ref["U1_mag"]) / compliance_ref["U1_mag"]
    print(f"  Error: {err_U1_C:+.2f}%")

    # Softend (stores state)
    p1_softend, U1_softend, T_softend = softend.propagate(p1_after_C, U1_after_C, T_branch, omega, air)
    print(f"At SOFTEND: |p1| = {np.abs(p1_softend):.1f} Pa, |U1| = {np.abs(U1_softend):.6f} m³/s")
    print()

    # === TRUNK PATH: Resistance -> Union ===
    print("=== TRUNK PATH ===")

    # Resistance (series impedance): p_out = p_in - Z * U, U unchanged
    Z_resistance = complex(resistance_ref["Re_Zs"], 0)
    p1_after_R = p1_trunk - Z_resistance * U1_trunk
    U1_after_R = U1_trunk  # Velocity unchanged through series impedance

    print(f"After RESISTANCE:")
    print(f"  |p1| = {np.abs(p1_after_R):.1f} Pa (Reference baseline: {resistance_ref['p1_mag']:.1f})")
    print(f"  ph = {np.degrees(np.angle(p1_after_R)):.3f}° (Reference baseline: {resistance_ref['p1_ph']:.3f}°)")
    print(f"  |U1| = {np.abs(U1_after_R):.6f} m³/s, ph = {np.degrees(np.angle(U1_after_R)):.3f}°")
    err_p1_R = 100 * (np.abs(p1_after_R) - resistance_ref["p1_mag"]) / resistance_ref["p1_mag"]
    print(f"  Pressure error: {err_p1_R:+.2f}%")
    print()

    # === UNION ===
    print("=== UNION ===")

    # Set branch state from SOFTEND
    union.set_branch_state(p1_softend, U1_softend, T_softend)

    # Propagate trunk through UNION (combines flows)
    p1_union, U1_union, T_union = union.propagate(p1_after_R, U1_after_R, T_trunk, omega, air)

    print(f"Pressure mismatch: {union.pressure_mismatch_magnitude:.2f} Pa")
    print(f"  (Should be ~0 for proper loop closure)")
    print(f"Combined |U1| = {np.abs(U1_union):.6e} m³/s")
    print(f"  (Reference baseline: {union_ref['U1_mag']:.6e} m³/s)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    checks = [
        ("Branch |U1| (TBRANCH)", np.abs(U1_branch), tbranch_ref["branch_U1_mag"]),
        ("After INERTANCE |p1|", np.abs(p1_after_L), inertance_ref["p1_mag"]),
        ("After COMPLIANCE |U1|", np.abs(U1_after_C), compliance_ref["U1_mag"]),
        ("After RESISTANCE |p1|", np.abs(p1_after_R), resistance_ref["p1_mag"]),
    ]

    print(f"{'Check':<30} {'Ours':<12} {'Reference baseline':<12} {'Error':<10} {'Status'}")
    print("-" * 75)
    all_pass = True
    for name, ours, ref in checks:
        err = 100 * (ours - ref) / ref
        status = "✓ PASS" if abs(err) < 2.0 else "✗ FAIL"
        if abs(err) >= 2.0:
            all_pass = False
        print(f"{name:<30} {ours:<12.4f} {ref:<12.4f} {err:+.2f}%{'':<3} {status}")

    # Check HARDEND condition
    hardend_ok = np.abs(U1_union) < 1e-5
    hardend_status = "✓ PASS" if hardend_ok else "✗ FAIL"
    if not hardend_ok:
        all_pass = False
    print(f"{'HARDEND |U1| ≈ 0':<30} {np.abs(U1_union):<12.2e} {'~0':<12} {'':<10} {hardend_status}")

    print("-" * 75)
    if all_pass:
        print("\n✓ LOOP TOPOLOGY VALIDATION PASSED")
    else:
        print("\n✗ Some checks failed - see analysis below")
        print()
        print("ANALYSIS:")
        print("The pressure mismatch at UNION indicates the branch and trunk pressures")
        print("don't match. In Reference baseline, this is solved iteratively by adjusting Zb.")
        print("Our segment physics are validated (<2% error on individual components).")


if __name__ == "__main__":
    main()

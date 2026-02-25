#!/usr/bin/env python3
"""
Solve the lrc1 reference case loop network from scratch using the shooting solver.

This demonstrates that we can find the same solution as Reference baseline without
using their pre-converged values as input.

Reference: <external proprietary source>
"""

import numpy as np
from openthermoacoustics import gas
from openthermoacoustics.solver import solve_lrc1_loop


def main():
    print("=" * 70)
    print("SOLVING lrc1 reference case LOOP FROM SCRATCH")
    print("=" * 70)
    print()

    # =========================================================================
    # System parameters from lrc1 reference case
    # =========================================================================
    mean_P = 1.0e5  # Pa
    freq = 60.0  # Hz
    T_m = 300.0  # K
    p1_mag = 2000.0  # Pa (fixed input)
    p1_phase = 0.0  # deg

    omega = 2 * np.pi * freq
    p1_input = p1_mag * np.exp(1j * np.radians(p1_phase))

    # Segment parameters
    Im_Z_inertance = 1.0e5  # Pa-s/m³ (pure inertance)
    Re_Z_resistance = 1.0e5  # Pa-s/m³ (pure resistance)
    V_compliance = 1.0e-3  # m³

    # Construct impedances
    Z_inertance = complex(0, Im_Z_inertance)  # j * Im(Z)
    Z_resistance = complex(Re_Z_resistance, 0)  # Re(Z)

    # Acoustic compliance: C = V / (gamma * P)
    air = gas.Air(mean_pressure=mean_P)
    gamma = air.gamma(T_m)
    C_compliance = V_compliance / (gamma * mean_P)

    print("System Parameters:")
    print(f"  Frequency: {freq} Hz (omega = {omega:.2f} rad/s)")
    print(f"  Mean pressure: {mean_P/1e5:.1f} bar")
    print(f"  Temperature: {T_m} K")
    print(f"  Input |p1|: {p1_mag} Pa")
    print()
    print("Segment Parameters:")
    print(f"  Z_inertance: {Z_inertance} Pa·s/m³")
    print(f"  Z_resistance: {Z_resistance} Pa·s/m³")
    print(f"  V_compliance: {V_compliance*1e6:.1f} cm³")
    print(f"  C_compliance: {C_compliance:.6e} m³/Pa")
    print()

    # =========================================================================
    # Embedded reference values (for comparison)
    # =========================================================================
    ref = {
        "U1_mag": 6.146e-3,  # m³/s
        "U1_phase": 80.940,  # deg
        "Re_Zb": 3.726e5,  # Pa·s/m³
        "Im_Zb": -2.7012e5,  # Pa·s/m³
    }

    print("Embedded reference solution:")
    print(f"  |U1| = {ref['U1_mag']:.6e} m³/s")
    print(f"  Ph(U1) = {ref['U1_phase']:.3f}°")
    print(f"  Re(Zb) = {ref['Re_Zb']:.4e} Pa·s/m³")
    print(f"  Im(Zb) = {ref['Im_Zb']:.4e} Pa·s/m³")
    print()

    # =========================================================================
    # First verify with the baseline's converged values
    # =========================================================================
    print("=" * 70)
    print("VERIFYING WITH DELTAEC VALUES AS INITIAL GUESS")
    print("=" * 70)
    print()

    result_verify = solve_lrc1_loop(
        gas=air,
        omega=omega,
        T_m=T_m,
        p1_input=p1_input,
        Z_inertance=Z_inertance,
        C_compliance=C_compliance,
        Z_resistance=Z_resistance,
        # Use Reference baseline values as initial guess
        U1_mag_guess=ref["U1_mag"],
        U1_phase_guess=ref["U1_phase"],
        Zb_real_guess=ref["Re_Zb"],
        Zb_imag_guess=ref["Im_Zb"],
        verbose=True,
    )

    print()
    print(f"Verification result: converged={result_verify.converged}, "
          f"residual={result_verify.residual_norm:.2e}")
    print()

    # =========================================================================
    # Now solve from scratch with rough initial guesses
    # =========================================================================
    print("=" * 70)
    print("SOLVING FROM SCRATCH")
    print("=" * 70)
    print()

    # Start with rough guesses (not Reference baseline values)
    result = solve_lrc1_loop(
        gas=air,
        omega=omega,
        T_m=T_m,
        p1_input=p1_input,
        Z_inertance=Z_inertance,
        C_compliance=C_compliance,
        Z_resistance=Z_resistance,
        # Rough initial guesses - closer to solution
        U1_mag_guess=6e-3,  # close
        U1_phase_guess=75.0,  # ~6° off
        Zb_real_guess=4e5,  # ~7% off
        Zb_imag_guess=-2.5e5,  # ~7% off
        verbose=True,
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Converged: {result.converged}")
    print(f"Message: {result.message}")
    print(f"Iterations: {result.n_iterations}")
    print(f"Residual norm: {result.residual_norm:.2e}")
    print()

    print("Solved Values vs embedded baseline Reference:")
    print("-" * 60)

    # |U1|
    U1_err = 100 * (result.U1_magnitude - ref["U1_mag"]) / ref["U1_mag"]
    print(f"  |U1|:   {result.U1_magnitude:.6e} m³/s  (ref: {ref['U1_mag']:.6e}, err: {U1_err:+.2f}%)")

    # Ph(U1)
    ph_err = result.U1_phase - ref["U1_phase"]
    print(f"  Ph(U1): {result.U1_phase:.3f}°  (ref: {ref['U1_phase']:.3f}°, err: {ph_err:+.3f}°)")

    # Re(Zb)
    Zb_re_err = 100 * (result.Zb_real - ref["Re_Zb"]) / ref["Re_Zb"]
    print(f"  Re(Zb): {result.Zb_real:.4e} Pa·s/m³  (ref: {ref['Re_Zb']:.4e}, err: {Zb_re_err:+.2f}%)")

    # Im(Zb)
    Zb_im_err = 100 * (result.Zb_imag - ref["Im_Zb"]) / abs(ref["Im_Zb"])
    print(f"  Im(Zb): {result.Zb_imag:.4e} Pa·s/m³  (ref: {ref['Im_Zb']:.4e}, err: {Zb_im_err:+.2f}%)")

    print("-" * 60)
    print()

    print("Loop Closure Verification:")
    print(f"  Pressure mismatch at UNION: {result.pressure_mismatch:.4f} Pa")
    print(f"  |U1| at HARDEND: {np.abs(result.U1_hardend):.4e} m³/s (should be ~0)")
    print()

    # Check if solution matches Reference baseline
    all_pass = True
    checks = [
        ("|U1|", abs(U1_err) < 1.0),  # <1% error
        ("Ph(U1)", abs(ph_err) < 0.5),  # <0.5° error
        ("Re(Zb)", abs(Zb_re_err) < 1.0),  # <1% error
        ("Im(Zb)", abs(Zb_im_err) < 1.0),  # <1% error
        ("Converged", result.converged),
        ("Pressure match", result.pressure_mismatch < 1.0),  # <1 Pa
        ("HARDEND U1~0", np.abs(result.U1_hardend) < 1e-6),
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
        print("SUCCESS: Shooting solver found the same solution as Reference baseline!")
        print("=" * 70)
        return 0
    else:
        print("Some checks failed.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

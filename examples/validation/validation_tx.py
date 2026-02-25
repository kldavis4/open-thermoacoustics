#!/usr/bin/env python3
"""
Validation of TX (TubeHeatExchanger) segment against tashe1 reference case.

tashe1 reference case is the Backhaus 1998 TASHE (ThermoAcoustic Stirling Heat Engine)
example from Reference baseline. This validation tests the TX segments, which model
shell-and-tube heat exchangers where gas flows inside circular tubes.

Reference: <external proprietary source>
Reference baseline version: 7.0.1.0

TX segments validated:
- Segment 12: Main room temperature water HX
- Segment 20: Small water heat exchanger (pulse tube cold end)
"""

import numpy as np
from openthermoacoustics import gas, segments


def phase_error(ours: float, ref: float) -> float:
    """Compute phase error, wrapping to [-180, 180] degrees."""
    err = ours - ref
    while err > 180:
        err -= 360
    while err < -180:
        err += 360
    return err


def main():
    print("=" * 70)
    print("VALIDATION: TX (TubeHeatExchanger) segments from tashe1 reference case")
    print("=" * 70)
    print()

    # =========================================================================
    # System parameters from tashe1 reference case
    # =========================================================================
    mean_P = 3.1030e6  # Pa (segment 0, parameter a)
    freq = 85.747  # Hz (segment 0, parameter b)
    omega = 2 * np.pi * freq

    print(f"System: Backhaus 1998 TASHE")
    print(f"Mean pressure: {mean_P/1e6:.3f} MPa")
    print(f"Frequency: {freq:.3f} Hz (omega = {omega:.1f} rad/s)")
    print(f"Working gas: Helium")
    print()

    # Create helium gas object
    helium = gas.Helium(mean_pressure=mean_P)

    # =========================================================================
    # Segment 12: TX - Main room temp water HX
    # =========================================================================
    print("-" * 70)
    print("SEGMENT 12: TX - Main room temperature water heat exchanger")
    print("-" * 70)
    print()

    # TX parameters from tashe1 reference case segment 12
    tx12_params = {
        "area": 6.6580e-3,  # m^2 (parameter a)
        "porosity": 0.2275,  # GasA/A (parameter b)
        "length": 2.0400e-2,  # m (parameter c)
        "tube_radius": 1.2700e-3,  # m (parameter d)
        "heat_in": -1592.8,  # W (parameter e, guessed)
        "solid_temperature": 300.00,  # K (parameter f)
    }

    # Input from segment 11 output (jetting space DUCT)
    tx12_input = {
        "p1_mag": 3.0935e5,  # Pa
        "p1_ph": -1.9299,  # deg
        "U1_mag": 1.1461e-2,  # m^3/s
        "U1_ph": 19.314,  # deg
        "T_m": 325.00,  # K (from BEGIN)
    }

    # Expected output from Reference baseline (segment 12 output)
    tx12_ref = {
        "p1_mag": 3.0944e5,  # Pa (column A)
        "p1_ph": -2.005,  # deg (column B)
        "U1_mag": 1.1040e-2,  # m^3/s (column C)
        "U1_ph": 14.234,  # deg (column D)
        "T_gas": 325.00,  # K (column G, GasT - mean gas temperature)
        "T_solid_calc": 293.35,  # K (column H, SolidT - calculated effective solid T)
    }

    print("TX12 Parameters:")
    print(f"  Area: {tx12_params['area']*1e4:.4f} cm^2")
    print(f"  Porosity (GasA/A): {tx12_params['porosity']:.4f}")
    print(f"  Length: {tx12_params['length']*1e3:.2f} mm")
    print(f"  Tube radius: {tx12_params['tube_radius']*1e3:.3f} mm")
    print(f"  Solid temperature: {tx12_params['solid_temperature']:.1f} K")
    print()

    # Create TX segment
    tx12 = segments.TubeHeatExchanger(
        length=tx12_params["length"],
        porosity=tx12_params["porosity"],
        tube_radius=tx12_params["tube_radius"],
        area=tx12_params["area"],
        solid_temperature=tx12_params["solid_temperature"],
        heat_in=tx12_params["heat_in"],
        name="TX12_main_water_HX",
    )

    # Construct complex input values
    p1_in_12 = tx12_input["p1_mag"] * np.exp(1j * np.radians(tx12_input["p1_ph"]))
    U1_in_12 = tx12_input["U1_mag"] * np.exp(1j * np.radians(tx12_input["U1_ph"]))

    print("Input state (from segment 11):")
    print(f"  |p1| = {np.abs(p1_in_12):.1f} Pa, Ph = {np.degrees(np.angle(p1_in_12)):.4f} deg")
    print(f"  |U1| = {np.abs(U1_in_12):.6f} m^3/s, Ph = {np.degrees(np.angle(U1_in_12)):.4f} deg")
    print(f"  T_m = {tx12_input['T_m']:.2f} K")
    print()

    # Propagate through TX12
    p1_out_12, U1_out_12, T_out_12 = tx12.propagate(
        p1_in_12, U1_in_12, tx12_input["T_m"], omega, helium
    )

    print("Output state:")
    print(f"  |p1| = {np.abs(p1_out_12):.1f} Pa (Reference baseline: {tx12_ref['p1_mag']:.1f})")
    print(f"  Ph(p1) = {np.degrees(np.angle(p1_out_12)):.4f} deg (Reference baseline: {tx12_ref['p1_ph']:.4f})")
    print(f"  |U1| = {np.abs(U1_out_12):.6f} m^3/s (Reference baseline: {tx12_ref['U1_mag']:.6f})")
    print(f"  Ph(U1) = {np.degrees(np.angle(U1_out_12)):.4f} deg (Reference baseline: {tx12_ref['U1_ph']:.4f})")
    print(f"  T_out = {T_out_12:.2f} K (Reference baseline GasT: {tx12_ref['T_gas']:.2f})")
    print()

    # Calculate errors for TX12
    tx12_errors = {
        "p1_mag": 100 * (np.abs(p1_out_12) - tx12_ref["p1_mag"]) / tx12_ref["p1_mag"],
        "p1_ph": phase_error(np.degrees(np.angle(p1_out_12)), tx12_ref["p1_ph"]),
        "U1_mag": 100 * (np.abs(U1_out_12) - tx12_ref["U1_mag"]) / tx12_ref["U1_mag"],
        "U1_ph": phase_error(np.degrees(np.angle(U1_out_12)), tx12_ref["U1_ph"]),
    }

    print("Errors:")
    print(f"  |p1| error: {tx12_errors['p1_mag']:+.3f}%")
    print(f"  Ph(p1) error: {tx12_errors['p1_ph']:+.3f} deg")
    print(f"  |U1| error: {tx12_errors['U1_mag']:+.3f}%")
    print(f"  Ph(U1) error: {tx12_errors['U1_ph']:+.3f} deg")
    print()

    # =========================================================================
    # Segment 20: TX - Small water heat exchanger
    # =========================================================================
    print("-" * 70)
    print("SEGMENT 20: TX - Small water heat exchanger (pulse tube cold end)")
    print("-" * 70)
    print()

    # TX parameters from tashe1 reference case segment 20
    tx20_params = {
        "area": 6.6580e-3,  # m^2 (parameter a)
        "porosity": 0.2690,  # GasA/A (parameter b)
        "length": 1.0160e-2,  # m (parameter c)
        "tube_radius": 2.2860e-3,  # m (parameter d)
        "heat_in": -2107.2,  # W (parameter e, sameas 19A)
        "solid_temperature": 290.00,  # K (parameter f)
    }

    # Input from segment 18 output (STKDUCT pulse tube)
    tx20_input = {
        "p1_mag": 2.6698e5,  # Pa
        "p1_ph": 1.2829,  # deg
        "U1_mag": 7.4895e-2,  # m^3/s
        "U1_ph": -69.424,  # deg
        "T_m": 325.00,  # K (TEnd from segment 18)
    }

    # Expected output from Reference baseline (segment 20 output)
    tx20_ref = {
        "p1_mag": 2.6590e5,  # Pa (column A)
        "p1_ph": 1.2172,  # deg (column B)
        "U1_mag": 7.5386e-2,  # m^3/s (column C)
        "U1_ph": -69.573,  # deg (column D)
        "T_gas": 325.00,  # K (column G, GasT - mean gas temperature)
        "T_solid_calc": 197.00,  # K (column H, SolidT - calculated effective solid T)
    }

    print("TX20 Parameters:")
    print(f"  Area: {tx20_params['area']*1e4:.4f} cm^2")
    print(f"  Porosity (GasA/A): {tx20_params['porosity']:.4f}")
    print(f"  Length: {tx20_params['length']*1e3:.2f} mm")
    print(f"  Tube radius: {tx20_params['tube_radius']*1e3:.3f} mm")
    print(f"  Solid temperature: {tx20_params['solid_temperature']:.1f} K")
    print()

    # Create TX segment
    tx20 = segments.TubeHeatExchanger(
        length=tx20_params["length"],
        porosity=tx20_params["porosity"],
        tube_radius=tx20_params["tube_radius"],
        area=tx20_params["area"],
        solid_temperature=tx20_params["solid_temperature"],
        heat_in=tx20_params["heat_in"],
        name="TX20_small_water_HX",
    )

    # Construct complex input values
    p1_in_20 = tx20_input["p1_mag"] * np.exp(1j * np.radians(tx20_input["p1_ph"]))
    U1_in_20 = tx20_input["U1_mag"] * np.exp(1j * np.radians(tx20_input["U1_ph"]))

    print("Input state (from segment 18):")
    print(f"  |p1| = {np.abs(p1_in_20):.1f} Pa, Ph = {np.degrees(np.angle(p1_in_20)):.4f} deg")
    print(f"  |U1| = {np.abs(U1_in_20):.6f} m^3/s, Ph = {np.degrees(np.angle(U1_in_20)):.4f} deg")
    print(f"  T_m = {tx20_input['T_m']:.2f} K")
    print()

    # Propagate through TX20
    p1_out_20, U1_out_20, T_out_20 = tx20.propagate(
        p1_in_20, U1_in_20, tx20_input["T_m"], omega, helium
    )

    print("Output state:")
    print(f"  |p1| = {np.abs(p1_out_20):.1f} Pa (Reference baseline: {tx20_ref['p1_mag']:.1f})")
    print(f"  Ph(p1) = {np.degrees(np.angle(p1_out_20)):.4f} deg (Reference baseline: {tx20_ref['p1_ph']:.4f})")
    print(f"  |U1| = {np.abs(U1_out_20):.6f} m^3/s (Reference baseline: {tx20_ref['U1_mag']:.6f})")
    print(f"  Ph(U1) = {np.degrees(np.angle(U1_out_20)):.4f} deg (Reference baseline: {tx20_ref['U1_ph']:.4f})")
    print(f"  T_out = {T_out_20:.2f} K (Reference baseline GasT: {tx20_ref['T_gas']:.2f})")
    print()

    # Calculate errors for TX20
    tx20_errors = {
        "p1_mag": 100 * (np.abs(p1_out_20) - tx20_ref["p1_mag"]) / tx20_ref["p1_mag"],
        "p1_ph": phase_error(np.degrees(np.angle(p1_out_20)), tx20_ref["p1_ph"]),
        "U1_mag": 100 * (np.abs(U1_out_20) - tx20_ref["U1_mag"]) / tx20_ref["U1_mag"],
        "U1_ph": phase_error(np.degrees(np.angle(U1_out_20)), tx20_ref["U1_ph"]),
    }

    print("Errors:")
    print(f"  |p1| error: {tx20_errors['p1_mag']:+.3f}%")
    print(f"  Ph(p1) error: {tx20_errors['p1_ph']:+.3f} deg")
    print(f"  |U1| error: {tx20_errors['U1_mag']:+.3f}%")
    print(f"  Ph(U1) error: {tx20_errors['U1_ph']:+.3f} deg")
    print()

    # =========================================================================
    # Validation Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    # Thresholds
    amp_threshold = 5.0  # % for amplitude
    phase_threshold = 5.0  # degrees for phase

    checks = [
        # TX12 outputs
        ("TX12 |p1|", tx12_errors["p1_mag"], amp_threshold, "%"),
        ("TX12 Ph(p1)", tx12_errors["p1_ph"], phase_threshold, "deg"),
        ("TX12 |U1|", tx12_errors["U1_mag"], amp_threshold, "%"),
        ("TX12 Ph(U1)", tx12_errors["U1_ph"], phase_threshold, "deg"),
        # TX20 outputs
        ("TX20 |p1|", tx20_errors["p1_mag"], amp_threshold, "%"),
        ("TX20 Ph(p1)", tx20_errors["p1_ph"], phase_threshold, "deg"),
        ("TX20 |U1|", tx20_errors["U1_mag"], amp_threshold, "%"),
        ("TX20 Ph(U1)", tx20_errors["U1_ph"], phase_threshold, "deg"),
    ]

    print(f"{'Parameter':<15} {'Error':<15} {'Threshold':<12} {'Status'}")
    print("-" * 50)

    all_pass = True
    for name, error, threshold, unit in checks:
        status = "PASS" if abs(error) < threshold else "FAIL"
        if abs(error) >= threshold:
            all_pass = False
        print(f"{name:<15} {error:+.3f} {unit:<8} {threshold:.1f} {unit:<6} {status}")

    print("-" * 50)

    if all_pass:
        print("\nALL CHECKS PASSED")
        print("TX segment implementation validates against Reference baseline tashe1 reference case")
    else:
        print("\nSome checks failed")
        print("See detailed output above for analysis")

    print()
    print("Notes:")
    print("- TX segments use circular pore (Bessel function) thermoviscous functions")
    print("- Our implementation uses solid temperature for gas property calculations")
    print("- Reference baseline reports GasT as the mean gas temperature (constant through TX)")
    print("- the baseline's SolidT column (H) shows calculated effective solid temperature")
    print("- TX12: Input SolidT=300K, calculated SolidT=293.35K")
    print("- TX20: Input SolidT=290K, calculated SolidT=197.00K")
    print("- Our output T_m equals solid_temperature parameter (conservative)")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

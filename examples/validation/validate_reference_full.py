#!/usr/bin/env python3
"""
Comprehensive Validation Against Embedded Reference Baseline

Tests all segments from the Hofler 1986 thermoacoustic refrigerator example:
- Segment 2: DUCT (ambient temperature)
- Segment 3: HX (ambient heat exchanger)
- Segment 4: STKSLAB (stack with temperature gradient)
- Segment 5: HX (cold heat exchanger)
- Segment 6: DUCT (cold temperature)
- Segment 7: CONE (tapered section)
- Segment 8: COMPLIANCE (end bulb)

Embedded reference: Hofler1 reference case
"""

import numpy as np
from openthermoacoustics import gas, segments, geometry
from openthermoacoustics.utils import acoustic_power


def compare_results(name, reference_baseline, openthermo, tolerance_pct=5.0):
    """Compare results and return pass/fail status."""
    results = []
    max_err = 0.0

    for param, (dec_val, ota_val) in reference_baseline.items():
        if "ph" in param.lower() or "phase" in param.lower():
            # Phase comparison (absolute degrees)
            err = abs(ota_val - dec_val)
            err_str = f"{err:+.3f}°"
            passed = err < 1.0  # 1 degree tolerance for phase
        else:
            # Magnitude comparison (percentage)
            if abs(dec_val) > 1e-10:
                err = 100 * (ota_val - dec_val) / dec_val
                err_str = f"{err:+.2f}%"
                passed = abs(err) < tolerance_pct
                max_err = max(max_err, abs(err))
            else:
                err = ota_val - dec_val
                err_str = f"{err:+.4f}"
                passed = abs(err) < 0.01

        status = "✓" if passed else "✗"
        results.append((param, dec_val, ota_val, err_str, status))

    return results, max_err


def print_comparison(name, results, max_err):
    """Print formatted comparison results."""
    print(f"\n{'='*70}")
    print(f"SEGMENT: {name}")
    print(f"{'='*70}")
    print(f"{'Parameter':<20} {'Reference baseline':<15} {'OpenThermo':<15} {'Error':<12} {'Status'}")
    print("-" * 70)

    all_passed = True
    for param, dec_val, ota_val, err_str, status in results:
        if status == "✗":
            all_passed = False
        print(f"{param:<20} {dec_val:<15.4g} {ota_val:<15.4g} {err_str:<12} {status}")

    print("-" * 70)
    if all_passed:
        print(f"✓ PASSED (max amplitude error: {max_err:.2f}%)")
    else:
        print(f"✗ NEEDS REVIEW (max amplitude error: {max_err:.2f}%)")

    return all_passed


def main():
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION: OpenThermoacoustics vs embedded reference baseline")
    print("Model: Hofler 1986 Thermoacoustic Refrigerator (Full System)")
    print("=" * 70)

    # Common setup
    helium = gas.Helium(mean_pressure=1.0e6)
    freq = 500.0
    omega = 2 * np.pi * freq

    all_passed = []

    # =========================================================================
    # Segment 2: DUCT (ambient temperature)
    # =========================================================================
    # Input from segment 1 (SURFACE)
    p1_in = 30000.0 * np.exp(1j * np.radians(0.0))
    U1_in = 4.9767e-4 * np.exp(1j * np.radians(-0.26946))
    T_in = 300.0

    # Reference baseline output
    dec_out = {
        "|p1|": (29739.0, None),
        "ph(p1)": (-0.17763, None),
        "|U1|": (2.7789e-3, None),
        "ph(U1)": (-79.991, None),
    }

    # Our calculation
    area = 1.134e-3
    perim = 0.1190
    length = 4.26e-2
    radius = np.sqrt(area / np.pi)

    duct = segments.Duct(length=length, radius=radius, geometry=geometry.CircularPore())
    p1_out, U1_out, T_out = duct.propagate(p1_in, U1_in, T_in, omega, helium)

    dec_out["|p1|"] = (29739.0, np.abs(p1_out))
    dec_out["ph(p1)"] = (-0.17763, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (2.7789e-3, np.abs(U1_out))
    dec_out["ph(U1)"] = (-79.991, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("DUCT (ambient)", dec_out, None)
    passed = print_comparison("2: DUCT (ambient temperature)", results, max_err)
    all_passed.append(("DUCT (ambient)", passed, max_err))

    # =========================================================================
    # Segment 3: HX (ambient heat exchanger)
    # =========================================================================
    # Input from segment 2
    p1_in = 29739.0 * np.exp(1j * np.radians(-0.17763))
    U1_in = 2.7789e-3 * np.exp(1j * np.radians(-79.991))
    T_in = 300.0

    # Reference baseline output
    dec_out = {
        "|p1|": (29570.0, None),
        "ph(p1)": (-0.12971, None),
        "|U1|": (3.0568e-3, None),
        "ph(U1)": (-81.875, None),
    }

    # HX parameters
    hx_area = 1.134e-3  # m² (sameas stack)
    hx_porosity = 0.6000
    hx_length = 6.35e-3
    hx_y0 = 1.9e-4

    hx = segments.HeatExchanger(
        length=hx_length,
        porosity=hx_porosity,
        hydraulic_radius=hx_y0,
        area=hx_area,
        geometry=geometry.ParallelPlate(),
        temperature=300.0,
    )
    p1_out, U1_out, T_out = hx.propagate(p1_in, U1_in, T_in, omega, helium)

    dec_out["|p1|"] = (29570.0, np.abs(p1_out))
    dec_out["ph(p1)"] = (-0.12971, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (3.0568e-3, np.abs(U1_out))
    dec_out["ph(U1)"] = (-81.875, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("HX (ambient)", dec_out, None)
    passed = print_comparison("3: HX (ambient heat exchanger)", results, max_err)
    all_passed.append(("HX (ambient)", passed, max_err))

    # =========================================================================
    # Segment 4: STKSLAB (Stack with temperature gradient)
    # =========================================================================
    # Input from segment 3
    p1_in = 29570.0 * np.exp(1j * np.radians(-0.12971))
    U1_in = 3.0568e-3 * np.exp(1j * np.radians(-81.875))
    T_in = 300.0
    T_out_target = 217.03

    # Reference baseline output
    dec_out = {
        "|p1|": (26103.0, None),
        "ph(p1)": (1.4277, None),
        "|U1|": (6.8004e-3, None),
        "ph(U1)": (-87.965, None),
    }

    # Stack parameters
    stack_area = 1.134e-3
    stack_porosity = 0.7240
    stack_length = 7.85e-2
    stack_y0 = 1.8e-4

    stack = segments.Stack(
        length=stack_length,
        porosity=stack_porosity,
        hydraulic_radius=stack_y0,
        area=stack_area,
        geometry=geometry.ParallelPlate(),
        T_cold=T_in,
        T_hot=T_out_target,
    )
    p1_out, U1_out, T_out = stack.propagate(p1_in, U1_in, T_in, omega, helium)

    dec_out["|p1|"] = (26103.0, np.abs(p1_out))
    dec_out["ph(p1)"] = (1.4277, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (6.8004e-3, np.abs(U1_out))
    dec_out["ph(U1)"] = (-87.965, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("STKSLAB", dec_out, None)
    passed = print_comparison("4: STKSLAB (Stack)", results, max_err)
    all_passed.append(("STKSLAB", passed, max_err))

    # =========================================================================
    # Segment 5: HX (cold heat exchanger)
    # =========================================================================
    # Input from segment 4
    p1_in = 26103.0 * np.exp(1j * np.radians(1.4277))
    U1_in = 6.8004e-3 * np.exp(1j * np.radians(-87.965))
    T_in = 217.03

    # Reference baseline output
    dec_out = {
        "|p1|": (25923.0, None),
        "ph(p1)": (1.4848, None),
        "|U1|": (6.9051e-3, None),
        "ph(U1)": (-88.059, None),
    }

    # Cold HX parameters
    cold_hx_area = 1.134e-3  # m² (sameas stack)
    cold_hx_porosity = 0.6700
    cold_hx_length = 2.54e-3
    cold_hx_y0 = 2.55e-4

    cold_hx = segments.HeatExchanger(
        length=cold_hx_length,
        porosity=cold_hx_porosity,
        hydraulic_radius=cold_hx_y0,
        area=cold_hx_area,
        geometry=geometry.ParallelPlate(),
        temperature=217.03,  # Gas temperature at cold side
    )
    p1_out, U1_out, T_out = cold_hx.propagate(p1_in, U1_in, T_in, omega, helium)

    dec_out["|p1|"] = (25923.0, np.abs(p1_out))
    dec_out["ph(p1)"] = (1.4848, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (6.9051e-3, np.abs(U1_out))
    dec_out["ph(U1)"] = (-88.059, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("HX (cold)", dec_out, None)
    passed = print_comparison("5: HX (cold heat exchanger)", results, max_err)
    all_passed.append(("HX (cold)", passed, max_err))

    # =========================================================================
    # Segment 6: DUCT (cold temperature)
    # =========================================================================
    # Input from segment 5
    p1_in = 25923.0 * np.exp(1j * np.radians(1.4848))
    U1_in = 6.9051e-3 * np.exp(1j * np.radians(-88.059))
    T_in = 217.03  # Cold temperature

    # Reference baseline output
    dec_out = {
        "|p1|": (1489.2, None),
        "ph(p1)": (1.5242, None),
        "|U1|": (8.6235e-3, None),
        "ph(U1)": (-88.211, None),
    }

    # Cold duct parameters
    cold_duct_area = 3.84e-4
    cold_duct_perim = 6.94e-2
    cold_duct_length = 0.1670
    cold_duct_radius = np.sqrt(cold_duct_area / np.pi)

    cold_duct = segments.Duct(
        length=cold_duct_length,
        radius=cold_duct_radius,
        geometry=geometry.CircularPore()
    )
    p1_out, U1_out, T_out = cold_duct.propagate(p1_in, U1_in, T_in, omega, helium)

    dec_out["|p1|"] = (1489.2, np.abs(p1_out))
    dec_out["ph(p1)"] = (1.5242, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (8.6235e-3, np.abs(U1_out))
    dec_out["ph(U1)"] = (-88.211, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("DUCT (cold)", dec_out, None)
    passed = print_comparison("6: DUCT (cold temperature)", results, max_err)
    all_passed.append(("DUCT (cold)", passed, max_err))

    # =========================================================================
    # Segment 7: CONE (tapered section)
    # =========================================================================
    # Input from segment 6
    p1_in = 1489.2 * np.exp(1j * np.radians(1.5242))
    U1_in = 8.6235e-3 * np.exp(1j * np.radians(-88.211))
    T_in = 217.03

    # Reference baseline output
    dec_out = {
        "|p1|": (4526.5, None),
        "ph(p1)": (-178.49, None),
        "|U1|": (8.3826e-3, None),
        "ph(U1)": (-88.196, None),
    }

    # Cone parameters
    cone_area_in = 3.84e-4
    cone_area_out = 1.16e-3
    cone_length = 6.68e-2

    cone = segments.Cone(
        length=cone_length,
        radius_in=np.sqrt(cone_area_in / np.pi),
        radius_out=np.sqrt(cone_area_out / np.pi),
        geometry=geometry.CircularPore(),
    )
    p1_out, U1_out, T_out = cone.propagate(p1_in, U1_in, T_in, omega, helium)

    # Handle phase wrapping for comparison
    ph_p_out = np.degrees(np.angle(p1_out))
    # Reference baseline shows -178.49, which is equivalent to 181.51
    if ph_p_out > 0 and dec_out["ph(p1)"][0] < -90:
        ph_p_out = ph_p_out - 360

    dec_out["|p1|"] = (4526.5, np.abs(p1_out))
    dec_out["ph(p1)"] = (-178.49, ph_p_out)
    dec_out["|U1|"] = (8.3826e-3, np.abs(U1_out))
    dec_out["ph(U1)"] = (-88.196, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("CONE", dec_out, None)
    passed = print_comparison("7: CONE (tapered section)", results, max_err)
    all_passed.append(("CONE", passed, max_err))

    # =========================================================================
    # Segment 8: COMPLIANCE (end bulb)
    # =========================================================================
    # Input from segment 7
    p1_in = 4526.5 * np.exp(1j * np.radians(-178.49))
    U1_in = 8.3826e-3 * np.exp(1j * np.radians(-88.196))
    T_in = 217.03

    # Reference baseline output - compliance should have same pressure, different velocity
    dec_out = {
        "|p1|": (4526.5, None),
        "ph(p1)": (-178.49, None),
        "|U1|": (6.7549e-4, None),
        "ph(U1)": (86.866, None),
    }

    # Compliance parameters
    compliance_volume = 1.06e-3  # m³

    compliance = segments.Compliance(
        volume=compliance_volume,
    )
    p1_out, U1_out, T_out = compliance.propagate(p1_in, U1_in, T_in, omega, helium)

    dec_out["|p1|"] = (4526.5, np.abs(p1_out))
    dec_out["ph(p1)"] = (-178.49, np.degrees(np.angle(p1_out)))
    dec_out["|U1|"] = (6.7549e-4, np.abs(U1_out))
    dec_out["ph(U1)"] = (86.866, np.degrees(np.angle(U1_out)))

    results, max_err = compare_results("COMPLIANCE", dec_out, None)
    passed = print_comparison("8: COMPLIANCE (end bulb)", results, max_err)
    all_passed.append(("COMPLIANCE", passed, max_err))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Segment':<25} {'Status':<10} {'Max Error'}")
    print("-" * 70)

    total_passed = 0
    for name, passed, max_err in all_passed:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:<25} {status:<10} {max_err:.2f}%")
        if passed:
            total_passed += 1

    print("-" * 70)
    print(f"Total: {total_passed}/{len(all_passed)} segments passed validation")
    print()

    if total_passed == len(all_passed):
        print("✓ ALL SEGMENTS VALIDATED SUCCESSFULLY")
    else:
        print(f"✗ {len(all_passed) - total_passed} segment(s) need review")


if __name__ == "__main__":
    main()

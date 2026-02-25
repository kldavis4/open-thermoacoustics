#!/usr/bin/env python3
"""
Validation of ANCHOR and INSULATE thermal mode control segments.

This validation tests:
1. Basic instantiation
2. Acoustic state passthrough (no modification)
3. ThermalMode enum values
4. Usage in a segment chain
5. Mode toggling with ANCHOR/INSULATE sequence
"""

import numpy as np

from openthermoacoustics import gas, segments
from openthermoacoustics.segments.anchor import ThermalMode


def main():
    print("=" * 70)
    print("VALIDATION: ANCHOR and INSULATE thermal mode control segments")
    print("=" * 70)
    print()

    # =========================================================================
    # Test 1: Basic instantiation
    # =========================================================================
    print("TEST 1: Basic instantiation")
    print("-" * 50)

    anchor = segments.Anchor(name="water_jacket")
    insulate = segments.Insulate(name="end_jacket")

    print(f"ANCHOR segment: {anchor}")
    print(f"  thermal_mode = {anchor.thermal_mode}")
    print(f"  length = {anchor.length}")

    print(f"INSULATE segment: {insulate}")
    print(f"  thermal_mode = {insulate.thermal_mode}")
    print(f"  length = {insulate.length}")

    test1_pass = (
        anchor.thermal_mode == ThermalMode.ANCHORED
        and insulate.thermal_mode == ThermalMode.INSULATED
        and anchor.length == 0.0
        and insulate.length == 0.0
    )
    print()
    print(f"Test 1: {'PASS' if test1_pass else 'FAIL'} (basic instantiation)")
    print()

    # =========================================================================
    # Test 2: Acoustic state passthrough
    # =========================================================================
    print("TEST 2: Acoustic state passthrough")
    print("-" * 50)

    # Use helium at 3 MPa
    helium = gas.Helium(mean_pressure=3e6)
    T_m = 300.0
    omega = 2 * np.pi * 50  # 50 Hz

    # Input acoustic state
    p1_in = 50000.0 + 10000.0j  # Pa
    U1_in = 5e-5 + 1e-5j  # m³/s

    # Propagate through ANCHOR
    p1_anchor, U1_anchor, T_anchor = anchor.propagate(p1_in, U1_in, T_m, omega, helium)

    print("ANCHOR passthrough:")
    print(f"  p1_in = {p1_in}")
    print(f"  p1_out = {p1_anchor}")
    print(f"  U1_in = {U1_in}")
    print(f"  U1_out = {U1_anchor}")
    print(f"  T_m_in = {T_m}")
    print(f"  T_m_out = {T_anchor}")

    # Check passthrough
    anchor_pass = (
        p1_anchor == p1_in and U1_anchor == U1_in and T_anchor == T_m
    )

    # Propagate through INSULATE
    p1_insulate, U1_insulate, T_insulate = insulate.propagate(
        p1_in, U1_in, T_m, omega, helium
    )

    print("INSULATE passthrough:")
    print(f"  p1_in = {p1_in}")
    print(f"  p1_out = {p1_insulate}")
    print(f"  U1_in = {U1_in}")
    print(f"  U1_out = {U1_insulate}")

    insulate_pass = (
        p1_insulate == p1_in and U1_insulate == U1_in and T_insulate == T_m
    )

    test2_pass = anchor_pass and insulate_pass
    print()
    print(f"Test 2: {'PASS' if test2_pass else 'FAIL'} (acoustic state unchanged)")
    print()

    # =========================================================================
    # Test 3: ThermalMode enum values
    # =========================================================================
    print("TEST 3: ThermalMode enum values")
    print("-" * 50)

    print(f"ThermalMode.INSULATED = {ThermalMode.INSULATED}")
    print(f"ThermalMode.INSULATED.value = '{ThermalMode.INSULATED.value}'")
    print(f"ThermalMode.ANCHORED = {ThermalMode.ANCHORED}")
    print(f"ThermalMode.ANCHORED.value = '{ThermalMode.ANCHORED.value}'")

    test3_pass = (
        ThermalMode.INSULATED.value == "insulated"
        and ThermalMode.ANCHORED.value == "anchored"
        and ThermalMode.INSULATED != ThermalMode.ANCHORED
    )
    print()
    print(f"Test 3: {'PASS' if test3_pass else 'FAIL'} (enum values)")
    print()

    # =========================================================================
    # Test 4: Usage in a segment chain
    # =========================================================================
    print("TEST 4: Usage in a segment chain")
    print("-" * 50)

    # Create a typical chain: DUCT -> ANCHOR -> DUCT -> INSULATE -> DUCT
    radius = np.sqrt(1e-4 / np.pi)  # radius for 1 cm² area
    duct1 = segments.Duct(length=0.1, radius=radius, name="duct1")
    anchor_seg = segments.Anchor(name="start_anchor")
    duct2 = segments.Duct(length=0.1, radius=radius, name="duct2_anchored")
    insulate_seg = segments.Insulate(name="end_anchor")
    duct3 = segments.Duct(length=0.1, radius=radius, name="duct3")

    # Propagate through chain
    p1, U1, T = p1_in, U1_in, T_m
    chain_segments = [duct1, anchor_seg, duct2, insulate_seg, duct3]
    modes_in_chain = []

    print("Segment chain propagation:")
    for seg in chain_segments:
        p1, U1, T = seg.propagate(p1, U1, T, omega, helium)
        if hasattr(seg, "thermal_mode"):
            modes_in_chain.append(seg.thermal_mode)
            print(f"  {seg._name}: mode = {seg.thermal_mode}")
        else:
            print(f"  {seg._name}: |p1| = {np.abs(p1):.2f} Pa")

    # Check that pressure dropped through ducts
    pressure_drop = np.abs(p1_in) - np.abs(p1)

    test4_pass = (
        pressure_drop > 0
        and len(modes_in_chain) == 2
        and modes_in_chain[0] == ThermalMode.ANCHORED
        and modes_in_chain[1] == ThermalMode.INSULATED
    )
    print(f"  Total pressure drop: {pressure_drop:.2f} Pa")
    print()
    print(f"Test 4: {'PASS' if test4_pass else 'FAIL'} (segment chain usage)")
    print()

    # =========================================================================
    # Test 5: Mode toggling sequence
    # =========================================================================
    print("TEST 5: Mode toggling sequence")
    print("-" * 50)

    # Create multiple ANCHOR/INSULATE segments
    seg1 = segments.ANCHOR(name="anchor1")
    seg2 = segments.INSULATE(name="insulate1")
    seg3 = segments.ANCHOR(name="anchor2")

    modes = [seg1.thermal_mode, seg2.thermal_mode, seg3.thermal_mode]
    expected = [ThermalMode.ANCHORED, ThermalMode.INSULATED, ThermalMode.ANCHORED]

    print("Mode sequence:")
    for i, (seg, mode) in enumerate(zip([seg1, seg2, seg3], modes)):
        print(f"  {seg._name}: {mode}")

    test5_pass = modes == expected
    print()
    print(f"Test 5: {'PASS' if test5_pass else 'FAIL'} (mode toggling)")
    print()

    # =========================================================================
    # Test 6: Reference baseline aliases
    # =========================================================================
    print("TEST 6: Reference baseline aliases")
    print("-" * 50)

    anchor_alias = segments.ANCHOR(name="alias_test")
    insulate_alias = segments.INSULATE(name="alias_test2")

    print(f"segments.ANCHOR is segments.Anchor: {segments.ANCHOR is segments.Anchor}")
    print(f"segments.INSULATE is segments.Insulate: {segments.INSULATE is segments.Insulate}")

    test6_pass = (
        segments.ANCHOR is segments.Anchor
        and segments.INSULATE is segments.Insulate
        and isinstance(anchor_alias, segments.Anchor)
        and isinstance(insulate_alias, segments.Insulate)
    )
    print()
    print(f"Test 6: {'PASS' if test6_pass else 'FAIL'} (Reference baseline aliases)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = (
        test1_pass and test2_pass and test3_pass
        and test4_pass and test5_pass and test6_pass
    )

    print(f"Test 1 (Basic instantiation):     {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Acoustic passthrough):    {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (ThermalMode enum):        {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (Segment chain usage):     {'PASS' if test4_pass else 'FAIL'}")
    print(f"Test 5 (Mode toggling):           {'PASS' if test5_pass else 'FAIL'}")
    print(f"Test 6 (Reference baseline aliases):         {'PASS' if test6_pass else 'FAIL'}")
    print()

    if all_pass:
        print("ANCHOR/INSULATE VALIDATION PASSED")
        print()
        print("Notes:")
        print("- ANCHOR sets ThermalMode.ANCHORED for subsequent segments")
        print("- INSULATE returns to default ThermalMode.INSULATED")
        print("- Acoustic state (p1, U1, T_m) passes through unchanged")
        print("- These segments control energy accounting mode (dH_tot/dx behavior)")
        print("- In ANCHORED mode: dH_tot/dx = dE/dx (heat removed locally)")
        print("- In INSULATED mode: dH_tot/dx = 0 (default)")
    else:
        print("Some tests FAILED")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

#!/usr/bin/env python3
"""
Investigate STKCIRC phase error.

Compare:
1. Circular vs parallel-plate geometry with same conditions
2. Engine mode (dT/dx < 0) vs refrigerator mode (dT/dx > 0)
3. Isothermal case (no temperature gradient)
"""

import numpy as np
from openthermoacoustics import gas
from openthermoacoustics.segments import Stack
from openthermoacoustics.geometry.circular import CircularPore
from openthermoacoustics.geometry.parallel_plate import ParallelPlate


def run_test(name, stack, p1_in, U1_in, T_in, omega, gas_obj):
    """Run a propagation test and return results."""
    p1_out, U1_out, T_out = stack.propagate(p1_in, U1_in, T_in, omega, gas_obj)

    Edot_in = 0.5 * np.real(p1_in * np.conj(U1_in))
    Edot_out = 0.5 * np.real(p1_out * np.conj(U1_out))

    return {
        "name": name,
        "p1_mag": np.abs(p1_out),
        "p1_phase": np.degrees(np.angle(p1_out)),
        "U1_mag": np.abs(U1_out),
        "U1_phase": np.degrees(np.angle(U1_out)),
        "T_out": T_out,
        "Edot_in": Edot_in,
        "Edot_out": Edot_out,
        "dEdot": Edot_out - Edot_in,
    }


def print_result(r):
    """Print test result."""
    print(f"  {r['name']}:")
    print(f"    |p1| = {r['p1_mag']:.4e} Pa, Ph(p1) = {r['p1_phase']:+.2f}°")
    print(f"    |U1| = {r['U1_mag']:.4e} m³/s, Ph(U1) = {r['U1_phase']:+.2f}°")
    print(f"    T_out = {r['T_out']:.1f} K")
    print(f"    Edot: {r['Edot_in']:.1f} W → {r['Edot_out']:.1f} W (ΔE = {r['dEdot']:+.1f} W)")
    print()


def main():
    print("=" * 70)
    print("INVESTIGATING STKCIRC PHASE ERROR")
    print("=" * 70)
    print()

    # Use 5inch reference case conditions
    mean_P = 1.3800e6  # Pa
    freq = 121.15  # Hz
    omega = 2 * np.pi * freq
    helium = gas.Helium(mean_pressure=mean_P)

    # Common stack parameters (from 5inch STKCIRC)
    length = 0.2790
    porosity = 0.8100
    area = 1.2920e-2
    r_h = 5.0000e-4  # hydraulic radius

    # Input state (from segment 3 HX output)
    p1_in = 7.1468e4 * np.exp(1j * np.radians(0.39052))
    U1_in = 9.6743e-2 * np.exp(1j * np.radians(-91.205))

    # Temperature conditions
    T_hot = 556.89  # K
    T_cold = 306.34  # K

    # Geometries
    circular = CircularPore()
    parallel = ParallelPlate()

    print("=" * 70)
    print("TEST 1: ISOTHERMAL (no temperature gradient)")
    print("=" * 70)
    print()

    # Isothermal at T_hot
    stack_circ_iso = Stack(
        length=length, porosity=porosity, hydraulic_radius=r_h,
        area=area, geometry=circular, name="circular_isothermal"
    )
    stack_para_iso = Stack(
        length=length, porosity=porosity, hydraulic_radius=r_h,
        area=area, geometry=parallel, name="parallel_isothermal"
    )

    r1 = run_test("Circular pore (isothermal)", stack_circ_iso, p1_in, U1_in, T_hot, omega, helium)
    r2 = run_test("Parallel plate (isothermal)", stack_para_iso, p1_in, U1_in, T_hot, omega, helium)

    print_result(r1)
    print_result(r2)

    print(f"  Phase difference (circular - parallel):")
    print(f"    ΔPh(p1) = {r1['p1_phase'] - r2['p1_phase']:+.4f}°")
    print(f"    ΔPh(U1) = {r1['U1_phase'] - r2['U1_phase']:+.4f}°")
    print()

    print("=" * 70)
    print("TEST 2: ENGINE MODE (T decreasing: hot→cold, dT/dx < 0)")
    print("=" * 70)
    print()

    # Engine mode: hot at input, cold at output
    # Our convention: T_cold at x=0, T_hot at x=L
    # So we set T_cold=T_hot, T_hot=T_cold to get dT/dx < 0
    stack_circ_eng = Stack(
        length=length, porosity=porosity, hydraulic_radius=r_h,
        area=area, geometry=circular,
        T_cold=T_hot, T_hot=T_cold,  # Hot at input (x=0)
        name="circular_engine"
    )
    stack_para_eng = Stack(
        length=length, porosity=porosity, hydraulic_radius=r_h,
        area=area, geometry=parallel,
        T_cold=T_hot, T_hot=T_cold,  # Hot at input (x=0)
        name="parallel_engine"
    )

    print(f"  Temperature gradient: dT/dx = {(T_cold - T_hot)/length:.1f} K/m")
    print()

    r3 = run_test("Circular pore (engine)", stack_circ_eng, p1_in, U1_in, T_hot, omega, helium)
    r4 = run_test("Parallel plate (engine)", stack_para_eng, p1_in, U1_in, T_hot, omega, helium)

    print_result(r3)
    print_result(r4)

    print(f"  Phase difference (circular - parallel):")
    print(f"    ΔPh(p1) = {r3['p1_phase'] - r4['p1_phase']:+.4f}°")
    print(f"    ΔPh(U1) = {r3['U1_phase'] - r4['U1_phase']:+.4f}°")
    print()

    print("=" * 70)
    print("TEST 3: REFRIGERATOR MODE (T increasing: cold→hot, dT/dx > 0)")
    print("=" * 70)
    print()

    # Refrigerator mode: cold at input, hot at output
    # Our convention: T_cold at x=0, T_hot at x=L
    # This is the natural convention, dT/dx > 0
    stack_circ_ref = Stack(
        length=length, porosity=porosity, hydraulic_radius=r_h,
        area=area, geometry=circular,
        T_cold=T_cold, T_hot=T_hot,  # Cold at input (x=0)
        name="circular_refrig"
    )
    stack_para_ref = Stack(
        length=length, porosity=porosity, hydraulic_radius=r_h,
        area=area, geometry=parallel,
        T_cold=T_cold, T_hot=T_hot,  # Cold at input (x=0)
        name="parallel_refrig"
    )

    # For refrigerator, input at cold end
    T_in_ref = T_cold

    print(f"  Temperature gradient: dT/dx = {(T_hot - T_cold)/length:.1f} K/m")
    print()

    r5 = run_test("Circular pore (refrig)", stack_circ_ref, p1_in, U1_in, T_in_ref, omega, helium)
    r6 = run_test("Parallel plate (refrig)", stack_para_ref, p1_in, U1_in, T_in_ref, omega, helium)

    print_result(r5)
    print_result(r6)

    print(f"  Phase difference (circular - parallel):")
    print(f"    ΔPh(p1) = {r5['p1_phase'] - r6['p1_phase']:+.4f}°")
    print(f"    ΔPh(U1) = {r5['U1_phase'] - r6['U1_phase']:+.4f}°")
    print()

    print("=" * 70)
    print("TEST 4: Compare f_nu and f_kappa at operating point")
    print("=" * 70)
    print()

    # Calculate thermoviscous functions
    from openthermoacoustics.utils import penetration_depth_viscous, penetration_depth_thermal

    rho = helium.density(T_hot)
    mu = helium.viscosity(T_hot)
    k = helium.thermal_conductivity(T_hot)
    cp = helium.specific_heat_cp(T_hot)

    delta_nu = penetration_depth_viscous(omega, rho, mu)
    delta_kappa = penetration_depth_thermal(omega, rho, k, cp)

    f_nu_circ = circular.f_nu(omega, delta_nu, r_h)
    f_kappa_circ = circular.f_kappa(omega, delta_kappa, r_h)

    f_nu_para = parallel.f_nu(omega, delta_nu, r_h)
    f_kappa_para = parallel.f_kappa(omega, delta_kappa, r_h)

    print(f"  At T = {T_hot} K:")
    print(f"    δ_ν = {delta_nu:.4e} m, δ_κ = {delta_kappa:.4e} m")
    print(f"    r_h/δ_ν = {r_h/delta_nu:.2f}, r_h/δ_κ = {r_h/delta_kappa:.2f}")
    print()
    print(f"  Circular pore:")
    print(f"    f_ν = {f_nu_circ:.4f} = {np.abs(f_nu_circ):.4f} ∠ {np.degrees(np.angle(f_nu_circ)):+.2f}°")
    print(f"    f_κ = {f_kappa_circ:.4f} = {np.abs(f_kappa_circ):.4f} ∠ {np.degrees(np.angle(f_kappa_circ)):+.2f}°")
    print()
    print(f"  Parallel plate:")
    print(f"    f_ν = {f_nu_para:.4f} = {np.abs(f_nu_para):.4f} ∠ {np.degrees(np.angle(f_nu_para)):+.2f}°")
    print(f"    f_κ = {f_kappa_para:.4f} = {np.abs(f_kappa_para):.4f} ∠ {np.degrees(np.angle(f_kappa_para)):+.2f}°")
    print()

    # The temperature gradient term coefficient
    sigma = helium.prandtl(T_hot)

    def gradient_coeff(f_nu, f_kappa):
        return (f_kappa - f_nu) / ((1 - f_nu) * (1 - sigma))

    coeff_circ = gradient_coeff(f_nu_circ, f_kappa_circ)
    coeff_para = gradient_coeff(f_nu_para, f_kappa_para)

    print(f"  Temperature gradient term coefficient (f_κ - f_ν)/((1-f_ν)(1-σ)):")
    print(f"    Circular: {coeff_circ:.4f} = {np.abs(coeff_circ):.4f} ∠ {np.degrees(np.angle(coeff_circ)):+.2f}°")
    print(f"    Parallel: {coeff_para:.4f} = {np.abs(coeff_para):.4f} ∠ {np.degrees(np.angle(coeff_para)):+.2f}°")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # Compare isothermal vs engine mode for circular
    print("Effect of temperature gradient (circular pore):")
    print(f"  Isothermal → Engine mode:")
    print(f"    ΔPh(p1) = {r3['p1_phase'] - r1['p1_phase']:+.2f}°")
    print(f"    ΔPh(U1) = {r3['U1_phase'] - r1['U1_phase']:+.2f}°")
    print()

    # Compare isothermal vs engine mode for parallel
    print("Effect of temperature gradient (parallel plate):")
    print(f"  Isothermal → Engine mode:")
    print(f"    ΔPh(p1) = {r4['p1_phase'] - r2['p1_phase']:+.2f}°")
    print(f"    ΔPh(U1) = {r4['U1_phase'] - r2['U1_phase']:+.2f}°")
    print()

    # Embedded reference
    print("Embedded reference (STKCIRC in 5inch reference case):")
    print(f"  Ph(p1) = +2.76°, Ph(U1) = -85.36°")
    print()
    print("Our circular pore engine mode:")
    print(f"  Ph(p1) = {r3['p1_phase']:+.2f}°, Ph(U1) = {r3['U1_phase']:+.2f}°")
    print()
    print("Phase errors:")
    print(f"  ΔPh(p1) = {r3['p1_phase'] - 2.76:+.2f}°")
    print(f"  ΔPh(U1) = {r3['U1_phase'] - (-85.36):+.2f}°")


if __name__ == "__main__":
    main()

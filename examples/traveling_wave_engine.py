#!/usr/bin/env python3
"""
Traveling-Wave Thermoacoustic Engine Example
Based on Backhaus-Swift TASHE (JASA 2000)

This example implements a simplified model of the Backhaus-Swift Thermoacoustic
Stirling Heat Engine (TASHE), demonstrating the key physics of traveling-wave
thermoacoustic engines and how they achieve high efficiency.

=============================================================================
WHY TRAVELING-WAVE ENGINES ACHIEVE CARNOT-LIKE EFFICIENCY
=============================================================================

The fundamental insight is the pressure-velocity phase relationship:

1. STANDING-WAVE ENGINES:
   - Pressure and velocity oscillations are 90 degrees out of phase
   - Gas parcels experience pressure changes while stationary (at velocity nodes)
   - Heat transfer is inherently irreversible (finite-time, finite-deltaT)
   - The stack operates with "imperfect thermal contact"
   - Typical efficiency: 10-30% of Carnot

2. TRAVELING-WAVE ENGINES:
   - Pressure and velocity oscillations are IN PHASE
   - Gas parcels experience compression while moving toward hot end
   - Gas parcels experience expansion while moving toward cold end
   - This mimics the Stirling cycle with near-isothermal compression/expansion
   - The regenerator can operate with "perfect thermal contact"
   - Typical efficiency: 40-60% of Carnot (Backhaus & Swift achieved ~42%)

The key equation for acoustic power production in a regenerator:

    dE2/dx = (beta * Im(f_kappa - f_nu)) / (|1-f_nu|^2 * omega * (1-Pr))
             * (dT_m/dx) * Re(p1 * conj(U1)) / (rho_m * A)

For maximum power production, we want Re(p1 * conj(U1)) to be LARGE and POSITIVE.
This happens when p1 and U1 are IN PHASE (traveling wave condition).

=============================================================================
THE ROLE OF THE REGENERATOR: STANDING-WAVE VS TRAVELING-WAVE
=============================================================================

In a STANDING-WAVE device, the porous element is called a "STACK":
- Hydraulic radius r_h ~ 1-5 * delta_kappa (thermal penetration depth)
- Imperfect thermal contact with the walls
- Relies on hysteresis in the p-V diagram for work extraction
- Position: placed at velocity antinode for max acoustic power

In a TRAVELING-WAVE device, the porous element is called a "REGENERATOR":
- Hydraulic radius r_h << delta_kappa (much finer mesh)
- Near-perfect thermal contact: gas temperature follows local solid temp
- Acts like a heat exchanger stack - gas enters at one temp, leaves at another
- The traveling-wave phasing allows quasi-isothermal expansion/compression
- Position: anywhere in the loop with proper acoustic impedance

Key parameter ratios:
- Standing-wave stack: r_h/delta_kappa ~ 1-4
- Traveling-wave regenerator: r_h/delta_kappa ~ 0.05-0.3

=============================================================================
HOW THE FEEDBACK LOOP CREATES THE TRAVELING WAVE
=============================================================================

The Backhaus-Swift design uses a toroidal feedback loop:

                Resonator stub (provides compliance, sets frequency)
                       |
           [TEE JUNCTION] <-- Junction compliance
                /            \\
               /              \\
   Main path --|              |-- Feedback inertance tube
               |              |
           Cold HX            |
               |              |
         Regenerator          |
               |              |
            Hot HX            |
               |              |
        Thermal Buffer        |
               |              |
       Secondary Cold HX      |
               |              |
               +------<-------+
                 (loop closure)

How it works:
1. The resonator stub is a quarter-wave section that sets the frequency
   and provides acoustic compliance (pressure antinode at closed end)

2. At the junction, pressure is the same for all branches (p1_junction)
   but velocity splits between the resonator and the feedback loop

3. The feedback inertance (narrow tube) provides an inductive impedance:
   - Delays the velocity relative to pressure
   - Creates a phase shift approaching -90 degrees

4. The compliance at the junction provides a capacitive effect:
   - Velocity leads pressure by up to +90 degrees

5. When properly tuned, the combination produces:
   - Traveling-wave phasing (p1 and U1 nearly in phase) in the regenerator
   - The right acoustic impedance for efficient power production

The acoustic power flows:
- Generated in the regenerator (thermal-to-acoustic conversion)
- Most loops back through the feedback inertance to sustain oscillations
- Some flows to a load (acoustic-to-mechanical converter) or is dissipated

=============================================================================
ACOUSTIC POWER FLOW THROUGH THE SYSTEM
=============================================================================

Energy conservation in steady state requires:

    Q_hot = W_acoustic_net + Q_cold + Q_losses

Where:
- Q_hot: Heat input at the hot heat exchanger
- W_acoustic_net: Net acoustic power output (to load)
- Q_cold: Heat rejected at cold heat exchangers
- Q_losses: Various dissipation mechanisms

In the Backhaus-Swift engine:
- ~4 kW heat input at hot HX
- ~1 kW acoustic power generated in regenerator
- ~0.7-0.8 kW circulates in feedback loop
- ~0.2-0.3 kW available for useful work
- Remaining heat rejected at cold HX

The high "looped power" to "output power" ratio (~3:1) is characteristic
of traveling-wave engines and is necessary to maintain the proper phasing.

References:
-----------
[1] Backhaus, S., & Swift, G. W. (2000). "A thermoacoustic-Stirling heat
    engine: Detailed study", JASA 107(6), 3148-3166.

[2] Backhaus, S., & Swift, G. W. (1999). "A thermoacoustic Stirling heat
    engine", Nature 399, 335-338.

[3] Swift, G. W. (2002). "Thermoacoustics: A Unifying Perspective for Some
    Engines and Refrigerators", Acoustical Society of America.

[4] de Blok, K. (2010). "Novel 4-stage traveling wave thermoacoustic power
    generator", ASME IMECE 2010.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from openthermoacoustics import gas, segments
from openthermoacoustics.solver.network import NetworkTopology
from openthermoacoustics.solver.loop_network import LoopNetwork
from openthermoacoustics.utils import acoustic_power


# =============================================================================
# BACKHAUS-SWIFT TASHE GEOMETRY PARAMETERS
# =============================================================================

def get_tashe_parameters() -> dict:
    """
    Return the geometric and operating parameters of the Backhaus-Swift TASHE.

    These values are taken from or derived from:
    Backhaus & Swift, JASA 107(6), 3148-3166 (2000), Table II

    Returns
    -------
    dict
        Dictionary containing all TASHE parameters.
    """
    params = {
        # Operating conditions
        "mean_pressure": 3.1e6,  # Pa (31 bar helium)
        "frequency": 80.0,  # Hz (nominal operating frequency)
        "T_cold": 298.0,  # K (ambient, 25 C)
        "T_hot": 998.0,  # K (725 C)

        # Main resonator (provides frequency setting and compliance)
        "resonator": {
            "length": 0.84,  # m (quarter-wave at ~80 Hz in helium)
            "diameter": 0.089,  # m
        },

        # Main cold heat exchanger (ambient side)
        "cold_hx": {
            "length": 0.054,  # m
            "diameter": 0.089,  # m
            "porosity": 0.52,
            "hydraulic_radius": 0.0004,  # m (shell-and-tube design)
        },

        # Regenerator (the heart of the engine)
        "regenerator": {
            "length": 0.075,  # m
            "diameter": 0.089,  # m
            "porosity": 0.72,
            "hydraulic_radius": 0.000042,  # m (42 microns - fine mesh)
            "material": "stainless steel 120-mesh screen",
        },

        # Hot heat exchanger
        "hot_hx": {
            "length": 0.025,  # m
            "diameter": 0.089,  # m
            "porosity": 0.50,
            "hydraulic_radius": 0.0004,  # m
        },

        # Thermal buffer tube (prevents streaming into feedback)
        "thermal_buffer": {
            "length": 0.19,  # m
            "diameter": 0.089,  # m
        },

        # Secondary cold HX (in feedback loop)
        "secondary_cold_hx": {
            "length": 0.027,  # m
            "diameter": 0.089,  # m
            "porosity": 0.52,
            "hydraulic_radius": 0.0004,  # m
        },

        # Feedback inertance (narrow tube for acoustic mass)
        "feedback_inertance": {
            "length": 0.26,  # m
            "diameter": 0.029,  # m (narrow!)
        },

        # Junction compliance
        "junction_compliance": {
            "volume": 0.0006,  # m^3 (600 cm^3)
        },
    }

    return params


# =============================================================================
# MAIN RESONATOR ANALYSIS (Standing wave section)
# =============================================================================

def analyze_main_resonator(params: dict, helium: gas.Helium) -> dict:
    """
    Analyze the main resonator section (quarter-wave stub).

    This section provides the acoustic compliance that sets the operating
    frequency and creates a pressure antinode (velocity node) at the closed end.

    Parameters
    ----------
    params : dict
        TASHE parameters from get_tashe_parameters().
    helium : gas.Helium
        Helium gas object at the operating pressure.

    Returns
    -------
    dict
        Analysis results including resonant frequency and impedances.
    """
    print("=" * 70)
    print("MAIN RESONATOR ANALYSIS")
    print("=" * 70)
    print()

    T_cold = params["T_cold"]
    resonator = params["resonator"]

    # Gas properties
    sound_speed = helium.sound_speed(T_cold)
    rho = helium.density(T_cold)

    # Resonator geometry
    L = resonator["length"]
    D = resonator["diameter"]
    A = np.pi * (D/2)**2

    print(f"Resonator stub:")
    print(f"  Length: {L*100:.1f} cm")
    print(f"  Diameter: {D*100:.1f} cm")
    print(f"  Area: {A*1e4:.2f} cm^2")
    print()

    # For a quarter-wave resonator (closed-open), f = a / (4L)
    # But this is connected at the junction, so it's more like a stub
    # that provides compliance at the junction

    f_quarter = sound_speed / (4 * L)
    print(f"Quarter-wave frequency (ideal): {f_quarter:.1f} Hz")
    print(f"Sound speed at {T_cold:.0f} K: {sound_speed:.1f} m/s")
    print()

    # The stub provides an acoustic compliance to the junction
    # For a closed-end stub of length L:
    # Z_stub = -j * Z_char * cot(k*L)
    # where Z_char = rho*a/A is the characteristic impedance

    Z_char = rho * sound_speed / A
    print(f"Characteristic impedance: {Z_char:.2e} Pa*s/m^3")

    # At the operating frequency
    f_op = params["frequency"]
    omega = 2 * np.pi * f_op
    k = omega / sound_speed

    # Stub impedance (ideal lossless)
    kL = k * L
    Z_stub = -1j * Z_char / np.tan(kL)

    print(f"\nAt operating frequency {f_op} Hz:")
    print(f"  k*L = {kL:.3f} rad ({np.degrees(kL):.1f} deg)")
    print(f"  Stub impedance = {Z_stub:.2e} Pa*s/m^3")
    print(f"  |Z_stub| = {abs(Z_stub):.2e} Pa*s/m^3")
    print(f"  Phase(Z_stub) = {np.degrees(np.angle(Z_stub)):.1f} deg")

    # For k*L near pi/2 (quarter wave), the stub provides large compliance
    # (Z approaches zero at resonance, Z is capacitive below resonance)

    if np.angle(Z_stub) < 0:
        print("  --> Stub is CAPACITIVE (provides compliance)")
    else:
        print("  --> Stub is INDUCTIVE (provides inertance)")

    print()

    # Build a network model of just the resonator
    network = NetworkTopology()

    # Create lossy duct for the resonator
    resonator_duct = segments.Duct(
        length=L,
        radius=D/2,
        name="resonator_stub"
    )
    network.add_segment(resonator_duct)

    # Propagate a wave through it to see the effect
    p1_junction = 10000.0 + 0j  # Pa, at junction
    U1_junction = p1_junction / Z_stub  # Volume velocity into stub

    results = network.propagate_all(
        p1_start=p1_junction,
        U1_start=U1_junction,
        T_m_start=T_cold,
        omega=omega,
        gas=helium,
    )

    profiles = network.get_global_profiles()

    print("Propagation through resonator stub:")
    print(f"  Input: p1 = {p1_junction:.0f} Pa, U1 = {U1_junction:.4e} m^3/s")
    print(f"  Output: p1 = {profiles['p1'][-1]:.0f} Pa, U1 = {profiles['U1'][-1]:.4e} m^3/s")
    print(f"  Acoustic power in: {acoustic_power(p1_junction, U1_junction):.4f} W")
    print(f"  Acoustic power out: {profiles['acoustic_power'][-1]:.4f} W")
    print()

    return {
        "f_quarter": f_quarter,
        "Z_char": Z_char,
        "Z_stub": Z_stub,
        "network": network,
        "profiles": profiles,
    }


# =============================================================================
# FEEDBACK BRANCH ANALYSIS
# =============================================================================

def analyze_feedback_branch(params: dict, helium: gas.Helium) -> dict:
    """
    Analyze the feedback branch (inertance + compliance).

    The feedback branch provides the phase shift needed to create
    traveling-wave conditions in the regenerator.

    Parameters
    ----------
    params : dict
        TASHE parameters.
    helium : gas.Helium
        Helium gas object.

    Returns
    -------
    dict
        Analysis results including feedback impedance.
    """
    print("=" * 70)
    print("FEEDBACK BRANCH ANALYSIS")
    print("=" * 70)
    print()

    T_cold = params["T_cold"]
    omega = 2 * np.pi * params["frequency"]

    # Gas properties
    rho = helium.density(T_cold)
    sound_speed = helium.sound_speed(T_cold)

    # Feedback inertance tube
    fb_params = params["feedback_inertance"]
    L_fb = fb_params["length"]
    D_fb = fb_params["diameter"]
    A_fb = np.pi * (D_fb/2)**2

    print(f"Feedback inertance tube:")
    print(f"  Length: {L_fb*100:.1f} cm")
    print(f"  Diameter: {D_fb*100:.1f} cm")
    print(f"  Area: {A_fb*1e4:.2f} cm^2")

    # Acoustic inertance
    L_acoustic = rho * L_fb / A_fb
    Z_inertance = 1j * omega * L_acoustic

    print(f"  Acoustic inertance: {L_acoustic:.2e} kg/m^4")
    print(f"  Inductive impedance: {Z_inertance:.2e} Pa*s/m^3")
    print(f"  |Z_L| = {abs(Z_inertance):.2e}")
    print()

    # Also include viscous resistance (Poiseuille)
    mu = helium.viscosity(T_cold)
    R_viscous = 8 * np.pi * mu * L_fb / A_fb**2
    print(f"  Viscous resistance: {R_viscous:.2e} Pa*s/m^3")
    print(f"  Q of inertance: {abs(Z_inertance) / R_viscous:.1f}")
    print()

    # Junction compliance
    V_comp = params["junction_compliance"]["volume"]
    C_acoustic = V_comp / (rho * sound_speed**2)
    Z_compliance = 1 / (1j * omega * C_acoustic)

    print(f"Junction compliance:")
    print(f"  Volume: {V_comp*1e6:.0f} cm^3")
    print(f"  Acoustic compliance: {C_acoustic:.2e} m^5/N")
    print(f"  Capacitive impedance: {Z_compliance:.2e} Pa*s/m^3")
    print()

    # The feedback loop impedance (simplified - series L and R)
    Z_feedback = Z_inertance + R_viscous

    print("Combined feedback impedance:")
    print(f"  Z_feedback = {Z_feedback:.2e} Pa*s/m^3")
    print(f"  |Z_feedback| = {abs(Z_feedback):.2e}")
    print(f"  Phase = {np.degrees(np.angle(Z_feedback)):.1f} deg")
    print()

    # For traveling-wave condition, we want the phase between p1 and U1
    # in the regenerator to be close to zero
    # The feedback loop creates this by providing the right impedance

    # Estimate the power circulating in the feedback loop
    # If p1 at junction is p1_j, then U1 into feedback is approximately:
    p1_j = 10000.0  # Pa (example)
    U1_feedback = p1_j / Z_feedback

    power_feedback = acoustic_power(p1_j, U1_feedback)
    print(f"Estimated power flow into feedback (at p1={p1_j:.0f} Pa):")
    print(f"  U1_feedback = {abs(U1_feedback)*1e3:.3f} x 10^-3 m^3/s")
    print(f"  Phase(U1) relative to p1: {np.degrees(np.angle(U1_feedback)):.1f} deg")
    print(f"  Acoustic power: {power_feedback:.2f} W")
    print()

    return {
        "L_acoustic": L_acoustic,
        "Z_inertance": Z_inertance,
        "R_viscous": R_viscous,
        "C_acoustic": C_acoustic,
        "Z_compliance": Z_compliance,
        "Z_feedback": Z_feedback,
    }


# =============================================================================
# REGENERATOR ANALYSIS (Traveling-wave section)
# =============================================================================

def analyze_regenerator(params: dict, helium: gas.Helium) -> dict:
    """
    Analyze the regenerator section with traveling-wave phasing.

    This is the heart of the engine where thermal energy is converted
    to acoustic power.

    Parameters
    ----------
    params : dict
        TASHE parameters.
    helium : gas.Helium
        Helium gas object.

    Returns
    -------
    dict
        Analysis results including power generation.
    """
    print("=" * 70)
    print("REGENERATOR ANALYSIS (Traveling-Wave Core)")
    print("=" * 70)
    print()

    T_cold = params["T_cold"]
    T_hot = params["T_hot"]
    omega = 2 * np.pi * params["frequency"]

    regen_params = params["regenerator"]
    L = regen_params["length"]
    D = regen_params["diameter"]
    porosity = regen_params["porosity"]
    r_h = regen_params["hydraulic_radius"]

    print(f"Regenerator geometry:")
    print(f"  Length: {L*100:.1f} cm")
    print(f"  Diameter: {D*100:.1f} cm")
    print(f"  Porosity: {porosity}")
    print(f"  Hydraulic radius: {r_h*1e6:.0f} microns")
    print()

    # Penetration depths at cold end
    rho_cold = helium.density(T_cold)
    mu = helium.viscosity(T_cold)
    kappa = helium.thermal_conductivity(T_cold)
    cp = helium.specific_heat_cp(T_cold)

    delta_nu = np.sqrt(2 * mu / (rho_cold * omega))
    delta_kappa = np.sqrt(2 * kappa / (rho_cold * cp * omega))

    print(f"Penetration depths at {T_cold:.0f} K:")
    print(f"  delta_nu (viscous): {delta_nu*1e6:.1f} microns")
    print(f"  delta_kappa (thermal): {delta_kappa*1e6:.1f} microns")
    print()
    print(f"Key ratios:")
    print(f"  r_h / delta_nu = {r_h/delta_nu:.3f}")
    print(f"  r_h / delta_kappa = {r_h/delta_kappa:.3f}")
    print()

    # For a traveling-wave regenerator, we want r_h << delta_kappa
    # This ensures good thermal contact
    if r_h / delta_kappa < 0.5:
        print("  --> GOOD: r_h << delta_kappa (excellent thermal contact)")
    else:
        print("  --> WARNING: r_h may be too large for ideal regenerator operation")
    print()

    # Temperature ratio
    T_ratio = T_hot / T_cold
    print(f"Temperature gradient:")
    print(f"  T_cold: {T_cold:.0f} K")
    print(f"  T_hot: {T_hot:.0f} K")
    print(f"  Temperature ratio: {T_ratio:.2f}")
    print(f"  Carnot efficiency: {(1 - T_cold/T_hot)*100:.1f}%")
    print()

    # Build regenerator network
    network = NetworkTopology()

    # Cold HX -> Regenerator -> Hot HX
    cold_hx_params = params["cold_hx"]
    cold_hx = segments.HeatExchanger(
        length=cold_hx_params["length"],
        porosity=cold_hx_params["porosity"],
        hydraulic_radius=cold_hx_params["hydraulic_radius"],
        temperature=T_cold,
        name="cold_hx"
    )
    network.add_segment(cold_hx)

    regenerator = segments.Stack(
        length=L,
        porosity=porosity,
        hydraulic_radius=r_h,
        T_cold=T_cold,
        T_hot=T_hot,
        name="regenerator"
    )
    network.add_segment(regenerator)

    hot_hx_params = params["hot_hx"]
    hot_hx = segments.HeatExchanger(
        length=hot_hx_params["length"],
        porosity=hot_hx_params["porosity"],
        hydraulic_radius=hot_hx_params["hydraulic_radius"],
        temperature=T_hot,
        name="hot_hx"
    )
    network.add_segment(hot_hx)

    print("Propagating through regenerator section...")
    print()

    # For traveling-wave, p1 and U1 should be in phase
    # Use representative values
    A_duct = np.pi * (D/2)**2
    Z_char = rho_cold * helium.sound_speed(T_cold) / (porosity * A_duct)

    p1_in = 50000.0 + 0j  # Pa (higher pressure for realistic power)

    # Different phase relationships to compare
    phases_to_test = [0, 30, 60, 90]  # degrees

    print(f"Testing different p1-U1 phase relationships:")
    print(f"{'Phase (deg)':^12} {'Power in (W)':^14} {'Power out (W)':^14} {'Delta Power (W)':^16}")
    print("-" * 58)

    results_by_phase = {}
    for phase_deg in phases_to_test:
        phase_rad = np.radians(phase_deg)
        U1_in = (p1_in / Z_char) * np.exp(1j * phase_rad)

        results = network.propagate_all(
            p1_start=p1_in,
            U1_start=U1_in,
            T_m_start=T_cold,
            omega=omega,
            gas=helium,
        )

        profiles = network.get_global_profiles()

        power_in = acoustic_power(p1_in, U1_in)
        power_out = profiles["acoustic_power"][-1]
        delta_power = power_out - power_in

        print(f"{phase_deg:^12} {power_in:^14.2f} {power_out:^14.2f} {delta_power:^+16.2f}")

        results_by_phase[phase_deg] = {
            "p1_in": p1_in,
            "U1_in": U1_in,
            "power_in": power_in,
            "power_out": power_out,
            "delta_power": delta_power,
            "profiles": profiles,
        }

    print()
    print("INTERPRETATION:")
    print("  - At 0 deg phase: p1 and U1 in phase (TRAVELING WAVE)")
    print("  - At 90 deg phase: p1 and U1 in quadrature (STANDING WAVE)")
    print("  - Positive delta power = ACOUSTIC POWER GENERATED")
    print("  - Maximum generation occurs near traveling-wave condition")
    print()

    return {
        "network": network,
        "results_by_phase": results_by_phase,
        "delta_nu": delta_nu,
        "delta_kappa": delta_kappa,
    }


# =============================================================================
# LOOP NETWORK DEMONSTRATION (using TBranch/Return)
# =============================================================================

def demonstrate_loop_network(params: dict, helium: gas.Helium) -> dict:
    """
    Demonstrate the TBranch/Return segments and LoopNetwork solver.

    This shows how to properly set up a traveling-wave engine loop
    using the branching topology features.

    Parameters
    ----------
    params : dict
        TASHE parameters.
    helium : gas.Helium
        Helium gas object.

    Returns
    -------
    dict
        Demonstration results.
    """
    print("=" * 70)
    print("LOOP NETWORK DEMONSTRATION (TBranch/Return)")
    print("=" * 70)
    print()

    T_cold = params["T_cold"]
    T_hot = params["T_hot"]
    omega = 2 * np.pi * params["frequency"]

    print("The LoopNetwork solver with TBranch/Return enables proper")
    print("loop topology modeling. Here's how it works:")
    print()

    # Create the main duct path (simplified)
    main_duct_1 = segments.Duct(
        length=0.3,
        radius=params["resonator"]["diameter"] / 2,
        name="main_duct_1"
    )

    # The TBranch diverts flow into the side branch
    side_area = np.pi * (params["feedback_inertance"]["diameter"] / 2) ** 2
    tbranch = segments.TBranch(
        side_area=side_area,
        side_fraction=0.5,  # 50% of flow goes to side branch
        name="tbranch"
    )

    # Main duct continues
    main_duct_2 = segments.Duct(
        length=0.2,
        radius=params["resonator"]["diameter"] / 2,
        name="main_duct_2"
    )

    # Return brings flow back from side branch
    return_seg = segments.Return(
        tbranch=tbranch,
        name="return"
    )

    # Main duct after return
    main_duct_3 = segments.Duct(
        length=0.3,
        radius=params["resonator"]["diameter"] / 2,
        name="main_duct_3"
    )

    print("Main path segments:")
    print(f"  1. {main_duct_1}")
    print(f"  2. {tbranch}")
    print(f"  3. {main_duct_2}")
    print(f"  4. {return_seg}")
    print(f"  5. {main_duct_3}")
    print()

    # Side branch segments (simplified thermoacoustic core)
    side_cold_hx = segments.HeatExchanger(
        length=params["cold_hx"]["length"],
        porosity=params["cold_hx"]["porosity"],
        hydraulic_radius=params["cold_hx"]["hydraulic_radius"],
        temperature=T_cold,
        name="side_cold_hx"
    )

    side_regenerator = segments.Stack(
        length=params["regenerator"]["length"],
        porosity=params["regenerator"]["porosity"],
        hydraulic_radius=params["regenerator"]["hydraulic_radius"],
        T_cold=T_cold,
        T_hot=T_hot,
        name="side_regenerator"
    )

    side_hot_hx = segments.HeatExchanger(
        length=params["hot_hx"]["length"],
        porosity=params["hot_hx"]["porosity"],
        hydraulic_radius=params["hot_hx"]["hydraulic_radius"],
        temperature=T_hot,
        name="side_hot_hx"
    )

    # Create a SideBranch container
    side_branch = segments.SideBranch(
        segments=[side_cold_hx, side_regenerator, side_hot_hx],
        name="thermoacoustic_core"
    )

    print("Side branch segments (thermoacoustic core):")
    for seg in side_branch.segments:
        print(f"  - {seg}")
    print(f"  Total length: {side_branch.total_length * 100:.1f} cm")
    print()

    # Build LoopNetwork
    network = LoopNetwork()
    network.add_segment(main_duct_1)
    network.add_segment(tbranch)
    network.add_segment(main_duct_2)
    network.add_segment(return_seg)
    network.add_segment(main_duct_3)

    # Add the side branch
    network.add_branch(
        tbranch_index=1,  # Index of tbranch in segment list
        segments=[side_cold_hx, side_regenerator, side_hot_hx],
        return_index=3,  # Index of return in segment list
    )

    print(f"Built: {network}")
    print()

    # Set up initial conditions
    rho_cold = helium.density(T_cold)
    A_main = np.pi * (params["resonator"]["diameter"] / 2) ** 2
    a_cold = helium.sound_speed(T_cold)
    Z_char = rho_cold * a_cold / A_main

    p1_start = 10000.0 + 0j  # Pa
    U1_start = p1_start / Z_char  # In-phase for traveling wave

    print("Propagating through loop network...")
    print(f"  Initial: p1 = {abs(p1_start):.0f} Pa, U1 = {abs(U1_start)*1e3:.4f} x 10^-3 m^3/s")
    print()

    # Attempt propagation
    try:
        # Set a small number of iterations for demonstration
        network.max_iterations = 10
        network.loop_closure_tolerance = 1e-3  # Relax tolerance

        results = network.propagate_all(
            p1_start=p1_start,
            U1_start=U1_start,
            T_m_start=T_cold,
            omega=omega,
            gas=helium,
        )

        print("Loop network propagation SUCCEEDED!")
        print()

        # Show results
        print("Main path results:")
        for result in results:
            p1_exit = result.p1[-1]
            U1_exit = result.U1[-1]
            power = result.acoustic_power[-1]
            print(f"  {result.segment.name:<15}: |p1| = {abs(p1_exit):.0f} Pa, "
                  f"|U1| = {abs(U1_exit):.4e} m^3/s, Power = {power:.3f} W")

        # Verify mass conservation
        conservation = network.verify_mass_conservation()
        print()
        print("Mass conservation check:")
        print(f"  Max split error: {conservation['max_split_error']:.2e}")
        print(f"  Max combine error: {conservation['max_combine_error']:.2e}")

        # Get side branch results
        if network.branch_results:
            print()
            print("Side branch (thermoacoustic core) results:")
            branch_profiles = network.get_branch_profiles(0)
            for i, result in enumerate(network.branch_results[0]):
                p1_exit = result.p1[-1]
                U1_exit = result.U1[-1]
                power = result.acoustic_power[-1]
                print(f"  {result.segment.name:<15}: |p1| = {abs(p1_exit):.0f} Pa, "
                      f"Power = {power:.3f} W")

        return {
            "network": network,
            "results": results,
            "converged": True,
        }

    except RuntimeError as e:
        print(f"Loop closure did not converge: {e}")
        print()
        print("This is expected for this simplified demonstration.")
        print("Full convergence requires:")
        print("  - Accurate impedance matching between branches")
        print("  - Correct phasing conditions")
        print("  - Proper frequency tuning")
        print()
        print("The TBranch/Return segments and LoopNetwork solver provide")
        print("the framework for solving these problems iteratively.")

        return {
            "network": network,
            "converged": False,
        }


# =============================================================================
# COMPLETE ENGINE ANALYSIS
# =============================================================================

def analyze_complete_engine(params: dict, helium: gas.Helium) -> dict:
    """
    Analyze the complete traveling-wave engine.

    This function demonstrates how all the components work together
    to produce acoustic power from a temperature difference.

    Parameters
    ----------
    params : dict
        TASHE parameters.
    helium : gas.Helium
        Helium gas object.

    Returns
    -------
    dict
        Complete analysis results.
    """
    print("=" * 70)
    print("COMPLETE TRAVELING-WAVE ENGINE ANALYSIS")
    print("=" * 70)
    print()

    T_cold = params["T_cold"]
    T_hot = params["T_hot"]
    omega = 2 * np.pi * params["frequency"]

    # Build the segments for the traveling-wave loop
    # We'll propagate through them manually since the NetworkTopology
    # has issues with lumped elements (Inertance) in the integrator

    segment_list = []

    # 1. Cold heat exchanger (at junction)
    cold_hx = segments.HeatExchanger(
        length=params["cold_hx"]["length"],
        porosity=params["cold_hx"]["porosity"],
        hydraulic_radius=params["cold_hx"]["hydraulic_radius"],
        temperature=T_cold,
        name="cold_hx"
    )
    segment_list.append(cold_hx)

    # 2. Regenerator (thermoacoustic power production)
    regenerator = segments.Stack(
        length=params["regenerator"]["length"],
        porosity=params["regenerator"]["porosity"],
        hydraulic_radius=params["regenerator"]["hydraulic_radius"],
        T_cold=T_cold,
        T_hot=T_hot,
        name="regenerator"
    )
    segment_list.append(regenerator)

    # 3. Hot heat exchanger
    hot_hx = segments.HeatExchanger(
        length=params["hot_hx"]["length"],
        porosity=params["hot_hx"]["porosity"],
        hydraulic_radius=params["hot_hx"]["hydraulic_radius"],
        temperature=T_hot,
        name="hot_hx"
    )
    segment_list.append(hot_hx)

    # 4. Thermal buffer tube (at hot temperature)
    tb_params = params["thermal_buffer"]
    buffer_tube = segments.Duct(
        length=tb_params["length"],
        radius=tb_params["diameter"]/2,
        name="thermal_buffer"
    )
    segment_list.append(buffer_tube)

    # 5. Secondary cold HX (returns to cold)
    sec_cold_hx = segments.HeatExchanger(
        length=params["secondary_cold_hx"]["length"],
        porosity=params["secondary_cold_hx"]["porosity"],
        hydraulic_radius=params["secondary_cold_hx"]["hydraulic_radius"],
        temperature=T_cold,
        name="secondary_cold_hx"
    )
    segment_list.append(sec_cold_hx)

    # 6. Feedback inertance (lumped element)
    fb_params = params["feedback_inertance"]
    feedback = segments.Inertance(
        length=fb_params["length"],
        radius=fb_params["diameter"]/2,
        include_resistance=True,
        name="feedback_inertance"
    )
    segment_list.append(feedback)

    print(f"Engine network built with {len(segment_list)} segments:")
    for seg in segment_list:
        print(f"  - {seg}")
    print()

    # Now propagate with traveling-wave phasing
    # Use direct propagate() calls on each segment

    rho_cold = helium.density(T_cold)
    A_main = np.pi * (params["cold_hx"]["diameter"]/2)**2 * params["cold_hx"]["porosity"]
    a_cold = helium.sound_speed(T_cold)
    Z_char = rho_cold * a_cold / A_main

    # Use a typical pressure amplitude
    p1_start = 50000.0 + 0j  # Pa

    # For traveling wave, U1 is in phase with p1
    U1_start = p1_start / Z_char

    print("Forward propagation through the loop:")
    print(f"  Initial: p1 = {abs(p1_start):.0f} Pa, U1 = {abs(U1_start)*1e3:.3f} x 10^-3 m^3/s")
    print(f"  Phase difference: {np.degrees(np.angle(U1_start) - np.angle(p1_start)):.1f} deg")
    print()

    # Propagate through each segment
    results = []
    p1_current = p1_start
    U1_current = U1_start
    T_current = T_cold

    for seg in segment_list:
        p1_in = p1_current
        U1_in = U1_current
        T_in = T_current
        power_in = acoustic_power(p1_in, U1_in)

        # Propagate through segment
        p1_out, U1_out, T_out = seg.propagate(p1_in, U1_in, T_in, omega, helium)
        power_out = acoustic_power(p1_out, U1_out)

        results.append({
            "segment": seg,
            "p1_in": p1_in,
            "U1_in": U1_in,
            "T_in": T_in,
            "p1_out": p1_out,
            "U1_out": U1_out,
            "T_out": T_out,
            "power_in": power_in,
            "power_out": power_out,
            "delta_power": power_out - power_in,
        })

        # Update for next segment
        p1_current = p1_out
        U1_current = U1_out
        T_current = T_out

    print("Results at each segment exit:")
    print(f"{'Segment':<20} {'|p1| (Pa)':>12} {'|U1| (m^3/s)':>14} {'Phase diff (deg)':>16} {'Power (W)':>12}")
    print("-" * 78)

    for res in results:
        p1_exit = res["p1_out"]
        U1_exit = res["U1_out"]
        power = res["power_out"]
        phase_diff = np.degrees(np.angle(U1_exit) - np.angle(p1_exit))

        print(f"{res['segment'].name:<20} {abs(p1_exit):>12.1f} {abs(U1_exit):>14.4e} "
              f"{phase_diff:>+16.1f} {power:>12.3f}")

    print("-" * 78)

    # Check loop closure
    p1_end = results[-1]["p1_out"]
    U1_end = results[-1]["U1_out"]

    print()
    print("LOOP CLOSURE CHECK:")
    print(f"  Start: p1 = {p1_start:.0f} Pa, U1 = {U1_start:.4e} m^3/s")
    print(f"  End:   p1 = {p1_end:.0f} Pa, U1 = {U1_end:.4e} m^3/s")
    print()
    print(f"  |p1_end / p1_start| = {abs(p1_end / p1_start):.4f}")
    print(f"  Phase(p1_end - p1_start) = {np.degrees(np.angle(p1_end / p1_start)):.1f} deg")
    print()

    # Acoustic power balance
    power_in = acoustic_power(p1_start, U1_start)
    power_out = acoustic_power(p1_end, U1_end)
    power_generated = power_out - power_in

    print("ACOUSTIC POWER BALANCE:")
    print(f"  Power at start: {power_in:.2f} W")
    print(f"  Power at end: {power_out:.2f} W")
    print(f"  Net power generated: {power_generated:+.2f} W")
    print()

    # Find where power is generated/lost
    print("Power generation by segment:")
    for res in results:
        delta_p = res["delta_power"]
        print(f"  {res['segment'].name:<20}: {delta_p:+.3f} W")

    print()
    print("IMPORTANT NOTE:")
    print("-" * 50)
    print("The large numbers above demonstrate WHY loop closure is essential.")
    print("Without proper feedback, acoustic energy accumulates exponentially.")
    print("In a real engine:")
    print("  1. The loop MUST close: p1_end = p1_start, U1_end = U1_start")
    print("  2. This constrains the operating frequency")
    print("  3. Steady state has power generated = power dissipated")
    print()
    print("The forward propagation shows the DIRECTION of power flow")
    print("but not the MAGNITUDE (which requires loop solution).")
    print()

    return {
        "segments": segment_list,
        "results": results,
        "p1_start": p1_start,
        "U1_start": U1_start,
        "p1_end": p1_end,
        "U1_end": U1_end,
        "power_generated": power_generated,
    }


# =============================================================================
# FREQUENCY ESTIMATION FOR LOOP CLOSURE
# =============================================================================

def estimate_loop_frequency(params: dict, helium: gas.Helium) -> dict:
    """
    Estimate the operating frequency for loop closure.

    For a traveling-wave engine to operate, the acoustic state must
    return to its starting value after traversing the loop. This
    requires a specific relationship between frequency and geometry.

    This function uses a simplified lumped-element model to estimate
    the operating frequency.

    Parameters
    ----------
    params : dict
        TASHE parameters.
    helium : gas.Helium
        Helium gas object.

    Returns
    -------
    dict
        Frequency estimation results.
    """
    print("=" * 70)
    print("LOOP FREQUENCY ESTIMATION")
    print("=" * 70)
    print()

    T_cold = params["T_cold"]
    rho = helium.density(T_cold)
    a = helium.sound_speed(T_cold)

    # Resonator stub parameters
    L_res = params["resonator"]["length"]
    A_res = np.pi * (params["resonator"]["diameter"]/2)**2

    # Feedback inertance
    L_fb = params["feedback_inertance"]["length"]
    A_fb = np.pi * (params["feedback_inertance"]["diameter"]/2)**2

    # Compliance
    V_comp = params["junction_compliance"]["volume"]

    # Simplified model: LC resonator
    # The resonator stub provides additional compliance
    # The feedback provides inertance

    # Acoustic inertance of feedback
    L_acoustic = rho * L_fb / A_fb

    # Acoustic compliance of junction volume
    C_acoustic = V_comp / (rho * a**2)

    # For a simple LC circuit: f = 1 / (2*pi*sqrt(L*C))
    f_simple = 1 / (2 * np.pi * np.sqrt(L_acoustic * C_acoustic))

    print("Simplified LC model:")
    print(f"  Feedback inertance L = {L_acoustic:.2e} kg/m^4")
    print(f"  Junction compliance C = {C_acoustic:.2e} m^5/N")
    print(f"  Resonance frequency = {f_simple:.1f} Hz")
    print()

    # The resonator stub also contributes compliance
    # For a quarter-wave stub: effective compliance ~ V_eff / (rho * a^2)
    # where V_eff ~ A * L / 3 for a distributed system
    V_eff_stub = A_res * L_res / 3
    C_stub = V_eff_stub / (rho * a**2)

    C_total = C_acoustic + C_stub
    f_with_stub = 1 / (2 * np.pi * np.sqrt(L_acoustic * C_total))

    print("Including resonator stub compliance:")
    print(f"  Stub effective compliance = {C_stub:.2e} m^5/N")
    print(f"  Total compliance = {C_total:.2e} m^5/N")
    print(f"  Estimated frequency = {f_with_stub:.1f} Hz")
    print()

    # The target design frequency
    f_design = params["frequency"]
    print(f"Design operating frequency: {f_design:.1f} Hz")
    print(f"Estimation error: {abs(f_with_stub - f_design)/f_design * 100:.1f}%")
    print()

    # More accurate model would require solving the complete loop equation
    print("NOTE: A more accurate frequency would require:")
    print("  1. Full distributed-element model of all segments")
    print("  2. Loop closure condition: p1_end = p1_start, U1_end = U1_start")
    print("  3. Complex eigenvalue solution (frequency is generally complex)")
    print("  4. Iteration to find the real frequency where loop closes")
    print()

    return {
        "f_simple": f_simple,
        "f_with_stub": f_with_stub,
        "f_design": f_design,
        "L_acoustic": L_acoustic,
        "C_acoustic": C_acoustic,
        "C_total": C_total,
    }


# =============================================================================
# CONVERGENCE CHALLENGES FOR LOOP NETWORKS
# =============================================================================

def discuss_loop_solver_challenges() -> None:
    """
    Discuss the challenges in solving loop network equations.

    Traveling-wave engine simulation requires solving for loop closure,
    which is a challenging nonlinear eigenvalue problem.
    """
    print("=" * 70)
    print("CHALLENGES IN LOOP NETWORK CONVERGENCE")
    print("=" * 70)
    print()

    print("""
The complete solution of a traveling-wave engine loop requires:

1. LOOP CLOSURE CONDITION
   The acoustic state must return to its starting point:
     p1(x=0) = p1(x=L_loop)
     U1(x=0) = U1(x=L_loop)

   This gives 4 real equations (real and imaginary parts of p1 and U1).

2. UNKNOWN PARAMETERS
   - Frequency (omega) - 1 parameter
   - Overall amplitude (can be normalized) - 1 parameter
   - Phase reference - 1 parameter

   Net: Need to solve for 1 free parameter (frequency) with 4 constraints,
   which means 3 constraints must be automatically satisfied by the physics.

3. COMPLEXITY OF THE EQUATIONS
   - Nonlinear coupling between frequency and acoustic field
   - Temperature-dependent gas properties
   - Distributed (non-lumped) elements
   - Lossy elements (complex eigenvalue)

4. NUMERICAL CHALLENGES
   - Multiple solutions (different modes)
   - Sensitive to initial guesses
   - Can converge to unphysical solutions
   - May require pseudo-arclength continuation

5. AVAILABLE FEATURES (OpenThermoacoustics now includes):

   a) TBranch segment: Splits acoustic power at junction
      - Enforces: p1 same in all branches
      - Enforces: sum(U1_out) = U1_in
      - Stores reference to Return segment
      - Adjustable side_fraction for iteration

   b) Return segment: Brings branch back to main path
      - Connects back to the TBranch location
      - Adds the returning U1 to the main path
      - Provides pressure mismatch for loop closure check

   c) SideBranch class: Container for side branch segments
      - Groups segments forming the side branch
      - Convenient propagate() through all segments

   d) LoopNetwork class: Manages loop topology
      - Builds complete network graph with branches
      - Sets up loop closure equations
      - Iteratively adjusts side fractions
      - Provides mass conservation verification

6. REMAINING CHALLENGES FOR FULL CONVERGENCE:
   The current implementation provides the FRAMEWORK, but solving
   for accurate loop closure still requires:

   - More sophisticated iteration algorithms (Newton-Raphson)
   - Frequency as an unknown (complex eigenvalue)
   - Better initial guess strategies
   - Pseudo-arclength continuation for robustness

   The demonstration in this example shows that while the loop
   topology can be modeled, achieving full convergence for the
   Backhaus-Swift geometry requires additional numerical work.
""")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    """Main function demonstrating traveling-wave engine analysis."""

    print()
    print("=" * 70)
    print("TRAVELING-WAVE THERMOACOUSTIC ENGINE ANALYSIS")
    print("Based on Backhaus-Swift TASHE (JASA 2000)")
    print("=" * 70)
    print()

    # Get engine parameters
    params = get_tashe_parameters()

    # Create working gas
    helium = gas.Helium(mean_pressure=params["mean_pressure"])

    print(f"Working gas: Helium at {params['mean_pressure']/1e6:.1f} MPa "
          f"({params['mean_pressure']/1e5:.0f} bar)")
    print(f"Operating frequency: {params['frequency']} Hz")
    print(f"Temperature: {params['T_cold']:.0f} K (cold) to {params['T_hot']:.0f} K (hot)")
    print(f"Temperature ratio: {params['T_hot']/params['T_cold']:.2f}")
    print(f"Carnot efficiency: {(1 - params['T_cold']/params['T_hot'])*100:.1f}%")
    print()

    # Run analyses
    resonator_results = analyze_main_resonator(params, helium)
    feedback_results = analyze_feedback_branch(params, helium)
    regenerator_results = analyze_regenerator(params, helium)
    loop_demo_results = demonstrate_loop_network(params, helium)
    engine_results = analyze_complete_engine(params, helium)
    frequency_results = estimate_loop_frequency(params, helium)

    # Discuss challenges
    discuss_loop_solver_challenges()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print("This example demonstrated:")
    print("  1. Why traveling-wave engines achieve high efficiency")
    print("     (p1 and U1 in phase enables quasi-isothermal cycles)")
    print()
    print("  2. The role of the regenerator in traveling-wave vs standing-wave")
    print("     (Fine mesh for thermal contact vs coarse for imperfect contact)")
    print()
    print("  3. How the feedback loop creates traveling-wave phasing")
    print("     (Inertance + compliance provide correct phase shift)")
    print()
    print("  4. Acoustic power flow through the system")
    print("     (Power generated in regenerator, most circulates in loop)")
    print()

    print("Forward propagation results:")
    print(f"  Estimated frequency: ~{frequency_results['f_with_stub']:.0f} Hz (simplified LC model)")
    print(f"  Design frequency: {params['frequency']:.0f} Hz")
    print()
    print("  NOTE: The forward propagation shows large amplification because")
    print("  we are NOT solving for loop closure. In a real traveling-wave")
    print("  engine, the loop must close on itself, which constrains the")
    print("  frequency and amplitude. The analysis above shows the COMPONENTS")
    print("  correctly but requires a loop solver for quantitative results.")
    print()

    print("Available features in OpenThermoacoustics:")
    print("  - TBranch/Return segments for loop topology (IMPLEMENTED)")
    print("  - SideBranch container for side branch management (IMPLEMENTED)")
    print("  - LoopNetwork solver with iterative closure (IMPLEMENTED)")
    print()
    print("Still needed for full convergence:")
    print("  - Newton-Raphson or secant method for faster convergence")
    print("  - Complex eigenvalue solver for frequency")
    print("  - Better impedance matching algorithms")
    print()
    print("See also: examples/traveling_wave_stub.py for geometry overview")
    print()


if __name__ == "__main__":
    main()

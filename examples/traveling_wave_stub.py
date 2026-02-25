#!/usr/bin/env python3
"""
Example: Traveling-Wave Engine Geometry Stub

NOTE: For a more comprehensive traveling-wave engine example with full
physics explanations and analysis, see:

    examples/traveling_wave_engine.py

That example includes:
- Detailed explanation of why traveling-wave engines achieve high efficiency
- Analysis of the Backhaus-Swift TASHE parameters
- Forward propagation through all engine components
- Acoustic power flow calculations
- Discussion of loop solver challenges

This stub file provides a simpler overview of the geometry and parameters
for reference.

Original stub purpose:
---------------------
This is a geometry reference for a traveling-wave thermoacoustic engine
based on the Backhaus-Swift design (JASA 2000). The TBranch/Return segments
and LoopNetwork solver are now available for loop topology modeling.

Physics Background:
-------------------
A traveling-wave thermoacoustic engine differs fundamentally from a
standing-wave engine in several ways:

Standing-Wave vs. Traveling-Wave:
--------------------------------
1. Standing-Wave Engine:
   - Pressure and velocity are 90 degrees out of phase
   - Stack operates with imperfect heat transfer
   - Typical onset temperature ratio: 1.3 - 2.0
   - Efficiency limited to ~20% of Carnot

2. Traveling-Wave Engine:
   - Pressure and velocity are in phase
   - Regenerator operates with nearly reversible heat transfer
   - Can achieve much lower onset temperature ratios
   - Efficiency can approach 50% of Carnot or higher

The Backhaus-Swift Engine:
--------------------------
The breakthrough 1998/2000 design by Backhaus and Swift achieved
traveling-wave phasing by using a toroidal (looped) geometry:

   Compliance (thermal buffer tube)
       |
   Inertance (feedback inertance)
       |
   [Branch point] ---- Resonator (quarter-wave stub)
       |
   Regenerator
       |
   Hot Heat Exchanger
       |
   Thermal Buffer Tube
       |
   Cold Heat Exchanger
       |
   [Return to branch point]

Key features:
- The resonator provides pressure gain and frequency determination
- The feedback inertance and compliance provide proper phasing
- The regenerator experiences traveling-wave acoustic field
- Temperature ratios as low as T_h/T_c ~ 1.1 are possible

Topology Requirements:
----------------------
To model this properly, we need:
1. TBRANCH segment - splits acoustic power between paths
2. RETURN segment - reconnects a branch back to the main path
3. Proper handling of complex impedance matching

These are not yet implemented in OpenThermoacoustics.

References:
-----------
Backhaus, S., & Swift, G. W. (2000). "A thermoacoustic-Stirling heat engine:
    Detailed study", JASA 107(6), 3148-3166.

Backhaus, S., & Swift, G. W. (1999). "A thermoacoustic Stirling heat engine",
    Nature 399, 335-338.

Swift, G. W. (2002). "Thermoacoustics: A Unifying Perspective for Some Engines
    and Refrigerators", Acoustical Society of America.
"""

import numpy as np

from openthermoacoustics import gas, segments


def backhaus_swift_geometry() -> dict:
    """
    Define the geometry parameters of the Backhaus-Swift engine.

    Returns a dictionary with all geometric parameters based on the
    2000 JASA paper. These parameters can be used once TBRANCH/RETURN
    segments are implemented.

    Returns
    -------
    dict
        Dictionary containing all geometric parameters.
    """
    # All dimensions from Backhaus & Swift JASA 2000, Table II

    geometry = {
        "name": "Backhaus-Swift TASHE (2000)",
        "reference": "JASA 107(6), 3148-3166 (2000)",

        # Operating conditions
        "mean_pressure": 3.1e6,  # Pa (31 bar)
        "gas": "helium",
        "frequency": 80.0,  # Hz (typical operating frequency)

        # Resonator (quarter-wave stub)
        "resonator": {
            "length": 1.0,  # m (approximate)
            "diameter": 0.089,  # m
            "description": "Quarter-wave resonator providing acoustic compliance",
        },

        # Main cold heat exchanger (ambient HX)
        "main_cold_hx": {
            "length": 0.054,  # m
            "diameter": 0.089,  # m
            "porosity": 0.52,
            "hydraulic_radius": 0.0004,  # m (approximate)
            "temperature": 300.0,  # K (ambient)
            "description": "Copper shell-and-tube heat exchanger",
        },

        # Regenerator
        "regenerator": {
            "length": 0.075,  # m
            "diameter": 0.089,  # m
            "porosity": 0.72,
            "hydraulic_radius": 0.000042,  # m (42 micron)
            "material": "stainless steel mesh",
            "mesh_number": 120,  # screens per inch
            "description": "Stacked stainless steel screens",
        },

        # Hot heat exchanger
        "hot_hx": {
            "length": 0.025,  # m (approximate)
            "diameter": 0.089,  # m
            "porosity": 0.50,
            "hydraulic_radius": 0.0004,  # m
            "temperature_typical": 725.0,  # K (typical hot-side temp)
            "description": "Electric heater cartridges in nickel block",
        },

        # Thermal buffer tube (between hot HX and feedback)
        "thermal_buffer_tube": {
            "length": 0.19,  # m
            "diameter": 0.089,  # m
            "description": "Prevents streaming of hot gas into feedback",
        },

        # Secondary cold heat exchanger (in feedback loop)
        "secondary_cold_hx": {
            "length": 0.027,  # m
            "diameter": 0.089,  # m
            "porosity": 0.52,
            "temperature": 300.0,  # K
            "description": "Removes heat from feedback loop",
        },

        # Feedback inertance
        "feedback_inertance": {
            "length": 0.26,  # m
            "diameter": 0.029,  # m
            "description": "Provides inertance for traveling-wave phasing",
        },

        # Compliance (at junction)
        "compliance": {
            "volume": 0.0006,  # m^3 (approximate)
            "description": "Compliance at tee junction",
        },

        # Branch junction
        "junction": {
            "type": "tee",
            "description": "Connects resonator, feedback, and regenerator",
        },
    }

    return geometry


def display_geometry() -> None:
    """Display the Backhaus-Swift engine geometry."""
    print("=" * 70)
    print("Traveling-Wave Engine Geometry Stub")
    print("Backhaus-Swift TASHE Configuration")
    print("=" * 70)
    print()

    geom = backhaus_swift_geometry()

    print(f"Engine: {geom['name']}")
    print(f"Reference: {geom['reference']}")
    print()

    print("Operating Conditions:")
    print(f"  Mean pressure: {geom['mean_pressure']/1e6:.1f} MPa ({geom['mean_pressure']/1e5:.0f} bar)")
    print(f"  Working gas: {geom['gas']}")
    print(f"  Operating frequency: {geom['frequency']:.1f} Hz")
    print()

    print("Component Geometry:")
    print("-" * 50)

    for component, params in geom.items():
        if isinstance(params, dict) and "length" in params:
            print(f"\n{component.replace('_', ' ').title()}:")
            if "description" in params:
                print(f"  Description: {params['description']}")
            if "length" in params:
                print(f"  Length: {params['length']*100:.1f} cm")
            if "diameter" in params:
                print(f"  Diameter: {params['diameter']*100:.1f} cm")
            if "porosity" in params:
                print(f"  Porosity: {params['porosity']:.2f}")
            if "hydraulic_radius" in params:
                print(f"  Hydraulic radius: {params['hydraulic_radius']*1000:.3f} mm")
            if "temperature" in params:
                print(f"  Temperature: {params['temperature']:.0f} K")
            if "temperature_typical" in params:
                print(f"  Typical temperature: {params['temperature_typical']:.0f} K")

        elif component == "compliance" and isinstance(params, dict):
            print(f"\n{component.replace('_', ' ').title()}:")
            if "description" in params:
                print(f"  Description: {params['description']}")
            if "volume" in params:
                print(f"  Volume: {params['volume']*1e6:.0f} cm^3")

        elif component == "junction" and isinstance(params, dict):
            print(f"\n{component.replace('_', ' ').title()}:")
            print(f"  Type: {params['type']}")
            if "description" in params:
                print(f"  Description: {params['description']}")

    print()


def placeholder_calculation() -> None:
    """
    Placeholder showing how segments would be connected.

    This demonstrates the intended API once TBRANCH/RETURN are implemented.
    """
    print("=" * 70)
    print("Placeholder: Segment Creation (Linear Path Only)")
    print("=" * 70)
    print()

    # Working gas
    mean_pressure = 3.1e6  # Pa (31 bar)
    helium = gas.Helium(mean_pressure=mean_pressure)
    T_cold = 300.0
    T_hot = 725.0

    sound_speed = helium.sound_speed(T_cold)
    print(f"Helium at {mean_pressure/1e6:.1f} MPa:")
    print(f"  Sound speed at {T_cold:.0f} K: {sound_speed:.1f} m/s")
    print(f"  Sound speed at {T_hot:.0f} K: {helium.sound_speed(T_hot):.1f} m/s")
    print()

    # Create individual segments (these could be connected in a full model)
    print("Creating individual segments...")
    print()

    # Regenerator
    regen = segments.Stack(
        length=0.075,
        porosity=0.72,
        hydraulic_radius=0.000042,
        T_cold=T_cold,
        T_hot=T_hot,
        name="regenerator",
    )
    print(f"Regenerator: {regen}")

    # Hot heat exchanger
    hot_hx = segments.HeatExchanger(
        length=0.025,
        porosity=0.50,
        hydraulic_radius=0.0004,
        temperature=T_hot,
        name="hot_hx",
    )
    print(f"Hot HX: {hot_hx}")

    # Cold heat exchanger
    cold_hx = segments.HeatExchanger(
        length=0.054,
        porosity=0.52,
        hydraulic_radius=0.0004,
        temperature=T_cold,
        name="cold_hx",
    )
    print(f"Cold HX: {cold_hx}")

    # Thermal buffer tube (as a duct)
    buffer_tube = segments.Duct(
        length=0.19,
        radius=0.089/2,
        name="thermal_buffer",
    )
    print(f"Buffer tube: {buffer_tube}")

    # Feedback inertance
    feedback = segments.Inertance(
        length=0.26,
        radius=0.029/2,
        include_resistance=True,
        name="feedback_inertance",
    )
    print(f"Feedback inertance: {feedback}")

    # Compliance
    compliance = segments.Compliance(
        volume=0.0006,
        name="junction_compliance",
    )
    print(f"Compliance: {compliance}")

    # Resonator
    resonator = segments.Duct(
        length=1.0,
        radius=0.089/2,
        name="resonator",
    )
    print(f"Resonator: {resonator}")

    print()
    print("-" * 50)
    print()

    print("TOPOLOGY NOTE:")
    print()
    print("The Backhaus-Swift engine has a LOOPED topology:")
    print()
    print("                 Resonator (lambda/4)")
    print("                     |")
    print("           [TEE JUNCTION] <-- Compliance")
    print("                /        \\")
    print("               /          \\")
    print("    Cold HX --+            +-- Feedback Inertance")
    print("        |                          |")
    print("    Regenerator                    |")
    print("        |                          |")
    print("    Hot HX                         |")
    print("        |                          |")
    print("    Buffer Tube                    |")
    print("        |                          |")
    print("    Secondary Cold HX              |")
    print("        |                          |")
    print("        +----------<---------------+")
    print("              (loop closure)")
    print()
    print("To model this properly requires:")
    print("  1. TBRANCH segment - acoustic power splitter")
    print("  2. UNION/RETURN segment - loop closure")
    print("  3. Complex eigenvalue solver for loop impedance matching")
    print()
    print("These features are planned for future development.")


def estimate_performance() -> None:
    """Provide rough performance estimates based on published results."""
    print()
    print("=" * 70)
    print("Expected Performance (from Literature)")
    print("=" * 70)
    print()

    print("Backhaus-Swift TASHE Performance (JASA 2000):")
    print("-" * 50)
    print()
    print("Operating point:")
    print("  Mean pressure: 3.1 MPa (31 bar)")
    print("  Frequency: ~80 Hz")
    print("  Hot temperature: 725 K")
    print("  Cold temperature: 300 K")
    print("  Temperature ratio: 2.42")
    print()
    print("Performance:")
    print("  Heat input: ~4 kW (at max power)")
    print("  Acoustic power: ~1 kW")
    print("  Thermal efficiency: ~30% (of Carnot)")
    print("  Carnot efficiency: 59%")
    print("  Actual efficiency: ~18%")
    print()
    print("Onset temperature ratio: ~1.06 (remarkable!)")
    print("  Compare to standing-wave engines: ~1.3-2.0")
    print()
    print("Key to high performance:")
    print("  - Traveling-wave phasing in regenerator")
    print("  - High-porosity, fine-mesh regenerator")
    print("  - Proper acoustic impedance matching")
    print()


def main() -> None:
    """Main entry point for the traveling-wave stub example."""
    display_geometry()
    placeholder_calculation()
    estimate_performance()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("This stub demonstrates:")
    print("  1. The geometry of a traveling-wave thermoacoustic engine")
    print("  2. How individual segments would be created")
    print("  3. Why TBRANCH/RETURN segments are needed for loop topology")
    print()
    print("For a complete analysis with physics explanations, run:")
    print("  python examples/traveling_wave_engine.py")
    print()
    print("That example includes:")
    print("  - Why traveling-wave engines achieve Carnot-like efficiency")
    print("  - Role of regenerator in traveling-wave vs standing-wave")
    print("  - How the feedback loop creates traveling-wave phasing")
    print("  - Acoustic power flow through the system")
    print("  - Frequency estimation and loop closure challenges")
    print()
    print("Available features (now implemented):")
    print("  - TBranch segment for power splitting")
    print("  - Return segment for loop closure")
    print("  - SideBranch container for branch management")
    print("  - LoopNetwork solver with iterative closure")
    print()
    print("Future work still needed for full convergence:")
    print("  - Newton-Raphson iteration for faster convergence")
    print("  - Complex eigenvalue solver for frequency")
    print("  - Better impedance matching algorithms")
    print("  - Gedeon streaming calculations")
    print()
    print("Once fully converged, this would enable:")
    print("  - Full traveling-wave engine simulation")
    print("  - Optimization of regenerator parameters")
    print("  - Prediction of onset and efficiency")
    print()


if __name__ == "__main__":
    main()

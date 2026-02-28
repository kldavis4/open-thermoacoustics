# OpenThermoacoustics

OpenThermoacoustics is a Python library for 1D thermoacoustic modeling based on Rott/Swift linear theory.  
It solves complex pressure/volume-velocity fields in acoustic networks (ducts, stacks, heat exchangers, and related elements) for design-space exploration and validation workflows.

## Installation

```bash
pip install openthermoacoustics
```

Optional extras:

```bash
pip install "openthermoacoustics[viz]"   # plotting utilities
pip install "openthermoacoustics[dev]"   # tests, lint, typing tools
```

## Quick Start

```python
import openthermoacoustics as ota

# Working gas: helium at 30 bar
gas = ota.gas.Helium(mean_pressure=3e6)

# Build a simple resonator
network = ota.solver.NetworkTopology()
network.add_segment(ota.segments.Duct(length=1.0, radius=0.05))

# Solve for a resonant mode (shooting method)
solver = ota.solver.ShootingSolver(network, gas)
result = solver.solve(
    guesses={"frequency": 500.0, "p1_phase": 0.0},
    targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
    options={"T_m_start": 300.0},
)

print(f"Resonant frequency: {result.frequency:.2f} Hz")
```

## What's Included in v0.1.1

- Standing-wave engine validation helpers (onset and profile workflows)
- Standing-wave refrigerator validation helpers, including:
  - `tijani_refrigerator_config()`
  - short-stack cooling/COP utilities in `openthermoacoustics.validation`
- Example scripts for engine/refrigerator benchmark runs
- Plotting helpers in `openthermoacoustics.viz`

## Core Capabilities

### Gas Models
- Helium, Argon, Nitrogen, Air
- Temperature-dependent transport properties

### Geometry Models
- Circular pore
- Parallel plate
- Wire screen approximation

### Segment Types
- Duct, Cone
- Stack, HeatExchanger
- Compliance, Inertance
- Additional branch/transducer elements in `openthermoacoustics.segments`

### Solvers
- Network shooting solver for resonant boundary-value problems
- Distributed segment integration with thermoviscous effects

## Model Scope and Limitations

- Linear 1D thermoacoustic model (frequency-domain, small-signal regime)
- Best suited for early-stage design, sensitivity studies, and benchmark validation
- Does **not** model nonlinear saturation/limit cycles or full CFD-scale effects
- Results depend on topology assumptions, loss models, and boundary-condition fidelity

## Documentation

- Docs index: `docs/index.md`
- Validation docs: `docs/validation/`

## References

1. Swift, G.W. (2017). *Thermoacoustics: A Unifying Perspective for Some Engines and Refrigerators*.
2. Swift, G.W. (1988). "Thermoacoustic engines." *JASA*.
3. Rott, N. (1969). "Damped and thermally driven acoustic oscillations in wide and narrow tubes."

## License

MIT

# OpenThermoacoustics

A Python implementation of thermoacoustic engine and refrigerator design tools, based on Rott's linear thermoacoustic theory as extended by Swift.

## Overview

OpenThermoacoustics provides a solver for analyzing thermoacoustic devices by modeling them as 1D acoustic networks. It propagates complex oscillating pressure and volumetric velocity through connected segments (ducts, stacks, heat exchangers, etc.) using the linearized thermoacoustic equations.

This project aims to provide an open-source thermoacoustic design tool, implementing well-documented physics from published literature in a clean, modern, extensible Python codebase.

Full docs index: [docs/index.md](docs/index.md)

## Installation

```bash
pip install -e .
```

For development with testing tools:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import openthermoacoustics as ota

# Create working gas: helium at 30 bar
gas = ota.gas.Helium(mean_pressure=3e6)

# Build a simple closed-closed resonator
network = ota.solver.NetworkTopology()
network.add_segment(ota.segments.Duct(length=1.0, radius=0.05))

# Solve for resonant frequency
solver = ota.solver.ShootingSolver(network, gas)
result = solver.solve(
    guesses={'p1_amplitude': 1e4, 'frequency': 500},
    targets={'U1_end_real': 0, 'U1_end_imag': 0},
    options={'T_m_start': 300}
)

print(f"Resonant frequency: {result.frequency:.2f} Hz")
```

## Visualization

Install with visualization extras:
```bash
pip install -e ".[viz]"
```

Plot global profiles with optional segment boundaries and save PNGs:
```python
import openthermoacoustics as ota

gas = ota.gas.Helium(mean_pressure=3e6)
network = ota.solver.NetworkTopology()
network.add_segment(ota.segments.Duct(length=1.0, radius=0.05))

solver = ota.solver.ShootingSolver(network, gas)
result = solver.solve(
    guesses={"frequency": 500.0, "U1_imag": 0.0},
    targets={"U1_end_real": 0, "U1_end_imag": 0},
    options={"T_m_start": 300.0},
)

ota.viz.plot_profiles(
    result,
    segment_results=network.results,
    save_path="profiles.png",
)
ota.viz.plot_phasor_profiles(
    result,
    segment_results=network.results,
    save_path="phasors.png",
)

# Generic sweep plotting utility
import numpy as np
freqs = np.linspace(200, 800, 25)
vals = (freqs - result.frequency) ** 2
ota.viz.plot_frequency_sweep(
    freqs,
    vals,
    ylabel="Synthetic residual",
    save_path="sweep.png",
)
```

Troubleshooting:
- If `pip install -e ".[viz]"` succeeds but `python` cannot import matplotlib, install with the same interpreter: `python -m pip install -e ".[viz]"`.
- Do not include `HardEnd`/`SoftEnd` segments in `NetworkTopology` for this solver path. Enforce end conditions through solver `targets` instead.

## Features

### Gas Properties

Ideal gas models with temperature-dependent transport properties:
- **Helium** - monatomic, γ = 5/3
- **Argon** - monatomic, γ = 5/3
- **Nitrogen** - diatomic, γ = 1.4
- **Air** - diatomic, γ = 1.4 (Sutherland's law for viscosity)

### Pore Geometries

Thermoviscous functions f_ν and f_κ for:
- **CircularPore** - Bessel function solution
- **ParallelPlate** - tanh solution
- **WireScreen** - mesh/screen approximation

### Network Segments

- **Duct** - uniform circular tube
- **Cone** - linearly tapered tube
- **Stack** - porous medium with temperature gradient
- **HeatExchanger** - fixed temperature boundary
- **Compliance** - lumped acoustic volume
- **Inertance** - lumped acoustic mass
- **HardEnd** - closed boundary (U1 = 0)
- **SoftEnd** - open boundary (p1 = 0)

### Solver

Shooting method using scipy.optimize.root to find operating conditions that satisfy boundary constraints.

## Core Equations

The solver propagates the state vector [p₁, U₁] through each segment using:

**Momentum equation:**
```
dp₁/dx = -(jωρₘ / A(1 - fᵥ)) · U₁
```

**Continuity equation:**
```
dU₁/dx = -(jωA / ρₘa²) · [1 + (γ-1)fκ] · p₁ + (fκ - fᵥ)/((1-fᵥ)(1-σ)) · (dTₘ/dx / Tₘ) · U₁
```

## References

1. Swift, G.W. (2017). *Thermoacoustics: A Unifying Perspective for Some Engines and Refrigerators*, 2nd ed. Springer.
2. Swift, G.W. (1988). "Thermoacoustic engines." J. Acoust. Soc. Am. 84(4), 1145–1180.
3. Rott, N. (1969). "Damped and thermally driven acoustic oscillations in wide and narrow tubes." Z. Angew. Math. Phys. 20, 230–243.

## Development

Run tests:
```bash
pytest tests/ -v
```

## Publishing

This repository includes a GitHub Actions release workflow at
`/Users/kelly/projects/personal/deltaec/.github/workflows/publish.yml`.

- Manual run (`workflow_dispatch`) publishes to **TestPyPI**
- GitHub Release `published` event publishes to **PyPI**

### One-time setup (Trusted Publisher)

1. In PyPI and TestPyPI, create a Trusted Publisher for this GitHub repo.
2. Map environments:
   - `pypi` environment -> PyPI trusted publisher
   - `testpypi` environment -> TestPyPI trusted publisher
3. In GitHub repo settings, create environments `pypi` and `testpypi`
   (optionally require approvals).

### Release flow

1. Bump version in `pyproject.toml`.
2. Merge to `main`.
3. Run workflow manually to TestPyPI and verify install.
4. Create GitHub release (for example `v0.1.1`) to publish to PyPI.

Documentation quality gate for changes:
- If a change affects public API or behavior, update matching docs in `docs/`.
- At minimum, update the relevant `docs/api/*` page and one usage guide (`getting-started`, `modeling-workflows`, or `troubleshooting`) when applicable.
- If validation behavior changed, update `docs/validation/index.md` and `docs/validation/example-matrix.md`.
- See the full checklist in [docs/contributing-docs.md](docs/contributing-docs.md).

## License

MIT License

# Getting Started

## Purpose

Get from a clean checkout to a successful first solve and optional visualization.

## Prerequisites

- Python 3.10+
- A virtual environment
- Repository checkout at `/Users/kelly/projects/personal/reference_baseline`

## Install

```bash
cd /Users/kelly/projects/personal/reference_baseline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[viz]"
python -m pip install -e ".[config]"
```

## First Solve

```python
import openthermoacoustics as ota

gas = ota.gas.Helium(mean_pressure=101325.0)

network = ota.solver.NetworkTopology()
network.add_segment(ota.segments.Duct(length=1.0, radius=0.025))

solver = ota.solver.ShootingSolver(network, gas)
result = solver.solve(
    guesses={"frequency": 500.0, "U1_imag": 0.0},
    targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
    options={"T_m_start": 300.0},
)

print(result.converged, result.frequency, result.residual_norm)
```

Expected behavior:
- `result.converged` is typically `True` for this simple case.
- `frequency` and profiles are populated in `result`.

## First Plot

```python
ota.viz.plot_profiles(
    result,
    segment_results=network.results,
    save_path="profiles.png",
    show=False,
)
```

## Key Notes

- Use `python -m pip ...` to ensure install and runtime interpreter match.
- For this solver path, enforce boundary conditions via `targets`.
- Do not add `HardEnd`/`SoftEnd` directly into `NetworkTopology` for the integrated propagation path.

## Next Reading

- [Concepts](./concepts.md)
- [Modeling Workflows](./modeling-workflows.md)
- [Troubleshooting](./troubleshooting.md)


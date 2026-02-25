# Modeling Workflows

## Purpose

Provide repeatable modeling patterns for common OpenThermoacoustics tasks.

## Workflow 1: Single-Duct Resonator

Use when:
- You need a baseline resonant behavior check.

```python
import openthermoacoustics as ota

gas = ota.gas.Helium(mean_pressure=101325.0)
network = ota.solver.NetworkTopology()
network.add_segment(ota.segments.Duct(length=1.0, radius=0.025))

result = ota.solver.ShootingSolver(network, gas).solve(
    guesses={"frequency": 500.0, "U1_imag": 0.0},
    targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
    options={"T_m_start": 300.0},
)
```

## Workflow 2: Engine-Style Chain with Stack and Heat Exchanger

Use when:
- You want physically meaningful thermoacoustic gain/loss behavior.

```python
import openthermoacoustics as ota
from openthermoacoustics.geometry import ParallelPlate

gas = ota.gas.Helium(mean_pressure=3.0e6)
network = ota.solver.NetworkTopology()
network.add_segment(ota.segments.Duct(length=0.3, radius=0.02))
network.add_segment(
    ota.segments.HeatExchanger(
        length=0.01,
        porosity=0.7,
        hydraulic_radius=2.0e-4,
        temperature=300.0,
        area=1.2e-3,
    )
)
network.add_segment(
    ota.segments.Stack(
        length=0.08,
        porosity=0.72,
        hydraulic_radius=1.8e-4,
        area=1.2e-3,
        geometry=ParallelPlate(),
        T_cold=300.0,
        T_hot=500.0,
    )
)

result = ota.solver.ShootingSolver(network, gas).solve(
    guesses={"frequency": 120.0, "U1_imag": 0.0},
    targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
    options={"T_m_start": 300.0},
)
```

## Workflow 3: Config-Driven Modeling

Use when:
- You want declarative models in JSON/YAML.

```python
from openthermoacoustics.config import run_from_config

result = run_from_config("example.json")
print(result.frequency, result.converged)
```

Minimal JSON shape:

```json
{
  "gas": {"type": "helium", "mean_pressure": 101325.0},
  "segments": [
    {"type": "duct", "length": 1.0, "radius": 0.025}
  ],
  "solver": {
    "guesses": {"frequency": 500.0, "U1_imag": 0.0},
    "targets": {"U1_end_real": 0.0, "U1_end_imag": 0.0},
    "options": {"T_m_start": 300.0}
  }
}
```

## Workflow 4: Visualization and Export

Use when:
- You need inspectable output for reports or debugging.

```python
import numpy as np
import openthermoacoustics as ota

ota.viz.plot_profiles(result, segment_results=network.results, save_path="profiles.png")
ota.viz.plot_phasor_profiles(result, segment_results=network.results, save_path="phasors.png")

freqs = np.linspace(200, 800, 25)
vals = (freqs - result.frequency) ** 2
ota.viz.plot_frequency_sweep(freqs, vals, ylabel="Synthetic residual", save_path="sweep.png")
```

## Workflow 5: Validation Against Known Cases

Use when:
- You need confidence from repository-safe regression checks.

Run selected scripts from `examples/`, for example:

```bash
python examples/test_shooting_method.py
python examples/validation/validate_interface_conditions.py
python examples/validation/validate_radiation_branches.py
```

## Common Modeling Rules

- Keep guesses/targets dimensionality equal.
- Start simple; add segment complexity incrementally.
- Use explicit `area` on stack/HX-like segments in realistic cases.
- Use stable temperature boundary assumptions for stacks where appropriate.

## Next Reading

- [Concepts](./concepts.md)
- [Troubleshooting](./troubleshooting.md)

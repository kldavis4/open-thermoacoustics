# Engine API

## Module

`openthermoacoustics.engine`

## Primary Class

`Network`

Purpose:
- High-level builder around `NetworkTopology` + `ShootingSolver`.

Core methods:
- `add(segment)`
- `solve(...)`
- `solve_closed_closed(...)`
- `solve_closed_open(...)`

Key solve parameters:
- `p1_amplitude`, `p1_phase`
- `frequency` and `solve_frequency`
- `targets` (for end constraints)
- `T_m_start`, `tol`, `maxiter`, `method`

Example:

```python
import openthermoacoustics as ota

gas = ota.gas.Helium(mean_pressure=101325.0)
engine = ota.Network(gas=gas, frequency_guess=500.0)
engine.add(ota.segments.Duct(length=1.0, radius=0.025))

result = engine.solve(
    p1_amplitude=1e4,
    frequency=500.0,
    solve_frequency=False,
    targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
)
```

Notes:
- Use explicit real/imag targets where possible.
- For this solver path, prefer boundary conditions through targets.


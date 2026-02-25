# Solver API

## Module

`openthermoacoustics.solver`

## Public Exports

- `integrate_segment`
- `NetworkTopology`
- `SegmentResult`
- `ShootingSolver`
- `SolverResult`
- `LoopNetwork`
- `TBranchLoopSolver`
- `TBranchLoopResult`
- `solve_lrc1_loop`

## `NetworkTopology`

Purpose:
- Assemble ordered segment chains and propagate acoustic state.

Common methods:
- `add_segment(segment)`
- `propagate_all(...)`
- `get_global_profiles()`
- `get_endpoint_values()`

## `ShootingSolver`

Purpose:
- Solve unknown parameters by matching outlet targets via root finding.

Entry point:
- `solve(guesses, targets, options=None)`

Rule:
- number of `guesses` must equal number of `targets`.

Example:

```python
import openthermoacoustics as ota

gas = ota.gas.Helium(mean_pressure=101325.0)
network = ota.solver.NetworkTopology()
network.add_segment(ota.segments.Duct(length=1.0, radius=0.025))

solver = ota.solver.ShootingSolver(network, gas)
result = solver.solve(
    guesses={"frequency": 500.0, "U1_imag": 0.0},
    targets={"U1_end_real": 0.0, "U1_end_imag": 0.0},
    options={"T_m_start": 300.0, "maxiter": 100},
)
```

## `SolverResult`

Key fields:
- `frequency`, `omega`
- `x_profile`, `p1_profile`, `U1_profile`, `T_m_profile`, `acoustic_power`
- `converged`, `message`, `n_iterations`, `residual_norm`
- `guesses_final`

## Loop Solvers

Use when modeling looped networks and branch-closure constraints:
- `LoopNetwork`
- `TBranchLoopSolver`
- `solve_lrc1_loop`

Reference examples:
- `examples/test_shooting_method.py`
- `examples/validation/validate_interface_conditions.py`

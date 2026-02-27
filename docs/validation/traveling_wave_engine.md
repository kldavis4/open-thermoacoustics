# Traveling-Wave Engine Validation (Distributed Loop)

## Scope

This case connects `TBranchLoopSolver` to the distributed segment integrator so
looped topologies can be solved with real thermoacoustic segments (ducts, heat
exchangers, regenerator/stack with gradient), not only lumped impedances.

Topology solved:

`TBRANCH -> trunk(resonator -> HARDEND)` and `TBRANCH -> branch(AHX1 -> regenerator -> HHX -> TBT -> AHX2 -> feedback duct) -> UNION`

The loop solver shoots on:

- `|U1|`, `phase(U1)`, `Re(Zb)`, `Im(Zb)`

to satisfy:

- `Re/Im(pressure mismatch at UNION) = 0`
- `Re/Im(U1 at HARDEND) = 0`

## Files

- Bridge module: `src/openthermoacoustics/solver/distributed_loop.py`
- Loop solver: `src/openthermoacoustics/solver/tbranch_loop_solver.py`
- Validation API: `src/openthermoacoustics/validation/traveling_wave_engine.py`
- Example: `examples/traveling_wave_engine.py`
- Optimization sweep: `scripts/optimize_traveling_wave.py`
- Tests: `tests/test_distributed_loop.py`, `tests/test_traveling_wave_engine.py`

## Current Status

- Minimum viable distributed-loop solve is implemented and tested.
- Fixed-frequency solves converge robustly in baseline sweeps (`50..250 Hz`).
- Best residual frequency region can be identified from sweep.
- Regenerator phase is reported as a diagnostic for traveling-wave character.
- A first-pass temperature sweep API is added with a net-gain proxy:
  `regenerator power delta - other segment losses`.
- A proxy onset detector is available from gain-proxy zero crossing.
- A tuned candidate configuration is included for regression tracking and
  currently shows proxy onset in a low ratio band (~`1.18-1.20` in coarse sweeps).

## Notes

- This is the plumbing milestone (distributed propagation + loop shooting).
- Full onset mapping for the traveling-wave engine with complex-frequency growth
  criterion is the next milestone and builds directly on this bridge.

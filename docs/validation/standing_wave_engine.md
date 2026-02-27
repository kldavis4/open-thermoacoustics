# Standing-Wave Engine Validation

## Scope

This case validates a canonical standing-wave thermoacoustic prime-mover layout:

`[Duct_L] -> [Cold HX] -> [Stack] -> [Hot HX] -> [Duct_R]`

with closed-end velocity boundary conditions enforced through the shooting targets
(`U1_end_real = 0`, `U1_end_imag = 0`).

Primary onset criterion in this validation is the **complex-frequency crossing**:
- damped mode: `f_imag > 0`
- growing mode (above onset): `f_imag < 0`
- onset: `f_imag` crosses through zero

`dW_stack` is retained as a secondary diagnostic trend only.

### Why `f_imag` Is Primary

For closed-closed standing-wave resonators, local acoustic-power differences can
be misleading because power flow redistributes with standing-wave phase and can
change sign locally even when the global mode remains damped. The complex-frequency
growth rate `f_imag` directly measures modal damping/growth and is therefore the
physically correct onset criterion for this benchmark.

## Configurations

- Gas: helium, mean pressure `1.0 MPa`
- Geometry:
  - Left duct: `L = 0.30 m`, `r = 0.035 m`
  - Cold HX: `L = 0.02 m`, fixed `T = 300 K`
  - Stack: `L = 0.10 m`, parallel-plate pores, linear `T(x)` from `300 K` to `T_hot`
  - Hot HX: `L = 0.02 m`, fixed `T = T_hot`
  - Right duct: `L = 0.30 m`, `r = 0.035 m`
- Stack matrix:
  - Plate spacing `2*y0 = 0.8 mm` (`y0 = 0.4 mm`)
  - Plate thickness `0.1 mm`
  - Porosity `0.8 / 0.9 = 0.8889`
- Sweep: `T_hot = 300..800 K`

### Controls

- Symmetric control: `left/right = 0.30/0.30 m` (no onset <= 800 K expected)
- Shifted control: `left/right = 0.10/0.50 m` (no onset <= 800 K expected)
- Geometry-sensitive reference: `left/right = 0.45/0.15 m`, with onset near
  `890 K` (`ratio ~2.97`)

### Optimized Benchmark (Current Primary)

- Left duct: `1.00 m`
- Right duct: `0.20 m`
- Cold HX / Stack / Hot HX: unchanged (`0.02 / 0.10 / 0.02 m`)
- Plate spacing: `1.2 mm` (`y0 = 0.6 mm`)
- Plate thickness: `0.1498 mm` (porosity `~0.889`)
- Mean pressure: `1.0 MPa`
- Complex-frequency onset crossing: `T_hot ≈ 603.8 K`
- Onset ratio: `T_hot/T_cold ≈ 2.013`

## Files

- Example script: `examples/standing_wave_engine.py`
- Test suite: `tests/test_standing_wave_engine.py`
- Reusable API: `src/openthermoacoustics/validation/standing_wave_engine.py`
- Optimization workflow: `scripts/optimize_onset.py`

## Validation Checks

The test suite implements:

1. Isothermal resonant frequency against `a/(2*L_total)` (15% bound).
2. Complex-frequency onset detection (`f_imag` sign crossing) on optimized benchmark.
3. Symmetric negative control (no onset below 800 K).
4. Shifted negative control (`left_duct=0.10 m`, `right_duct=0.50 m`) no onset below 800 K.
5. Geometry-sensitive reference onset near `890 K`.
6. Above-onset margin check (`f_imag < 0` at onset+50 K for benchmark).
7. Frequency shift monotonicity with increasing `T_hot` in baseline.
8. Power-profile reasonableness (finite and bounded).

## Current Result Summary

- Frequency trend is stable and monotonic with temperature.
- Isothermal resonance is within the expected engineering tolerance.
- Symmetric baseline (0.30/0.30 ducts): no onset crossing below 800 K
  (kept as intentional negative control).
- Geometry-sensitive reference (0.45/0.15 ducts): onset near `890 K`,
  confirming stack-position sensitivity.
- Optimized benchmark now crosses onset near `603.8 K` (`ratio ~2.01`), within the
  target standing-wave validation window (`1.3..2.5`).
- Symmetric and 0.10/0.50 controls remain damped through `800 K`.

## Implementation Notes

- The integrator was updated to:
  - call geometry thermoviscous functions using the repository API
    (`f_nu(omega, delta_nu, hydraulic_radius)` and `f_kappa(...)`)
  - use segment-provided local temperature models (`temperature_at` / fixed `temperature`)
- These changes are required for this standing-wave configuration to run with
  parallel-plate geometry and fixed-temperature heat exchangers.
- A complex-frequency solver path was added to provide a physically grounded onset
  metric independent of acoustic-power proxy sign conventions.
- `scripts/optimize_onset.py` performs the full multi-phase search:
  stack position, `y0`, resonator length, and fine onset interpolation, and writes
  plots to `examples/output/`.

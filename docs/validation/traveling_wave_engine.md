# Traveling-Wave Engine Validation (Distributed Loop)

## Scope

This benchmark connects `TBranchLoopSolver` to distributed segment integration
for looped topologies:

`TBRANCH -> trunk(resonator -> HARDEND)` and  
`TBRANCH -> branch(AHX1 -> regenerator -> HHX -> TBT -> AHX2 -> feedback) -> UNION`

The current solver uses a 4x4 shooting system:

- Unknowns: `|U1|`, `phase(U1)`, `Re(Zb)`, `Im(Zb)`
- Targets: `Re/Im(p_mismatch at UNION)=0`, `Re/Im(U1_hardend)=0`

Onset is currently reported with a tightened gain proxy, not complex-frequency
zero crossing. This is intentional for now and documented below.

## Key Outputs

Implemented in `src/openthermoacoustics/validation/traveling_wave_engine.py`:

- `find_onset_ratio_proxy`: coarse+fine gain-proxy onset ratio
- `compute_regenerator_phase_profile`: inlet/mid/outlet phase diagnostics
- `compute_loop_power_profile`: boundary acoustic powers by segment
- `compute_efficiency_estimate`: first-order efficiency estimate with Carnot bound
- `sweep_efficiency_estimate`: temperature sweep of efficiency metrics
- `estimate_loop_frequency_range`: quarter-wave + loop-loading sanity range

## Tuned Candidate (Current Primary TW Regression Case)

- `mean_pressure = 4.0 MPa`
- `resonator_length = 0.8 m`
- `feedback_radius = 0.03 m`
- `feedback_length = 0.5 m`
- `tbt_length = 0.25 m`
- `regenerator_hydraulic_radius = 0.12 mm`

Observed (proxy method):

- Onset ratio: `~1.16` at `120 Hz` (`T_hot/T_cold`)
- Regenerator phase at `T_hot=600 K`: approximately `-104°` (inlet),
  `-96°` (mid), `-93°` (outlet)
- Net gain proxy at `600 K`: positive

Interpretation:

- The configuration clearly outperforms the standing-wave benchmark on onset
  ratio.
- Phase is still closer to standing-wave-like than ideal traveling-wave (`~0°`),
  so this is a promising but not fully optimized traveling-wave state.

## Efficiency and Power Budget

The current efficiency estimate is a conservative first-order metric:

- `eta_carnot = 1 - T_cold/T_hot`
- `W_regen = Delta W` across regenerator
- `W_useful = -Delta W` in resonator section (non-negative clipped)
- `eta_thermal_est = eta_carnot * (W_useful / W_regen)` clipped to `[0, eta_carnot]`

This keeps second-law consistency by construction and is useful for relative
screening during geometry sweeps. It is not a full second-order enthalpy-flux
efficiency model.

## Comparison to Published Ranges

| Metric | This benchmark | Literature context |
|---|---:|---:|
| Onset ratio `T_hot/T_cold` | ~1.16 (proxy) | ~1.06 to 1.22 reported for optimized TW systems |
| Frequency | solver sweep around 120 Hz | order-of-magnitude consistent with low-100 Hz TW engines |
| Regenerator phase(p,u) | around -90° to -105° | ideal TW behavior is near 0° |

Caveat: this geometry is not a direct replica of Backhaus & Swift hardware, so
comparisons are qualitative/dimensionless.

## Status and Next Step

- Distributed-loop plumbing is complete and tested.
- Power/phase/efficiency diagnostics are in place for optimization loops.
- Next milestone: add true complex-frequency loop onset (`f_imag` zero crossing)
  to replace gain-proxy onset as the primary criterion.

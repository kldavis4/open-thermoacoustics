# Traveling-Wave Engine Validation (Distributed Loop)

## Scope

This benchmark connects `TBranchLoopSolver` to distributed segment integration
for looped topologies:

`TBRANCH -> trunk(resonator -> HARDEND)` and  
`TBRANCH -> branch(AHX1 -> regenerator -> HHX -> TBT -> AHX2 -> feedback) -> UNION`

The base loop closure uses a 4x4 shooting system:

- Unknowns: `|U1|`, `phase(U1)`, `Re(Zb)`, `Im(Zb)`
- Targets: `Re/Im(p_mismatch at UNION)=0`, `Re/Im(U1_hardend)=0`

Onset now supports both:

- Gain-proxy crossing (`net_gain_proxy = 0`)
- Complex-frequency crossing (`f_imag = 0`)
- Determinant-based complex-frequency crossing (`det(M)=0`)

## Key Outputs

Implemented in `src/openthermoacoustics/validation/traveling_wave_engine.py`:

- `find_onset_ratio_proxy`: coarse+fine gain-proxy onset ratio
- `solve_traveling_wave_engine_complex_frequency`: complex-frequency solve
- `sweep_traveling_wave_complex_frequency`: temperature continuation sweep
- `compute_trunk_transfer_matrix` / `compute_branch_transfer_matrix`: per-path
  linear transfer matrices
- `build_boundary_matrix`: 3x3 loop-closure matrix `M(omega)`
- `solve_traveling_wave_engine_determinant_complex_frequency`: 2x2
  determinant solve for `(f_real, f_imag)`
- `sweep_traveling_wave_determinant_complex_frequency`: determinant-based
  continuation sweep
- `sweep_traveling_wave_determinant_complex_frequency_multimode`: strict
  determinant branch tracking across multiple seeded modes with physical
  signature constraints
- `evaluate_traveling_wave_boundary_determinant`: determinant diagnostics at a
  specified complex frequency
- `recover_mode_shape`: null-space mode reconstruction from `M(omega*)`
- `detect_onset_from_complex_frequency`: onset ratio from `f_imag` crossing
  (with configurable deadband tolerance for numerical noise)
  and optional residual-quality filtering.
- `sweep_traveling_wave_complex_frequency_multimode`: runs multiple anchored
  branches and selects the best candidate branch.
- `summarize_multimode_selection`: compact selected-vs-alternate branch report.
- `compute_regenerator_phase_profile`: inlet/mid/outlet phase diagnostics
- `compute_loop_power_profile`: boundary acoustic powers by segment
- `compute_net_acoustic_power`: net acoustic production/dissipation from full
  branch+trunk power deltas
- `compute_stored_energy`: first-order stored acoustic energy estimate from
  distributed profiles
- `compute_energy_balance_growth_rate`: growth-rate estimate
  `f_imag ~= -W_net/(4*pi*E_stored)`
- `sweep_energy_balance_growth_rate`: temperature sweep of
  `W_net`, `E_stored`, and energy-balance `f_imag`
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

Complex-frequency formulation details (augmented 5x5):

- Unknowns: `f_real`, `f_imag`, `|U1_input|`, `Re(Zb)`, `Im(Zb)`
- Fixed gauge:
  `p1_input = p_norm + 0j` (phase convention),
  `phase(U1_input)` from real-frequency reference
- Targets:
  `Re/Im(p_mismatch)=0`,
  `Re/Im(U1_hardend)=0`,
  `Re(p1_trunk_end)-p_norm=0` (normalization)
- Continuation:
  sweep in `T_hot`, re-solving real-frequency reference each step for phase,
  and seeding `(f_real, f_imag, |U1|, Zb)` from prior point.
- Mode validation in sweep:
  candidate branches are scored not only by residual/frequency continuity, but
  also by regenerator phase consistency and loop power-flow sign consistency
  relative to an identified anchor mode at the first sweep point.

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

Energy-balance onset diagnostics are also available from real-frequency 4x4
solutions:

- `W_net = sum(Delta W_segment)` (branch + trunk)
- `E_stored` from pressure and kinetic first-order energy integrals
- `f_imag,energy ~= -W_net/(4*pi*E_stored)`

Interpretation with the existing sign convention:

- `f_imag,energy > 0`: decaying mode (below onset)
- `f_imag,energy = 0`: onset
- `f_imag,energy < 0`: growing mode (above onset)

## Comparison to Published Ranges

| Metric | This benchmark | Literature context |
|---|---:|---:|
| Onset ratio `T_hot/T_cold` | ~1.16 (proxy) | ~1.06 to 1.22 reported for optimized TW systems |
| Frequency | solver sweep around 120 Hz | order-of-magnitude consistent with low-100 Hz TW engines |
| Regenerator phase(p,u) | around -90° to -105° | ideal TW behavior is near 0° |

Caveat: this geometry is not a direct replica of Backhaus & Swift hardware, so
comparisons are qualitative/dimensionless.

## Determinant Formulation

For loop eigenmodes we now support a normalization-free formulation:

- Build trunk and branch transfer matrices:
  `[p_out, U_out]^T = T(omega) [p_in, U_in]^T`
- Assemble boundary matrix:
  `M(omega) * [p1, U_trunk, U_branch]^T = 0`
- Solve `det(M)=0` as a 2-real-equation root in `(f_real, f_imag)`.

This removes the 5x5 normalization ambiguity and gives a direct eigenvalue
criterion. The existing 4x4 and 5x5 solvers are still available for
cross-checking and branch-selection diagnostics.

Strict branch-identification layer:

- Runs multiple determinant branches per temperature point.
- Scores candidates by residual, frequency continuity, and two physical
  signatures:
  regenerator phase consistency and branch/trunk dominant power-flow sign.
- Locks the selected branch to the identified physical mode instead of relying
  only on frequency continuity penalties.

Complex-frequency sensitivity diagnostic:

- `scripts/determinant_landscape.py` generates:
  - `examples/output/det_landscape_600K.png`
  - `examples/output/transfer_matrix_sensitivity.txt`
- Integrator now uses full complex `omega` in penetration depths
  (`delta_nu`, `delta_kappa`) for thermoviscous evaluation, so determinant
  and transfer matrices respond to nonzero `f_imag`.

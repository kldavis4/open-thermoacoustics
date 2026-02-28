# Standing-Wave Refrigerator Validation

## Scope

This benchmark validates driven refrigerator-mode behavior using the same
distributed thermoacoustic segment physics used for engine benchmarks.

Topology:

`[drive] -> left duct -> hot HX -> stack -> cold HX -> right duct -> [hard end]`

Gas and operating point:

- Helium, `p_m = 1.0 MPa`
- `T_hot = 300 K`
- `T_cold = 270 K` (default benchmark point)
- Drive ratio `|p1|/p_m = 0.03`

## Method

The solver uses a fixed-amplitude driven standing-wave mode:

- Solve for resonance with shooting unknowns: `frequency`, `p1_phase`
- Targets: `Re(U1_end)=0`, `Im(U1_end)=0`
- Scale the linear solution to the requested drive pressure

Refrigerator performance now uses an acoustic-power short-stack path as the
primary metric:

- Solve mode shape with full distributed segment model
- Compute stack acoustic input from solved boundary powers:
  `W_stack = W_hot_side - W_cold_side`
- Compute short-stack cooling from a Tijani-style `Γ` relation evaluated at
  stack center using solved mode quantities
- Compute `COP_stack = Q_cold_short_stack / W_stack`

The direct H2 boundary formulation is still computed and returned, but treated
as diagnostic/experimental because it is not yet energy-conservative across the
stack in this implementation.

Diagnostic H2 components:

- `H2 = E_dot + H_streaming + Q_conduction`
- `E_dot = 0.5 * Re(p1 * conj(U1))`
- `H_streaming` uses thermoviscous `f_nu`, `f_kappa`, Prandtl number, and
  imposed stack gradient `dT/dx`
- `Q_conduction = -(k_gas * A_gas + k_solid * A_solid) * dT/dx`

Primary benchmark outputs:

- `Q_cold = Q_cold_short_stack`
- `W_input = W_stack` (acoustic power absorbed by stack)
- `COP = Q_cold / W_input`
- `COP_Carnot = T_cold / (T_hot - T_cold)`
- `COP_relative = COP / COP_Carnot`

Additional diagnostics returned:

- `cooling_power_h2` (H2 boundary difference)
- `cooling_power_proxy` (legacy acoustic proxy)
- full per-segment acoustic power table

## Current Benchmark Behavior

At default settings (`T_hot=300 K`, `T_cold=270 K`, drive ratio `3%`):

- Resonance is in the expected low-kHz range for the current compact geometry.
- Selected short-stack cooling power is positive.
- COP stays below Carnot.
- Cooling power scales approximately as drive-ratio squared.
- H2 diagnostic values are available but not used for pass/fail metrics.

## API

Implemented in:
`/Users/kelly/projects/personal/deltaec/src/openthermoacoustics/validation/standing_wave_refrigerator.py`

- `StandingWaveRefrigeratorConfig`
- `default_standing_wave_refrigerator_config`
- `tijani_refrigerator_config` (Tijani et al. approximate benchmark)
- `build_standing_wave_refrigerator_network`
- `solve_standing_wave_refrigerator`
- `compute_cooling_power`
- `compute_refrigerator_cop`
- `compute_refrigerator_performance_short_stack`
- `tijani_cooling_power_short_stack`
- `tijani_acoustic_power_short_stack`
- `sweep_drive_ratio`
- `sweep_cold_temperature`

## Tijani Approximate Validation

Reference target: Tijani et al. (Cryogenics 42, 2002), helium at 10 bar,
~400 Hz, ~2% drive.

Implemented approximation keeps the stack geometry/operating point and uses a
uniform-radius half-wave resonator for compatibility with the current topology:

- `p_m = 1.0 MPa`
- `T_hot = 287.5 K`, `T_cold = 212.5 K`
- `drive_ratio = 0.02`
- stack: `L = 0.085 m`, `2y0 = 0.3 mm`, porosity `~0.833`
- tube radius `0.019 m`
- resonator length tuned via geometry to produce frequency near 400 Hz
- short-stack cooling is evaluated from solved mode shape + stack acoustic input
  power; H2 boundary value is reported for diagnostics

Validation checks:

- resonant frequency within 20% of 400 Hz
- `COP_stack` within broad published-consistent bounds
- `Q_cold` scales approximately as `D^2` at small drive
- `COP` decreases with increasing temperature lift `ΔT`

Representative result at Tijani nominal point (`T_hot=287.5 K`,
`T_cold=212.5 K`, `D=2%`):

- `f ≈ 427.3 Hz`
- `W_stack ≈ 3.30 W`
- `Q_cold_short_stack ≈ 4.19 W` (published design target ~4 W)
- `COP_short_stack ≈ 1.27` (published short-stack theory ~1.3)
- `Q_cold_h2_boundary ≈ 4.00 W` (diagnostic)

Example script:
`/Users/kelly/projects/personal/deltaec/examples/standing_wave_refrigerator_tijani.py`

Outputs:

- `/Users/kelly/projects/personal/deltaec/examples/output/tijani_diagnostic.txt`
- `/Users/kelly/projects/personal/deltaec/examples/output/tijani_Qcold_vs_drive.png`
- `/Users/kelly/projects/personal/deltaec/examples/output/tijani_Qcold_scaling.png`
- `/Users/kelly/projects/personal/deltaec/examples/output/tijani_cop_vs_drive.png`
- `/Users/kelly/projects/personal/deltaec/examples/output/tijani_cop_vs_deltaT.png`

## Example Outputs

`/Users/kelly/projects/personal/deltaec/examples/standing_wave_refrigerator.py`
generates:

- `/Users/kelly/projects/personal/deltaec/examples/output/sw_refrigerator_cop_vs_drive.png`
- `/Users/kelly/projects/personal/deltaec/examples/output/sw_refrigerator_cooling_power_vs_delta_T.png`
- `/Users/kelly/projects/personal/deltaec/examples/output/sw_refrigerator_cop_vs_delta_T.png`

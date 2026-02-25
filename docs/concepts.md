# Concepts

## Purpose

Define the modeling conventions and solver mechanics used throughout OpenThermoacoustics.

## Core State and Variables

- Oscillatory quantities use complex phasors.
- Typical propagated state includes:
  - `p1`: complex pressure amplitude (Pa)
  - `U1`: complex volumetric velocity amplitude (m^3/s)
  - `T_m`: mean temperature (K)
- Position coordinate: `x` (m)
- Angular frequency: `omega = 2*pi*f` (rad/s)

## Units

SI units are used across the project:
- Pressure: Pa
- Length: m
- Temperature: K
- Volume flow: m^3/s
- Power: W

## Network Model

- A system is assembled as a sequence of segments in `NetworkTopology`.
- Segments are propagated in order to compute global profiles.
- `ShootingSolver` uses root-finding to satisfy outlet target conditions.

## Guesses and Targets

The shooting problem is formed with:
- `guesses`: unknown parameters to be adjusted
- `targets`: required outlet constraints

Important rule:
- Number of guesses must equal number of targets.

Typical closed-end target pair:
- `U1_end_real = 0`
- `U1_end_imag = 0`

## Boundary Conditions

Current practical usage for this solver path:
- Apply boundaries via solver `targets`, not by inserting boundary segments into integrated propagation chains.

## Thermoviscous Modeling

Gas transport and pore-geometry functions are used in segment ODEs:
- Gas models in `openthermoacoustics.gas`
- Geometry models in `openthermoacoustics.geometry`

These influence attenuation, phase, and power flow in ducts/stacks/HX segments.

## Output Interpretation

`SolverResult` provides:
- `frequency`, `omega`
- global profiles (`x_profile`, `p1_profile`, `U1_profile`, `T_m_profile`)
- `acoustic_power`
- convergence diagnostics (`converged`, `message`, `n_iterations`, `residual_norm`)

## Next Reading

- [Getting Started](./getting-started.md)
- [Modeling Workflows](./modeling-workflows.md)
- [Troubleshooting](./troubleshooting.md)


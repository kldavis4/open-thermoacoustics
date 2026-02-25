# Example Matrix

## Purpose

Map committed example scripts to intent, subsystem coverage, and practical use.

Policy:
- This matrix only includes repository-safe scripts.
- Scripts may contain embedded neutral reference baselines for regression checks.

## Workflow and Scenario Scripts (`examples/`)

| Script | Purpose | Primary Coverage | Typical Use |
|---|---|---|---|
| `standing_wave_resonator.py` | standing-wave baseline | `Duct`, solver | first sanity run |
| `quarter_wave_resonator.py` | quarter-wave behavior | solver targets | open/closed behavior check |
| `helmholtz_resonator.py` | lumped resonator behavior | `Compliance`, `Inertance` | low-order validation |
| `lossy_duct_attenuation.py` | attenuation trend | `Duct`, gas/geometry effects | loss model check |
| `standing_wave_engine_onset.py` | onset behavior | stack/HX/duct interplay | sensitivity study |
| `traveling_wave_engine.py` | traveling-wave scenario | multi-segment chain | system behavior exploration |
| `traveling_wave_stub.py` | traveling-wave prototype | topology + tuning | early design iteration |
| `diagnose_energy_equation.py` | energy-equation diagnostics | `StackEnergy` | thermal debugging |
| `test_shooting_method.py` | solver operation sanity | `ShootingSolver` | regression sanity |

## Validation Scripts (`examples/validation/`)

| Script | Purpose | Primary Coverage | Typical Use |
|---|---|---|---|
| `validate_acoustic_state.py` | state/derived quantity checks | `AcousticState` | API correctness |
| `validate_anchor.py` | thermal mode controls | `Anchor`, `Insulate` | thermal constraints |
| `validate_enclosed_transducer.py` | enclosed transducer behavior | `EnclosedTransducer` | electroacoustic checks |
| `validate_interface_conditions.py` | interface conventions | `NetworkTopology` | continuity behavior check |
| `validate_px.py` | power-law HX validation | `PX` | HX variant checks |
| `validate_radiation_branches.py` | radiation branch behavior | `OpenBranch`, `PistonBranch` | radiation impedance checks |
| `validate_side_branch_transducer.py` | side-branch transducers | `SideBranchTransducer` | branch source checks |
| `validate_stkpowerlw.py` | power-law regenerator validation | `StackPowerLaw` | regenerator behavior checks |
| `validate_vespeaker.py` | voltage-driven transducer behavior | `VESPEAKER` alias | source model check |
| `validate_vxq1.py` | variable heat-flux HX (1-pass) | `VXQ1` | heat-flux control checks |
| `validate_vxq2.py` | variable heat-flux HX (2-pass) | `VXQ2` | heat-flux control checks |
| `validate_vxt1.py` | variable-temperature HX (1-pass) | `VXT1` | thermal boundary checks |
| `validate_vxt2.py` | variable-temperature HX (2-pass) | `VXT2` | thermal boundary checks |

## Operational Guidance

1. Start with `test_shooting_method.py` and `validate_interface_conditions.py`.
2. Run only segment-family scripts relevant to your changes.
3. Use scenario scripts after segment-level checks pass.
4. Keep proprietary parity checks separate from repository-safe CI or release gates.

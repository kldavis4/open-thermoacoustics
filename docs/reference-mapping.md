# Reference Mapping

## Purpose

Map legacy segment names and concepts to OpenThermoacoustics classes for migration and parity work.

## Core Mapping

| Legacy Concept | OpenThermoacoustics Class/Alias | Notes |
|---|---|---|
| `DUCT` | `segments.Duct` | uniform tube |
| `CONE` | `segments.Cone` | tapered tube |
| `STKSLAB` | `segments.Stack` + `geometry.ParallelPlate` | parallel-plate stack |
| `STKRECT` | `segments.Stack` + `geometry.RectangularPore` | rectangular pore stack |
| `STKCIRC` | `segments.Stack` + `geometry.CircularPore` | circular pore stack |
| `STKPIN` | stack-like segment with `geometry.PinArray` | pin-array pore model |
| `STKSCREEN` | `segments.StackScreen` | wire-screen regenerator |
| `STKDUCT` | `segments.StackDuct` (`STKDUCT`) | duct+stack coupled segment |
| `STKCONE` | `segments.StackCone` (`STKCONE`) | tapered stack segment |
| `STKPOWERLW` | `segments.StackPowerLaw` (`STKPOWERLW`) | power-law regenerator |
| `HX` | `segments.HeatExchanger` | fixed solid-temperature model |
| `SX` | `segments.ScreenHeatExchanger` (`SX`) | screen HX |
| `TX` | `segments.TubeHeatExchanger` (`TX`) | tubular HX |
| `PX` | `segments.PowerLawHeatExchanger` (`PX`) | power-law HX |
| `COMPLIANCE` | `segments.Compliance` | lumped volume |
| `INERTANCE` | `segments.Inertance` | lumped mass |
| `IMPEDANCE` | `segments.Impedance` | imposed impedance |
| `MINOR` | `segments.Minor` (`MINOR`) | minor-loss element |
| `HARDEND` | `segments.HardEnd` | boundary marker |
| `SOFTEND` | `segments.SoftEnd` | boundary marker |
| `OPNEND` | `segments.OpenEnd` | open-end model |
| `JOIN` | `segments.Join` (`JOIN`) | topology join |
| `BRANCH` | `segments.ImpedanceBranch` (`BRANCH`) | branch with impedance |
| `TBRANCH` | `segments.TBranchImpedance` / `segments.TBranch` | solver context dependent |
| `UNION` | `segments.Union` | branch reconnection |
| `IESPEAKER` | `segments.IESPEAKER` alias to `segments.Transducer` | current-driven use pattern |
| `VESPEAKER` | `segments.VESPEAKER` alias to `segments.Transducer` | voltage-driven use pattern |
| `IDUCER/VDUCER` | `segments.IDUCER`, `segments.VDUCER` | side-branch transducer aliases |
| `IEDUCER/VEDUCER` | `segments.IEDUCER`, `segments.VEDUCER` | enclosed transducer aliases |
| `OPNBRANCH/PISTBRANCH` | `segments.OpenBranch`, `segments.PistonBranch` | radiation branches |
| `ANCHOR/INSULATE` | `segments.Anchor`, `segments.Insulate` | thermal mode control |
| `VXT1/VXT2` | `segments.VXT1`, `segments.VXT2` | variable temperature HX variants |
| `VXQ1/VXQ2` | `segments.VXQ1`, `segments.VXQ2` | variable heat flux HX variants |

## Migration Notes

1. Start migration with linear chains (`Duct`, `Cone`, `Stack`, `HeatExchanger`) before branch/loop features.
2. Match pore geometry explicitly (`ParallelPlate`, `RectangularPore`, `CircularPore`, `WireScreen`, `PinArray`).
3. For current solver path, impose boundaries with solver targets instead of boundary segment insertion in integrated chains.
4. Validate each migrated subsystem using corresponding scripts in `examples/validation/`.

## Verification Path

After mapping a model:
1. run nearest segment validation script
2. run scenario-level validation script
3. compare with current project validation expectations and script outputs

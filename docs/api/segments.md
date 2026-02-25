# Segments API

## Module

`openthermoacoustics.segments`

## Purpose

Catalog of segment classes used to build thermoacoustic networks.

## Core Segments

- `Segment` (base)
- `Duct`
- `Cone`
- `Stack`
- `StackEnergy`
- `HeatExchanger`
- `Compliance`
- `Inertance`
- `HardEnd`
- `SoftEnd`
- `OpenEnd`
- `Impedance`
- `Transducer`

## Branch, Join, and Topology Segments

- `Join`, `JOIN`
- `TBranch`, `Return`, `SideBranch`
- `TBranchImpedance`, `Union`, `SoftEndWithState`
- `ImpedanceBranch`, `BRANCH`
- `OpenBranch`, `OPNBRANCH`
- `PistonBranch`, `PISTBRANCH`

## reference baseline-Compatible Regenerator and HX Segments

- `StackScreen`
- `ScreenHeatExchanger`, `SX`
- `StackDuct`, `STKDUCT`
- `StackCone`, `STKCONE`
- `StackPowerLaw`, `STKPOWERLW`
- `PowerLawHeatExchanger`, `PX`
- `TubeHeatExchanger`, `TX`
- `Surface`, `SURFACE`
- `Minor`, `MINOR`

## reference baseline-Compatible Transducer and Thermal-Control Segments

- `SideBranchTransducer`, `IDUCER`, `VDUCER`
- `SideBranchSpeaker`, `ISPEAKER`, `VSPEAKER`
- `EnclosedTransducer`, `IEDUCER`, `VEDUCER`
- `IESPEAKER`, `VESPEAKER` (aliases to `Transducer`)
- `Anchor`, `ANCHOR`
- `Insulate`, `INSULATE`
- `ThermalMode`
- `VariableTemperatureHeatExchanger`, `VXT1`
- `VariableTemperatureHeatExchanger2Pass`, `VXT2`
- `VariableHeatFluxHeatExchanger`, `VXQ1`
- `VariableHeatFluxHeatExchanger2Pass`, `VXQ2`

## Basic Usage

```python
import openthermoacoustics as ota

network = ota.solver.NetworkTopology()
network.add_segment(ota.segments.Duct(length=0.5, radius=0.02))
network.add_segment(ota.segments.Cone(length=0.1, radius_in=0.02, radius_out=0.01))
```

## Practical Notes

- For many realistic models, explicitly specify cross-sectional `area` where supported.
- Some boundary or specialized segments are not appropriate for all solver paths.
- Prefer validating complex segment chains using matching scripts in `examples/validation/`.

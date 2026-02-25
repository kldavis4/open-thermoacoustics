# Gas API

## Module

`openthermoacoustics.gas`

## Public Exports

- `Gas`
- `Helium`
- `Air`
- `Argon`
- `Nitrogen`
- `Xenon`
- `GasMixture`
- `helium_argon`
- `helium_xenon`

## Usage

```python
import openthermoacoustics as ota

he = ota.gas.Helium(mean_pressure=3.0e6)
xe = ota.gas.Xenon(mean_pressure=3.0e6)
mix = ota.gas.helium_xenon(mean_pressure=3.0e6, mole_fraction_helium=0.7)
```

## Notes

- All gas models are designed for thermoacoustic calculations with SI units.
- Mean pressure is a required modeling input for density-dependent behavior.
- Mixture helpers are convenient entry points; use `GasMixture` for custom composition workflows.


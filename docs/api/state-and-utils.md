# State and Utils API

## Modules

- `openthermoacoustics.state`
- `openthermoacoustics.utils`
- `openthermoacoustics.viz`

## `AcousticState`

Class:
- `AcousticState`

Purpose:
- container and helper operations for acoustic/thermal state data.

## Utility Functions

From `openthermoacoustics.utils`:
- `specific_gas_constant(gas_name)`
- `penetration_depth_viscous(...)`
- `penetration_depth_thermal(...)`
- `acoustic_power(p1, U1)`
- `complex_to_state(p1, U1)`
- `state_to_complex(y)`
- `wavelength(frequency, sound_speed)`
- `wavenumber(frequency, sound_speed)`

## Visualization Functions

From `openthermoacoustics.viz`:
- `plot_profiles(...)`
- `plot_phasor_profiles(...)`
- `plot_frequency_sweep(...)`

Usage:

```python
import openthermoacoustics as ota

fig, axes = ota.viz.plot_profiles(result, segment_results=network.results, show=False)
```

PNG export:

```python
ota.viz.plot_profiles(result, segment_results=network.results, save_path="profiles.png")
```


# Geometry API

## Module

`openthermoacoustics.geometry`

## Public Exports

- `Geometry`
- `CircularPore`
- `ParallelPlate`
- `RectangularPore`
- `WireScreen`
- `PinArray`

## Purpose

Geometry classes provide thermoviscous correction behavior used by stack/regenerator/HX segment models.

## Usage

```python
from openthermoacoustics.geometry import ParallelPlate
from openthermoacoustics.segments import Stack

geom = ParallelPlate()
stack = Stack(
    length=0.08,
    porosity=0.72,
    hydraulic_radius=180e-6,
    area=1.134e-3,
    geometry=geom,
    T_cold=300.0,
    T_hot=217.0,
)
```

## Notes

- Choose geometry classes consistent with physical pore shape.
- Use explicit segment area for realistic system models.


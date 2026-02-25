# Config API

## Module

`openthermoacoustics.config`

## Public API

- `ConfigError`
- `load_config(filepath)`
- `save_config(network, filepath)`
- `parse_config(config_dict)`
- `run_from_config(filepath)`

## Supported Formats

- `.json`
- `.yaml` / `.yml` (requires optional YAML dependency)

## Minimal Config

```json
{
  "gas": {"type": "helium", "mean_pressure": 101325.0},
  "segments": [
    {"type": "duct", "length": 1.0, "radius": 0.025}
  ],
  "solver": {
    "guesses": {"frequency": 500.0, "U1_imag": 0.0},
    "targets": {"U1_end_real": 0.0, "U1_end_imag": 0.0},
    "options": {"T_m_start": 300.0}
  }
}
```

## Runtime Usage

```python
from openthermoacoustics.config import run_from_config

result = run_from_config("model.json")
print(result.frequency, result.converged)
```

## Important Behavior

- `solver.guesses.frequency` and `solver.guesses.p1_phase` are both forwarded when both are present.
- Validate types carefully; invalid fields raise `ConfigError`.


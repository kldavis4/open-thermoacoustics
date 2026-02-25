# Validation Guide

## Purpose

Document how to run and interpret validation scripts under `examples/` for physics checks and reference baseline parity tracking.

## Prerequisites

```bash
cd /Users/kelly/projects/personal/reference_baseline
source .venv/bin/activate
python -m pip install -e ".[dev,viz,config]"
```

## Quick Validation Runs

Public, repository-safe checks:

```bash
python examples/test_shooting_method.py
python examples/validation/validate_acoustic_state.py
python examples/validation/validate_anchor.py
python examples/validation/validate_enclosed_transducer.py
python examples/validation/validate_interface_conditions.py
python examples/validation/validate_px.py
python examples/validation/validate_radiation_branches.py
python examples/validation/validate_side_branch_transducer.py
python examples/validation/validate_stkpowerlw.py
python examples/validation/validate_vespeaker.py
python examples/validation/validate_vxq1.py
python examples/validation/validate_vxq2.py
python examples/validation/validate_vxt1.py
python examples/validation/validate_vxt2.py
```

## Proprietary Comparison Scripts

Strict-cleanliness policy:
- Validation scripts in this repository use embedded neutral reference baselines.
- No proprietary install paths or proprietary file dependencies are required.

## How to Interpret Results

Expected script behavior:
- Script exits successfully (`0`) and prints metrics/summary values.
- Reported errors or deviation percentages should be within script-specific tolerances.

When a validation fails:
1. Re-run the script to confirm determinism.
2. Run the nearest lower-complexity validation script.
3. Check parameter assumptions (gas pressure, area, temperature profile).
4. Compare against recent parity expectations documented in current validation notes.

## Validation Families

- `validate_*`: focused module or segment validations.
- `solve_*`: workflow demonstrations with solve targets.

## Coverage Tracking

Use these docs together:
- [Example Matrix](./example-matrix.md)
- [Reference Mapping](../reference-mapping.md)

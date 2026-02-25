# Troubleshooting

## Purpose

Fast diagnosis for common setup, solver, and visualization issues.

## 1) `matplotlib` import fails after install

Symptoms:
- `pip install -e ".[viz]"` reports success
- `python -c "import matplotlib"` fails

Cause:
- install and runtime interpreters are different.

Fix:

```bash
which python
python -V
python -m pip -V
python -m pip install -e ".[viz]"
python -c "import matplotlib; print(matplotlib.__version__)"
```

## 2) `.venv` has no pip

Symptoms:
- `python -m pip -V` -> `No module named pip`

Fix:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```

Fallback:
- recreate venv and reinstall dependencies.

## 3) `HardEnd`/`SoftEnd` segment errors during propagation

Symptoms:
- errors such as missing `transfer` during solver propagation.

Cause:
- boundary segments used in a path that expects integrable transfer behavior.

Fix:
- enforce outlet boundary conditions using solver `targets`.
- use physical segments (`Duct`, `Stack`, `HeatExchanger`, etc.) in propagation chain.

## 4) Solver fails to converge

Symptoms:
- `result.converged == False`
- large residuals or unstable frequency updates

Checks:
- guesses and targets count must match.
- initial frequency guess should be physically plausible.
- simplify model (single duct), then reintroduce complexity.
- verify temperatures/areas/porosity/hydraulic radius are realistic.

Actions:
- adjust `tol`, `maxiter`, and guess values.
- add diagnostics with intermediate visualizations.

## 5) Guess/target dimensionality error

Symptoms:
- `Number of guesses (...) must equal number of targets (...)`

Fix:
- add/remove guess variables or targets so counts match exactly.

## 6) Configuration parsing errors

Symptoms:
- `ConfigError` during `load_config` or `run_from_config`

Checks:
- required keys: `gas`, `segments`
- valid `type` names for gas and segments
- numeric fields are valid numeric types
- file extension is supported (`.json`, `.yaml`, `.yml`)

## 7) PNG not written

Checks:
- use `save_path="output.png"` in viz calls
- ensure target directory is writable
- check for exceptions from matplotlib backend/runtime

## 8) Low confidence in physical realism

Actions:
- run parity/validation scripts in `examples/validation/`
- compare outputs to current validation baseline used by your team
- test sensitivity to frequency, pressure, and geometry perturbations

## Escalation Path

When uncertain:
1. Reduce to a minimal single-segment model.
2. Verify convergence and profile shapes.
3. Reintroduce segments one by one.
4. Use visualization and validation scripts after each increment.

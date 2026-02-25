# OpenThermoacoustics Documentation Plan

## 1) Objective

Build complete, maintainable documentation for all user-facing and contributor-facing functionality in OpenThermoacoustics, including:
- API usage
- segment and solver behavior
- reference baseline parity and migration context
- validation and testing workflows
- troubleshooting and known limitations

This plan is written as an execution roadmap, not just an outline.

## 2) Scope

In scope:
- Package modules under `src/openthermoacoustics/`
- All public exports from:
  - `openthermoacoustics.__init__`
  - `openthermoacoustics.segments.__init__`
  - `openthermoacoustics.gas.__init__`
  - `openthermoacoustics.geometry.__init__`
  - `openthermoacoustics.solver.__init__`
- Configuration workflows (`config.py`)
- Visualization workflows (`viz.py`)
- Example and validation scripts in `examples/`

Out of scope:
- Internal implementation details not needed for usage or contribution
- Third-party dependency internals (SciPy/NumPy internals)

## 3) Audiences and Documentation Outcomes

Audience A: New users
- Outcome: install, run a first model, interpret outputs.

Audience B: Applied users (design/validation)
- Outcome: model realistic systems, use correct segments, validate against references.

Audience C: Contributors
- Outcome: add features safely, run tests, preserve conventions and parity behavior.

## 4) Deliverables (Documentation Set)

### 4.1 Core Guides

1. `docs/getting-started.md`
- Installation matrix (`base`, `dev`, `viz`, `config`, `all`)
- First solve walkthrough
- Common environment pitfalls (`python -m pip`, venv mismatch)

2. `docs/concepts.md`
- Phasor/state conventions
- Units and sign conventions
- Network topology model
- Boundary conditions via targets vs boundary segments

3. `docs/modeling-workflows.md`
- Closed-closed resonator workflow
- Closed-open workflow
- Stack/regenerator workflow
- Loop/branch workflow
- Config-file driven workflow

4. `docs/visualization.md`
- `plot_profiles`, `plot_phasor_profiles`, `plot_frequency_sweep`
- Figure object usage and PNG output
- Segment boundary overlays

5. `docs/troubleshooting.md`
- Solver non-convergence patterns
- Guess/target mismatch errors
- Segment compatibility caveats
- Environment/import issues

### 4.2 API Reference

6. `docs/api/engine.md`
- `Network` class usage and solve helpers.

7. `docs/api/solver.md`
- `NetworkTopology`, `ShootingSolver`, `SolverResult`
- `loop_network` and `tbranch_loop_solver` usage notes.

8. `docs/api/gas.md`
- All gas models and `Mixture`.

9. `docs/api/geometry.md`
- `Circular`, `ParallelPlate`, `Rectangular`, `Screen`, `PinArray`.

10. `docs/api/segments.md`
- Segment catalog grouped by category:
  - basic propagation (`Duct`, `Cone`, `Stack`, `HeatExchanger`, etc.)
  - boundaries and impedance
  - branches and joins
  - transducers
  - reference baseline-compatible aliases (`STK*`, `SX`, `TX`, `PX`, `VXT*`, etc.)

11. `docs/api/config.md`
- schema for YAML/JSON
- parsing and run flows
- examples with solver options and targets

12. `docs/api/state-and-utils.md`
- `AcousticState` and utility functions.

### 4.3 Validation and Parity

13. `docs/validation/index.md`
- how to run validations in `examples/`
- expected outputs and interpretation
- relationship to maintained parity notes/baselines

14. `docs/validation/example-matrix.md`
- map each example script to:
  - purpose
  - covered modules
  - expected pass criteria

15. `docs/reference-mapping.md`
- mapping table from reference baseline segment names to OpenThermoacoustics classes
- caveats where behavior is intentionally approximate or different

### 4.4 Contributor Docs

16. `docs/contributing-docs.md`
- doc style guide
- how to add/update examples
- doc review checklist
- release checklist for docs

## 5) Information Architecture

Recommended docs tree:

```text
docs/
  index.md
  getting-started.md
  concepts.md
  modeling-workflows.md
  visualization.md
  troubleshooting.md
  reference-mapping.md
  contributing-docs.md
  api/
    engine.md
    solver.md
    gas.md
    geometry.md
    segments.md
    config.md
    state-and-utils.md
  validation/
    index.md
    example-matrix.md
```

README should remain concise and link into this structure.

## 6) Execution Phases

### Phase 0: Inventory and ownership
- Create API inventory from `__all__` exports.
- Mark undocumented symbols.
- Assign owners/reviewers.

Exit criteria:
- Inventory table complete and reviewed.

### Phase 1: User-critical guides
- Deliver:
  - `getting-started.md`
  - `concepts.md`
  - `modeling-workflows.md`
  - `troubleshooting.md`
- Update README links.

Exit criteria:
- New user can install, solve, visualize, and troubleshoot without source diving.

### Phase 2: API reference completeness
- Deliver all `docs/api/*.md`.
- Ensure every public class/function has:
  - purpose
  - parameters/units
  - return values
  - minimal example
  - known caveats

Exit criteria:
- 100% coverage of public exports.

### Phase 3: Validation and parity docs
- Deliver `docs/validation/*` and `reference-mapping.md`.
- Connect to existing parity assets and examples.

Exit criteria:
- Users can reproduce parity checks with documented commands.

### Phase 4: Contributor quality gates
- Add docs update checklist to contribution workflow.
- Require doc updates for new exported APIs.

Exit criteria:
- Documentation maintenance becomes part of normal review.

## 7) Quality Standards

Every doc page must include:
- purpose statement
- prerequisites
- runnable example(s)
- expected output/behavior
- limitations and caveats
- links to adjacent docs

Technical standards:
- all units explicit (SI unless stated)
- consistent variable names (`p1`, `U1`, `T_m`, `omega`)
- no ambiguity around target/guess semantics
- absolute clarity on solver expectations and failure modes

## 8) Traceability Matrix (Code -> Docs)

Map each module family to primary docs:
- `engine.py` -> `api/engine.md`, `modeling-workflows.md`
- `solver/*` -> `api/solver.md`, `concepts.md`, `troubleshooting.md`
- `segments/*` -> `api/segments.md`, `reference-mapping.md`
- `gas/*` -> `api/gas.md`
- `geometry/*` -> `api/geometry.md`
- `config.py` -> `api/config.md`, `getting-started.md`
- `viz.py` -> `visualization.md`
- `state.py`, `utils.py` -> `api/state-and-utils.md`
- `examples/*` -> `validation/example-matrix.md`, `validation/index.md`

## 9) Implementation Checklist

1. Create docs skeleton files and cross-links.
2. Populate user-critical pages first.
3. Generate API inventory and fill API pages.
4. Add command blocks for each validation example family.
5. Add troubleshooting entries from known issues encountered in real runs.
6. Run a docs QA pass:
   - command blocks are runnable
   - links resolve
   - terminology consistent
7. Merge with reviewer signoff from at least:
   - one solver owner
   - one segments/parity owner

## 10) Suggested Timeline

- Week 1: Phase 0 + Phase 1
- Week 2: Phase 2 (engine/solver/gas/geometry/config/viz)
- Week 3: Phase 2 (segments), Phase 3
- Week 4: Phase 4 + QA hardening

## 11) Success Metrics

- Public API documentation coverage: 100%
- Broken internal docs links: 0
- New-user first successful solve time: <= 15 minutes from clean checkout
- Validation reproduction: documented for all scripts in `examples/validation/`
- Documentation update included in all feature PRs touching public API

# Contributing Docs

## Purpose

Define how documentation changes are authored, reviewed, and shipped with code changes.

## When Docs Updates Are Required

Docs updates are required when a change affects any of:
- public API surface (`openthermoacoustics.*` exports)
- behavior of solver inputs/outputs
- segment parameters, defaults, or conventions
- config schema or supported keys
- validation workflow or example script behavior
- troubleshooting-relevant runtime behavior

## Required Update Targets

For changes to public behavior, update at least:
1. relevant page in `docs/api/`
2. relevant workflow/concept/troubleshooting page
3. `docs/validation/*` if validation commands or interpretation changed

If no docs update is needed, PR description must include a short justification.

## Documentation Quality Checklist

- Purpose statement is present.
- Prerequisites are explicit.
- Commands/examples are runnable.
- Units and variable names are consistent (`p1`, `U1`, `T_m`, `omega`).
- Caveats/limitations are documented.
- Links resolve and point to current files.

## Review Checklist (Docs Reviewer)

1. Does the doc accurately match current code behavior?
2. Are examples aligned with current solver usage constraints?
3. Are failure modes and troubleshooting covered where relevant?
4. Are parity/validation references consistent with current scripts?
5. Are migration implications documented for reference baseline-facing changes?

## Release Checklist (Docs)

Before release/tag:
1. Run core quickstart commands from `docs/getting-started.md`.
2. Run representative validation commands from `docs/validation/index.md`.
3. Confirm API pages exist for all exported module families.
4. Confirm docs index links all major pages.

## Style Guidelines

- Keep examples minimal but complete.
- Use SI units and avoid implicit assumptions.
- Prefer direct statements over narrative explanation.
- Document known limitations instead of implying unsupported behavior works.


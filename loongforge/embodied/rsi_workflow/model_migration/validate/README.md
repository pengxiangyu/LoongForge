# Validate

The validation-module directory of the model-migration workspace. It is responsible for checking the accuracy of the final output, including logic consistency and precision alignment, to make sure the migration result is correct and usable — the last gate before a migration output is merged.

## Responsibilities

- **Logic consistency check**: verifies that the migrated model matches the original open-source model in structure, data flow, and inference path.
- **Precision alignment check**: compares the outputs of the pre- and post-migration models under identical inputs and weights, confirming the numerical precision is aligned (within the allowed tolerance).
- **Usability check**: confirms the migration output can be trained, evaluated, and released inside LoongForge with reproducible results.

## Contents

- Validation scripts: implementations of the logic comparison, precision comparison, and smoke tests.
- Validation configuration: check items and thresholds for different models/tasks.
- Validation reports: results and conclusions of each validation run, used as the merge basis.

## Implemented checker boundary

All six checkers are implemented, and each judges exactly one question:

- `IntakeChecker` / `AnatomyChecker` — independently verify their own sealed package.
- `ConstructChecker` — slot/target correspondence, required segments, weight coverage;
  delegates to the frozen structure judge.
- `ParityChecker` — per-step loss deviation of the migration; delegates to the frozen
  numeric gate.
- `TrainResumeChecker` — whether a checkpoint save/resume changes the trajectory. Steps
  before the boundary must be identical (same weights, same sample order); steps after it
  are judged by the same frozen numeric gate.
- `EvalChecker` — whether the closed-loop eval report is complete (model factory, payload
  builder, action decoder all evidenced), every task measured, and no task regressed
  against its baseline. It runs no simulator; it judges the report.

`numeric_gate` is the single definition of the numeric verdict — `ParityChecker` and
`TrainResumeChecker` both call it rather than re-deriving a deviation.
`base.py` contains the checker protocol and source-digest helper, `gate_runner.py`
aggregates results fail-closed, and `accept_gate.py` decides final acceptance.

The composite validator hash binds checker source paths, `validate/base.py`,
`core/orchestrator/judger.py` and declared dependency sources — including the frozen judge a
checker delegates to, so swapping in a more lenient judge changes the hash.
See `../docs/intake-v2-handoff.md` and `../docs/anatomy-spec.md` for v2 and model-specific checks.

Both implemented validators now emit `validator_version="2"`.
Anatomy verification requires an externally trusted `spec_manifest_sha256`
(`--spec-manifest-sha256` in the CLI), not a digest taken from the bundle's lock.
v1 task/behavior schemas remain supported with newly generated v2 receipts.
Historical `stages.verify.*` v1 sealed packages require their pinned old runtime;
historical sealed-package replay compatibility is not implemented or claimed.

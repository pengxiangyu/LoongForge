# Supervising a sample_vla run

What to watch while the migration runs, in the order it can go wrong.

## Before anything is written

- `python -m validate verify` holds. A judgement issued against an unlocked judge is not a
  judgement; if the lock broke, find out what changed the frozen files before re-freezing.
- `core/invoke/probe_skills.py` passes. Both skill cards must be mounted: the framework delegates
  structure analysis and precision work entirely, and discovering a missing card during F4 leaves
  a half-written tree behind.

## F3 scaffold

- `check_tree.json` reports no missing slot and no surplus file. A surplus file is not a harmless
  extra — it is how coverage gets padded.
- Watch `reuse_required` in the scaffold report: those are the target framework's shared files.
  They must not be regenerated from a template, which is why the scaffold never overwrites.

## F4 migrate

- `kept` and `needs_merge` in the stage report are not failures but they are not done either:
  a shared file left a `.f4-proposal` beside it, waiting for a reviewed merge.
- Every `reimplement` entry stays open until the alignment gate passes. The writer marks the
  target shape; it does not write the logic, and it should not pretend to.
- `rewrites matched nothing` from `adapt` means a rename silently did not apply. Left alone, the
  migrated file keeps importing the original repository and fails F5 for an invisible reason.
- The weight report must show no unmapped key. A checkpoint that loads while missing parameters
  is the failure mode this gate exists for.

## F5 align

- The four-same check comes first and is not negotiable. If it fails, fix the condition it names;
  comparing losses anyway produces a number that looks like evidence.
- The self-test must pass in the same round as the verdict. A gate that has not just rejected a
  broken case is indistinguishable from a gate that always passes.
- A failing gate goes back to F4 with the skill's per-slot hints. It never goes to
  `thresholds.yaml`: that file is hashed, and editing it fails the lock instead of passing the gate.
- Watch `no_amplify` even when the three limits pass. A deviation that keeps growing means the two
  paths are drifting apart, and a short run would have hidden it.

## When it stalls

If several rounds produce no movement, stop and report the blocking slot rather than continuing to
rewrite. Repeated rounds on the same slot usually mean the anatomy placed it wrong, which is an
F2 problem and cannot be fixed by writing the file again.

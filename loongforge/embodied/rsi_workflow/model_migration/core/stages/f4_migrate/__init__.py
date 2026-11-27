"""F4: write the original implementation into the target tree. Same logic, different structure.

  map/            the per-slot source-to-target mapping and its schema
  mapping.yaml    the mapping this migration actually used
  writers/        the four write actions: copy, adapt, wrap, reimplement
  dataset_bridge  places the data side per model-adaptation's plan
  weights/        checkpoint key and layout conversion, in one place
  handoff.py      hands the slots a writer cannot finish to a coding agent, under a typed request
  migrate.py      the stage entry: dispatch the mapping, convert the weights, log every write

Mapping comes before writing, deliberately. Looking for a target location while writing is how
two slots end up in one file and how a module ends up with no slot at all; the mapping is
checked against the anatomy and the tree spec first, and only then is anything written.

Two write outcomes are open by design and leave the stage: `reimplement` marks a target shape
without inventing its logic, and a shared framework file gets a `.f4-proposal` instead of being
overwritten. `handoff.py` is how those reach a coding agent with a scoped write allowlist.
"""


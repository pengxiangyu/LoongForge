"""F5: prove the migrated project reproduces the original's loss, or say where it does not.

  four_same_check.py  same data, same order, same weights, same launch -- checked first
  align_driver.py     drives both paths through precision-alignment and the frozen gate
  align_gate.py       routes the verdict: accept, or back to F4 with the skill's hints
  cases/              one case that must pass and one that must fail, for the self-test

The division of labour is the point. Collecting losses, comparing them and attributing a
deviation is the skill's work. Deciding whether the result is good enough is the frozen gate's,
against the frozen thresholds. This stage only makes sure the comparison is meaningful before it
happens, and that a failure goes back to the implementation rather than to the threshold file.
"""

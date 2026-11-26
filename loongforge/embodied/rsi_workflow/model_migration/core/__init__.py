"""Execution layer: initialise a task, isolate it, run the stages, accept the result.

`invoke/` is the only place that talks to the two skill cards; `stages/f1_script` ... `f5_align`
are the framework side of the five functions. Judging is not here -- it lives in `validate/`,
frozen, so that the side being judged cannot relax the verdict.
"""

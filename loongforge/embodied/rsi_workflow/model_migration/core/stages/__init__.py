"""The framework side of the five functions, one directory per stage.

  f1_intake    lock the source, discover the entrypoint, triage assets, probe and seal
  f2_anatomy   capture one training step twice and freeze the behaviour contract
  f3_scaffold  turn the accepted slot mapping into the anatomy, then generate the skeleton
  f4_migrate   write the original implementation into the target tree, slot by slot
  f5_align     check the four-same conditions, drive the comparison, route the verdict

Each stage reads what the previous one wrote and writes what the next one reads; none of them
decides whether the migration passed. Judging lives in `validate/`, frozen, and the four
acceptance gates are aggregated by `validate/accept_gate.py` -- acceptance is a verdict, not a
sixth stage.
"""

from .base import StageContext, StageResult, StageRunner, StageStatus

__all__ = [
    "StageContext",
    "StageResult",
    "StageRunner",
    "StageStatus",
]


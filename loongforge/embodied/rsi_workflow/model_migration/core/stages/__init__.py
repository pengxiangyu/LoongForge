"""Unified 6-stage pipeline package (`intake`, `anatomy`, `scaffold`, `adapt`, `verify`, `deliver`).
The framework side of the five functions, one directory per stage.

  f1_script    parse the unknown model, discover slots, regenerate the launch script
  f2_anatomy   split one training step into segments and place the slots
  f3_scaffold  generate the target project skeleton from the anatomy
  f4_migrate   write the original implementation into the target tree
  f5_align     check the four-same conditions, drive the comparison, route the verdict

Each stage reads what the previous one wrote and writes what the next one reads; none of them
decides whether the migration passed.
"""

from .base import StageContext, StageResult, StageRunner, StageStatus

__all__ = [
    "StageContext",
    "StageResult",
    "StageRunner",
    "StageStatus",
]
=======


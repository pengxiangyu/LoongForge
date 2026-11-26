"""Validation layer of the VLA migration framework: the judges, their inputs and their lock.

The modules split along what they are allowed to depend on:

  structure_judge  does the target tree correspond, slot by slot, to the anatomy
  align_gate       does the migrated loss trace match the original's, step by step
  stats, trajectory, adapter, record  the data the two judges read, and how it is frozen
  frozen/          the copies that actually judge, plus the hash lock over them
  selftest/        proves the judges accept a correct case and reject a broken one

A judgement always runs against `frozen/`, never against the modules above it: the development
copy may be edited while a migration is in flight, the frozen one may not.
Independent artifact checkers; no stage orchestration."""

from .anatomy_check import AnatomyChecker
from .base import CheckContext, GateChecker, validator_source_sha256
from .construct_check import ConstructChecker
from .eval_check import EvalChecker
from .intake_check import IntakeChecker
from .parity_check import ParityChecker
from .train_resume_check import TrainResumeChecker

__all__ = [
    "AnatomyChecker", "CheckContext", "GateChecker", "validator_source_sha256",
    "ConstructChecker", "EvalChecker", "IntakeChecker", "ParityChecker", "TrainResumeChecker",
    "adapter", "align_gate", "record", "stats", "structure_judge", "trajectory"
]
"""Validation layer: the judges, their inputs, their lock, and the runner that aggregates them.

Two families of judge live here, and they answer different questions:

  structure_judge, numeric_gate    is the migration correct -- slot-by-slot correspondence, and
                                  step-by-step loss agreement. Frozen under `frozen/` with a
                                  hash lock, so the side being judged cannot relax the verdict.
  intake_check, anatomy_check,    are the sealed artifacts of each stage internally consistent.
  construct/parity/train_resume/  Each is an independent checker emitting a bound GateResult.
  eval_check

  stats, trajectory, adapter, record   the data the first family reads, and how it is frozen
  gate_runner                     runs the second family fail-closed and aggregates the gates
  selftest/                       proves the judges accept a correct case and reject a broken one

Nothing is imported eagerly. The checkers need pydantic and the contract layer; the frozen
judges need the standard library and nothing else. Importing this package pulls in whichever of
the two you actually touch, so `python -m validate judge` and `core/orchestrator/*` stay
dependency-light while `validate.intake_check` still works.
"""

from typing import TYPE_CHECKING, Any

_CHECKERS = {
    "AnatomyChecker": "anatomy_check",
    "ConstructChecker": "construct_check",
    "EvalChecker": "eval_check",
    "IntakeChecker": "intake_check",
    "ParityChecker": "parity_check",
    "TrainResumeChecker": "train_resume_check",
    "CheckContext": "base",
    "GateChecker": "base",
    "validator_source_sha256": "base",
    "GateRunner": "gate_runner",
}

if TYPE_CHECKING:  # import-time names for type checkers only
    from .anatomy_check import AnatomyChecker
    from .base import CheckContext, GateChecker, validator_source_sha256
    from .construct_check import ConstructChecker
    from .eval_check import EvalChecker
    from .gate_runner import GateRunner
    from .intake_check import IntakeChecker
    from .parity_check import ParityChecker
    from .train_resume_check import TrainResumeChecker


def __getattr__(name: str) -> Any:
    """Resolve a checker on first use, so importing `validate` costs nothing by itself."""
    module_name = _CHECKERS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "AnatomyChecker", "CheckContext", "ConstructChecker", "EvalChecker", "GateChecker",
    "GateRunner", "IntakeChecker", "ParityChecker", "TrainResumeChecker",
    "validator_source_sha256",
]

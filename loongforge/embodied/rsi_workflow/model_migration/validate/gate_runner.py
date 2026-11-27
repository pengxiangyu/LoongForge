"""Runs the independent checkers and aggregates their results into one gate decision.

The design keeps judging out of the stages: a stage writes artifacts, this layer decides whether
they hold. Two rules make that decision trustworthy rather than advisory:

  fail-closed   a checker that raises does not vanish from the tally. It becomes an `error`
                result with reason code `CHECKER_CRASH`, so a crash can never read as a pass.
  bound         `trusted_validators` must be supplied: each result carries the commit, the
                manifest digest and the validator's own hash, and `judger.decide` rejects a
                result whose validator is not the trusted one. Passing `allow_untrusted=True`
                is for exercising this module itself, never for a real verdict.

`required_gates` is what makes a missing checker a failure instead of a silent omission: the
decision needs a result for every gate on that list.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from core.orchestrator.judger import (
    VALID_CHECKS,
    GateBinding,
    GateDecision,
    GateEvidence,
    GateResult,
    decide,
)

from .base import CheckContext, GateChecker


class GateRunner:
    """Executes registered independent checkers and aggregates via `judger.decide`."""

    def __init__(
        self,
        checkers: Iterable[GateChecker] = (),
        *,
        trusted_validators: Mapping[str, str] | None = None,
        required_gates: Iterable[str] | None = None,
        allow_untrusted: bool = False,
    ) -> None:
        if isinstance(required_gates, str):
            raise TypeError("required_gates must be an iterable of gate IDs, not str")
        self.checkers: tuple[GateChecker, ...] = tuple(checkers)
        if trusted_validators is None and not allow_untrusted:
            raise ValueError(
                "trusted_validators is required for fail-closed verification "
                "(pass allow_untrusted=True only for explicit test overrides)"
            )
        self.trusted_validators = (
            dict(trusted_validators) if trusted_validators is not None else None
        )
        self.allow_untrusted = allow_untrusted
        if required_gates is not None:
            self.required_gates: tuple[str, ...] = tuple(required_gates)
        else:
            self.required_gates = tuple(checker.gate_id for checker in self.checkers)

    def _crash_result(self, checker: GateChecker, context: CheckContext, error: Exception) -> GateResult:
        try:
            validator_sha = checker.validator_sha256()
        except Exception:                      # a checker too broken to hash is still reported
            validator_sha = "f" * 64
        check_name = checker.check_name if checker.check_name in VALID_CHECKS else "intake"
        return GateResult(
            gate_id=checker.gate_id,
            check=check_name,
            status="error",
            binding=GateBinding(
                commit=context.commit,
                manifest_sha256=context.manifest_sha256,
                validator_id=checker.validator_id,
                validator_sha256=validator_sha,
            ),
            reason_codes=("CHECKER_CRASH",),
            first_mismatch=f"{checker.gate_id}:{type(error).__name__}",
            summary=f"checker {checker.gate_id!r} raised {type(error).__name__}: {error}",
            evidence=(GateEvidence(key="exception", detail=f"{type(error).__name__}: {error}"),),
        )

    def evaluate(self, context: CheckContext) -> GateDecision:
        results: list[GateResult] = []
        for checker in self.checkers:
            try:
                results.append(checker.check(context))
            except Exception as error:
                results.append(self._crash_result(checker, context, error))
        return decide(
            results,
            expected_commit=context.commit,
            expected_manifest_sha256=context.manifest_sha256,
            trusted_validators=self.trusted_validators,
            required_gates=self.required_gates,
        )


__all__ = ["GateRunner"]

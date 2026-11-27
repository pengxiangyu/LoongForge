"""Independent checker for the `construct` gate: does the tree match the anatomy.

The judgement itself is not re-implemented here. This checker delegates to the frozen
`structure_judge`, which is the single source of that verdict, and only adds what a gate
result needs on top: the provenance binding (commit, manifest digest, this validator's own
hash) and the translation of a report into a `GateOutcome`.

Why the delegated module is bound into the digest: the judge is what decides, so a checker
that hashed only itself could be swapped for a lenient judge without the hash moving.
`validator_sha256` therefore covers `structure_judge.py` as well.

Outcome mapping:
  PASS -> pass
  FAIL -> repairable   the target implementation is what has to change, which is F4's job
  a missing input is `error`, never `pass`

`context.parameters` accepts either parsed objects or paths to parse:
  anatomy       STEP_ANATOMY.yaml (mapping or path)   required
  target_root   root of the generated project         required
  weight_report weight conversion report              optional
  generic       generic-file whitelist                optional
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.orchestrator.judger import GateBinding, GateEvidence, GateResult

from . import structure_judge
from .base import CheckContext, validator_source_sha256

_DELEGATED = ("structure_judge.py",)
_MAX_EVIDENCE = 20


def _load(value: Any) -> Any:
    """Accept a parsed mapping or a path to a YAML document."""
    if isinstance(value, (str, Path)):
        import yaml

        return yaml.safe_load(Path(value).read_text(encoding="utf-8"))
    return value


def _evidence(problems: list[str], extra: dict[str, str]) -> tuple[GateEvidence, ...]:
    items = [
        GateEvidence(key=f"problem[{index}]", detail=problem)
        for index, problem in enumerate(problems[:_MAX_EVIDENCE])
    ]
    items += [GateEvidence(key=key, detail=detail) for key, detail in sorted(extra.items())]
    return tuple(items)


class ConstructChecker:
    """Checks slot/target correspondence, required segments, and weight coverage."""

    gate_id = "construct"
    check_name = "construct"
    validator_id = "validate.construct_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(
            type(self),
            extra_sources=[Path(__file__).resolve().with_name(name) for name in _DELEGATED],
        )

    def check(self, context: CheckContext) -> GateResult:
        params = context.parameters
        anatomy = _load(params.get("anatomy"))
        target_root = params.get("target_root")
        missing = [
            name
            for name, value in (("anatomy", anatomy), ("target_root", target_root))
            if value is None
        ]
        if missing:
            return GateResult(
                gate_id=self.gate_id,
                check=self.check_name,
                status="error",
                binding=self._binding(context),
                summary=f"construct gate cannot run: missing input {', '.join(missing)}",
                reason_codes=("MISSING_INPUT",),
                first_mismatch=f"{self.gate_id}:missing:{','.join(missing)}",
                evidence=_evidence([], {"missing": ", ".join(missing)}),
            )

        report = structure_judge.judge(
            anatomy,
            target_root,
            weight_report=_load(params.get("weight_report")),
            generic=tuple(params.get("generic") or structure_judge.GENERIC_FILES),
        )
        problems = list(report["problems"])
        passed = report["verdict"] == "PASS"
        return GateResult(
            gate_id=self.gate_id,
            check=self.check_name,
            status="pass" if passed else "repairable",
            binding=self._binding(context),
            summary=(
                f"structure judge passed over {report['slots']} slot(s)"
                if passed
                else f"structure judge found {len(problems)} problem(s)"
            ),
            reason_codes=() if passed else ("STRUCTURE_MISMATCH",),
            first_mismatch=None if passed else f"{self.gate_id}:{problems[0]}",
            evidence=_evidence(problems, {"target_root": str(report["target_root"])}),
            metrics={"slots": report["slots"], "problems": len(problems)},
        )

    def _binding(self, context: CheckContext) -> GateBinding:
        return GateBinding(
            commit=context.commit,
            manifest_sha256=context.manifest_sha256,
            validator_id=self.validator_id,
            validator_sha256=self.validator_sha256(),
        )


__all__ = ["ConstructChecker"]
"""Independent checker for the `parity` gate: does the port reproduce the original's loss.

Like `construct_check`, this checker owns no judgement of its own. The verdict comes from the
frozen `numeric_gate`, which is the single source of the numeric limits, and both
`numeric_gate.py` and the `stats.py` it reads are bound into this validator's digest -- a
checker that hashed only itself could be paired with a relaxed gate without the hash moving.

Outcome mapping, following what the numeric gate's own three failure modes mean:
  PASS    -> pass
  FAIL    -> repairable   a limit was exceeded or the deviation amplifies; F4 fixes the target
  ERROR   -> blocked      the two traces are not the same experiment, so there is nothing to
                           compare; that is not something the migration code can repair

`context.parameters` accepts either parsed objects or paths to parse:
  src_losses    {step: loss} of the reference run        required
  tgt_losses    {step: loss} of the migrated run        required
  thresholds    thresholds.yaml (mapping or path)       required
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.orchestrator.judger import GateBinding, GateEvidence, GateResult

from . import numeric_gate
from .base import CheckContext, validator_source_sha256

_DELEGATED = ("numeric_gate.py", "stats.py")
_MAX_EVIDENCE = 20

_OUTCOME = {"PASS": "pass", "FAIL": "repairable", "ERROR": "blocked"}
_REASON = {
    "PASS": (),
    "FAIL": ("NUMERIC_LIMIT_EXCEEDED",),
    "ERROR": ("TRACES_NOT_COMPARABLE",),
}


def _load(value: Any) -> Any:
    """Accept a parsed mapping or a path to a YAML document."""
    if isinstance(value, (str, Path)):
        import yaml

        return yaml.safe_load(Path(value).read_text(encoding="utf-8"))
    return value


class ParityChecker:
    """Checks per-step loss deviation against the frozen thresholds."""

    gate_id = "parity"
    check_name = "parity"
    validator_id = "validate.parity_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(
            type(self),
            extra_sources=[Path(__file__).resolve().with_name(name) for name in _DELEGATED],
        )

    def check(self, context: CheckContext) -> GateResult:
        params = context.parameters
        src = params.get("src_losses")
        tgt = params.get("tgt_losses")
        thresholds = _load(params.get("thresholds"))
        missing = [
            name
            for name, value in (
                ("src_losses", src),
                ("tgt_losses", tgt),
                ("thresholds", thresholds),
            )
            if value is None
        ]
        binding = GateBinding(
            commit=context.commit,
            manifest_sha256=context.manifest_sha256,
            validator_id=self.validator_id,
            validator_sha256=self.validator_sha256(),
        )
        if missing:
            return GateResult(
                gate_id=self.gate_id,
                check=self.check_name,
                status="error",
                binding=binding,
                summary=f"parity gate cannot run: missing input {', '.join(missing)}",
                reason_codes=("MISSING_INPUT",),
                first_mismatch=f"{self.gate_id}:missing:{','.join(missing)}",
                evidence=(
                    GateEvidence(key="missing", detail=", ".join(missing)),
                ),
            )

        report = numeric_gate.evaluate(src, tgt, thresholds)
        verdict = report["verdict"]
        problems = list(report["problems"])
        evidence = [
            GateEvidence(key=f"problem[{index}]", detail=problem)
            for index, problem in enumerate(problems[:_MAX_EVIDENCE])
        ]
        evidence.append(
            GateEvidence(
                key="limits",
                detail=json.dumps(report["limits"], ensure_ascii=False, sort_keys=True),
            )
        )
        return GateResult(
            gate_id=self.gate_id,
            check=self.check_name,
            status=_OUTCOME[verdict],
            binding=binding,
            summary=(
                f"numeric gate {verdict} over {len(report.get('per_step') or [])} step(s)"
                if verdict == "PASS"
                else f"numeric gate {verdict} with {len(problems)} problem(s)"
            ),
            reason_codes=_REASON[verdict],
            first_mismatch=None if verdict == "PASS" else f"{self.gate_id}:{problems[0]}",
            evidence=tuple(evidence),
            metrics=dict(report.get("metrics") or {}),
        )


__all__ = ["ParityChecker"]
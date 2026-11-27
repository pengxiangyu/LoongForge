"""Independent checker for the `train_resume` gate: does the run survive a save and a resume.

Parity asks whether the port matches the original. This asks a different question about the
same migrated code: whether stopping it and restarting from a checkpoint changes anything. A
run that diverges across a resume boundary is unusable however good its loss curve looks,
because the divergence silently becomes the new trajectory.

Three things are checked, and no numeric limit is re-derived here:

  declared     the resumed trace must say which step and which checkpoint it restarted from.
  pre-resume   before the boundary both runs used the same weights and the same sample order,
               so the losses must be equal to `pre_tol` (0.0 by default). This is strictly
               stronger than the parity band on purpose: it is what catches a resume that
               reloads the model but not the data order or the RNG state.
  post-resume  after the boundary the deviation is judged by the frozen numeric gate against
               the frozen thresholds. `parity_check` judges migration parity with that gate;
               this checker judges resume continuity with it. Neither re-implements it.

`context.parameters` accepts a trace as a parsed mapping or a path to load:
  continuous   trace of the uninterrupted run               required
  resumed      trace of the run started from a checkpoint   required
  thresholds   thresholds.yaml (mapping or path)           required
  pre_tol      tolerance for the pre-resume identity       optional, default 0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.orchestrator.judger import GateBinding, GateEvidence, GateResult

from . import numeric_gate, trajectory
from .base import CheckContext, validator_source_sha256

# stats.py is bound because numeric_gate reads it; the deviation itself is never re-derived here.
_DELEGATED = ("numeric_gate.py", "stats.py", "trajectory.py")
_MAX_EVIDENCE = 20


def _load(value: Any) -> Any:
    """Accept a parsed mapping or a path to a YAML/JSON document."""
    if isinstance(value, (str, Path)):
        import yaml

        return yaml.safe_load(Path(value).read_text(encoding="utf-8"))
    return value


def _trace(value: Any) -> dict[str, Any]:
    """A trace mapping, validated against the shared trace schema."""
    trace = trajectory.load(value) if isinstance(value, (str, Path)) else value
    if not isinstance(trace, dict):
        raise ValueError("a trace must be a mapping or a path to one")
    if trace.get("schema") != trajectory.SCHEMA:
        raise ValueError(f"trace schema is {trace.get('schema')!r}, expected "
                         f"{trajectory.SCHEMA!r}")
    missing = [f for f in trajectory.RUN_FIELDS if f not in (trace.get("run") or {})]
    if missing:
        raise ValueError(f"trace run fingerprint lacks {missing}")
    return trace


class TrainResumeChecker:
    """Checks that a checkpoint save/resume leaves the trajectory unchanged."""

    gate_id = "train_resume"
    check_name = "train_resume"
    validator_id = "validate.train_resume_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(
            type(self),
            extra_sources=[Path(__file__).resolve().with_name(name) for name in _DELEGATED],
        )

    def check(self, context: CheckContext) -> GateResult:
        params = context.parameters
        binding = GateBinding(
            commit=context.commit,
            manifest_sha256=context.manifest_sha256,
            validator_id=self.validator_id,
            validator_sha256=self.validator_sha256(),
        )
        missing = [
            name
            for name, value in (
                ("continuous", params.get("continuous")),
                ("resumed", params.get("resumed")),
                ("thresholds", params.get("thresholds")),
            )
            if value is None
        ]
        if missing:
            return self._result(binding, "error", ("MISSING_INPUT",),
                                f"train_resume gate cannot run: missing input "
                                f"{', '.join(missing)}",
                                [", ".join(missing)], {}, first="missing input")
        try:
            continuous = _trace(params["continuous"])
            resumed = _trace(params["resumed"])
        except (ValueError, OSError) as error:
            return self._result(binding, "error", ("UNREADABLE_TRACE",),
                                f"train_resume gate cannot read a trace: {error}",
                                [str(error)], {}, first="unreadable trace")

        problems, metrics = self._judge(
            continuous, resumed, _load(params["thresholds"]),
            float(params.get("pre_tol", 0.0)),
        )
        if problems:
            return self._result(binding, "repairable", ("RESUME_DIVERGENCE",),
                                f"resume changed the trajectory in {len(problems)} way(s)",
                                problems, metrics, first=problems[0])
        return self._result(binding, "pass", (),
                            f"resume is transparent across step "
                            f"{metrics['from_step']} ({metrics['shared_pre']} pre-resume "
                            f"step(s) identical)", [], metrics, first="passed")

    def _judge(self, continuous, resumed, thresholds, pre_tol):
        """Returns (problems, metrics) for one save/resume pair."""
        declared = (resumed.get("run") or {}).get("resume") or {}
        from_step = declared.get("from_step")
        ckpt_hash = declared.get("ckpt_hash")
        if from_step is None or not ckpt_hash:
            return ([f'the resumed trace does not declare where it restarted from: run.resume '
                     f'needs from_step and ckpt_hash, got {sorted(declared)}'],
                    {"from_step": from_step})

        src, tgt = trajectory.losses(continuous), trajectory.losses(resumed)
        from_step = int(from_step)
        problems: list[str] = []
        if from_step not in src:
            problems.append(f'resume boundary {from_step} is not a step the continuous run '
                            f'recorded (it covers {min(src)}..{max(src)})')
        pre = sorted(set(s for s in set(src) & set(tgt) if s < from_step))
        post = sorted(set(s for s in set(src) & set(tgt) if s >= from_step))
        if not pre:
            problems.append(f'no step before the resume boundary {from_step} is present in '
                            f'both traces, so the resume cannot be checked')
        if not post:
            problems.append(f'the resumed run records no step at or after {from_step}, so it '
                            f'never actually continued')

        worst_pre = 0.0
        for step in pre:
            delta = abs(src[step] - tgt[step])
            worst_pre = max(worst_pre, delta)
            if delta > pre_tol:
                problems.append(f'pre-resume step {step} differs by {delta:.6g} (tolerance '
                                f'{pre_tol:.6g}): the resume did not restore the same state')
        metrics = {"from_step": from_step, "shared_pre": len(pre), "shared_post": len(post),
                   "pre_max_abs_diff": worst_pre}

        if post:
            gate = numeric_gate.evaluate(
                {s: src[s] for s in post}, {s: tgt[s] for s in post}, thresholds
            )
            metrics["post_verdict"] = gate["verdict"]
            metrics.update({k: v for k, v in (gate.get("metrics") or {}).items()})
            problems += [f'post-resume {p}' for p in gate["problems"]]
        return problems, metrics

    def _result(self, binding, status, reason_codes, summary, problems, metrics, *, first):
        evidence = [
            GateEvidence(key=f"problem[{index}]", detail=problem)
            for index, problem in enumerate(problems[:_MAX_EVIDENCE])
        ]
        return GateResult(
            gate_id=self.gate_id,
            check=self.check_name,
            status=status,
            binding=binding,
            summary=summary,
            reason_codes=reason_codes,
            first_mismatch=None if status == "pass" else f"{self.gate_id}:{first}",
            evidence=tuple(evidence),
            metrics=dict(metrics),
        )


__all__ = ["TrainResumeChecker"]
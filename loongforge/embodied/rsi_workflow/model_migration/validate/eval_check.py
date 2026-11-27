"""Independent checker for the `task_eval` gate: did the closed loop actually run, and is it
no worse than the baseline.

This checker runs no simulator and imports nothing from the target framework. It judges the
*report* the migration produced, which is the only thing a judge can look at without
reproducing the whole evaluation stack. That makes the gate deliberately narrow, and the
narrowness is the point: it refuses to let an eval that skipped a stage, or that has no
baseline to compare against, read as a pass.

Four things are checked:

  present      there is a report, it names the model and the suite, and it covers at least one
               task.
  triad        the closed loop has three parts -- model factory, payload builder, action
               decoder -- and the report must carry evidence for all three. A report that only
               shows a metric cannot tell a working decoder from a silently skipped one.
  executed     every task reports a numeric metric and none of them errored or was skipped.
  no regress   every task's metric is compared against the baseline and does not drop by more
               than `regression_tol`. No baseline is `blocked`, not `pass`: an uncompared
               metric is not evidence.

`context.parameters` accepts a report as a parsed mapping or a path to load:
  report           the closed-loop eval report                required
  baseline         {task: metric} of the reference run        optional
  regression_tol   tolerated relative drop per task           optional, default 0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.orchestrator.judger import GateBinding, GateEvidence, GateResult

from .base import CheckContext, validator_source_sha256

TRIAD = ("model_factory", "payload_builder", "action_decoder")
_DELEGATED = ()
_MAX_EVIDENCE = 20

# A task in one of these states produced no measurement, so it cannot support a verdict.
_UNMEASURED = ("error", "errored", "failed", "skipped", "skip", "timeout")


def _load(value: Any) -> Any:
    """Accept a parsed mapping or a path to a YAML/JSON document."""
    if isinstance(value, (str, Path)):
        import yaml

        return yaml.safe_load(Path(value).read_text(encoding="utf-8"))
    return value


def _metric(task: Any) -> tuple[float | None, str | None]:
    """(metric, state) of one task entry, tolerating a bare number as shorthand."""
    if isinstance(task, (int, float)) and not isinstance(task, bool):
        return float(task), None
    if not isinstance(task, dict):
        return None, "malformed"
    state = str(task.get("status") or "").strip().lower() or None
    value = task.get("metric")
    if value is None:
        value = task.get("success_rate")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), state
    return None, state


class EvalChecker:
    """Checks the closed-loop eval report for completeness and non-regression."""

    gate_id = "task_eval"
    check_name = "task_eval"
    validator_id = "validate.eval_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(type(self))

    def check(self, context: CheckContext) -> GateResult:
        params = context.parameters
        binding = GateBinding(
            commit=context.commit,
            manifest_sha256=context.manifest_sha256,
            validator_id=self.validator_id,
            validator_sha256=self.validator_sha256(),
        )
        report = _load(params.get("report"))
        if report is None:
            return self._result(binding, "error", ("MISSING_INPUT",),
                                "task_eval gate cannot run: no report supplied",
                                ["report"], {}, first="missing input")

        problems, metrics = self._judge(
            report, _load(params.get("baseline")),
            float(params.get("regression_tol", 0.0)),
        )
        if metrics.get("baseline") is False:
            return self._result(binding, "blocked", ("EVAL_BASELINE_MISSING",),
                                "the eval report has no baseline to compare against",
                                problems, metrics, first="baseline missing")
        if problems:
            return self._result(binding, "repairable", ("EVAL_INCOMPLETE",),
                                f"the eval report is not usable in {len(problems)} way(s)",
                                problems, metrics, first=problems[0])
        return self._result(binding, "pass", (),
                            f"{metrics['tasks']} task(s) measured, triad complete, no "
                            f"regression against baseline", [], metrics, first="passed")

    def _judge(self, report, baseline, regression_tol):
        problems: list[str] = []
        metrics: dict[str, Any] = {}
        if not isinstance(report, dict):
            return (["the eval report is not a mapping"], metrics)

        for field in ("model_tag", "suite"):
            if not report.get(field):
                problems.append(f'eval report does not name its {field}')
        tasks = report.get("tasks")
        if not isinstance(tasks, dict) or not tasks:
            problems.append("eval report records no task result")
            tasks = {}
        metrics["tasks"] = len(tasks)

        triad = report.get("triad") or {}
        absent = [part for part in TRIAD if not triad.get(part)]
        if absent:
            problems.append(f'eval report has no evidence for: {", ".join(absent)}')
        metrics["triad"] = [part for part in TRIAD if triad.get(part)]

        measured: dict[str, float] = {}
        for name in sorted(tasks):
            value, state = _metric(tasks[name])
            if state in _UNMEASURED:
                problems.append(f'task {name} did not produce a measurement (status {state})')
            elif value is None:
                problems.append(f'task {name} reports no numeric metric')
            else:
                measured[name] = value
        metrics["measured"] = len(measured)

        metrics["baseline"] = baseline is not None
        if baseline is None:
            problems.append("no baseline supplied, so no task can be compared")
            return problems, metrics
        if not isinstance(baseline, dict) or not baseline:
            problems.append("the baseline is empty, so no task can be compared")
            return problems, metrics

        for name in sorted(measured):
            if name not in baseline:
                problems.append(f'task {name} has no baseline entry')
                continue
            reference, _ = _metric(baseline[name])
            if reference is None:
                problems.append(f'baseline entry for task {name} is not numeric')
                continue
            if reference == 0:
                drop = 0.0 if measured[name] == 0 else float("inf")
            else:
                drop = (reference - measured[name]) / abs(reference)
            if drop > regression_tol:
                problems.append(f'task {name} regressed: {measured[name]:.6g} against a '
                                f'baseline of {reference:.6g} (drop {drop:.4%} exceeds '
                                f'tolerance {regression_tol:.4%})')
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


__all__ = ["EvalChecker"]
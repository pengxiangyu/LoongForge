"""Final acceptance gate: the four gates of the design, and the bundle that is handed over.

This is a judge, not a stage. The migration itself is F1-F5; acceptance is decided here, in the
validation layer, from the verdicts the other judges already produced.

A migration is done when all four hold, and the four are deliberately judged by different
things, so that passing one cannot be mistaken for passing another:

  coverage       every module the original has lands on a slot        frozen structure judge
  structure      slot and target file correspond exactly, no surplus  frozen structure judge
  numeric        per-step loss deviation within one percent, not amplifying   frozen numeric gate
  self_contained the target project starts training with no dependency on the original repo
                 -- checked here, because it is the only gate about what the tree does *not*
                 contain

`accept()` refuses to improvise: a gate with no verdict is `missing`, not `pass`. The
self-contained check is the one it runs itself, scanning the migrated files for imports that
still reach into the original repository.

This module stays free of pydantic and of `core.contracts` on purpose: judging has to stay
import-light so it can run before any model-specific dependency is installed.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

GATES: tuple[str, ...] = ("coverage", "structure", "numeric", "self_contained")
GateStatus = Literal["pass", "fail", "missing"]


@dataclass(frozen=True, slots=True)
class DeliveryBundle:
    """What is handed over once the four gates hold; the design's deliverable list."""

    model_tag: str
    target_dir: str
    gates: dict[str, GateStatus]
    verdict: Literal["PASS", "FAIL"]
    problems: tuple[str, ...] = ()
    evidence_index: tuple[str, ...] = ()
    report_path: str | None = None
    ci_trigger_command: str = "/ok-to-test --suite embodied"
    metadata: dict[str, str] = field(default_factory=dict)


def _module_roots(source_repo: Path) -> set[str]:
    """Top-level importable names of the original repository, as import prefixes to forbid."""
    roots: set[str] = set()
    for child in source_repo.iterdir():
        if child.name.startswith((".", "_")):
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            roots.add(child.name)
        elif child.suffix == ".py":
            roots.add(child.stem)
    if (source_repo / "src").is_dir():
        roots.add("src")
    return roots


def check_self_contained(target_dir: Path, source_repo: Path | None) -> list[str]:
    """Imports in the migrated tree that still reach into the original repository."""
    if source_repo is None or not source_repo.is_dir():
        return ["no source repository supplied; self-containment cannot be checked"]
    forbidden = _module_roots(source_repo)
    if not forbidden:
        return [f"{source_repo} exposes no importable module; nothing to check against"]
    problems: list[str] = []
    for path in sorted(Path(target_dir).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            problems.append(f"{path}: cannot be parsed: {error}")
            continue
        rel = path.relative_to(target_dir).as_posix()
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in forbidden:
                    problems.append(f"{rel}:{node.lineno} still imports the original repository "
                                    f"({name})")
    return problems


def accept(
    *,
    model_tag: str,
    target_dir: Path,
    structure_verdict: dict[str, Any] | None,
    align_report: dict[str, Any] | None,
    source_repo: Path | None = None,
    evidence: tuple[str, ...] = (),
    report_path: str | None = None,
) -> DeliveryBundle:
    """Aggregate the three judges' verdicts, run the self-contained check, and bundle the result.

    `structure_verdict` is one report from `python -m validate judge`; `align_report` is the
    last round of `work/<task>/precision/align_report.json`.
    """
    gates: dict[str, GateStatus] = dict.fromkeys(GATES, "missing")
    problems: list[str] = []

    if structure_verdict is None:
        problems.append("no structure verdict; run `python -m validate judge`")
    else:
        status: GateStatus = "pass" if structure_verdict.get("verdict") == "PASS" else "fail"
        # The frozen judge covers both gates; coverage failures are the "no slot" problems.
        gates["structure"] = status
        gates["coverage"] = status
        problems += [f"structure: {p}" for p in structure_verdict.get("problems") or []]

    if align_report is None:
        problems.append("no alignment report; run core/stages/f5_align/align_driver.py")
    else:
        last = align_report[-1] if isinstance(align_report, list) else align_report
        gates["numeric"] = "pass" if last.get("verdict") == "PASS" else "fail"
        problems += [f"numeric: {p}" for p in last.get("problems") or []]

    contained = check_self_contained(Path(target_dir), source_repo)
    gates["self_contained"] = "pass" if not contained else "fail"
    problems += [f"self_contained: {p}" for p in contained]

    verdict = "PASS" if all(gates[g] == "pass" for g in GATES) else "FAIL"
    return DeliveryBundle(
        model_tag=model_tag, target_dir=str(target_dir), gates=gates, verdict=verdict,
        problems=tuple(problems), evidence_index=evidence, report_path=report_path,
    )


def _load(path: str | None) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="accept_gate.py", description=__doc__)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--target", required=True, help="root of the migrated project")
    p.add_argument("--structure-verdict", help="report from `python -m validate judge`")
    p.add_argument("--align-report", help="work/<task>/precision/align_report.json")
    p.add_argument("--source-repo", help="original repository, for the self-contained check")
    p.add_argument("--evidence", nargs="*", default=[], help="paths to index in the bundle")
    p.add_argument("--out", help="write the bundle here as well as to stdout")
    a = p.parse_args(argv)
    bundle = accept(
        model_tag=a.model_tag, target_dir=Path(a.target),
        structure_verdict=_load(a.structure_verdict), align_report=_load(a.align_report),
        source_repo=Path(a.source_repo) if a.source_repo else None,
        evidence=tuple(a.evidence), report_path=a.out,
    )
    payload = {"verdict": bundle.verdict, "model_tag": bundle.model_tag,
               "target_dir": bundle.target_dir, "gates": bundle.gates,
               "problems": list(bundle.problems), "evidence_index": list(bundle.evidence_index),
               "ci_trigger_command": bundle.ci_trigger_command}
    text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if bundle.verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
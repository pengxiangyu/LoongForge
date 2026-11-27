"""Hands the slots F4 cannot write by itself to a coding agent, under a typed request.

Two of the four write actions do not finish on their own, by design:

  reimplement       the writer marks the target shape and stops; deciding what the target
                    structure should compute needs a reader of both codebases
  needs_merge       a shared framework file already carries the other embodied models, so the
                    `.f4-proposal` beside it is merged by review, not automatically

`build_request()` collects exactly those slots out of the F4 stage report and turns them into one
request: which worktree to write in, which paths may be touched, which skill playbooks to follow,
and what the previous attempt got wrong. The source tree is passed read-only and is not in the
write allowlist, so an agent cannot "fix" an alignment failure by editing the original model.

The agent's receipt is not a verdict. It records what was written; whether the result is accepted
is decided afterwards by the frozen structure judge and, for the loss, by F5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.invoke import EmbodiedSkillName

OPEN_STATES: tuple[str, ...] = ("needs_implementation", "needs_merge")


@dataclass(frozen=True, slots=True)
class SkillBinding:
    """Pinned reference to a skill playbook, read by the agent and never imported as code."""

    name: EmbodiedSkillName
    skill_path: str
    repo_commit: str
    role: str


@dataclass(frozen=True, slots=True)
class MigrationHandoffRequest:
    """One dispatch to a coding agent, scoped to the slots F4 left open."""

    request_id: str
    model_tag: str
    read_only_source_dir: Path
    target_dir: Path
    open_slots: tuple[dict[str, Any], ...]
    allowed_write_globs: tuple[str, ...]
    skills: tuple[SkillBinding, ...] = ()
    anatomy_path: str | None = None
    mapping_path: str | None = None
    repair_hints: tuple[dict[str, str], ...] = ()
    budget_turns: int = 30

    def __post_init__(self) -> None:
        if not self.open_slots:
            raise ValueError("nothing to hand off: no slot is in an open state")
        if not self.allowed_write_globs:
            raise ValueError("a handoff without a write allowlist cannot be scoped")
        source = str(self.read_only_source_dir)
        if any(glob.startswith(source) for glob in self.allowed_write_globs):
            raise ValueError("the read-only source tree must not be writable")


@dataclass(frozen=True, slots=True)
class MigrationHandoffReceipt:
    """What an agent session reports back; subject to the frozen judges afterwards."""

    request_id: str
    written: tuple[str, ...] = ()
    still_open: tuple[str, ...] = ()
    summary: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class CodingAgentAdapter(Protocol):
    """Protocol for invoking an external coding agent CLI."""

    backend_name: str

    def execute(self, request: MigrationHandoffRequest) -> MigrationHandoffReceipt:
        """Run the agent inside `request.target_dir` and return what it wrote."""
        ...


def build_request(
    report: dict[str, Any],
    *,
    request_id: str,
    source_dir: Path,
    target_dir: Path,
    skills: tuple[SkillBinding, ...] = (),
    anatomy_path: str | None = None,
    mapping_path: str | None = None,
    repair_hints: tuple[dict[str, str], ...] = (),
) -> MigrationHandoffRequest:
    """Build a handoff from an F4 stage report (`migrate.py --report`).

    The write allowlist is derived from the open slots themselves: an agent asked to implement
    three files may write those three files and the proposal beside them, nothing else.
    """
    open_slots = tuple(record for record in report.get("writes") or []
                       if record.get("status") in OPEN_STATES)
    globs: list[str] = []
    for record in open_slots:
        globs.append(record["dst"])
        if record.get("status") == "needs_merge":
            globs.append(record["dst"] + ".f4-proposal")
    return MigrationHandoffRequest(
        request_id=request_id,
        model_tag=report.get("model_tag", ""),
        read_only_source_dir=source_dir,
        target_dir=target_dir,
        open_slots=open_slots,
        allowed_write_globs=tuple(dict.fromkeys(globs)),
        skills=skills,
        anatomy_path=anatomy_path,
        mapping_path=mapping_path,
        repair_hints=repair_hints,
    )


__all__ = [
    "CodingAgentAdapter",
    "MigrationHandoffReceipt",
    "MigrationHandoffRequest",
    "OPEN_STATES",
    "SkillBinding",
    "build_request",
]

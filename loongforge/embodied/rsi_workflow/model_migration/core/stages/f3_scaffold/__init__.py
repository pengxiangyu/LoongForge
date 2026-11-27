"""F3: generate the target project skeleton, in one-to-one correspondence with the anatomy.

  scaffold.py     creates the tree from the anatomy and TREE_SPEC.yaml
  TREE_SPEC.yaml  the only description of the target layout; swapping frameworks edits this file
  check_tree.py   missing-slot and surplus-file check over the result
  templates/      one blank template per slot: responsibility and signatures, no logic

Two ways to reach that anatomy, and they end at the same file format:

  hand-written  a `STEP_ANATOMY.yaml` with segments and slots, passed to `scaffold.py --anatomy`
  from F1/F2    `ScaffoldPlan.from_intake()` reads the sealed intake -- and, when supplied, the
                sealed anatomy -- through the independent checkers, then `plan_to_anatomy()`
                turns the accepted per-extension-point slot mapping into the same shape

Neither path is allowed to invent a landing site. `from_intake` refuses to build on anything the
checkers did not accept; `plan_to_anatomy` refuses a slot the layout spec does not declare and a
target file the analysis did not claim, reporting the problem instead of guessing. The two axes
it needs -- which segment a slot belongs to, which extension point owns it -- both live in
TREE_SPEC.yaml, so pointing this stage at another framework stays a one-file change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

LoongForgeExtensionPoint: TypeAlias = Literal[
    "model_registry",
    "data_and_transforms",
    "config_map",
    "trainer_builder",
    "eval_adapter",
    "regression_runner",
]

LOONGFORGE_EXTENSION_POINTS: tuple[LoongForgeExtensionPoint, ...] = (
    "model_registry",
    "data_and_transforms",
    "config_map",
    "trainer_builder",
    "eval_adapter",
    "regression_runner",
)
REQUIRED_SEGMENTS: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
SEGMENT_IDS: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S3.5", "S4", "S5", "S6", "S7")
OPTIONAL_BRANCHES: tuple[str, ...] = ("W", "M", "E-infer", "E-eval")


@dataclass(frozen=True, slots=True)
class ExtensionPointBinding:
    """Maps one LoongForge extension point to concrete target paths and symbols."""

    extension_point: LoongForgeExtensionPoint
    registry_path: str
    target_files: tuple[str, ...] = ()
    reuse_components: tuple[str, ...] = ()
    notes: str = ""
    side: str = ""
    slot_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScaffoldPlan:
    """Declarative integration plan targeting `baidu-baige/LoongForge`."""

    model_name: str
    target_framework_commit: str
    intake_manifest_sha256: str
    anatomy_manifest_sha256: str
    bindings: tuple[ExtensionPointBinding, ...] = ()
    recommended_skills: tuple[str, ...] = (
        "model-adaptation",
        "precision-alignment",
        "vla-model-eval-adapter",
    )
    metadata: dict[str, str] = field(default_factory=dict)
    anatomy_reference_status: Literal["verified", "unverified_external"] = "unverified_external"

    @classmethod
    def from_intake(
        cls, *, intake_package: Path, intake_manifest_sha256: str,
        anatomy_manifest_sha256: str, source_commit: str,
        anatomy_package_dir: Path | None = None, spec_manifest_sha256: str | None = None,
    ) -> ScaffoldPlan:
        """Consume only independently accepted v2 mappings; no hardcoded slot IDs."""
        from core.artifacts import PackageRef, verify
        from core.contracts import IntakeTaskPackV2, ModelAnalysisReport, load_intake_task_pack
        from core.contracts.common import load_yaml_document
        from validate.base import CheckContext
        from validate.intake_check import IntakeChecker
        from validate.anatomy_check import AnatomyChecker
        from pydantic import TypeAdapter
        from core.contracts.common import Sha256
        TypeAdapter(Sha256).validate_python(anatomy_manifest_sha256)
        gate = IntakeChecker().check(CheckContext(
            commit=source_commit, manifest_sha256=intake_manifest_sha256, package_dir=intake_package,
        ))
        if gate.status != "pass":
            raise ValueError(f"ScaffoldPlan requires accepted intake: {gate.reason_codes}")
        verified = verify(PackageRef(package_dir=intake_package, manifest_sha256=intake_manifest_sha256))
        task = load_intake_task_pack(intake_package / "inputs/task.yaml")
        if not isinstance(task, IntakeTaskPackV2):
            raise ValueError("v1 intake has no soft slot mapping")
        reference_status = "unverified_external"
        if anatomy_package_dir is not None:
            anatomy_gate = AnatomyChecker().check(CheckContext(
                commit=source_commit, manifest_sha256=anatomy_manifest_sha256, package_dir=anatomy_package_dir,
                parameters={"intake_package_dir": str(intake_package), "intake_manifest_sha256": intake_manifest_sha256,
                            "spec_manifest_sha256": spec_manifest_sha256},
            ))
            if anatomy_gate.status != "pass":
                raise ValueError(f"ScaffoldPlan requires accepted anatomy: {anatomy_gate.reason_codes}")
            anatomy_manifest_sha256 = anatomy_gate.binding.manifest_sha256
            reference_status = "verified"
        report = ModelAnalysisReport.model_validate(
            load_yaml_document(intake_package / "analysis/model_analysis.json"),
        )
        bindings = []
        for extension in task.slot_catalog.extensions:
            slots = [x for x in report.slots if x.extension_point == extension.extension_point]
            bindings.append(ExtensionPointBinding(
                extension_point=extension.extension_point, registry_path=extension.registry_path,
                target_files=tuple(dict.fromkeys(path for slot in slots for path in slot.target_files)),
                reuse_components=tuple(extension.reuse_components), side=extension.side,
                slot_ids=tuple(x.slot_id for x in slots),
                notes="; ".join(x.reason for x in slots if x.reason),
            ))
        return cls(
            model_name=task.source.model_variant, target_framework_commit=task.target.commit,
            intake_manifest_sha256=verified.ref.manifest_sha256,
            anatomy_manifest_sha256=anatomy_manifest_sha256, bindings=tuple(bindings),
            anatomy_reference_status=reference_status,
            metadata={"compatibility_profile": "internal", "handoff_boundary": "handoff_only"},
        )


def plan_to_anatomy(
    plan: ScaffoldPlan,
    *,
    actions: dict[str, str] | None = None,
    tree_spec_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Turn an accepted `ScaffoldPlan` into the anatomy that `scaffold()` and the judge read.

    Returns `(anatomy, problems)`. A non-empty `problems` list means the plan cannot be built
    from without guessing, and the caller must stop: every entry is a slot the layout spec does
    not declare, a target file the analysis never claimed, or a required segment left uncovered.
    `actions` overrides the per-slot migration action (default `adapt`), which F4 reads back.
    """
    from .scaffold import TREE_SPEC, load_yaml, resolve_target

    spec = load_yaml(tree_spec_path or TREE_SPEC)
    slot_segments: dict[str, str] = spec["slot_segments"]
    owners = {point: set(body["slots"]) for point, body in spec["extension_points"].items()}
    actions = dict(actions or {})
    segments: dict[str, list[dict[str, Any]]] = {}
    branches: dict[str, list[dict[str, Any]]] = {}
    problems: list[str] = []

    for binding in plan.bindings:
        owned = owners.get(binding.extension_point, set())
        for slot_id in binding.slot_ids:
            where = f"{binding.extension_point}.{slot_id}"
            if slot_id not in slot_segments:
                problems.append(f"{where}: slot is not declared in TREE_SPEC.slot_segments")
                continue
            if slot_id not in owned:
                problems.append(f"{where}: TREE_SPEC assigns this slot to another extension point")
                continue
            target = resolve_target(spec, slot_id, plan.model_name, None)
            if target not in binding.target_files:
                problems.append(f"{where}: layout target {target!r} is not among the analysed "
                                f"target files {list(binding.target_files)}")
                continue
            entry = {"slot": slot_id, "src": binding.registry_path, "dst": target,
                     "action": actions.get(slot_id, "adapt"),
                     "reason": binding.notes or f"slot reported by the F1 analysis for {where}"}
            segment = slot_segments[slot_id]
            owner = segments if segment in SEGMENT_IDS else branches
            owner.setdefault(segment, []).append(entry)

    anatomy: dict[str, Any] = {
        "version": 4,
        "model_tag": plan.model_name,
        "source": {"repo": plan.metadata.get("source_repo", ""),
                   "commit": plan.target_framework_commit},
        "segments": [{"id": sid, "required": sid in REQUIRED_SEGMENTS,
                      "slots": segments.get(sid, []),
                      **({} if segments.get(sid) else
                         {"empty_reason": f"the F1 analysis reported no slot for {sid}"})}
                     for sid in SEGMENT_IDS],
        "branches": [{"id": bid, "enabled": bool(branches.get(bid)),
                      **({"slots": branches[bid]} if branches.get(bid) else
                         {"empty_reason": f"the F1 analysis reported no slot for branch {bid}"})}
                     for bid in OPTIONAL_BRANCHES],
        "provenance": {"intake_manifest_sha256": plan.intake_manifest_sha256,
                       "anatomy_manifest_sha256": plan.anatomy_manifest_sha256,
                       "anatomy_reference_status": plan.anatomy_reference_status},
    }
    problems += [f"required segment {sid} has no slot: the plan does not cover it"
                 for sid in REQUIRED_SEGMENTS if not segments.get(sid)]
    return anatomy, problems


__all__ = [
    "ExtensionPointBinding",
    "LOONGFORGE_EXTENSION_POINTS",
    "LoongForgeExtensionPoint",
    "OPTIONAL_BRANCHES",
    "REQUIRED_SEGMENTS",
    "SEGMENT_IDS",
    "ScaffoldPlan",
    "plan_to_anatomy",
]


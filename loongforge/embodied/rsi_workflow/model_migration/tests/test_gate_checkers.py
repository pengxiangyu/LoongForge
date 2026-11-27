"""The construct, parity, train_resume and task_eval gates must judge, not merely exist.

None of the four owns a verdict: they delegate to the frozen structure judge and numeric gate
and wrap the report in a provenance-bound `GateResult`. What needs pinning down is the seam --
that a real report reaches `judger.decide`, that each verdict maps to the outcome it means,
and that the binding is enforced rather than decorative.
"""

import json
from pathlib import Path

from core.orchestrator.judger import decide
from validate import (
    CheckContext,
    ConstructChecker,
    EvalChecker,
    GateRunner,
    ParityChecker,
    TrainResumeChecker,
)
from validate.accept_gate import GATES, accept
from validate.trajectory import losses

REPO_ROOT = Path(__file__).parents[1]
THRESHOLDS = REPO_ROOT / "validate" / "thresholds.yaml"
ALIGNED_CASE = REPO_ROOT / "validate" / "selftest" / "cases" / "structure_aligned.json"
ALIGN_CASES = REPO_ROOT / "core" / "stages" / "f5_align" / "cases"

COMMIT = "7f3c2a9d4e5b6f70128394aa65b7c8d9e0f1a2b3"
MANIFEST = "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace"


def _context(package_dir, **parameters):
    return CheckContext(
        commit=COMMIT,
        manifest_sha256=MANIFEST,
        package_dir=package_dir,
        parameters=parameters,
    )


def _materialise(root, rel_paths):
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return root


def _aligned_anatomy():
    return json.loads(ALIGNED_CASE.read_text(encoding="utf-8"))


def _align_case(name):
    case = json.loads((ALIGN_CASES / name).read_text(encoding="utf-8"))
    return losses(case["src"]), losses(case["tgt"])


def test_construct_gate_passes_an_aligned_tree_and_binds_the_judge(tmp_path):
    case = _aligned_anatomy()
    tree = _materialise(tmp_path / "target", case["tree"])
    checker = ConstructChecker()

    result = checker.check(
        _context(
            tmp_path,
            anatomy=case["anatomy"],
            target_root=tree,
            weight_report=case["weight_report"],
        )
    )

    assert result.status == "pass"
    assert result.gate_id == "construct"
    assert result.binding.validator_id == "validate.construct_check"
    assert result.binding.validator_sha256 == checker.validator_sha256()
    assert result.metrics["slots"] == len(case["tree"]) - 1  # __init__.py carries no slot

    decision = decide(
        [result],
        expected_commit=COMMIT,
        expected_manifest_sha256=MANIFEST,
        trusted_validators={checker.validator_id: checker.validator_sha256()},
    )
    assert decision.status == "pass"


def test_construct_gate_is_repairable_when_a_slot_file_is_missing(tmp_path):
    case = _aligned_anatomy()
    tree = _materialise(tmp_path / "target", case["tree"])
    (tree / "model/sample_vla/modules/action_dit.py").unlink()

    result = ConstructChecker().check(
        _context(
            tmp_path,
            anatomy=case["anatomy"],
            target_root=tree,
            weight_report=case["weight_report"],
        )
    )

    assert result.status == "repairable"
    assert result.reason_codes == ("STRUCTURE_MISMATCH",)
    assert "missing slot file" in result.first_mismatch


def test_parity_gate_maps_each_align_verdict_to_its_outcome(tmp_path):
    import yaml

    thresholds = yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8"))
    checker = ParityChecker()
    expected = {
        "aligned_case.json": ("pass", ()),
        "broken_case.json": ("repairable", ("NUMERIC_LIMIT_EXCEEDED",)),
    }
    for name, (status, reason_codes) in expected.items():
        src, tgt = _align_case(name)
        result = checker.check(
            _context(tmp_path, src_losses=src, tgt_losses=tgt, thresholds=thresholds)
        )
        assert result.status == status, name
        assert result.reason_codes == reason_codes, name


def test_parity_gate_blocks_when_the_traces_are_not_comparable(tmp_path):
    import yaml

    thresholds = yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8"))
    result = ParityChecker().check(
        _context(
            tmp_path,
            src_losses={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
            tgt_losses={7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
            thresholds=thresholds,
        )
    )

    assert result.status == "blocked"
    assert result.reason_codes == ("TRACES_NOT_COMPARABLE",)


def test_gate_runner_refuses_a_result_whose_validator_is_not_trusted(tmp_path):
    case = _aligned_anatomy()
    tree = _materialise(tmp_path / "target", case["tree"])
    checker = ConstructChecker()
    context = _context(
        tmp_path,
        anatomy=case["anatomy"],
        target_root=tree,
        weight_report=case["weight_report"],
    )

    runner = GateRunner(
        [checker],
        trusted_validators={checker.validator_id: "0" * 64},
        required_gates=["construct"],
    )
    decision = runner.evaluate(context)

    assert decision.status == "error"
    assert "VALIDATOR_SHA_MISMATCH" in decision.reason_codes


def test_accept_gate_reports_missing_and_failed_gates_never_pass(tmp_path):
    bundle = accept(
        model_tag="sample_vla",
        target_dir=_materialise(tmp_path / "target", ["model/registry.py"]),
        structure_verdict=None,
        align_report=None,
        source_repo=None,
    )

    assert bundle.verdict == "FAIL"
    assert bundle.gates["structure"] == "missing"
    assert bundle.gates["coverage"] == "missing"
    assert bundle.gates["numeric"] == "missing"
    assert set(bundle.gates) == set(GATES)
    assert any("no structure verdict" in problem for problem in bundle.problems)


def _details(result):
    """The problem texts a checker recorded, read back out of its evidence."""
    return [item.detail for item in result.evidence if item.key.startswith("problem[")]


def _trace(steps, *, resume=None):
    run = {
        "data_id": "sample_vla/val_shard_000:sha256:3f1c",
        "order_hash": "sampler=seq,seed=1234:sha256:9ab2",
        "ckpt_hash": "sha256:0d41d8cd98f00b204e9800998ecf8427",
        "launch": "micro_batch=2 precision=bf16 seq_len=2048 seed=1234 lr=2.5e-5",
    }
    if resume is not None:
        run["resume"] = resume
    return {
        "schema": "vma-loss-trace@1",
        "run": run,
        "steps": [{"step": step, "loss": loss} for step, loss in sorted(steps.items())],
    }


def test_train_resume_passes_when_the_resume_is_transparent(tmp_path):
    import yaml

    continuous = {step: 2.0 - step * 0.01 for step in range(1, 21)}
    resumed = dict(continuous)
    checker = TrainResumeChecker()

    result = checker.check(
        _context(
            tmp_path,
            continuous=_trace(continuous),
            resumed=_trace(resumed, resume={"from_step": 6, "ckpt_hash": "sha256:0d41d8"}),
            thresholds=yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8")),
        )
    )

    assert result.status == "pass"
    assert result.metrics["from_step"] == 6
    assert result.metrics["shared_pre"] == 5
    assert result.metrics["pre_max_abs_diff"] == 0.0
    assert result.metrics["post_verdict"] == "PASS"


def test_train_resume_is_repairable_when_the_pre_resume_steps_differ(tmp_path):
    import yaml

    continuous = {step: 2.0 - step * 0.01 for step in range(1, 21)}
    resumed = dict(continuous)
    resumed[3] += 0.5  # the resume restored the weights but not the sample order
    checker = TrainResumeChecker()

    result = checker.check(
        _context(
            tmp_path,
            continuous=_trace(continuous),
            resumed=_trace(resumed, resume={"from_step": 6, "ckpt_hash": "sha256:0d41d8"}),
            thresholds=yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8")),
        )
    )

    assert result.status == "repairable"
    assert result.reason_codes == ("RESUME_DIVERGENCE",)
    assert "pre-resume step 3" in result.first_mismatch


def test_train_resume_is_repairable_when_the_resume_is_not_declared(tmp_path):
    import yaml

    steps = {step: 2.0 - step * 0.01 for step in range(1, 21)}
    result = TrainResumeChecker().check(
        _context(
            tmp_path,
            continuous=_trace(steps),
            resumed=_trace(steps),
            thresholds=yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8")),
        )
    )

    assert result.status == "repairable"
    assert "does not declare where it restarted" in result.first_mismatch


def test_eval_gate_passes_a_complete_report_with_no_regression(tmp_path):
    report = {
        "model_tag": "sample_vla",
        "suite": "libero",
        "triad": {
            "model_factory": "eval/factories/sample_vla_factory.py",
            "payload_builder": "libero",
            "action_decoder": "ee6d",
        },
        "tasks": {"libero_goal/0": 0.92, "libero_goal/1": 0.88},
    }
    baseline = {"libero_goal/0": 0.92, "libero_goal/1": 0.88}

    result = EvalChecker().check(
        _context(tmp_path, report=report, baseline=baseline)
    )

    assert result.status == "pass"
    assert result.metrics["tasks"] == 2


def test_eval_gate_blocks_without_a_baseline_and_flags_a_skipped_triad_part(tmp_path):
    report = {
        "model_tag": "sample_vla",
        "suite": "libero",
        "triad": {"model_factory": "eval/factories/sample_vla_factory.py"},
        "tasks": {"libero_goal/0": 0.92},
    }

    blocked = EvalChecker().check(_context(tmp_path, report=report))
    assert blocked.status == "blocked"
    assert blocked.reason_codes == ("EVAL_BASELINE_MISSING",)
    # The incomplete triad is recorded as a problem even though the baseline dominates.
    assert any("action_decoder" in detail for detail in _details(blocked))


def test_eval_gate_is_repairable_when_a_task_regressed(tmp_path):
    report = {
        "model_tag": "sample_vla",
        "suite": "libero",
        "triad": {"model_factory": "f", "payload_builder": "p", "action_decoder": "a"},
        "tasks": {"libero_goal/0": 0.50, "libero_goal/1": {"status": "skipped"}},
    }
    baseline = {"libero_goal/0": 0.92}

    result = EvalChecker().check(
        _context(tmp_path, report=report, baseline=baseline, regression_tol=0.01)
    )

    assert result.status == "repairable"
    assert result.reason_codes == ("EVAL_INCOMPLETE",)
    details = _details(result)
    assert any("regressed" in detail for detail in details)
    assert any("skipped" in detail for detail in details)
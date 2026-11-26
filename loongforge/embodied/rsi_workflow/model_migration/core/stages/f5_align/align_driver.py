"""Drives both training paths through the skill and the frozen gate, and writes the align report.

The sequence, and why it is this one:

  1. the self-test runs first. A gate that has not just been shown a failing case is not evidence.
  2. collect each path's loss trace through `precision-alignment` (the skill owns collection).
  3. check the four-same fingerprints of the two collected traces. Checking after collection is
     deliberate: what matters is what the runs actually did, not what the adapters declared.
  4. freeze both traces. The original's trace is the baseline of the whole argument, and a
     re-collected baseline can turn a failure into a pass without touching the migration.
  5. ask the skill to compare and attribute. Its verdict is recorded as evidence.
  6. judge with the frozen gate against the frozen thresholds. That verdict is the one that counts.

Every round is appended to `align_report.json` rather than replacing it, so the sequence of fix
attempts stays readable after the fact.

Exit codes: 0 PASS, 2 FAIL, 3 the comparison was not possible, 4 a skill call is staged.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.invoke import PENDING, PendingSkillCall                                  # noqa: E402
from core.invoke.call_precision_alignment import (collect_precision,               # noqa: E402
                                                  compare_and_analyze)
from core.stages.f5_align.four_same_check import check                             # noqa: E402
from validate import record, trajectory                                            # noqa: E402
from validate.selftest.run_selftest import run_selftest                            # noqa: E402

REPORT_NAME = 'align_report.json'


def _append(path, round_report):
    """Append a round to the align report, numbering it by how many rounds came before."""
    existing = json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else []
    round_report['round'] = len(existing) + 1
    existing.append(round_report)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(existing, indent=1, ensure_ascii=False) + '\n',
                          encoding='utf-8')
    return round_report['round']


def judge(src_trace_path, tgt_trace_path, thresholds):
    from validate.frozen import align_gate_frozen, freeze
    freeze.require_unlocked_judges()
    src, tgt = trajectory.load(src_trace_path), trajectory.load(tgt_trace_path)
    same = check(trajectory.run_meta(src), trajectory.run_meta(tgt))
    if not same['ok']:
        return {'verdict': 'ERROR', 'problems': same['problems'], 'four_same': same}, same
    gate = align_gate_frozen.evaluate(trajectory.losses(src), trajectory.losses(tgt), thresholds)
    gate['four_same'] = same
    return gate, same


def drive(src_adapter, tgt_adapter, workdir, thresholds, skip_selftest=False):
    """One alignment round. Returns the round report; raises PendingSkillCall while waiting."""
    round_report = {'paths': {'src': src_adapter.name, 'tgt': tgt_adapter.name}}
    if not skip_selftest:
        selftest = run_selftest()
        round_report['selftest'] = {'verdict': selftest['verdict'],
                                    'problems': selftest['problems']}
        if selftest['verdict'] != 'PASS':
            round_report['verdict'] = 'ERROR'
            round_report['problems'] = ['the judges failed their own self-test; nothing was judged']
            return round_report
    traces = {}
    for adapter in (src_adapter, tgt_adapter):
        result = collect_precision(adapter.run_cmd, adapter.workdir, adapter.steps, workdir)
        traces[adapter.name] = result['loss_trace']
    precision_dir = Path(workdir) / 'precision'
    frozen = {name: record.freeze(path, precision_dir, f'loss_trace_{name}.json')
              for name, path in traces.items()}
    round_report['traces'] = {name: str(precision_dir / f'loss_trace_{name}.json')
                              for name in traces}
    round_report['frozen'] = frozen
    src_path = precision_dir / f'loss_trace_{src_adapter.name}.json'
    tgt_path = precision_dir / f'loss_trace_{tgt_adapter.name}.json'
    round_report['evidence'] = compare_and_analyze(src_path, tgt_path, workdir)
    gate, _ = judge(src_path, tgt_path, thresholds)
    round_report.update(gate)
    return round_report


def main(argv=None):
    p = argparse.ArgumentParser(prog='align_driver.py', description=__doc__)
    p.add_argument('--adapters', required=True,
                   help='JSON with the two paths; see validate/adapter.py')
    p.add_argument('--workdir', required=True, help='work/<task_id>')
    p.add_argument('--thresholds', help='defaults to the frozen thresholds.yaml')
    p.add_argument('--skip-selftest', action='store_true',
                   help='only for debugging the driver itself; a real verdict needs the self-test')
    p.add_argument('--out', help=f'defaults to <workdir>/precision/{REPORT_NAME}')
    a = p.parse_args(argv)
    import yaml
    from validate.adapter import load_pair
    thresholds_path = a.thresholds or (Path(__file__).resolve().parents[3] / 'validate' / 'frozen'
                                       / 'thresholds.yaml')
    thresholds = yaml.safe_load(Path(thresholds_path).read_text(encoding='utf-8'))
    src_adapter, tgt_adapter = load_pair(a.adapters)
    try:
        round_report = drive(src_adapter, tgt_adapter, a.workdir, thresholds, a.skip_selftest)
    except PendingSkillCall as pending:
        print(pending.instructions())
        return PENDING
    out = a.out or Path(a.workdir) / 'precision' / REPORT_NAME
    _append(out, round_report)
    print(json.dumps(round_report, indent=1, ensure_ascii=False))
    return {'PASS': 0, 'FAIL': 2, 'ERROR': 3}[round_report['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())

"""Routes the numeric verdict: accept it, or send it back to F4 with the skill's hints.

A failing gate has exactly one legitimate response -- change the target implementation -- so this
module turns a FAIL into work items rather than into a judgement call. It asks
precision-alignment for `propose_fix`, keys the hints by slot so each one lands on the writer that
owns it, and appends them to the task's work log. The thresholds are not an input it can touch:
they are hashed in `validate/frozen/`, and a run that edits them fails the lock instead of passing
the gate.

Exit codes: 0 the gate passed, 2 back to F4 (hints written), 3 the gate could not judge,
4 the skill call is staged and waiting.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.invoke import PENDING, PendingSkillCall                              # noqa: E402
from core.invoke.call_precision_alignment import propose_fix                   # noqa: E402

NEXT = {'PASS': 'accept', 'FAIL': 'back_to_f4', 'ERROR': 'stop'}


def route(gate_report, hints):
    """Decide what happens next and attach the per-slot hints when it is a failure."""
    verdict = gate_report['verdict']
    decision = {'verdict': verdict, 'next': NEXT[verdict],
                'metrics': gate_report.get('metrics', {}),
                'problems': gate_report.get('problems', []), 'hints': hints or []}
    if verdict == 'FAIL':
        decision['slots'] = sorted({h['slot'] for h in (hints or []) if h.get('slot')})
    return decision


def work_log(decision, path):
    lines = [f'## F5 align: {decision["verdict"]} -> {decision["next"]}',
             f'metrics: {json.dumps(decision["metrics"], ensure_ascii=False)}']
    lines += [f'- problem: {p}' for p in decision['problems']]
    lines += [f'- fix {h.get("slot")}: {h.get("reason")} -> {h.get("suggestion")}'
              for h in decision['hints']]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n\n')


def main(argv=None):
    p = argparse.ArgumentParser(prog='align_gate.py', description=__doc__)
    p.add_argument('--gate-report', required=True, help='report from `python -m validate align`')
    p.add_argument('--compare-result', help='compare_and_analyze result, input to propose_fix')
    p.add_argument('--workdir', required=True, help='work/<task_id>')
    p.add_argument('--work-log', help='append the decision here (logs/WORK_LOG.md)')
    p.add_argument('--out')
    a = p.parse_args(argv)
    gate = json.loads(Path(a.gate_report).read_text(encoding='utf-8'))
    hints = []
    if gate['verdict'] == 'FAIL' and a.compare_result:
        compare = json.loads(Path(a.compare_result).read_text(encoding='utf-8'))
        try:
            hints = propose_fix(compare, a.workdir).get('hints') or []
        except PendingSkillCall as pending:
            print(pending.instructions())
            return PENDING
    decision = route(gate, hints)
    if a.work_log:
        work_log(decision, a.work_log)
    text = json.dumps(decision, indent=1, ensure_ascii=False) + '\n'
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding='utf-8')
    print(text, end='')
    return {'PASS': 0, 'FAIL': 2, 'ERROR': 3}[decision['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())

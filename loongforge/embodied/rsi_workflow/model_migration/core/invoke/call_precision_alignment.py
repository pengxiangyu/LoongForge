"""The one place that calls precision-alignment. F5 reaches the skill only through here.

The skill collects the per-step loss of each path, compares the two traces and attributes the
deviation. What it returns is evidence: `compare_and_analyze` carries its own `verdict` field, and
F5 records it but does not act on it -- the numeric verdict belongs to the frozen gate against the
frozen thresholds, which is the only way a one-percent limit stays a one-percent limit.

`propose_fix` closes the loop: its hints are keyed by slot, so a failing gate routes straight back
to the F4 writer that owns that slot instead of starting a free-form search for the cause.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.invoke import PENDING, PendingSkillCall, call               # noqa: E402

SKILL = 'precision-alignment'


def collect_precision(run_cmd, workdir, steps, task_workdir):
    """Returns {'loss_trace': path} for one training path."""
    return call(SKILL, 'collect_precision',
                {'run_cmd': run_cmd, 'workdir': str(workdir), 'steps': int(steps)}, task_workdir)


def compare_and_analyze(trace_a, trace_b, task_workdir):
    """Returns {'diff_max', 'diff_p95', 'diff_mean', 'verdict', 'evidence'} as evidence only."""
    return call(SKILL, 'compare_and_analyze',
                {'trace_a': str(trace_a), 'trace_b': str(trace_b)}, task_workdir)


def propose_fix(compare_result, task_workdir):
    """Returns {'hints': [{slot, reason, suggestion}, ...]} to route back to F4."""
    return call(SKILL, 'propose_fix', {'compare_result': compare_result}, task_workdir)


def main(argv=None):
    p = argparse.ArgumentParser(prog='call_precision_alignment.py', description=__doc__)
    p.add_argument('--capability', required=True,
                   choices=('collect_precision', 'compare_and_analyze', 'propose_fix'))
    p.add_argument('--workdir', required=True, help='work/<task_id>')
    p.add_argument('--run-cmd', help='collect_precision: the launch command, verbatim')
    p.add_argument('--run-workdir', help='collect_precision: directory the command runs in')
    p.add_argument('--steps', type=int, help='collect_precision: steps to collect')
    p.add_argument('--trace-a', help='compare_and_analyze: the original path trace')
    p.add_argument('--trace-b', help='compare_and_analyze: the target path trace')
    p.add_argument('--compare-result', help='propose_fix: the compare_and_analyze result file')
    p.add_argument('--out')
    a = p.parse_args(argv)
    try:
        if a.capability == 'collect_precision':
            result = collect_precision(a.run_cmd, a.run_workdir, a.steps, a.workdir)
        elif a.capability == 'compare_and_analyze':
            result = compare_and_analyze(a.trace_a, a.trace_b, a.workdir)
        else:
            compare = json.loads(Path(a.compare_result).read_text(encoding='utf-8'))
            result = propose_fix(compare, a.workdir)
    except PendingSkillCall as pending:
        print(pending.instructions())
        return PENDING
    text = json.dumps(result, indent=1, ensure_ascii=False) + '\n'
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

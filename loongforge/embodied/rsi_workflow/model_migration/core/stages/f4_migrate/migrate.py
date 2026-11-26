"""F4 stage entry: dispatch the mapping, place the data side, convert the weights, log every write.

The order is not arbitrary. The mapping is checked before anything is written, because a table
that does not hold together will otherwise be discovered halfway through a half-written tree. The
data side goes through model-adaptation rather than through the generic writers, because its
correctness is a pipeline question, not a file-moving one. The weights are converted last, since
their report is an input to the structure judge.

Nothing here decides whether F4 succeeded. The verdict comes from the frozen structure judge, run
afterwards (`python -m validate judge`); this stage reports what it wrote and which slots still
need a human: `needs_merge` for the shared framework files, `needs_implementation` for the
rewrites.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.stages.f3_scaffold.scaffold import TREE_SPEC, load_yaml                  # noqa: E402
from core.stages.f4_migrate.map.build_mapping import check                         # noqa: E402
from core.stages.f4_migrate.writers import dispatch                                # noqa: E402

OPEN_STATES = ('needs_merge', 'needs_implementation')


def run(anatomy, spec, mapping, source_root, target_root, force=False, skip=()):
    """Dispatch every mapping entry. Returns the stage report."""
    entries = [e for e in (mapping.get('entries') or []) if e['seg'] not in skip]
    problems = check(mapping.get('entries') or [], anatomy, spec)
    report = {'verdict': 'FAIL' if problems else 'PASS', 'model_tag': anatomy['model_tag'],
              'target_root': str(target_root), 'writes': [], 'open': [], 'problems': problems}
    if problems:
        return report
    shared = set(spec.get('shared_targets') or [])
    source = mapping.get('source') or anatomy.get('source') or {}
    for entry in entries:
        record = dispatch(entry, source_root, target_root, source, shared, force)
        report['writes'].append(record)
        if record['status'] == 'error':
            report['problems'].append(f'{record["seg"]}.{record["slot"]}: {record.get("note")}')
        elif record['status'] in OPEN_STATES:
            report['open'].append({'dst': record['dst'], 'status': record['status'],
                                   'note': record.get('note')})
    report['verdict'] = 'PASS' if not report['problems'] else 'FAIL'
    return report


def work_log(report, path):
    """Append one line per write to the task's work log; failures and open items included."""
    lines = [f'## F4 migrate {report["model_tag"]} -> {report["target_root"]}',
             f'verdict: {report["verdict"]}, writes: {len(report["writes"])}, '
             f'open: {len(report["open"])}']
    for record in report['writes']:
        lines.append(f'- {record["seg"]}.{record["slot"]} {record["action"]}: {record["dst"]} '
                     f'[{record["status"]}]'
                     + (f' {record["note"]}' if record.get('note') else ''))
    for problem in report['problems']:
        lines.append(f'- FAILED: {problem}')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n\n')


def main(argv=None):
    p = argparse.ArgumentParser(prog='migrate.py', description=__doc__)
    p.add_argument('--anatomy', required=True, help='STEP_ANATOMY.yaml from F2')
    p.add_argument('--mapping', required=True, help='mapping.yaml from map/build_mapping.py')
    p.add_argument('--source-repo', required=True, help='read-only root of the original model')
    p.add_argument('--target', required=True, help='root of the target project from F3')
    p.add_argument('--tree-spec', default=str(TREE_SPEC))
    p.add_argument('--skip-data', action='store_true',
                   help='leave S1 to dataset_bridge.py, which routes it through the skill')
    p.add_argument('--force', action='store_true', help='replace files written by an earlier round')
    p.add_argument('--work-log', help='append the write log here (work/<task_id>/logs/WORK_LOG.md)')
    p.add_argument('--report')
    a = p.parse_args(argv)
    report = run(load_yaml(a.anatomy), load_yaml(a.tree_spec), load_yaml(a.mapping),
                 a.source_repo, a.target, a.force, skip=('S1',) if a.skip_data else ())
    if a.work_log:
        work_log(report, a.work_log)
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())

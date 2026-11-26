"""Places the data side of the migration from model-adaptation's plan. It does not rewrite it.

The data pipeline is where a migration goes wrong without failing: a different sampling order, a
dropped normalisation, a frame window off by one, and training still converges -- just not to the
original's loss, and not for a reason visible in the diff. Analysing that is what the
model-adaptation skill is for, so this bridge asks for the plan and places exactly what comes
back.

The one thing it does on its own is check the plan against the structure source of truth: every
plan target must be an S1 slot the anatomy already declared. A plan that wants to put a transform
somewhere no slot accounts for stops the stage instead of adding a file the structure gate would
later report as surplus.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.invoke import PENDING, PendingSkillCall                                 # noqa: E402
from core.invoke.call_model_adaptation import adapt_dataset_pipeline              # noqa: E402
from core.stages.f3_scaffold.scaffold import load_yaml                            # noqa: E402
from core.stages.f4_migrate.writers import dispatch                               # noqa: E402

DATA_SEGMENT = 'S1'


def data_slots(anatomy):
    """{target path: slot id} for the observation-assembly segment."""
    for seg in anatomy.get('segments') or []:
        if seg['id'] == DATA_SEGMENT:
            return {slot['dst']: slot['slot'] for slot in seg.get('slots') or []}
    return {}


def to_entries(plan, anatomy):
    """Turn the skill's plan into mapping-shaped entries, rejecting undeclared targets."""
    slots, entries, problems = data_slots(anatomy), [], []
    for item in plan:
        dst = item.get('dst')
        if dst not in slots:
            problems.append(f'plan targets {dst!r}, which no {DATA_SEGMENT} slot declares')
            continue
        entries.append({'seg': DATA_SEGMENT, 'slot': slots[dst], 'src': item['src'], 'dst': dst,
                        'action': item.get('action', 'adapt'),
                        'reason': item.get('notes') or 'model-adaptation dataset plan',
                        'rewrites': item.get('rewrites') or []})
    missing = sorted(set(slots) - {e['dst'] for e in entries})
    problems += [f'plan covers no source for the declared slot {dst!r}' for dst in missing]
    return entries, problems


def bridge(anatomy, plan, source_root, target_root, source, shared=(), force=False):
    entries, problems = to_entries(plan, anatomy)
    records = [] if problems else [dispatch(e, source_root, target_root, source, shared, force)
                                   for e in entries]
    problems += [f'{r["slot"]}: {r.get("note")}' for r in records if r['status'] == 'error']
    return {'verdict': 'PASS' if not problems else 'FAIL', 'writes': records,
            'problems': problems}


def main(argv=None):
    p = argparse.ArgumentParser(prog='dataset_bridge.py', description=__doc__)
    p.add_argument('--anatomy', required=True)
    p.add_argument('--source-repo', required=True, help='read-only root of the original model')
    p.add_argument('--target', required=True, help='root of the target project')
    p.add_argument('--workdir', required=True, help='work/<task_id>, where the skill call is staged')
    p.add_argument('--plan', help='a plan file, instead of calling the skill again')
    p.add_argument('--force', action='store_true')
    p.add_argument('--report')
    a = p.parse_args(argv)
    anatomy = load_yaml(a.anatomy)
    if a.plan:
        plan = json.loads(Path(a.plan).read_text(encoding='utf-8'))
        plan = plan['plan'] if isinstance(plan, dict) else plan
    else:
        try:
            plan = adapt_dataset_pipeline(a.source_repo, Path(a.target) / 'data', a.workdir)['plan']
        except PendingSkillCall as pending:
            print(pending.instructions())
            return PENDING
    report = bridge(anatomy, plan, a.source_repo, a.target, anatomy.get('source', {}),
                    force=a.force)
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())

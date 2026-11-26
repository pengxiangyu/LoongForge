"""F3: generates the target project skeleton from the anatomy, slot by slot.

The only inputs are `STEP_ANATOMY.yaml` (what this model actually has) and `TREE_SPEC.yaml`
(where the target framework puts it). The correspondence lives in those two files and nowhere
else: the templates carry no slot-to-path knowledge, so there is no second copy to drift.

What is generated is a boundary, not an implementation: each file states which segment and slot
it belongs to and the signatures that segment owes the step, and F4 fills in the logic.

Two things it refuses to do. It does not generate a file for a slot the anatomy did not hit, so
coverage cannot be padded. And it never overwrites a file that already exists -- the shared
targets of the target framework (`model/registry.py`, `optimizer/optimizer.py`, ...) already
carry the other embodied models, and for a migration those are to be reused, not rewritten.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from validate.structure_judge import slots_of                         # noqa: E402

STAGE_DIR = Path(__file__).resolve().parent
TREE_SPEC = STAGE_DIR / 'TREE_SPEC.yaml'
TEMPLATES = STAGE_DIR / 'templates'


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def resolve_target(spec, slot_id, model, name):
    """The target path a slot must land on, per TREE_SPEC; None when the slot is not declared."""
    pattern = spec['slot_targets'].get(slot_id)
    if pattern is None:
        return None
    return pattern.replace('{model}', model).replace('{name}', name or '')


def plan(anatomy, spec):
    """One entry per slot plus the problems that make the anatomy unbuildable."""
    model = anatomy['model_tag']
    entries, problems = [], []
    for seg_id, slot in slots_of(anatomy):
        slot_id = slot['slot']
        expected = resolve_target(spec, slot_id, model, slot.get('name'))
        if expected is None:
            problems.append(f'{seg_id}.{slot_id}: not declared in TREE_SPEC.slot_targets')
            continue
        if slot.get('dst') != expected:
            problems.append(f'{seg_id}.{slot_id}: anatomy dst {slot.get("dst")!r} does not match '
                            f'the TREE_SPEC target {expected!r}')
            continue
        template = spec['slot_templates'].get(slot_id)
        if template is None:
            problems.append(f'{seg_id}.{slot_id}: no template declared in TREE_SPEC')
            continue
        entries.append({'seg': seg_id, 'slot': slot_id, 'name': slot.get('name'), 'dst': expected,
                        'template': template, 'action': slot.get('action'),
                        'shared': expected in (spec.get('shared_targets') or [])})
    return entries, problems


def camel(text):
    """sample_vla -> SampleVla, for the class names the templates declare."""
    return ''.join(part[:1].upper() + part[1:] for part in (text or '').split('_') if part)


def render(template_path, model, name, entry):
    text = Path(template_path).read_text(encoding='utf-8')
    for token, value in (('{model}', model), ('{Model}', camel(model)), ('{name}', name or ''),
                         ('{Name}', camel(name)), ('{seg}', entry['seg']),
                         ('{slot}', entry['slot'])):
        text = text.replace(token, value)
    return text


def _touch_packages(target_root, rel_path, created):
    """Drop an empty __init__.py into every package directory on the way to a generated file."""
    parts = Path(rel_path).parent.parts
    for depth in range(1, len(parts) + 1):
        marker = Path(target_root, *parts[:depth], '__init__.py')
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text('', encoding='utf-8')
            created.append(str(marker.relative_to(target_root)).replace('\\', '/'))


def scaffold(anatomy, spec, target_root, force=False):
    """Create the tree. Returns a report; `verdict` is FAIL if the plan does not hold."""
    target_root = Path(target_root)
    entries, problems = plan(anatomy, spec)
    report = {'verdict': 'FAIL' if problems else 'PASS', 'model_tag': anatomy['model_tag'],
              'target_root': str(target_root), 'slots': len(entries), 'generated': [], 'kept': [],
              'reuse_required': [], 'packages': [], 'problems': problems}
    if problems:
        return report
    for entry in entries:
        path = target_root / entry['dst']
        if entry['shared']:
            report['reuse_required'].append(entry['dst'])
        if path.exists() and not force:
            report['kept'].append(entry['dst'])
            continue
        template_path = TEMPLATES / entry['template']
        if not template_path.exists():
            report['problems'].append(f'{entry["slot"]}: template is missing: {template_path}')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(template_path, anatomy['model_tag'], entry.get('name'), entry),
                        encoding='utf-8')
        report['generated'].append(entry['dst'])
        _touch_packages(target_root, entry['dst'], report['packages'])
    report['verdict'] = 'PASS' if not report['problems'] else 'FAIL'
    return report


def main(argv=None):
    p = argparse.ArgumentParser(prog='scaffold.py', description=__doc__)
    p.add_argument('--anatomy', required=True, help='STEP_ANATOMY.yaml from F2')
    p.add_argument('--target', required=True, help='where the target project is generated')
    p.add_argument('--tree-spec', default=str(TREE_SPEC))
    p.add_argument('--force', action='store_true',
                   help='overwrite existing files; off by default so reuse beats rewrite')
    p.add_argument('--report', help='write the scaffold report here as well as to stdout')
    a = p.parse_args(argv)
    report = scaffold(load_yaml(a.anatomy), load_yaml(a.tree_spec), a.target, a.force)
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())

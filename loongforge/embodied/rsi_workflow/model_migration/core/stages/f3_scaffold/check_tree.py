"""F3 check: the generated tree and the anatomy must correspond exactly, both ways.

A missing slot file means the skeleton does not cover what the model has. A surplus file means
something landed in the tree that no slot accounts for -- which is how coverage gets padded with
unrelated files -- so it fails just as hard. The check runs the same correspondence the frozen
structure judge runs later in F4, imported rather than re-coded, so passing here and failing the
gate cannot happen for different reasons.

Writes `check_tree.json` next to this file by default; exit 0 PASS, 2 FAIL.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from validate.structure_judge import check_tree, expected_files     # noqa: E402

STAGE_DIR = Path(__file__).resolve().parent


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def run(anatomy, target_root, generic):
    problems = check_tree(anatomy, target_root, tuple(generic))
    return {'verdict': 'PASS' if not problems else 'FAIL', 'target_root': str(target_root),
            'expected': sorted(expected_files(anatomy)), 'generic_files': list(generic),
            'problems': problems}


def main(argv=None):
    p = argparse.ArgumentParser(prog='check_tree.py', description=__doc__)
    p.add_argument('--anatomy', required=True, help='STEP_ANATOMY.yaml from F2')
    p.add_argument('--target', required=True, help='root of the generated project')
    p.add_argument('--tree-spec', default=str(STAGE_DIR / 'TREE_SPEC.yaml'),
                   help='read the generic-file whitelist from here')
    p.add_argument('--out', default=str(STAGE_DIR / 'check_tree.json'))
    a = p.parse_args(argv)
    spec = load_yaml(a.tree_spec)
    report = run(load_yaml(a.anatomy), a.target, spec.get('generic_files') or ['__init__.py'])
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())

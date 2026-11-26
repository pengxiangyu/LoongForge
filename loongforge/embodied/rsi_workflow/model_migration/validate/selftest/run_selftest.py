"""Runs the positive and negative cases against the frozen judges.

Two families of cases, both declaring the verdict they must receive:

  alignment  a pair of loss traces, read from `core/stages/f5_align/cases/` by default, judged
             by the frozen alignment gate with the frozen thresholds.
  structure  an anatomy plus the file list of a tree, materialised in a temporary directory and
             judged by the frozen structure judge.

Exit code 0 only if every case got the verdict it declared; 2 if any case did not, 3 if a case
could not be judged at all.
"""
import argparse
import json
from pathlib import Path
import tempfile

CASES_DIR = Path(__file__).resolve().parent / 'cases'
ALIGN_CASES_DIR = (Path(__file__).resolve().parents[2] / 'core' / 'stages' / 'f5_align' / 'cases')


def _thresholds():
    import yaml
    path = Path(__file__).resolve().parents[1] / 'frozen' / 'thresholds.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _align_case(case, thresholds):
    from ..frozen import align_gate_frozen
    from ..trajectory import losses
    report = align_gate_frozen.evaluate(losses(case['src']), losses(case['tgt']), thresholds)
    return report['verdict'], report['problems']


def _structure_case(case):
    from ..frozen import structure_judge
    with tempfile.TemporaryDirectory() as tmp:
        for rel in case['tree']:
            path = Path(tmp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('', encoding='utf-8')
        report = structure_judge.judge(case['anatomy'], tmp, case.get('weight_report'))
    return report['verdict'], report['problems']


def _run(case, thresholds):
    kind = 'align' if 'src' in case else 'structure'
    try:
        verdict, problems = (_align_case(case, thresholds) if kind == 'align'
                             else _structure_case(case))
    except Exception as exc:                                   # a case that cannot be judged
        return {'name': case['name'], 'kind': kind, 'expect': case['expect'],
                'verdict': 'ERROR', 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    return {'name': case['name'], 'kind': kind, 'expect': case['expect'], 'verdict': verdict,
            'ok': verdict == case['expect'], 'problems': problems}


def run_selftest(align_cases_dir=None):
    thresholds = _thresholds()
    paths = sorted(CASES_DIR.glob('*.json')) + sorted(Path(align_cases_dir or ALIGN_CASES_DIR)
                                                      .glob('*.json'))
    if not paths:
        return {'verdict': 'ERROR', 'cases': [], 'problems': ['no self-test cases found']}
    results = [_run(json.loads(p.read_text(encoding='utf-8')), thresholds) for p in paths]
    failed = [r for r in results if not r['ok']]
    verdict = 'PASS' if not failed else ('ERROR' if any(r['verdict'] == 'ERROR' for r in failed)
                                         else 'FAIL')
    return {'verdict': verdict, 'cases': results,
            'problems': [f'{r["name"]}: expected {r["expect"]}, got {r["verdict"]}'
                         for r in failed]}


def main(argv=None):
    p = argparse.ArgumentParser(prog='run_selftest.py', description=__doc__)
    p.add_argument('--cases', help='directory of alignment cases (default: f5_align/cases)')
    p.add_argument('--out', help='write the report here as well as to stdout')
    a = p.parse_args(argv)
    report = run_selftest(a.cases)
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding='utf-8')
    print(text, end='')
    return {'PASS': 0, 'FAIL': 2, 'ERROR': 3}[report['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())

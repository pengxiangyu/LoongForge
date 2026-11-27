"""python -m validate {judge,align,freeze,verify,selftest} -- the only way a verdict is issued.

judge   structure verdict over a generated/migrated tree (F3 check, F4 gate)
align   numeric verdict over two loss traces (F5 gate)
freeze  copy the judges into validate/frozen/ and write the hash lock
verify  re-hash the frozen judges against the lock
selftest  prove the judges accept a correct case and reject a broken one

Every subcommand judges with the frozen copies and verifies the lock first. Exit codes follow
the gates: 0 PASS, 2 FAIL, 3 ERROR (unusable input, or a lock that no longer holds).
"""
import argparse
import json
from pathlib import Path

EXIT = {'PASS': 0, 'FAIL': 2, 'ERROR': 3}


def _load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def _emit(report, out):
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding='utf-8')
    print(text, end='')
    return EXIT[report['verdict']]


def _judge(a):
    from .frozen import freeze, structure_judge
    freeze.require_unlocked_judges()
    anatomy = _load_yaml(a.anatomy)
    weights = json.loads(Path(a.weights).read_text(encoding='utf-8')) if a.weights else None
    generic = tuple(a.generic) if a.generic else structure_judge.GENERIC_FILES
    report = structure_judge.judge(anatomy, a.target, weight_report=weights, generic=generic)
    report['anatomy'] = str(a.anatomy)
    return _emit(report, a.out)


def _align(a):
    from . import trajectory
    from .frozen import freeze, numeric_gate_frozen
    freeze.require_unlocked_judges()
    thresholds = _load_yaml(a.thresholds or Path(__file__).resolve().parent / 'frozen'
                            / 'thresholds.yaml')
    src, tgt = trajectory.load(a.src_trace), trajectory.load(a.tgt_trace)
    report = numeric_gate_frozen.evaluate(trajectory.losses(src), trajectory.losses(tgt),
                                          thresholds)
    report['traces'] = {'src': str(a.src_trace), 'tgt': str(a.tgt_trace)}
    report['run'] = {'src': trajectory.run_meta(src), 'tgt': trajectory.run_meta(tgt)}
    return _emit(report, a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m validate', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    j = sub.add_parser('judge', help='structure verdict over a target tree')
    j.add_argument('--anatomy', required=True, help='STEP_ANATOMY.yaml, the structure source of truth')
    j.add_argument('--target', required=True, help='root of the generated/migrated project')
    j.add_argument('--weights', help='weight conversion report from f4_migrate/weights/')
    j.add_argument('--generic', nargs='+', help='files that carry no slot (default: __init__.py)')
    j.add_argument('--out', help='write the report here as well as to stdout')
    g = sub.add_parser('align', help='numeric verdict over two loss traces')
    g.add_argument('--src-trace', required=True, help='trace of the original model')
    g.add_argument('--tgt-trace', required=True, help='trace of the migrated project')
    g.add_argument('--thresholds', help='defaults to the frozen thresholds.yaml')
    g.add_argument('--out')
    sub.add_parser('freeze', help='copy the judges into validate/frozen/ and lock them')
    sub.add_parser('verify', help='re-hash the frozen judges against the lock')
    s = sub.add_parser('selftest', help='prove the judges catch a broken case')
    s.add_argument('--cases', help='directory of alignment cases (default: f5_align/cases)')
    s.add_argument('--out')
    a = p.parse_args(argv)

    if a.cmd == 'judge':
        return _judge(a)
    if a.cmd == 'align':
        return _align(a)
    if a.cmd in ('freeze', 'verify'):
        from .frozen.freeze import main as freeze_main
        return freeze_main([a.cmd])
    from .selftest.run_selftest import main as selftest_main
    argv = (['--cases', a.cases] if a.cases else []) + (['--out', a.out] if a.out else [])
    return selftest_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())

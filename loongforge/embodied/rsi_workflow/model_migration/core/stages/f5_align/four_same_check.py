"""The four-same precondition: a loss comparison that is not run under it means nothing.

Two training runs that saw different samples, in a different order, from different weights, or
with different launch arguments will produce different losses whether or not the migration is
correct. Comparing them anyway yields a number that looks like evidence, which is worse than no
number at all. So all four are checked before anything is collected, and any mismatch stops the
stage -- there is no "approximately the same data" path.

Each condition names what to fix, because the four fail for different reasons:

  data_id     the dataset identity, shard list and statistics file, normalisation included
              -> back to asset verification: the data is not the same data
  order_hash  sample order and batch composition (same sampler, same seed)
              -> fix the data-side configuration
  ckpt_hash   both paths load one checkpoint, with every key mapped
              -> back to F4: the weight conversion is incomplete
  launch      batch size, precision, sequence length, seed, optimizer hyper-parameters
              -> align the launch script
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from validate.trajectory import load, run_meta                       # noqa: E402

REMEDIES = {
    'data_id': 'the two paths do not read the same data; return to asset verification',
    'order_hash': 'sample order or batch composition differs; fix the data-side configuration',
    'ckpt_hash': 'the paths load different weights; return to F4 and check the weight conversion',
    'launch': 'launch arguments differ; align the launch script before comparing losses',
}


def check(src_run, tgt_run):
    """Compare two four-same fingerprints. Returns {'ok', 'problems', 'fields'}."""
    problems, fields = [], {}
    for field, remedy in REMEDIES.items():
        same = src_run.get(field) == tgt_run.get(field)
        fields[field] = {'same': same, 'src': src_run.get(field), 'tgt': tgt_run.get(field)}
        if not same:
            problems.append(f'{field} differs: {remedy}')
    return {'ok': not problems, 'problems': problems, 'fields': fields}


def main(argv=None):
    p = argparse.ArgumentParser(prog='four_same_check.py', description=__doc__)
    p.add_argument('--src-trace', required=True, help='trace of the original model')
    p.add_argument('--tgt-trace', required=True, help='trace of the migrated project')
    p.add_argument('--out')
    a = p.parse_args(argv)
    report = check(run_meta(load(a.src_trace)), run_meta(load(a.tgt_trace)))
    report['traces'] = {'src': str(a.src_trace), 'tgt': str(a.tgt_trace)}
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

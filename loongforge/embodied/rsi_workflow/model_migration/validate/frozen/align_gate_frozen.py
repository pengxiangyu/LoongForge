"""Alignment gate: does the migrated project reproduce the original model's loss, step by step.

The gate takes the two traces and the threshold file and returns a verdict. It never widens a
limit: `thresholds.yaml` is the single source of the judgement, and when the gate fails the
answer is to fix the target implementation (back to F4), not to touch the thresholds.

Three ways to fail, in the order they are checked:

  comparable   the traces must cover the same steps after warmup. Different step sets mean the
               two runs are not the same experiment, so there is nothing to compare.
  limits       diff_max, diff_p95 and diff_mean must each stay inside their limit.
  no_amplify   the deviation must not grow along the trajectory, even inside the limits.

`evaluate` works on already-loaded dicts so that the frozen copy of this module stays free of
any dependency on the framework's YAML specs; the CLI below does the file reading.
"""
from . import stats

METRICS = ('diff_max', 'diff_p95', 'diff_mean')


def _comparability(src, tgt, warmup):
    problems = []
    only_src = sorted(set(src) - set(tgt))
    only_tgt = sorted(set(tgt) - set(src))
    if only_src:
        problems.append(f'steps recorded only by the original path: {only_src[:5]}')
    if only_tgt:
        problems.append(f'steps recorded only by the target path: {only_tgt[:5]}')
    if len(set(src) & set(tgt)) <= warmup:
        problems.append(f'only {len(set(src) & set(tgt))} shared steps, warmup drops {warmup}')
    return problems


def evaluate(src_losses, tgt_losses, thresholds):
    """Judge two {step: loss} maps. Returns a report dict with verdict PASS, FAIL or ERROR."""
    limits = thresholds['limits']
    extra = thresholds.get('extra', {})
    warmup = int(extra.get('warmup_steps', 0))
    eps = float(extra.get('eps', 1e-12))
    report = {'verdict': 'ERROR', 'warmup_steps': warmup, 'metrics': {}, 'limits': dict(limits),
              'problems': []}
    report['problems'] += _comparability(src_losses, tgt_losses, warmup)
    if report['problems']:
        return report
    pairs = stats.relative_diffs(src_losses, tgt_losses, eps=eps)
    diffs = [d for step, d in pairs[warmup:]]
    report['metrics'] = stats.summarize(diffs)
    report['per_step'] = [{'step': s, 'diff': d} for s, d in pairs]
    for metric in METRICS:
        if metric not in limits:
            report['problems'].append(f'thresholds.yaml declares no limit for {metric}')
            return report
        if report['metrics'][metric] > limits[metric]:
            report['problems'].append(
                f'{metric} {report["metrics"][metric]:.6g} exceeds {limits[metric]:.6g}')
    if extra.get('no_amplify', True):
        report['amplifies'] = stats.amplifies(diffs, tol=float(extra.get('amplify_tol', 0.0)))
        if report['amplifies']:
            report['problems'].append('deviation grows along the trajectory (no_amplify)')
    report['verdict'] = 'PASS' if not report['problems'] else 'FAIL'
    return report

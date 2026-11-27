"""Deviation statistics for the alignment gate; standard library only.

The gate compares two per-step loss traces. Every statistic here is computed on the
relative deviation of one step,

    diff_i = |loss_src_i - loss_tgt_i| / max(|loss_src_i|, eps)

so the verdict does not depend on the scale of the loss. `warmup_steps` steps are dropped
before anything is computed: the first optimizer steps after initialisation jitter far more
than the steady state and would make a sound migration look broken.

`amplifies` is the one check that is not a threshold: even when max, p95 and mean all sit
inside their limits, a deviation that keeps growing along the trajectory means the two paths
are drifting apart rather than agreeing, so it is judged a failure.
"""
import statistics


def relative_diffs(src, tgt, eps=1e-12):
    """Per-step relative deviation of two {step: loss} maps, ordered by step.

    Only steps present in both traces are used; the caller is responsible for checking that
    the traces cover the same steps (numeric_gate does).
    """
    return [(step, abs(src[step] - tgt[step]) / max(abs(src[step]), eps))
            for step in sorted(set(src) & set(tgt))]


def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 1]; `values` need not be sorted."""
    if not values:
        raise ValueError('percentile of an empty sample')
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(diffs):
    """{diff_max, diff_p95, diff_mean, steps} of a list of relative deviations."""
    if not diffs:
        raise ValueError('no steps left after warmup; nothing to summarize')
    return {'diff_max': max(diffs), 'diff_p95': percentile(diffs, 0.95),
            'diff_mean': statistics.fmean(diffs), 'steps': len(diffs)}


def amplifies(diffs, tol=0.0):
    """True when the deviation grows along the trajectory.

    The sample is split in half; the later half's mean must not exceed the earlier half's
    (plus `tol`, which exists only to absorb floating-point noise). Fewer than four steps
    cannot show a trend, so they never count as amplifying.
    """
    if len(diffs) < 4:
        return False
    half = len(diffs) // 2
    first, second = statistics.fmean(diffs[:half]), statistics.fmean(diffs[half:])
    return second > first + tol

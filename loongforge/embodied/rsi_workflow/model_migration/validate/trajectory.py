"""The per-step loss trace exchanged between the two training paths, and its on-disk schema.

A trace is what `precision-alignment` collects from one run (the original model, or the
migrated target project) and what the alignment gate reads. It carries two things:

  run    the four-same fingerprint of the run (data id, sample-order hash, checkpoint hash,
         launch arguments). `four_same_check` compares these before any loss is compared,
         because a loss diff only means something when both runs saw the same data in the
         same order with the same weights and the same launch arguments.
  steps  one entry per optimizer step: the step index and the loss scalar.

Steps are stored as a list to keep the file readable in a diff, and loaded into a {step: loss}
map because the gate only ever looks up by step. A duplicated step index is an error rather
than a last-one-wins overwrite: it usually means two runs were concatenated into one trace.
"""
import json
from pathlib import Path

SCHEMA = 'vma-loss-trace@1'
RUN_FIELDS = ('data_id', 'order_hash', 'ckpt_hash', 'launch')


def load(path):
    """Read a trace file and reject anything that is not a complete trace."""
    trace = json.loads(Path(path).read_text(encoding='utf-8'))
    if trace.get('schema') != SCHEMA:
        raise ValueError(f'{path}: schema is {trace.get("schema")!r}, expected {SCHEMA!r}')
    missing = [f for f in RUN_FIELDS if f not in trace.get('run', {})]
    if missing:
        raise ValueError(f'{path}: run fingerprint lacks {missing}')
    if not trace.get('steps'):
        raise ValueError(f'{path}: no steps recorded')
    return trace


def losses(trace):
    """{step: loss} of a loaded trace."""
    out = {}
    for entry in trace['steps']:
        step = int(entry['step'])
        if step in out:
            raise ValueError(f'step {step} recorded twice')
        out[step] = float(entry['loss'])
    return out


def run_meta(trace):
    """The four-same fingerprint, with no extra keys, for comparison between two paths."""
    return {f: trace['run'][f] for f in RUN_FIELDS}


def write(path, run, steps):
    """Write a trace; `steps` is an iterable of (step, loss) pairs."""
    missing = [f for f in RUN_FIELDS if f not in run]
    if missing:
        raise ValueError(f'run fingerprint lacks {missing}')
    trace = {'schema': SCHEMA, 'run': dict(run),
             'steps': [{'step': int(s), 'loss': float(v)} for s, v in steps]}
    Path(path).write_text(json.dumps(trace, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return trace

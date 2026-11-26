"""Freezes a collected trace so that a later comparison cannot be rerun against a moved target.

The original model's trace is the baseline of the whole alignment argument. If it can be
re-collected between two judgements, a failing comparison can be made to pass by re-recording
the baseline instead of fixing the migration. `freeze` therefore copies the trace into the task's
precision directory and records its sha256; `verify` refuses to proceed once the content behind
a frozen name has changed.
"""
import hashlib
import json
from pathlib import Path

from . import trajectory

MANIFEST = 'traces.lock.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(trace_path, out_dir, name):
    """Copy a validated trace to `out_dir/name` and add it to the lock manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = trajectory.load(trace_path)
    target = out_dir / name
    target.write_text(json.dumps(trace, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    lock_path = out_dir / MANIFEST
    lock = json.loads(lock_path.read_text(encoding='utf-8')) if lock_path.exists() else {}
    lock[name] = {'sha256': sha256(target), 'run': trajectory.run_meta(trace),
                  'steps': len(trace['steps']), 'source': str(trace_path)}
    lock_path.write_text(json.dumps(lock, indent=1, ensure_ascii=False, sort_keys=True) + '\n',
                         encoding='utf-8')
    return lock[name]


def verify(out_dir):
    """Re-hash every frozen trace; returns the names whose content no longer matches the lock."""
    out_dir = Path(out_dir)
    lock_path = out_dir / MANIFEST
    if not lock_path.exists():
        return [f'no {MANIFEST} in {out_dir}']
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    problems = []
    for name, entry in sorted(lock.items()):
        path = out_dir / name
        if not path.exists():
            problems.append(f'frozen trace is gone: {name}')
        elif sha256(path) != entry['sha256']:
            problems.append(f'frozen trace was rewritten: {name}')
    return problems

"""Adapts "a training run" into the two things the gate needs: a loss trace and a fingerprint.

F5 drives two paths — the original model's repository and the migrated target project — and the
only difference the gate is allowed to see is the loss. Everything else about a path (where it
runs, how it is launched, which data and checkpoint it loads) is described by one Adapter, so
that the four-same check compares declarations instead of guessing from log files.

An adapter is data, not code: `align_driver` reads a JSON file holding one object per path and
hands `request()` to `precision-alignment` for collection, `four_same()` to `four_same_check`
for the precondition check. Nothing here launches a process; collection belongs to the skill.
"""
import json
from pathlib import Path

from .trajectory import RUN_FIELDS


class Adapter:
    """One of the two training paths.

    name        'src' for the original model, 'tgt' for the migrated project (free-form label).
    run_cmd     the launch command, verbatim, as the skill should run it.
    workdir     the directory the command runs in.
    steps       how many optimizer steps to collect.
    trace       where the collected trace is expected to land.
    fingerprint the four-same declaration: data_id, order_hash, ckpt_hash, launch.
    """

    def __init__(self, name, run_cmd, workdir, steps, trace, fingerprint):
        missing = [f for f in RUN_FIELDS if f not in fingerprint]
        if missing:
            raise ValueError(f'{name}: fingerprint lacks {missing}')
        self.name = name
        self.run_cmd = run_cmd
        self.workdir = workdir
        self.steps = int(steps)
        self.trace = trace
        self.fingerprint = dict(fingerprint)

    @classmethod
    def from_dict(cls, obj):
        return cls(obj['name'], obj['run_cmd'], obj['workdir'], obj['steps'], obj['trace'],
                   obj['fingerprint'])

    def request(self):
        """The `collect_precision` payload declared by CONTRACT_precision_alignment.yaml."""
        return {'run_cmd': self.run_cmd, 'workdir': self.workdir, 'steps': self.steps}

    def four_same(self):
        return dict(self.fingerprint)


def load_pair(path):
    """Read {'src': {...}, 'tgt': {...}} and return the two adapters in that order."""
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    missing = [k for k in ('src', 'tgt') if k not in obj]
    if missing:
        raise ValueError(f'{path}: declares no {missing} path')
    return Adapter.from_dict(obj['src']), Adapter.from_dict(obj['tgt'])

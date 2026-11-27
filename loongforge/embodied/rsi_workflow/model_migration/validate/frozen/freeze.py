"""Freezes the judging closure and verifies the lock before any verdict is issued.

Run `freeze` once, before the migration writes its first file: it copies the judges out of
`validate/` into this directory and writes `JUDGE_LOCK.json`. Every later judgement calls
`verify` first; a mismatch is an error, not a warning, because a judge that can be edited by the
side being judged is not a judge.

Re-freezing is deliberately a separate, explicit action: it is how a reviewer changes the
judging criteria between tasks, and it rewrites the lock so the change is visible in the diff.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

FROZEN_DIR = Path(__file__).resolve().parent
LIVE_DIR = FROZEN_DIR.parent
# live name -> frozen name; the frozen numeric gate is renamed so the two are never confused.
CLOSURE = {'structure_judge.py': 'structure_judge.py',
           'numeric_gate.py': 'numeric_gate_frozen.py',
           'stats.py': 'stats.py',
           'thresholds.yaml': 'thresholds.yaml'}
LOCK = FROZEN_DIR / 'JUDGE_LOCK.json'
RULE = ('a failing verdict is fixed in the target project; no file in validate/frozen/ may be '
        'edited to make it pass')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    """Copy the closure into this directory and rewrite the lock. Returns the lock object."""
    files = {}
    for live_name, frozen_name in sorted(CLOSURE.items()):
        source = LIVE_DIR / live_name
        if not source.exists():
            raise FileNotFoundError(f'nothing to freeze: {source}')
        target = FROZEN_DIR / frozen_name
        target.write_bytes(source.read_bytes())
        files[frozen_name] = sha256(target)
    lock = {'locked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'closure': dict(sorted(CLOSURE.items())), 'files': files, 'rule': RULE}
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return lock


def verify():
    """Returns the problems with the lock; an empty list means a verdict may be issued."""
    if not LOCK.exists():
        return [f'no {LOCK.name}; run validate/frozen/freeze.py before judging']
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    problems = []
    for name, digest in sorted(lock['files'].items()):
        path = FROZEN_DIR / name
        if not path.exists():
            problems.append(f'frozen file is gone: {name}')
        elif sha256(path) != digest:
            problems.append(f'frozen file was modified: {name}')
    for frozen_name in CLOSURE.values():
        if frozen_name not in lock['files']:
            problems.append(f'frozen file is not covered by the lock: {frozen_name}')
    return problems


def require_unlocked_judges():
    """Raise unless the lock holds; called by every judgement entry point."""
    problems = verify()
    if problems:
        raise SystemExit('refusing to judge, the frozen judges changed:\n  ' + '\n  '.join(problems))


def main(argv=None):
    p = argparse.ArgumentParser(prog='freeze.py', description=__doc__)
    p.add_argument('action', choices=('freeze', 'verify'))
    action = p.parse_args(argv).action
    if action == 'freeze':
        lock = freeze()
        print(json.dumps({'locked_at': lock['locked_at'], 'files': sorted(lock['files'])}))
        return 0
    problems = verify()
    print(json.dumps({'ok': not problems, 'problems': problems}, indent=1))
    return 0 if not problems else 3


if __name__ == '__main__':
    raise SystemExit(main())

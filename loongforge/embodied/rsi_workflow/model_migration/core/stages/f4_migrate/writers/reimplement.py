"""`reimplement`: the original is bound to its own stack, so the slot is written to the target shape.

This writer deliberately does not produce an implementation. It cannot: deciding what the target
structure should compute is the part of the migration that needs a reader of both codebases, and a
file this writer invented would look migrated while being a guess.

What it does is make the gap explicit and reviewable. It keeps the F3 template at the target path,
appends the source reference and the stated reason, and reports `needs_implementation`, which the
stage report carries forward. The structure gate still sees the file, so the tree stays in
correspondence; the alignment gate is what proves the rewrite was faithful, and until it passes
the slot is not done.

`reason` is mandatory for exactly this action: an unexplained rewrite cannot be reviewed against
the original.
"""
from pathlib import Path

from . import place

MARKER = '# --- F4 reimplement ---'


def render(entry, template_text):
    note = [MARKER,
            f'# Segment {entry["seg"]}, slot {entry["slot"]}: rewritten to the target structure.',
            f'# Original: {entry["src"]}',
            f'# Reason: {entry["reason"]}',
            '# The original logic must be reproduced here; the F5 alignment gate is what proves it.',
            '']
    return template_text.rstrip('\n') + '\n\n\n' + '\n'.join(note)


def write(entry, src_root, target_root, source, shared=(), force=False):
    if not entry.get('reason'):
        return {'seg': entry['seg'], 'slot': entry['slot'], 'action': 'reimplement',
                'dst': entry['dst'], 'status': 'error',
                'note': 'reimplement requires a reason: what bound the original to its own stack'}
    path = Path(target_root) / entry['dst']
    if not path.exists():
        return {'seg': entry['seg'], 'slot': entry['slot'], 'action': 'reimplement',
                'dst': entry['dst'], 'status': 'error',
                'note': f'no F3 skeleton at {entry["dst"]}; run f3_scaffold/scaffold.py first'}
    text = path.read_text(encoding='utf-8')
    if MARKER in text:
        record = {'seg': entry['seg'], 'slot': entry['slot'], 'action': 'reimplement',
                  'dst': entry['dst'], 'status': 'needs_implementation',
                  'note': 'already marked for rewrite; the implementation is still outstanding'}
        return record
    record = place(entry, target_root, render(entry, text), set(shared), force=True)
    record['status'] = 'needs_implementation'
    record['note'] = 'target shape marked; the rewrite itself is not automated'
    return record

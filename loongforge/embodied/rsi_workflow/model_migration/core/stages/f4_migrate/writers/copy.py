"""`copy`: logic and interface already match, so the file moves verbatim.

Verbatim means verbatim. The only thing added is the provenance header, so that a reviewer can
tell a copied file from an adapted one without diffing against the source repository. If a file
needs an import rewritten or a class renamed to fit the target framework, it is an `adapt`, not a
`copy` with edits -- that distinction is what makes the mapping table readable.
"""
from pathlib import Path

from . import place, provenance


def write(entry, src_root, target_root, source, shared=(), force=False):
    src = Path(src_root) / entry['src']
    if not src.exists():
        return {'seg': entry['seg'], 'slot': entry['slot'], 'action': 'copy', 'dst': entry['dst'],
                'status': 'error', 'note': f'source does not exist: {src}'}
    body = src.read_text(encoding='utf-8')
    text = provenance(entry, source) + body if entry['dst'].endswith('.py') else body
    record = place(entry, target_root, text, set(shared), force)
    record['lines'] = body.count('\n') + 1
    return record

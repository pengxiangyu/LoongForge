"""`adapt`: same logic, different names. Imports, class names and registration are rewritten.

The rewrites are declared in the mapping entry, not inferred here:

    - {seg: S2, slot: vision_slot, src: ..., dst: ..., action: adapt,
       reason: "...",
       rewrites: [{from: "from src.layers import", to: "from ...model.common import"}]}

Declaring them has a point beyond reproducibility: every rewrite is applied as a literal string
replacement and counted, and a rewrite that matched nothing is reported. A silently ineffective
rename is how an adapted file keeps importing the original repository and passes the structure
gate while failing the alignment gate for a reason nobody can find.

What this writer will not do is touch the computation. A changed formula, reduction or dtype is a
`reimplement` with a stated reason, so it shows up in the mapping table instead of hiding in a
rename list.
"""
from pathlib import Path

from . import place, provenance


def apply_rewrites(body, rewrites):
    """Literal replacements, in order. Returns (text, applied, ineffective)."""
    applied, ineffective = [], []
    for rule in rewrites:
        old, new = rule['from'], rule['to']
        hits = body.count(old)
        if hits:
            body = body.replace(old, new)
            applied.append({'from': old, 'to': new, 'hits': hits})
        else:
            ineffective.append(old)
    return body, applied, ineffective


def write(entry, src_root, target_root, source, shared=(), force=False):
    src = Path(src_root) / entry['src']
    if not src.exists():
        return {'seg': entry['seg'], 'slot': entry['slot'], 'action': 'adapt', 'dst': entry['dst'],
                'status': 'error', 'note': f'source does not exist: {src}'}
    body, applied, ineffective = apply_rewrites(src.read_text(encoding='utf-8'),
                                                entry.get('rewrites') or [])
    text = provenance(entry, source) + body if entry['dst'].endswith('.py') else body
    record = place(entry, target_root, text, set(shared), force)
    record['rewrites'] = applied
    if ineffective:
        record['status'] = 'error'
        record['note'] = f'rewrites matched nothing in the source: {ineffective}'
    return record

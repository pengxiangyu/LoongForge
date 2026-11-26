"""`wrap`: the target framework already has an equivalent, so call it instead of porting it.

This is the action that keeps a migration from duplicating the framework. The optimizers, the
schedulers, the parallel strategies, the checkpoint layer and most language backbones already
exist under `loongforge/embodied/`; porting the original repository's copy of one would leave two
implementations of the same thing, and the migrated model would not benefit from fixes to either.

The entry declares what to reuse and which names to keep:

    - {seg: S7, slot: optimizer_slot, src: ..., dst: optimizer/optimizer.py, action: wrap,
       reason: "the framework optimizer is equivalent; the original hyper-parameter semantics
                are preserved by the wrapper",
       reuse: {module: loongforge.embodied.optimizer.optimizer,
               exports: [build_optimizer], alias: {build_optimizer: build_adam}}}

`exports` are the names the wrapper re-exports; `alias` maps a target-framework name to the name
the original model used, so call sites migrated by `copy` or `adapt` keep working unchanged. A
wrap without `reuse` is rejected: "reuse something" is not a specification.
"""
from . import place, provenance


def render(entry):
    reuse = entry['reuse']
    module, exports = reuse['module'], list(reuse.get('exports') or [])
    alias = reuse.get('alias') or {}
    lines = [provenance(entry, {'repo': module}).rstrip('\n'),
             f'"""Reuses {module}; the names below keep the interface the original model used."""',
             '', f'from {module} import ' + ', '.join(exports) if exports else f'import {module}',
             '']
    for target_name, original_name in alias.items():
        lines.append(f'{original_name} = {target_name}')
    if alias:
        lines.append('')
    exported = sorted(set(exports) | set(alias.values()))
    lines.append('__all__ = [' + ', '.join(repr(n) for n in exported) + ']')
    return '\n'.join(lines) + '\n'


def write(entry, src_root, target_root, source, shared=(), force=False):
    reuse = entry.get('reuse')
    problem = None
    if not reuse or not reuse.get('module'):
        problem = 'wrap requires `reuse.module`: which existing implementation is being reused'
    elif not (reuse.get('exports') or reuse.get('alias')):
        problem = 'wrap requires `reuse.exports` or `reuse.alias`: which names it re-exports'
    if problem:
        return {'seg': entry['seg'], 'slot': entry['slot'], 'action': 'wrap', 'dst': entry['dst'],
                'status': 'error', 'note': problem}
    record = place(entry, target_root, render(entry), set(shared), force)
    record['reuses'] = reuse['module']
    return record

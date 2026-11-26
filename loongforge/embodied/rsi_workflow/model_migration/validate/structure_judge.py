"""Structure judge: is the target project in strict one-to-one correspondence with the anatomy.

Everything the judge knows comes from `STEP_ANATOMY.yaml` (the structure source of truth, F2)
and from the generated tree itself. It answers four questions, and a problem in any of them is
a failure — the judge never ranks or weighs them:

  tree        every slot in the anatomy has its target file, and the tree holds no file that no
              slot accounts for. A surplus file is a failure too: otherwise coverage could be
              faked by dropping unrelated files into the tree.
  required    the required segments are all present. A model that genuinely lacks one still has
              to say so, which is what the next check is for.
  empty       an empty segment or a disabled branch must carry `empty_reason`, so that "nothing
              landed here" is a recorded conclusion rather than an omission.
  weights     the checkpoint conversion left no source key unmapped, which is how a model loads
              successfully while silently missing parameters.

`judge` takes parsed objects, not paths, so that this module depends on the standard library
only and its frozen copy keeps judging after the framework around it changes.
"""
import os

REQUIRED_SEGMENTS = ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7')
GENERIC_FILES = ('__init__.py',)


def _is_generic(rel, generic):
    """Generic framework files (package markers, shared utilities) carry no slot."""
    return rel in generic or os.path.basename(rel) in generic


def slots_of(anatomy):
    """(segment id, slot dict) for every slot of every segment and every enabled branch."""
    for seg in anatomy.get('segments') or []:
        for slot in seg.get('slots') or []:
            yield seg['id'], slot
    for branch in anatomy.get('branches') or []:
        if branch.get('enabled'):
            for slot in branch.get('slots') or []:
                yield branch['id'], slot


def expected_files(anatomy):
    """{target path: owning slot} declared by the anatomy, rejecting two slots on one file."""
    out = {}
    for seg_id, slot in slots_of(anatomy):
        dst = slot['dst']
        if dst in out:
            raise ValueError(f'{dst} is claimed by both {out[dst]} and {seg_id}.{slot["slot"]}')
        out[dst] = f'{seg_id}.{slot["slot"]}'
    return out


def actual_files(target_root, generic=GENERIC_FILES):
    """Relative paths of the python files in the tree that a slot is supposed to own."""
    found = set()
    for root, _, files in os.walk(target_root):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), target_root).replace(os.sep, '/')
            if name.endswith('.py') and not _is_generic(rel, generic):
                found.add(rel)
    return found


def check_tree(anatomy, target_root, generic=GENERIC_FILES):
    expected = set(expected_files(anatomy))
    actual = actual_files(target_root, generic)
    return ([f'missing slot file: {p}' for p in sorted(expected - actual)]
            + [f'surplus file with no slot: {p}' for p in sorted(actual - expected)])


def check_required_segments(anatomy):
    seen = {seg['id'] for seg in anatomy.get('segments') or []}
    return [f'missing required segment: {sid}' for sid in REQUIRED_SEGMENTS if sid not in seen]


def check_empty_reasons(anatomy):
    problems = []
    for seg in anatomy.get('segments') or []:
        if not seg.get('slots') and not seg.get('empty_reason'):
            problems.append(f'empty segment without empty_reason: {seg["id"]}')
    for branch in anatomy.get('branches') or []:
        if not branch.get('enabled') and not branch.get('empty_reason'):
            problems.append(f'disabled branch without empty_reason: {branch["id"]}')
        if branch.get('enabled') and not branch.get('slots'):
            problems.append(f'branch {branch["id"]} is enabled but holds no slot')
    return problems


def check_weight_coverage(weight_report):
    """A weight report is optional only while no checkpoint has been converted yet."""
    if weight_report is None:
        return ['no weight conversion report; run f4_migrate/weights/convert_weights.py']
    problems = [f'unmapped source weight key: {k}' for k in (weight_report.get('unmapped') or [])]
    if weight_report.get('status') != 'ok':
        problems.append(f'weight conversion status is {weight_report.get("status")!r}')
    return problems


def judge(anatomy, target_root, weight_report=None, generic=GENERIC_FILES):
    """Returns {'verdict': 'PASS'|'FAIL', 'problems': [...]} over all four checks."""
    problems = (check_tree(anatomy, target_root, generic)
                + check_required_segments(anatomy)
                + check_empty_reasons(anatomy)
                + check_weight_coverage(weight_report))
    return {'verdict': 'PASS' if not problems else 'FAIL', 'problems': problems,
            'target_root': str(target_root), 'slots': len(expected_files(anatomy))}

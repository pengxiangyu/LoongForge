"""Converts a checkpoint from the original key space to the target's, and refuses to lose a key.

The rule that matters: a source key no rule covers is an error, not something to skip. Dropping
keys quietly produces a checkpoint that loads, trains and converges somewhere else -- the single
hardest migration bug to find from a loss curve. Keys that genuinely should not travel are listed
under `drop` in `weight_map.yaml`, each with a reason, so "not mapped" and "deliberately left
behind" stay distinguishable.

`--keys` converts a key list instead of a checkpoint (no torch needed). It exercises exactly the
mapping, which is where the mistakes are, and is how the conversion is checked before a real
checkpoint is available.
"""
import argparse
import json
from pathlib import Path


def rename(key, rule):
    return rule['dst'] + key[len(rule['src']):]


def plan_keys(keys, wmap):
    """Returns (renames, dropped, unmapped); the first rule whose prefix matches wins."""
    renames, dropped, unmapped = {}, {}, []
    for key in keys:
        for rule in wmap.get('drop') or []:
            if key.startswith(rule['src']):
                dropped[key] = rule['reason']
                break
        else:
            for rule in wmap['maps']:
                if key.startswith(rule['src']):
                    renames[key] = {'dst': rename(key, rule), 'transpose': rule.get('transpose')}
                    break
            else:
                unmapped.append(key)
    return renames, dropped, unmapped


def report_for(keys, wmap):
    """The conversion report: what was renamed, dropped, left unmapped or collided."""
    renames, dropped, unmapped = plan_keys(keys, wmap)
    targets = [v['dst'] for v in renames.values()]
    collisions = sorted({t for t in targets if targets.count(t) > 1})
    status = 'ok'
    if unmapped and wmap.get('unmapped_policy', 'fail') == 'fail':
        status = 'failed'
    if collisions:
        status = 'failed'
    return {'status': status, 'keys_src': len(keys), 'keys_dst': len(set(targets)),
            'renames': {k: v['dst'] for k, v in sorted(renames.items())},
            'dropped': dropped, 'unmapped': unmapped, 'collisions': collisions}


def convert(src_ckpt, wmap, dst_path):
    """Convert a real checkpoint. Returns the same report, plus the parameter counts."""
    import torch
    state = torch.load(src_ckpt, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state) if isinstance(state, dict) else state
    renames, dropped, unmapped = plan_keys(sorted(state), wmap)
    report = report_for(sorted(state), wmap)
    if report['status'] != 'ok':
        return report
    out = {}
    for key, how in renames.items():
        tensor = state[key]
        out[how['dst']] = tensor.permute(*how['transpose']) if how['transpose'] else tensor
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst_path)
    report['params_src'] = sum(t.numel() for k, t in state.items() if k not in dropped)
    report['params_dst'] = sum(t.numel() for t in out.values())
    report['dst'] = str(dst_path)
    if report['params_src'] != report['params_dst']:
        report['status'] = 'failed'
        report['unmapped'] = report['unmapped'] or ['parameter count changed during conversion']
    return report


def main(argv=None):
    p = argparse.ArgumentParser(prog='convert_weights.py', description=__doc__)
    p.add_argument('--weight-map', default=str(Path(__file__).resolve().parent / 'weight_map.yaml'))
    p.add_argument('--src-ckpt', help='checkpoint to convert')
    p.add_argument('--dst-ckpt', help='where the converted checkpoint is written')
    p.add_argument('--keys', help='JSON list of source keys: checks the mapping without torch')
    p.add_argument('--out', help='write the weight report here; the structure judge reads it')
    a = p.parse_args(argv)
    import yaml
    wmap = yaml.safe_load(Path(a.weight_map).read_text(encoding='utf-8'))
    if a.keys:
        report = report_for(json.loads(Path(a.keys).read_text(encoding='utf-8')), wmap)
    elif a.src_ckpt and a.dst_ckpt:
        report = convert(a.src_ckpt, wmap, a.dst_ckpt)
    else:
        p.error('pass either --keys, or both --src-ckpt and --dst-ckpt')
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['status'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())

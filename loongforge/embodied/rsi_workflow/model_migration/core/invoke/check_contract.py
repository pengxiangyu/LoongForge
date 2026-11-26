"""Checks a skill request and a skill response against the contract, before anything reads them.

The rule the rest of the framework relies on: no guessing field names. A capability's input keys
are exactly the contract's input keys (a `str?` value marks the key optional), and a response is
only handed back once every declared output key is present. A field that is missing stops the
stage with the name of the field, which is a far cheaper failure than a stage that runs on a
`None` it picked up from a renamed key.

Only the presence of top-level keys is checked. The nested shapes in the contract document what
the skill returns; re-deriving them here would be the framework re-implementing the skill, which
is the one thing it must not do.
"""
import argparse
import json
from pathlib import Path


def load_contract(path):
    import yaml
    contract = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for field in ('version', 'skill', 'capabilities', 'status'):
        if field not in contract:
            raise ValueError(f'{path}: contract declares no {field}')
    return contract


def _capability(contract, capability):
    spec = contract['capabilities'].get(capability)
    if spec is None:
        raise ValueError(f'{contract["skill"]} declares no capability {capability!r}; '
                         f'it has {sorted(contract["capabilities"])}')
    return spec


def _required_keys(declared):
    return [k for k, v in (declared or {}).items() if not str(v).endswith('?')]


def check_input(contract, capability, payload):
    spec = _capability(contract, capability)
    declared = spec.get('input') or {}
    problems = [f'missing input key: {k}' for k in _required_keys(declared) if k not in payload]
    problems += [f'input key not in the contract: {k}' for k in sorted(payload) if k not in declared]
    return problems


def check_output(contract, capability, response):
    spec = _capability(contract, capability)
    problems = []
    status = response.get('status')
    if status not in contract['status']:
        problems.append(f'status is {status!r}, the contract allows {contract["status"]}')
    if status == 'partial' and not response.get('uncovered'):
        problems.append('status is partial but `uncovered` does not list what was left out')
    if status != 'ok':
        return problems
    result = response.get('result')
    if not isinstance(result, dict):
        problems.append('result is missing or is not an object')
        return problems
    problems += [f'missing output key: {k}' for k in _required_keys(spec.get('output') or {})
                 if k not in result]
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(prog='check_contract.py', description=__doc__)
    p.add_argument('--contract', required=True)
    p.add_argument('--capability', required=True)
    p.add_argument('--response', required=True, help='a staged response file to validate')
    a = p.parse_args(argv)
    response = json.loads(Path(a.response).read_text(encoding='utf-8'))
    problems = check_output(load_contract(a.contract), a.capability, response)
    print(json.dumps({'ok': not problems, 'problems': problems}, indent=1, ensure_ascii=False))
    return 0 if not problems else 3


if __name__ == '__main__':
    raise SystemExit(main())

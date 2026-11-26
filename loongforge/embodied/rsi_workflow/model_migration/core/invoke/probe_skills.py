"""Checks both skill cards are mounted and match their contract, before a migration starts.

The framework delegates structure analysis and precision work entirely; starting a task with a
skill card missing would mean discovering it halfway through F4 or F5, with a half-written target
project on disk. So the probe runs first and the pipeline stops on a missing card rather than
improvising a replacement.

What it verifies, per skill: the mount point exists under `skills/`, `SKILL.md` is there and not
empty, the rule files it ships are readable, and the contract's `skill` field names this card.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.invoke import CONTRACTS, INVOKE_DIR                        # noqa: E402
from core.invoke.check_contract import load_contract                 # noqa: E402

SKILLS_DIR = INVOKE_DIR.parents[1] / 'skills'


def probe(skill, contract_path):
    report = {'skill': skill, 'mount': str(SKILLS_DIR / skill), 'problems': []}
    card = SKILLS_DIR / skill / 'SKILL.md'
    if not card.exists():
        report['problems'].append(f'skill card is not mounted: {card}')
    elif not card.read_text(encoding='utf-8').strip():
        report['problems'].append(f'skill card is empty: {card}')
    rules = sorted(p.name for p in (SKILLS_DIR / skill / 'rules').glob('*.md'))
    report['rules'] = rules
    if not rules:
        report['problems'].append(f'skill ships no rule files: {SKILLS_DIR / skill / "rules"}')
    try:
        contract = load_contract(contract_path)
    except (OSError, ValueError) as exc:
        report['problems'].append(f'contract is unusable: {exc}')
        return report
    report['contract'] = {'path': str(contract_path), 'version': contract['version'],
                          'capabilities': sorted(contract['capabilities'])}
    if contract['skill'] != skill:
        report['problems'].append(f'contract names {contract["skill"]!r}, mounted as {skill!r}')
    return report


def probe_all():
    skills = [probe(skill, INVOKE_DIR / name) for skill, name in sorted(CONTRACTS.items())]
    problems = [f'{s["skill"]}: {p}' for s in skills for p in s['problems']]
    return {'verdict': 'PASS' if not problems else 'FAIL', 'skills': skills, 'problems': problems}


def main(argv=None):
    p = argparse.ArgumentParser(prog='probe_skills.py', description=__doc__)
    p.add_argument('--out', help='write the report here as well as to stdout')
    a = p.parse_args(argv)
    report = probe_all()
    text = json.dumps(report, indent=1, ensure_ascii=False) + '\n'
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['verdict'] == 'PASS' else 3


if __name__ == '__main__':
    raise SystemExit(main())

"""The one place that calls model-adaptation. F4 reaches the skill only through here.

Of the three capabilities on the contract, F4 uses `adapt_dataset_pipeline`: the original data
pipeline is the part of a VLA model where a silent behaviour change (a different sampling order,
a dropped normalisation, a reshuffled frame window) still trains and still converges, just not to
the same loss. The skill owns that analysis; `dataset_bridge.py` only places what it returns.

The other two capabilities are F1/F2 territory and are exposed here so that stage does not grow
a second caller of its own.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.invoke import PENDING, PendingSkillCall, call               # noqa: E402

SKILL = 'model-adaptation'


def analyze_model_structure(source_repo, workdir, source_commit=None):
    payload = {'source_repo': str(source_repo)}
    if source_commit:
        payload['source_commit'] = source_commit
    return call(SKILL, 'analyze_model_structure', payload, workdir)


def dissect_step(source_repo, components, workdir):
    return call(SKILL, 'dissect_step',
                {'source_repo': str(source_repo), 'components': components}, workdir)


def adapt_dataset_pipeline(source_repo, target_data_root, workdir):
    """Returns {'plan': [{src, dst, action, notes}, ...]} for the data side of the migration."""
    return call(SKILL, 'adapt_dataset_pipeline',
                {'source_repo': str(source_repo), 'target_data_root': str(target_data_root)},
                workdir)


def main(argv=None):
    p = argparse.ArgumentParser(prog='call_model_adaptation.py', description=__doc__)
    p.add_argument('--capability', required=True,
                   choices=('analyze_model_structure', 'dissect_step', 'adapt_dataset_pipeline'))
    p.add_argument('--source-repo', required=True)
    p.add_argument('--workdir', required=True, help='work/<task_id>')
    p.add_argument('--source-commit')
    p.add_argument('--components', help='JSON file with the component list (dissect_step)')
    p.add_argument('--target-data-root', help='data/ root of the target project')
    p.add_argument('--out', help='write the skill result here')
    a = p.parse_args(argv)
    try:
        if a.capability == 'analyze_model_structure':
            result = analyze_model_structure(a.source_repo, a.workdir, a.source_commit)
        elif a.capability == 'dissect_step':
            components = json.loads(Path(a.components).read_text(encoding='utf-8'))
            result = dissect_step(a.source_repo, components, a.workdir)
        else:
            result = adapt_dataset_pipeline(a.source_repo, a.target_data_root, a.workdir)
    except PendingSkillCall as pending:
        print(pending.instructions())
        return PENDING
    text = json.dumps(result, indent=1, ensure_ascii=False) + '\n'
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding='utf-8')
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Transport between the framework and the two skills: a request file in, a response file out.

A skill is a playbook the coding assistant follows, not a library the framework can import, so a
call cannot block on a return value. It is staged on disk instead:

  1. the caller submits a request -> `<workdir>/invoke/<skill>.<capability>.<digest>.request.json`
  2. the assistant loads the skill, does the work, writes `<...>.response.json`
  3. the caller reruns and the response is validated against the contract and returned

The file name carries a digest of the payload, and the response must echo it. Two consequences:
the same capability called twice with different arguments (collecting from each of the two
training paths, for instance) keeps two independent request/response pairs, and a response left
over from an earlier version of a request is rejected rather than silently reused.

A call whose response is not there yet exits with code 4 (PENDING) and prints what to run -- that
is a normal state of the pipeline, not a failure, and `core/orchestrator/resume.sh` picks up from
exactly there.
"""
import hashlib
import json
from pathlib import Path
from typing import Literal, TypeAlias

from .check_contract import check_input, check_output, load_contract

PENDING = 4
CONTRACTS = {'model-adaptation': 'CONTRACT_model_adaptation.yaml',
             'precision-alignment': 'CONTRACT_precision_alignment.yaml'}
INVOKE_DIR = Path(__file__).resolve().parent

# The skill names the framework may pin. The two with contracts above are mounted under
# `skills/`; `vla-model-eval-adapter` is an optional external playbook with no contract here, so
# it can be referenced by a handoff but not called through `call()`.
EmbodiedSkillName: TypeAlias = Literal[
    'model-adaptation',
    'precision-alignment',
    'vla-model-eval-adapter',
]


class PendingSkillCall(Exception):
    """The request is staged; the skill has not answered yet."""

    def __init__(self, skill, capability, request_path, response_path):
        super().__init__(f'{skill}.{capability}: waiting for {response_path}')
        self.skill = skill
        self.capability = capability
        self.request_path = request_path
        self.response_path = response_path

    def instructions(self):
        return (f'load the {self.skill} skill, run its {self.capability} capability on\n'
                f'  {self.request_path}\n'
                f'and write the contract-shaped answer to\n  {self.response_path}')


def payload_sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode('utf-8')).hexdigest()


def call(skill, capability, payload, workdir):
    """Submit a request if needed, otherwise validate and return the skill's output.

    Raises PendingSkillCall when the response is missing or answers an older request, and
    ValueError when the payload or the response does not match the contract.
    """
    contract = load_contract(INVOKE_DIR / CONTRACTS[skill])
    problems = check_input(contract, capability, payload)
    if problems:
        raise ValueError(f'{skill}.{capability} request violates the contract: {problems}')
    digest = payload_sha(payload)
    stem = Path(workdir) / 'invoke' / f'{skill}.{capability}.{digest[:12]}'
    stem.parent.mkdir(parents=True, exist_ok=True)
    request = {'skill': skill, 'capability': capability, 'payload': payload,
               'payload_sha256': digest}
    request_path, response_path = Path(f'{stem}.request.json'), Path(f'{stem}.response.json')
    request_path.write_text(json.dumps(request, indent=1, ensure_ascii=False) + '\n',
                            encoding='utf-8')
    if not response_path.exists():
        raise PendingSkillCall(skill, capability, request_path, response_path)
    response = json.loads(response_path.read_text(encoding='utf-8'))
    if response.get('payload_sha256') != request['payload_sha256']:
        raise PendingSkillCall(skill, capability, request_path, response_path)
    problems = check_output(contract, capability, response)
    if problems:
        raise ValueError(f'{skill}.{capability} response violates the contract: {problems}')
    return response['result']

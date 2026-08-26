from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    REPO_ROOT / '.github' / 'workflows' / 'issue-intake.yml',
    REPO_ROOT / '.github' / 'workflows' / 'pr-intake.yml',
)


@pytest.mark.parametrize('path', WORKFLOWS)
def test_intake_workflow_separates_model_and_write_permissions(path: Path):
    document = yaml.load(path.read_text(), Loader=yaml.BaseLoader)

    assert document['permissions'] == {}
    classify = document['jobs']['classify']
    apply = document['jobs']['apply']
    assert all(value == 'read' for value in classify['permissions'].values())
    assert 'env' not in classify
    classify_step = next(step for step in classify['steps'] if step['name'].startswith('Classify'))
    assert 'ANTHROPIC_API_KEY' in classify_step['env']
    assert 'GITHUB_TOKEN' in classify_step['env']
    assert 'ANTHROPIC_API_KEY' not in str(apply)
    assert 'env' not in apply
    write_step = next(step for step in apply['steps'] if step['name'].startswith('Apply'))
    assert set(write_step['env']) == {'GITHUB_TOKEN'}
    assert any(value == 'write' for value in apply['permissions'].values())
    assert apply['needs'] == 'classify'


@pytest.mark.parametrize('path', WORKFLOWS)
def test_intake_workflow_only_interpolates_trusted_metadata(path: Path):
    text = path.read_text()

    assert 'github.event.issue.body' not in text
    assert 'github.event.issue.title' not in text
    assert 'github.event.pull_request.body' not in text
    assert 'github.event.pull_request.title' not in text


@pytest.mark.parametrize('path', WORKFLOWS)
def test_intake_workflow_actions_are_pinned_to_full_shas(path: Path):
    action_uses = re.findall(r'^\s*uses:\s*([^#\s]+)', path.read_text(), flags=re.MULTILINE)

    assert action_uses
    assert all(re.fullmatch(r'[^@]+@[0-9a-f]{40}', action) for action in action_uses)


def test_pr_intake_does_not_use_pull_request_target():
    text = WORKFLOWS[1].read_text()

    assert 'pull_request_target' not in yaml.load(text, Loader=yaml.BaseLoader)['on']

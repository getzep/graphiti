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
def test_intake_workflow_is_a_single_deterministic_job(path: Path):
    document = yaml.load(path.read_text(), Loader=yaml.BaseLoader)

    assert document['permissions'] == {}
    assert set(document['jobs']) == {'intake'}
    intake = document['jobs']['intake']
    assert 'env' not in intake

    # The decider and the apply step both use only the workflow token; no model
    # key or third-party dependency is involved anywhere in the file.
    run_steps = [step for step in intake['steps'] if 'run' in step]
    assert len(run_steps) == 2
    assert any('decide.py' in step['run'] for step in run_steps)
    assert any('apply.py' in step['run'] and '--write' in step['run'] for step in run_steps)
    for step in run_steps:
        assert set(step['env']) == {'GITHUB_TOKEN'}


@pytest.mark.parametrize('path', WORKFLOWS)
def test_intake_workflow_needs_no_llm_secret_or_pip_install(path: Path):
    text = path.read_text()

    assert 'INTAKE_API_KEY' not in text
    assert 'INTAKE_MODEL' not in text
    assert 'INTAKE_BASE_URL' not in text
    assert 'secrets.' not in text
    assert 'pip install' not in text


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


def test_issue_intake_permissions_and_triggers():
    document = yaml.load(WORKFLOWS[0].read_text(), Loader=yaml.BaseLoader)

    assert set(document['on']) == {'issues'}
    assert set(document['on']['issues']['types']) == {'opened', 'edited', 'reopened'}
    intake = document['jobs']['intake']
    assert intake['permissions'] == {'contents': 'read', 'issues': 'write'}
    assert 'issue_comment' not in str(document['on'])


def test_pr_intake_permissions():
    document = yaml.load(WORKFLOWS[1].read_text(), Loader=yaml.BaseLoader)

    assert document['jobs']['intake']['permissions'] == {
        'contents': 'read',
        'pull-requests': 'write',
        'issues': 'read',
    }


def test_stale_workflow_covers_every_promised_close_clock_label():
    # CONTRIBUTING.md and templates/pr_needs_rework.md promise a 14-day close
    # clock for every needs-* flag; the stale workflow must enforce all of them.
    stale = yaml.load(
        (REPO_ROOT / '.github' / 'workflows' / 'stale.yml').read_text(),
        Loader=yaml.BaseLoader,
    )
    step = stale['jobs']['stale']['steps'][0]['with']

    assert set(step['any-of-labels'].split(',')) == {
        'needs-info',
        'needs-issue',
        'needs-rfc',
        'needs-tests',
        'needs-rework',
    }
    assert set(step['exempt-issue-labels'].split(',')) == {'rfc-approved', 'security'}
    assert set(step['exempt-pr-labels'].split(',')) == {'rfc-approved', 'security'}


def test_pr_intake_runs_for_fork_pull_requests_without_touching_their_code():
    # pull_request_target is what lets the bot label fork pull requests. It is
    # safe only while every checkout pins the base branch and nothing reads the
    # pull request head, so those two facts are locked down together.
    text = WORKFLOWS[1].read_text()
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert 'pull_request_target' in document['on']
    assert 'pull_request' not in document['on']
    assert 'head.repo.full_name' not in text
    assert 'github.event.pull_request.head' not in text
    assert 'github.head_ref' not in text

    checkouts = [
        step
        for job in document['jobs'].values()
        for step in job['steps']
        if step.get('uses', '').startswith('actions/checkout@')
    ]
    assert checkouts
    for step in checkouts:
        assert step['with']['ref'] == '${{ github.event.pull_request.base.sha }}'
        assert step['with']['persist-credentials'] == 'false'


def test_ai_moderator_workflow_exists_and_reads_models():
    path = REPO_ROOT / '.github' / 'workflows' / 'ai-moderator.yml'
    document = yaml.load(path.read_text(), Loader=yaml.BaseLoader)

    job = document['jobs']['spam-detection']
    assert job['permissions']['models'] == 'read'

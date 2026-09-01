from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APPLY_PATH = REPO_ROOT / '.github' / 'intake' / 'apply.py'
SCENARIO_DIR = Path(__file__).resolve().parent / 'scenarios'
FAKE_REPO = 'mehulp93/graphiti-triage'
PRESERVED_LABEL = 'codex'


def load_apply():
    spec = importlib.util.spec_from_file_location('graphiti_intake_apply_scenarios', APPLY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


apply = load_apply()


class FakeGitHub:
    def __init__(self, responses: dict[tuple[str, str], object]):
        self.responses = responses
        self.requests: list[tuple[str, str, str, object]] = []

    def __call__(self, method: str, path: str, token: str, payload=None) -> object:
        self.requests.append((method, path, token, payload))
        return self.responses.get((method, path), {})


def scenario_paths() -> list[Path]:
    return sorted(SCENARIO_DIR.glob('*.json'))


def load_scenario(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['id'] == path.stem
    return payload


@pytest.mark.parametrize('path', scenario_paths(), ids=lambda path: path.stem)
def test_scenario_apply_json_labels_and_comment(path: Path):
    scenario = load_scenario(path)
    result = apply.apply_json(json.dumps(scenario['decision']))

    assert result.no_op is False
    assert result.labels == tuple(scenario['expect_labels'])
    for banned in scenario['forbid_labels']:
        assert banned not in result.labels

    expected_comment_id = scenario['expect_comment_id']
    assert result.comment_id == expected_comment_id
    if expected_comment_id is None:
        assert result.comment is None
    else:
        assert result.comment is not None
        assert apply.STICKY_MARKER in result.comment


@pytest.mark.parametrize('path', scenario_paths(), ids=lambda path: path.stem)
def test_scenario_apply_to_github_patch_payload(path: Path):
    scenario = load_scenario(path)
    result = apply.apply_json(json.dumps(scenario['decision']))
    issue_path = f'/repos/{FAKE_REPO}/issues/42'
    comments_path = f'{issue_path}/comments?per_page=100'
    github = FakeGitHub(
        {
            ('GET', issue_path): {'labels': [{'name': PRESERVED_LABEL}]},
            ('PATCH', issue_path): {},
            ('GET', comments_path): [],
            ('POST', f'{issue_path}/comments'): {'id': 101},
        }
    )

    summary = apply.apply_to_github(
        result,
        repo=FAKE_REPO,
        number=42,
        github_token='write-token',
        request_json=github,
    )

    patch = next(request for request in github.requests if request[0] == 'PATCH')
    expected = [*scenario['expect_labels'], PRESERVED_LABEL]
    assert patch[1] == issue_path
    assert patch[3] == {'labels': expected}
    for banned in scenario['forbid_labels']:
        assert banned not in patch[3]['labels']
    assert summary.labels_changed is True
    if scenario['expect_comment_id'] is None:
        assert summary.comment_action == 'none'
        assert all(request[0] != 'POST' for request in github.requests)
    else:
        assert summary.comment_action == 'created'
        post = next(request for request in github.requests if request[0] == 'POST')
        assert post[3] == {'body': result.comment}


def test_scenario_directory_covers_the_required_ids():
    expected = {
        'bug-incomplete-core',
        'bug-complete-core',
        'bug-mcp',
        'feature-needs-rfc',
        'docs',
        'question',
        'injection-banned',
        'security-misfile',
        'pr-needs-issue',
        'pr-needs-rfc',
        'pr-needs-tests',
    }
    assert {path.stem for path in scenario_paths()} == expected

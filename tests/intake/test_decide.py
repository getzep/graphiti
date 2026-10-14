from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DECIDE_PATH = REPO_ROOT / '.github' / 'intake' / 'decide.py'


def load_decide():
    spec = importlib.util.spec_from_file_location('graphiti_intake_decide', DECIDE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


decide = load_decide()


class FakeGitHub:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.requests: list[tuple[str, str]] = []

    def __call__(self, path: str, token: str) -> object:
        self.requests.append((path, token))
        return self.responses[path]


def issue_payload(**overrides) -> dict:
    payload = {
        'number': 42,
        'title': 'Report',
        'body': '',
        'user': {'login': 'reporter'},
        'labels': [],
    }
    payload.update(overrides)
    return payload


def fetch_issue(body: str, labels: list[str] | None = None, **overrides):
    github = FakeGitHub(
        {
            '/repos/getzep/graphiti/issues/42': issue_payload(
                body=body, labels=[{'name': label} for label in (labels or [])], **overrides
            ),
        }
    )
    return decide.fetch_intake_item(
        repo='getzep/graphiti',
        number=42,
        kind='issue',
        github_token='read-token',
        get_json=github,
    )


def fetch_pr(
    body: str, files: dict[str, list], issues: dict[int, object] | None = None, **overrides
):
    responses: dict[str, object] = {
        '/repos/getzep/graphiti/pulls/17': {
            'number': 17,
            'title': 'Change',
            'body': body,
            'user': {'login': 'contributor'},
            'labels': [],
            **overrides,
        },
    }
    for page, entries in files.items():
        responses[f'/repos/getzep/graphiti/pulls/17/files?per_page=100&page={page}'] = entries
    for number, payload in (issues or {}).items():
        responses[f'/repos/getzep/graphiti/issues/{number}'] = payload
    github = FakeGitHub(responses)
    item = decide.fetch_intake_item(
        repo='getzep/graphiti',
        number=17,
        kind='pull_request',
        github_token='read-token',
        get_json=github,
    )
    return item, github


BUG_FORM = """\
### Before you file

- [x] I searched existing issues and did not find a duplicate.
- [x] I read the contributing guide.

### Affected component

graphiti-core

### Bug description

Search returns stale results.

### Minimal reproduction

```python
print("repro")
```

### Expected behavior

Fresh results.

### Actual behavior

Stale results.

### Environment

Graphiti: 0.20
"""

FEATURE_FORM = """\
### Affected component

MCP server

### Size

{size}

### User problem

I cannot export data.

### Desired outcome

An export command.

### Proposed design

{proposal}

### Alternatives considered

Manual dumps.

### Compatibility and operational impact

None.
"""


def test_parse_form_sections_reads_labels_and_values():
    sections = decide.parse_form_sections(BUG_FORM)

    assert sections['Affected component'] == 'graphiti-core'
    assert sections['Bug description'] == 'Search returns stale results.'
    assert sections['Environment'] == 'Graphiti: 0.20'


def test_field_missing_treats_absent_blank_and_no_response_as_missing():
    sections = decide.parse_form_sections(
        '### Expected behavior\n\n_No response_\n\n### Actual behavior\n\n\n\n'
    )

    assert decide.field_missing(sections, 'Expected behavior') is True
    assert decide.field_missing(sections, 'Actual behavior') is True
    assert decide.field_missing(sections, 'Logs or traceback') is True
    assert decide.field_missing({'X': 'filled'}, 'X') is False


def test_checkbox_checked_matches_checked_and_unchecked_lines():
    body = '- [x] Tests are not applicable; reason given.\n- [ ] Feature flag\n'
    assert decide.checkbox_checked(body, 'Tests are not applicable') is True
    assert decide.checkbox_checked(body, 'Feature') is False
    assert decide.checkbox_checked('- [X] Tests are not applicable', 'Tests are not') is True
    assert decide.checkbox_checked('plain text', 'Tests') is False


def test_bug_issue_with_all_fields_is_clean():
    item = fetch_issue(BUG_FORM, labels=['bug'])
    decision = decide.decide_issue(item)

    assert decision['category'] == 'bug'
    assert decision['areas'] == ['scope:core']
    assert decision['labels'] == ['bug', 'scope:core']
    assert decision['comment_id'] is None
    assert decision['missing_fields'] == []


def test_bug_issue_missing_reproduction_and_environment_asks_repro():
    body = BUG_FORM.replace('### Minimal reproduction', '### Minimal repro').replace(
        '### Environment', '### Env'
    )
    item = fetch_issue(body, labels=['bug'])
    decision = decide.decide_issue(item)

    assert 'needs-info' in decision['labels']
    assert decision['comment_id'] == 'ask_repro'
    assert decision['missing_fields'] == ['reproduction', 'environment']


def test_small_feature_issue_is_clean():
    item = fetch_issue(
        FEATURE_FORM.format(size='Small improvement to existing behavior', proposal='Add a flag.'),
        labels=['feature'],
    )
    decision = decide.decide_issue(item)

    assert decision['labels'] == ['feature', 'scope:mcp']
    assert decision['comment_id'] is None


def test_large_feature_missing_proposal_needs_rfc_and_info():
    item = fetch_issue(
        FEATURE_FORM.format(
            size='Large feature requiring design approval', proposal='_No response_'
        ),
        labels=['feature'],
    )
    decision = decide.decide_issue(item)

    assert 'needs-rfc' in decision['labels']
    assert 'needs-info' in decision['labels']
    assert decision['comment_id'] == 'ask_rfc_fields'
    assert decision['missing_fields'] == ['proposal']


def test_large_feature_with_rfc_approved_skips_needs_rfc():
    item = fetch_issue(
        FEATURE_FORM.format(size='Large feature requiring design approval', proposal='Done.'),
        labels=['feature', 'rfc-approved'],
    )
    decision = decide.decide_issue(item)

    assert 'needs-rfc' not in decision['labels']


def test_documentation_issue_without_component_defaults_to_docs_scope():
    body = '### Documentation location\n\ndocs/foo.md\n\n### What needs to change?\n\nFix typo\n'
    item = fetch_issue(body, labels=['documentation'])
    decision = decide.decide_issue(item)

    assert decision['areas'] == ['scope:docs']
    assert decision['labels'] == ['documentation', 'scope:docs']


def test_question_missing_goal_asks_info():
    body = (
        '### Component\n\ngraphiti-core\n\n### What have you tried?\n\nReading the docs\n\n'
        '### Environment\n\nGraphiti 0.20\n'
    )
    item = fetch_issue(body, labels=['question'])
    decision = decide.decide_issue(item)

    assert 'needs-info' in decision['labels']
    assert decision['comment_id'] == 'ask_info'
    assert decision['missing_fields'] == ['goal']
    assert decision['areas'] == ['scope:core']


def test_issue_not_from_a_form_reports_all_required_fields_missing():
    item = fetch_issue('Something is broken, please help.', labels=['bug'])
    decision = decide.decide_issue(item)

    assert 'needs-info' in decision['labels']
    assert decision['missing_fields'] == [
        'description',
        'reproduction',
        'expected',
        'actual',
        'environment',
    ]


def test_main_writes_empty_decision_for_trusted_author(tmp_path, monkeypatch):
    item = decide.IntakeItem(
        kind='issue', number=5, title='t', body='b', author='m', author_association='MEMBER'
    )
    monkeypatch.setattr(decide, 'fetch_intake_item', lambda **_: item)
    monkeypatch.setenv('GITHUB_TOKEN', 'g')
    output = tmp_path / 'decision.json'

    code = decide.main(
        [
            '--repo',
            'getzep/graphiti',
            '--number',
            '5',
            '--kind',
            'issue',
            '--output',
            str(output),
        ]
    )

    assert code == 0
    assert output.read_text().strip() == '{}'


def test_main_writes_empty_decision_for_draft_pull_request(tmp_path, monkeypatch):
    item = decide.IntakeItem(
        kind='pull_request', number=5, title='t', body='b', author='c', draft=True
    )
    monkeypatch.setattr(decide, 'fetch_intake_item', lambda **_: item)
    monkeypatch.setenv('GITHUB_TOKEN', 'g')
    output = tmp_path / 'decision.json'

    code = decide.main(
        [
            '--repo',
            'getzep/graphiti',
            '--number',
            '5',
            '--kind',
            'pull_request',
            '--output',
            str(output),
        ]
    )

    assert code == 0
    assert output.read_text().strip() == '{}'


def test_pr_without_linked_issue_needs_issue_and_rework_comment():
    item, _ = fetch_pr(
        'No linked issue.',
        files={1: [{'filename': 'docs/guide.md', 'status': 'modified'}]},
    )
    decision = decide.decide_pull_request(item)

    assert 'needs-issue' in decision['labels']
    assert 'needs-tests' not in decision['labels']
    assert decision['comment_id'] == 'pr_needs_rework'
    assert decision['missing_fields'] == ['linked-issue']


def test_pr_linked_to_feature_issue_without_rfc_needs_rfc():
    item, _ = fetch_pr(
        'Fixes #42\n\n- [x] Tests are not applicable; config only.',
        files={1: [{'filename': 'graphiti_core/x.py', 'status': 'modified'}]},
        issues={42: {'number': 42, 'labels': [{'name': 'feature'}]}},
    )
    decision = decide.decide_pull_request(item)

    assert decision['category'] == 'feature'
    assert 'needs-rfc' in decision['labels']
    assert 'needs-issue' not in decision['labels']
    assert decision['comment_id'] == 'pr_needs_rfc'


def test_pr_linked_to_rfc_approved_feature_is_clean():
    item, _ = fetch_pr(
        'Fixes #42',
        files={1: [{'filename': 'tests/test_x.py', 'status': 'added'}]},
        issues={42: {'number': 42, 'labels': [{'name': 'feature'}, {'name': 'rfc-approved'}]}},
    )
    decision = decide.decide_pull_request(item)

    assert decision['category'] == 'feature'
    assert not [label for label in decision['labels'] if label.startswith('needs-')]
    assert decision['comment_id'] is None


def test_pr_changing_code_without_tests_needs_tests():
    item, _ = fetch_pr(
        'Fixes #42',
        files={1: [{'filename': 'graphiti_core/x.py', 'status': 'modified'}]},
        issues={42: {'number': 42, 'labels': [{'name': 'bug'}]}},
    )
    decision = decide.decide_pull_request(item)

    assert 'needs-tests' in decision['labels']
    assert decision['comment_id'] == 'pr_needs_tests'
    assert decision['missing_fields'] == ['tests']


def test_pr_with_a_test_file_does_not_need_tests():
    item, _ = fetch_pr(
        'Fixes #42',
        files={
            1: [
                {'filename': 'graphiti_core/x.py', 'status': 'modified'},
                {'filename': 'tests/test_x.py', 'status': 'added'},
            ]
        },
        issues={42: {'number': 42, 'labels': [{'name': 'bug'}]}},
    )
    decision = decide.decide_pull_request(item)

    assert 'needs-tests' not in decision['labels']
    assert decision['comment_id'] is None


def test_pr_with_tests_not_applicable_checkbox_does_not_need_tests():
    item, _ = fetch_pr(
        'Fixes #42\n\n- [x] Tests are not applicable; the reason is explained below.',
        files={1: [{'filename': 'graphiti_core/x.py', 'status': 'modified'}]},
        issues={42: {'number': 42, 'labels': [{'name': 'bug'}]}},
    )
    decision = decide.decide_pull_request(item)

    assert 'needs-tests' not in decision['labels']


def test_docs_only_pr_does_not_need_tests():
    item, _ = fetch_pr(
        'Fixes #42',
        files={1: [{'filename': 'docs/guide.md', 'status': 'modified'}]},
        issues={42: {'number': 42, 'labels': [{'name': 'documentation'}]}},
    )
    decision = decide.decide_pull_request(item)

    assert 'needs-tests' not in decision['labels']


def test_pr_missing_issue_and_tests_gets_needs_rework():
    item, _ = fetch_pr(
        'No links here.',
        files={1: [{'filename': 'graphiti_core/x.py', 'status': 'modified'}]},
    )
    decision = decide.decide_pull_request(item)

    assert 'needs-issue' in decision['labels']
    assert 'needs-tests' in decision['labels']
    assert 'needs-rework' in decision['labels']
    assert decision['comment_id'] == 'pr_needs_rework'
    assert decision['missing_fields'] == ['linked-issue', 'tests']


def test_pr_feature_category_from_checked_template_checkbox():
    item, _ = fetch_pr(
        'Fixes #42\n\n- [x] Feature (linked Feature issue already has `rfc-approved`)',
        files={1: [{'filename': 'tests/test_x.py', 'status': 'added'}]},
        issues={42: {'number': 42, 'labels': [{'name': 'rfc-approved'}]}},
    )
    decision = decide.decide_pull_request(item)

    assert decision['category'] == 'feature'
    assert 'needs-rfc' not in decision['labels']


# --- Fetch behavior ported from the classifier tests -------------------------


def test_fetch_pr_paginates_files_past_the_first_page():
    files_page_1 = [{'filename': 'graphiti_core/x.py', 'status': 'modified'} for _ in range(100)]
    files_page_2 = [
        {'filename': 'tests/test_x.py', 'status': 'added'},
        {'filename': 'mcp_server/y.py', 'status': 'added'},
        *[{'filename': f'graphiti_core/f{i}.py', 'status': 'added'} for i in range(28)],
    ]
    item, github = fetch_pr(
        'No linked issue here.',
        files={1: files_page_1, 2: files_page_2},
    )

    assert len(item.files) == 130
    # A scope only visible on a later page must still be derived.
    assert 'scope:mcp' in decide.derive_pr_scopes(item.files)
    files_requests = [path for path, _ in github.requests if '/files?' in path]
    assert len(files_requests) == 2
    assert files_requests[1].endswith('page=2')


def test_fetch_pr_skips_linked_issue_that_does_not_exist():
    responses = {
        '/repos/getzep/graphiti/pulls/17': {
            'number': 17,
            'title': 'Fix a thing',
            'body': 'Fixes #42000',
            'user': {'login': 'contributor'},
            'labels': [],
        },
        '/repos/getzep/graphiti/pulls/17/files?per_page=100&page=1': [
            {'filename': 'graphiti_core/nodes.py', 'status': 'modified'},
        ],
    }

    def get_json(path: str, token: str) -> object:
        if path == '/repos/getzep/graphiti/issues/42000':
            raise decide.GitHubNotFoundError('HTTP 404')
        return responses[path]

    item = decide.fetch_intake_item(
        repo='getzep/graphiti',
        number=17,
        kind='pull_request',
        github_token='read-token',
        get_json=get_json,
    )

    # A deleted or never-existing issue cannot satisfy the linked-issue
    # requirement, so needs-issue still applies.
    assert item.linked_issues == ()
    decision = decide.decide_pull_request(item)
    assert 'needs-issue' in decision['labels']


def test_fetch_pr_propagates_non_404_linked_issue_errors():
    def get_json(path: str, token: str) -> object:
        if path == '/repos/getzep/graphiti/issues/42':
            raise RuntimeError('HTTP 500 from GitHub')
        return {
            '/repos/getzep/graphiti/pulls/17': {
                'number': 17,
                'title': 'Fix a thing',
                'body': 'Fixes #42',
                'user': {'login': 'contributor'},
                'labels': [],
            },
            '/repos/getzep/graphiti/pulls/17/files?per_page=100&page=1': [],
        }[path]

    with pytest.raises(RuntimeError, match='HTTP 500'):
        decide.fetch_intake_item(
            repo='getzep/graphiti',
            number=17,
            kind='pull_request',
            github_token='read-token',
            get_json=get_json,
        )


def test_fetch_pr_ignores_linked_number_that_is_a_pull_request():
    github = FakeGitHub(
        {
            '/repos/getzep/graphiti/pulls/17': {
                'number': 17,
                'title': 'Fix a thing',
                'body': 'Fixes #49',
                'user': {'login': 'contributor'},
                'labels': [],
            },
            '/repos/getzep/graphiti/pulls/17/files?per_page=100&page=1': [],
            '/repos/getzep/graphiti/issues/49': {
                'number': 49,
                'labels': [],
                'pull_request': {'url': 'https://api.github.com/repos/getzep/graphiti/pulls/49'},
            },
        }
    )

    item = decide.fetch_intake_item(
        repo='getzep/graphiti',
        number=17,
        kind='pull_request',
        github_token='read-token',
        get_json=github,
    )

    # The issues endpoint also serves pull requests; `Fixes #49` pointing at a
    # PR is not a linked issue, so needs-issue still applies.
    assert item.linked_issues == ()
    decision = decide.decide_pull_request(item)
    assert 'needs-issue' in decision['labels']


@pytest.mark.parametrize('kind', ['not-an-issue', '', 'pr'])
def test_fetch_rejects_unknown_kind(kind: str):
    with pytest.raises(ValueError, match='kind'):
        decide.fetch_intake_item(
            repo='getzep/graphiti',
            number=1,
            kind=kind,
            github_token='read-token',
            get_json=lambda _path, _token: {},
        )

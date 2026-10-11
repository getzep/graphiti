#!/usr/bin/env python3
"""Deterministic GitHub intake decider.

Fetches an issue or pull request by number and writes a decision matching
decision.schema.json, using only objective facts: the issue-form sections, the
labels the forms apply, the changed file paths, and the linked-issue references
in the body. No LLM is involved, so the only credential needed is GITHUB_TOKEN
with read access. apply.py re-validates every decision against the schema and
an allowlist before anything reaches GitHub.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

INTAKE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = INTAKE_DIR / 'decision.schema.json'
GITHUB_API = 'https://api.github.com'
MAX_FILES = 100
# GitHub caps the files listing at 3,000 entries (100 per page), so 30 pages is
# the hard ceiling even for the largest pull requests.
MAX_FILE_PAGES = 30
LINKED_ISSUE_RE = re.compile(r'(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+#(\d+)\b')
NO_RESPONSE = '_No response_'
FORM_SECTION_RE = re.compile(r'^###\s+(.+?)\s*$', re.MULTILINE)
CHECKBOX_RE = re.compile(r'^\s*-\s*\[([xX ])\]\s*(.*)$')
LARGE_FEATURE_SIZE = 'Large feature requiring design approval'
FEATURE_CHECKBOX = 'Feature (linked Feature issue already has'
NO_TESTS_CHECKBOX = 'Tests are not applicable'

GetJson = Callable[[str, str], object]


@dataclass(frozen=True)
class IntakeItem:
    kind: str
    number: int
    title: str
    body: str
    author: str
    author_association: str = ''
    labels: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    linked_issues: tuple[dict[str, Any], ...] = ()
    draft: bool = False
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0


# Author associations we treat as trusted maintainers — their issues/PRs skip
# classification entirely (no decision, no labels).
TRUSTED_ASSOCIATIONS = frozenset({'OWNER', 'MEMBER'})


class GitHubNotFoundError(RuntimeError):
    """The requested GitHub resource does not exist (HTTP 404)."""


def _http_error(error: HTTPError) -> RuntimeError:
    body = error.read().decode('utf-8', errors='replace')
    message = f'HTTP {error.code} from {error.url}: {body[:1000]}'
    if error.code == 404:
        return GitHubNotFoundError(message)
    return RuntimeError(message)


def get_github_json(path: str, token: str) -> object:
    request = Request(
        f'{GITHUB_API}{path}',
        headers={
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {token}',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'graphiti-intake-decider',
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        raise _http_error(error) from error


def _labels(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        label['name']
        for label in payload.get('labels', [])
        if isinstance(label, dict) and isinstance(label.get('name'), str)
    )


def _linked_issue_numbers(body: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(match) for match in LINKED_ISSUE_RE.findall(body)))[:10]


def _linked_issues(
    repo: str, body: str, token: str, get_json: GetJson
) -> tuple[dict[str, Any], ...]:
    linked: list[dict[str, Any]] = []
    for number in _linked_issue_numbers(body):
        try:
            payload = get_json(f'/repos/{repo}/issues/{number}', token)
        except GitHubNotFoundError:
            # The referenced issue was deleted or never existed; it cannot
            # satisfy the linked-issue requirement, so treat it as absent.
            continue
        if not isinstance(payload, dict):
            raise RuntimeError(f'GitHub issue #{number} response was not an object')
        # The issues endpoint also serves pull requests; a `Fixes #N` pointing
        # at a PR is not a linked issue.
        if 'pull_request' in payload:
            continue
        linked.append({'number': number, 'labels': list(_labels(payload))})
    return tuple(linked)


# Objective facts derived in code, never left to judgment: scope comes from
# changed paths; missing-linked-issue is a hard compliance fact.
PATH_SCOPE_RULES = (
    ('mcp_server/', 'scope:mcp'),
    ('server/', 'scope:service'),
    ('.github/', 'scope:ci'),
    ('docs/', 'scope:docs'),
    ('examples/', 'scope:docs'),
    ('graphiti_core/', 'scope:core'),
)


def _scope_for_path(path: str) -> str:
    lower = path.lower()
    if lower.endswith('.md'):
        return 'scope:docs'
    if 'dockerfile' in lower or lower.endswith(('makefile', '.lock', '.toml')):
        return 'scope:ci'
    for prefix, scope in PATH_SCOPE_RULES:
        if path.startswith(prefix):
            return scope
    return 'scope:core'


def derive_pr_scopes(files: tuple[str, ...]) -> list[str]:
    scopes: list[str] = []
    for entry in files:
        name = entry.rsplit(' (', 1)[0]
        scope = _scope_for_path(name)
        if scope not in scopes:
            scopes.append(scope)
    return scopes[:5]


def fetch_intake_item(
    *,
    repo: str,
    number: int,
    kind: str,
    github_token: str,
    get_json: GetJson = get_github_json,
) -> IntakeItem:
    if kind not in {'issue', 'pull_request'}:
        raise ValueError('kind must be issue or pull_request')
    if number < 1:
        raise ValueError('number must be positive')

    endpoint = 'issues' if kind == 'issue' else 'pulls'
    payload = get_json(f'/repos/{repo}/{endpoint}/{number}', github_token)
    if not isinstance(payload, dict):
        raise RuntimeError('GitHub item response was not an object')

    body = str(payload.get('body') or '')
    user = payload.get('user')
    author = str(user.get('login') or '') if isinstance(user, dict) else ''
    association = str(payload.get('author_association') or '')

    files: tuple[str, ...] = ()
    linked_issues: tuple[dict[str, Any], ...] = ()
    if kind == 'pull_request':
        files_payload: list[Any] = []
        for page in range(1, MAX_FILE_PAGES + 1):
            page_payload = get_json(
                f'/repos/{repo}/pulls/{number}/files?per_page={MAX_FILES}&page={page}',
                github_token,
            )
            if not isinstance(page_payload, list):
                raise RuntimeError('GitHub pull request files response was not a list')
            files_payload.extend(page_payload)
            if len(page_payload) < MAX_FILES:
                break
        files = tuple(
            f'{file["filename"]} ({file.get("status", "modified")})'
            for file in files_payload
            if isinstance(file, dict) and isinstance(file.get('filename'), str)
        )
        linked_issues = _linked_issues(repo, body, github_token, get_json)

    return IntakeItem(
        kind=kind,
        number=number,
        title=str(payload.get('title') or ''),
        body=body,
        author=author,
        author_association=association,
        labels=_labels(payload),
        files=files,
        linked_issues=linked_issues,
        draft=bool(payload.get('draft')),
        additions=int(payload.get('additions') or 0),
        deletions=int(payload.get('deletions') or 0),
        changed_files=int(payload.get('changed_files') or 0),
    )


# --- Issue-form parsing -----------------------------------------------------

# GitHub renders a submitted issue form as `### <Label>\n\n<value>` sections.
# Dropdown options map to the scope taxonomy; values not listed (including
# 'Multiple components', 'Other', and 'General usage or architecture') carry no
# scope signal.
COMPONENT_SCOPES = {
    'graphiti-core': 'scope:core',
    'MCP server': 'scope:mcp',
    'REST server': 'scope:service',
    'Documentation or examples': 'scope:docs',
    'CI, Docker, or release': 'scope:ci',
}
# The bug and feature forms label the dropdown 'Affected component'; the
# question form labels it 'Component'.
COMPONENT_LABELS = ('Affected component', 'Component')

# Required sections per category, as (rendered label, missing_fields id) pairs
# in the order the forms render them.
ISSUE_REQUIRED_FIELDS = {
    'bug': (
        ('Bug description', 'description'),
        ('Minimal reproduction', 'reproduction'),
        ('Expected behavior', 'expected'),
        ('Actual behavior', 'actual'),
        ('Environment', 'environment'),
    ),
    'feature': (
        ('User problem', 'problem'),
        ('Desired outcome', 'outcome'),
    ),
    'question': (
        ('What are you trying to accomplish?', 'goal'),
        ('What have you tried?', 'attempted'),
        ('Environment', 'environment'),
    ),
    'documentation': (
        ('Documentation location', 'location'),
        ('What needs to change?', 'problem'),
    ),
}
RFC_REQUIRED_FIELDS = (
    ('Proposed design', 'proposal'),
    ('Alternatives considered', 'alternatives'),
    ('Compatibility and operational impact', 'impact'),
)


def parse_form_sections(body: str) -> dict[str, str]:
    matches = list(FORM_SECTION_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1).strip()] = body[match.end() : end].strip()
    return sections


def field_missing(sections: dict[str, str], label: str) -> bool:
    value = sections.get(label)
    if value is None:
        return True
    return not value.strip() or value.strip() == NO_RESPONSE


def checkbox_checked(body: str, text: str) -> bool:
    for line in body.splitlines():
        match = CHECKBOX_RE.match(line)
        if match and match.group(1).lower() == 'x' and match.group(2).strip().startswith(text):
            return True
    return False


def _issue_areas(sections: dict[str, str], category: str) -> list[str]:
    for label in COMPONENT_LABELS:
        value = sections.get(label)
        if value is not None:
            scope = COMPONENT_SCOPES.get(value.strip())
            areas = [scope] if scope else []
            break
    else:
        areas = []
    if category == 'documentation' and not areas:
        areas = ['scope:docs']
    return areas


def _decision(
    category: str,
    areas: list[str],
    flags: list[str],
    comment_id: str | None,
    missing_fields: list[str],
) -> dict[str, Any]:
    labels = sorted(([category] if category != 'other' else []) + areas + flags)
    return {
        'category': category,
        'areas': areas,
        'labels': labels,
        'comment_id': comment_id,
        'duplicate_issue_numbers': [],
        'missing_fields': missing_fields,
    }


def decide_issue(item: IntakeItem) -> dict[str, Any]:
    sections = parse_form_sections(item.body)
    category = next(
        (
            candidate
            for candidate in ('bug', 'feature', 'documentation', 'question')
            if candidate in item.labels
        ),
        'other',
    )
    areas = _issue_areas(sections, category)

    required = ISSUE_REQUIRED_FIELDS.get(category, ())
    missing = [field_id for label, field_id in required if field_missing(sections, label)]

    flags: list[str] = []
    large_feature = category == 'feature' and sections.get('Size', '').strip() == LARGE_FEATURE_SIZE
    if large_feature and 'rfc-approved' not in item.labels:
        flags.append('needs-rfc')
        missing.extend(
            field_id for label, field_id in RFC_REQUIRED_FIELDS if field_missing(sections, label)
        )
    if missing:
        flags.append('needs-info')

    if 'needs-rfc' in flags:
        comment_id: str | None = 'ask_rfc_fields'
    elif category == 'bug' and 'needs-info' in flags:
        comment_id = 'ask_repro'
    elif 'needs-info' in flags:
        comment_id = 'ask_info'
    else:
        comment_id = None
    return _decision(category, areas, flags, comment_id, missing)


def is_test_path(path: str) -> bool:
    name = path.rsplit('/', 1)[-1]
    return (
        path.startswith('tests/')
        or '/tests/' in path
        or name.startswith('test_')
        or name.endswith('_test.py')
        or name == 'conftest.py'
    )


def is_code_path(path: str) -> bool:
    if is_test_path(path):
        return False
    if path.lower().endswith('.md'):
        return False
    if path.startswith(('docs/', 'examples/', '.github/')):
        return False
    return _scope_for_path(path) != 'scope:ci'


def decide_pull_request(item: IntakeItem) -> dict[str, Any]:
    paths = [entry.rsplit(' (', 1)[0] for entry in item.files]
    areas = derive_pr_scopes(item.files)
    linked_labels = {label for issue in item.linked_issues for label in issue['labels']}

    missing: list[str] = []
    flags: list[str] = []
    if not item.linked_issues:
        flags.append('needs-issue')
        missing.append('linked-issue')

    if 'feature' in linked_labels:
        category = 'feature'
    elif 'bug' in linked_labels:
        category = 'bug'
    elif 'documentation' in linked_labels:
        category = 'documentation'
    elif checkbox_checked(item.body, FEATURE_CHECKBOX):
        category = 'feature'
    else:
        category = 'other'

    if category == 'feature' and 'rfc-approved' not in linked_labels:
        flags.append('needs-rfc')

    has_code = any(is_code_path(path) for path in paths)
    has_test = any(is_test_path(path) for path in paths)
    if has_code and not has_test and not checkbox_checked(item.body, NO_TESTS_CHECKBOX):
        flags.append('needs-tests')
        missing.append('tests')
    if 'needs-issue' in flags and 'needs-tests' in flags:
        flags.append('needs-rework')

    if 'needs-issue' in flags:
        comment_id: str | None = 'pr_needs_rework'
    elif 'needs-rfc' in flags:
        comment_id = 'pr_needs_rfc'
    elif 'needs-tests' in flags:
        comment_id = 'pr_needs_tests'
    else:
        comment_id = None

    missing_fields = [field for field in ('linked-issue', 'tests') if field in missing]
    return _decision(category, areas, flags, comment_id, missing_fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Decide triage for one GitHub issue or PR.')
    parser.add_argument('--repo', required=True, help='GitHub repository in owner/name form')
    parser.add_argument('--number', required=True, type=int, help='Issue or pull request number')
    parser.add_argument('--kind', required=True, choices=('issue', 'pull_request'))
    parser.add_argument('--output', required=True, type=Path, help='Decision JSON output path')
    args = parser.parse_args(argv)

    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        parser.error('GITHUB_TOKEN must be set')

    item = fetch_intake_item(
        repo=args.repo,
        number=args.number,
        kind=args.kind,
        github_token=github_token,
    )

    # Trusted maintainers and draft PRs need no triage: write the empty
    # decision, which apply.py treats as a clean no-op.
    if item.author_association in TRUSTED_ASSOCIATIONS or (
        item.kind == 'pull_request' and item.draft
    ):
        args.output.write_text('{}\n', encoding='utf-8')
        return 0

    decision = decide_pull_request(item) if item.kind == 'pull_request' else decide_issue(item)
    args.output.write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

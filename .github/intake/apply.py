#!/usr/bin/env python3
"""Intake apply layer.

Takes a decision (JSON) and turns it into allowlisted labels plus a templated
sticky comment. Never executes model text and never echoes an issue or pull
request body. GitHub writes happen only when the CLI is explicitly given
``--write`` and a write-scoped token.

Deciding steps (classify.py, later reproduce.py) emit decision.schema.json.
This file is the only place that may turn a decision into GitHub-facing output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

INTAKE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = INTAKE_DIR / 'decision.schema.json'
TEMPLATE_DIR = INTAKE_DIR / 'templates'
STICKY_MARKER = '<!-- graphiti-intake-bot -->'
# Must fit the widest legitimate item: a category, all five path-derived scope:*
# labels, every needs-* flag, and duplicate — truncating those silently loses
# signal, while the cap still bounds label spam from a hijacked decision.
MAX_LABELS = 12
MAX_COMMENT_CHARS = 4000
MAX_SUBSTITUTION_CHARS = 80

# Derived from the decision schema so it is the single source of truth for the
# label taxonomy; the allowlist can never silently drift from the schema enum.
_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
ALLOWLIST = frozenset(_SCHEMA['properties']['labels']['items']['enum'])
BANNED_LABELS = frozenset(
    {
        'rfc-approved',
        'good first issue',
        'help wanted',
        'triage/high',
        'triage/medium',
        'triage/low',
        'triage/skip',
        'wontfix',
        'spam',
        'ai-generated',
    }
)
LABEL_ALIASES = {
    'enhancement': 'feature',
    'slop-detected': 'needs-rework',
    'intake/needs-info': 'needs-info',
    'area/core': 'scope:core',
    'area/mcp': 'scope:mcp',
    'area/server': 'scope:service',
    'area/docs': 'scope:docs',
}
TYPE_LABELS = frozenset(_SCHEMA['properties']['category']['enum']) & ALLOWLIST
LABEL_ORDER = (
    'bug',
    'feature',
    'question',
    'documentation',
    'security',
    'scope:core',
    'scope:mcp',
    'scope:service',
    'scope:docs',
    'scope:ci',
    'needs-info',
    'needs-issue',
    'needs-rfc',
    'needs-tests',
    'needs-rework',
    'duplicate',
    'invalid',
)
MISSING_FIELD_COPY = {
    'reproduction': 'a minimal reproduction script or test case',
    'expected': 'expected behavior',
    'actual': 'actual behavior',
    'environment': 'environment details (versions, backend, models)',
    'logs': 'relevant logs or traceback, with secrets removed',
    'description': 'a short description of the problem',
    'problem': 'the user problem this would solve',
    'outcome': 'the desired outcome',
    'proposal': 'a proposed design',
    'alternatives': 'alternatives considered',
    'impact': 'compatibility and operational impact',
    'tests': 'tests for the behavior change',
    'linked-issue': 'a linked issue (`Fixes #<number>`)',
    'scope': 'a narrower, reviewable scope',
    'location': 'a docs URL or repository path',
}
SECRET_RE = re.compile(r'(?:ghp_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|sk-[A-Za-z0-9_-]{8,})')
URL_RE = re.compile(r'https?://[^\s)<>]+', re.IGNORECASE)
# Host allowlist checked against the parsed hostname, never a string prefix — a
# prefix check lets `https://help.getzep.com.evil.com/...` slip through.
ALLOWED_LINK_HOSTS = frozenset({'github.com', 'getzep.com'})
GITHUB_API = 'https://api.github.com'
MANAGED_LABELS = ALLOWLIST | frozenset(LABEL_ALIASES)

GitHubRequest = Callable[[str, str, str, object | None], object]


@dataclass(frozen=True)
class ApplyResult:
    labels: tuple[str, ...]
    comment: str | None
    comment_id: str | None
    dropped_labels: tuple[str, ...]
    no_op: bool
    sticky_marker: str = STICKY_MARKER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GitHubApplySummary:
    labels_changed: bool
    comment_action: str
    no_op: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))


def _comment_ids(schema: dict[str, Any]) -> frozenset[str]:
    variants = schema['properties']['comment_id']['oneOf']
    for variant in variants:
        if variant.get('type') == 'string':
            return frozenset(variant['enum'])
    raise RuntimeError('decision.schema.json is missing comment_id string enum')


def _validate_decision(data: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    required = schema['required']
    if any(key not in data for key in required):
        return None

    categories = set(schema['properties']['category']['enum'])
    area_enum = set(schema['properties']['areas']['items']['enum'])
    comment_ids = _comment_ids(schema)
    max_areas = schema['properties']['areas']['maxItems']
    max_labels = schema['properties']['labels']['maxItems']
    max_dupes = schema['properties']['duplicate_issue_numbers']['maxItems']
    max_missing = schema['properties']['missing_fields']['maxItems']
    max_label_len = schema['properties']['labels']['items']['maxLength']
    max_missing_len = schema['properties']['missing_fields']['items']['maxLength']

    category = data['category']
    areas = data['areas']
    labels = data['labels']
    comment_id = data['comment_id']
    duplicates = data['duplicate_issue_numbers']
    missing = data['missing_fields']

    if category not in categories:
        return None
    if not isinstance(areas, list) or len(areas) > max_areas:
        return None
    if any(area not in area_enum for area in areas):
        return None
    if len(set(areas)) != len(areas):
        return None
    if not isinstance(labels, list) or len(labels) > max_labels:
        return None
    if any(not isinstance(label, str) or len(label) > max_label_len for label in labels):
        return None
    if comment_id is not None and comment_id not in comment_ids:
        return None
    if not isinstance(duplicates, list) or len(duplicates) > max_dupes:
        return None
    if any(not isinstance(num, int) or isinstance(num, bool) or num < 1 for num in duplicates):
        return None
    if not isinstance(missing, list) or len(missing) > max_missing:
        return None
    if any(not isinstance(item, str) or len(item) > max_missing_len for item in missing):
        return None
    return data


def parse_decision(raw: str, schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return _validate_decision(data, schema or load_schema())


def _redact_secrets(text: str) -> str:
    return SECRET_RE.sub('[redacted]', text)


def _neutralize_html(text: str) -> str:
    # GitHub sanitizes HTML on render; escaping angle brackets (rather than
    # regex-stripping tags) can't be defeated by nested or malformed markup and
    # never silently drops surrounding text.
    return escape(text, quote=False)


def _host_allowed(host: str) -> bool:
    host = host.lower()
    return any(host == allowed or host.endswith('.' + allowed) for allowed in ALLOWED_LINK_HOSTS)


def _rewrite_urls(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        url = match.group(0).rstrip('.,;:')
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not _host_allowed(parsed.hostname or ''):
            return ''
        return url

    return URL_RE.sub(replace, text)


def sanitize_text(text: str, *, max_chars: int = MAX_SUBSTITUTION_CHARS) -> str:
    cleaned = _rewrite_urls(_redact_secrets(_neutralize_html(text)))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + '…'
    return cleaned


def normalize_labels(
    category: str, areas: list[str], requested: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    accepted: set[str] = set()
    dropped: list[str] = []

    if category in TYPE_LABELS:
        accepted.add(category)

    for area in areas:
        mapped = LABEL_ALIASES.get(area, area)
        if mapped in ALLOWLIST:
            accepted.add(mapped)
        else:
            dropped.append(area)

    for raw in requested:
        mapped = LABEL_ALIASES.get(raw, raw)
        if mapped in BANNED_LABELS or mapped not in ALLOWLIST:
            dropped.append(raw)
            continue
        accepted.add(mapped)

    ordered = tuple(label for label in LABEL_ORDER if label in accepted)[:MAX_LABELS]
    leftover = [label for label in sorted(accepted) if label not in ordered]
    dropped.extend(leftover)
    return ordered, tuple(dropped)


def _missing_field_lines(missing_fields: list[str]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in missing_fields:
        key = sanitize_text(raw, max_chars=40).lower().replace(' ', '-')
        copy = MISSING_FIELD_COPY.get(key)
        if copy is None or key in seen:
            continue
        seen.add(key)
        lines.append(f'- {copy}')
    if not lines:
        return '- a bit more detail so we can help'
    return '\n'.join(lines)


def _duplicate_links(numbers: list[int]) -> str:
    links = [
        f'https://github.com/getzep/graphiti/issues/{int(number)}'
        for number in numbers
        if number >= 1
    ]
    if not links:
        return 'an existing issue'
    if len(links) == 1:
        return links[0]
    return ', '.join(links[:-1]) + f', and {links[-1]}'


def render_comment(decision: dict[str, Any]) -> str | None:
    comment_id = decision['comment_id']
    if comment_id is None:
        return None
    template_path = TEMPLATE_DIR / f'{comment_id}.md'
    if not template_path.is_file():
        return None

    substitutions = {
        'category': sanitize_text(str(decision['category']), max_chars=32),
        'areas': sanitize_text(', '.join(decision['areas']) or 'unspecified', max_chars=80),
        'missing_fields': _missing_field_lines(decision['missing_fields']),
        'duplicate_issues': _duplicate_links(decision['duplicate_issue_numbers']),
    }
    body = template_path.read_text(encoding='utf-8')
    for key, value in substitutions.items():
        body = body.replace('{{' + key + '}}', value)
    if '{{' in body:
        return None

    if STICKY_MARKER not in body:
        body = STICKY_MARKER + '\n' + body
    body = _redact_secrets(body).strip() + '\n'
    if len(body) > MAX_COMMENT_CHARS:
        body = body[: MAX_COMMENT_CHARS - 1].rstrip() + '…\n'
    return body


def apply_decision(decision: dict[str, Any]) -> ApplyResult:
    labels, dropped = normalize_labels(
        decision['category'],
        list(decision['areas']),
        list(decision['labels']),
    )
    comment = render_comment(decision)
    return ApplyResult(
        labels=labels,
        comment=comment,
        comment_id=decision['comment_id'] if comment else None,
        dropped_labels=dropped,
        no_op=False,
    )


def apply_json(raw: str, schema: dict[str, Any] | None = None) -> ApplyResult:
    decision = parse_decision(raw, schema)
    if decision is None:
        return ApplyResult(
            labels=(),
            comment=None,
            comment_id=None,
            dropped_labels=(),
            no_op=True,
        )
    return apply_decision(decision)


def request_github_json(
    method: str,
    path: str,
    token: str,
    payload: object | None = None,
) -> object:
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    request = Request(
        f'{GITHUB_API}{path}',
        data=data,
        method=method,
        headers={
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'graphiti-intake-apply',
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            if response.status == 204:
                return None
            return json.load(response)
    except HTTPError as error:
        body = error.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'GitHub API returned HTTP {error.code}: {body[:1000]}') from error


def _current_labels(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        raise RuntimeError('GitHub issue response was not an object')
    labels = payload.get('labels')
    if not isinstance(labels, list):
        raise RuntimeError('GitHub issue response did not contain labels')
    return [
        label['name']
        for label in labels
        if isinstance(label, dict) and isinstance(label.get('name'), str)
    ]


def _bot_sticky_comments(payload: object) -> list[int]:
    if not isinstance(payload, list):
        raise RuntimeError('GitHub comments response was not a list')
    ids: list[int] = []
    for comment in payload:
        if not isinstance(comment, dict):
            continue
        user = comment.get('user')
        is_bot = isinstance(user, dict) and user.get('type') == 'Bot'
        body = comment.get('body')
        comment_id = comment.get('id')
        if (
            is_bot
            and isinstance(body, str)
            and STICKY_MARKER in body
            and isinstance(comment_id, int)
        ):
            ids.append(comment_id)
    return ids


def apply_to_github(
    result: ApplyResult,
    *,
    repo: str,
    number: int,
    github_token: str,
    request_json: GitHubRequest = request_github_json,
) -> GitHubApplySummary:
    if result.no_op:
        return GitHubApplySummary(labels_changed=False, comment_action='none', no_op=True)
    if number < 1:
        raise ValueError('number must be positive')

    issue_path = f'/repos/{repo}/issues/{number}'
    current = _current_labels(request_json('GET', issue_path, github_token, None))
    preserved = [label for label in current if label not in MANAGED_LABELS]
    desired = list(dict.fromkeys([*result.labels, *preserved]))
    labels_changed = set(desired) != set(current)
    if labels_changed:
        request_json('PATCH', issue_path, github_token, {'labels': desired})

    comments_path = f'{issue_path}/comments?per_page=100'
    sticky_ids = _bot_sticky_comments(request_json('GET', comments_path, github_token, None))
    comment_action = 'none'
    if result.comment is not None:
        if sticky_ids:
            request_json(
                'PATCH',
                f'/repos/{repo}/issues/comments/{sticky_ids[0]}',
                github_token,
                {'body': result.comment},
            )
            comment_action = 'updated'
            stale_ids = sticky_ids[1:]
        else:
            request_json(
                'POST',
                f'{issue_path}/comments',
                github_token,
                {'body': result.comment},
            )
            comment_action = 'created'
            stale_ids = []
    else:
        stale_ids = sticky_ids
        if stale_ids:
            comment_action = 'deleted'

    for comment_id in stale_ids:
        request_json(
            'DELETE',
            f'/repos/{repo}/issues/comments/{comment_id}',
            github_token,
            None,
        )

    return GitHubApplySummary(
        labels_changed=labels_changed,
        comment_action=comment_action,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Turn an intake decision JSON file into labels and a templated comment.'
    )
    parser.add_argument('decision', type=Path, help='Path to decision JSON from classify.py')
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        help='Write the apply result JSON here instead of stdout',
    )
    parser.add_argument('--repo', help='GitHub repository in owner/name form')
    parser.add_argument('--number', type=int, help='Issue or pull request number')
    parser.add_argument(
        '--write',
        action='store_true',
        help='Apply labels and the sticky comment to GitHub using GITHUB_TOKEN',
    )
    args = parser.parse_args(argv)

    raw = args.decision.read_text(encoding='utf-8')
    result = apply_json(raw)
    output: dict[str, Any] = result.to_dict()
    if args.write and not result.no_op:
        github_token = os.environ.get('GITHUB_TOKEN')
        if not github_token or not args.repo or not args.number:
            parser.error('--write requires GITHUB_TOKEN, --repo, and --number')
        output['github'] = apply_to_github(
            result,
            repo=args.repo,
            number=args.number,
            github_token=github_token,
        ).to_dict()

    payload = json.dumps(output, indent=2) + '\n'
    if args.output:
        args.output.write_text(payload, encoding='utf-8')
    else:
        sys.stdout.write(payload)
    # A no-op is the safe, expected outcome when a decision is invalid or empty
    # (e.g. the maintainer-skip sentinel): the airlock correctly applied nothing.
    # Exit 0 so it does not show as a failed job. Genuine faults raise instead.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

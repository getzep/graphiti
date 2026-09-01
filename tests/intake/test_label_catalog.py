from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / '.github' / 'scripts' / 'setup-triage-labels.sh'
SCHEMA_PATH = REPO_ROOT / '.github' / 'intake' / 'decision.schema.json'
APPLY_PATH = REPO_ROOT / '.github' / 'intake' / 'apply.py'
ALLOWED_LIVE_REPO = 'mehulp93/graphiti-triage'

CREATE_LABEL_RE = re.compile(r'^\s*create_label\s+"([^"]+)"', re.MULTILINE)

LOCKED_SCRIPT_LABELS = frozenset(
    {
        'feature',
        'scope:core',
        'scope:mcp',
        'scope:service',
        'scope:docs',
        'scope:ci',
        'needs-info',
        'needs-issue',
        'needs-rfc',
        'rfc-approved',
        'needs-tests',
        'needs-rework',
        'security',
        'duplicate',
        'invalid',
        'stale',
    }
)
SCOPE_LABELS = frozenset(
    {
        'scope:core',
        'scope:mcp',
        'scope:service',
        'scope:docs',
        'scope:ci',
    }
)
BOT_NEEDS_LABELS = frozenset(
    {
        'needs-info',
        'needs-issue',
        'needs-rfc',
        'needs-tests',
        'needs-rework',
    }
)


def load_apply():
    spec = importlib.util.spec_from_file_location('graphiti_intake_apply_catalog', APPLY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


apply = load_apply()


def script_labels() -> set[str]:
    return set(CREATE_LABEL_RE.findall(SCRIPT_PATH.read_text(encoding='utf-8')))


def test_setup_script_creates_the_locked_taxonomy():
    assert script_labels() == LOCKED_SCRIPT_LABELS


def test_schema_areas_are_the_five_scope_labels():
    schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
    assert set(schema['properties']['areas']['items']['enum']) == SCOPE_LABELS


def test_apply_allowlist_covers_bot_writable_taxonomy_and_excludes_rfc_approved():
    assert SCOPE_LABELS <= apply.ALLOWLIST
    assert BOT_NEEDS_LABELS <= apply.ALLOWLIST
    assert 'rfc-approved' not in apply.ALLOWLIST
    assert 'rfc-approved' in apply.BANNED_LABELS
    assert 'stale' not in apply.ALLOWLIST


def test_live_fork_label_list_only_targets_graphiti_triage():
    repo = os.environ.get('INTAKE_LABEL_REPO')
    if not repo:
        pytest.skip('INTAKE_LABEL_REPO is unset; skipping live GitHub label check')
    if repo != ALLOWED_LIVE_REPO:
        pytest.fail(
            f'INTAKE_LABEL_REPO={repo!r} is not allowed; '
            f'live checks must use {ALLOWED_LIVE_REPO} and never getzep/graphiti'
        )

    completed = subprocess.run(
        [
            'gh',
            'label',
            'list',
            '--repo',
            ALLOWED_LIVE_REPO,
            '--limit',
            '200',
            '--json',
            'name',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f'gh label list failed: {completed.stderr.strip() or completed.stdout.strip()}')

    names = {item['name'] for item in json.loads(completed.stdout)}
    missing = sorted(LOCKED_SCRIPT_LABELS - names)
    assert missing == [], f'fork is missing labels: {missing}'

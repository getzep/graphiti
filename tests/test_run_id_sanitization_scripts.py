from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_om_closeout_report import _RUN_ID_MAX_LEN as CLOSEOUT_RUN_ID_MAX_LEN
from scripts.build_om_closeout_report import (
    _validate_run_id as validate_closeout_run_id,
)
from scripts.build_om_closeout_report import (
    build_report,
)
from scripts.owned_paths_preflight import _RUN_ID_MAX_LEN as PREFLIGHT_RUN_ID_MAX_LEN
from scripts.owned_paths_preflight import OWNED_PATHS
from scripts.owned_paths_preflight import _validate_run_id as validate_preflight_run_id

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_SCRIPT = REPO_ROOT / 'scripts' / 'owned_paths_preflight.py'
CLOSEOUT_SCRIPT = REPO_ROOT / 'scripts' / 'build_om_closeout_report.py'
OWNED_PATHS_SNAPSHOT_SHA256 = '99d60c9093e8060a636110203021c7b47b8565b3d9443baab88358cd10c20a38'


@pytest.mark.parametrize('run_id', ['alpha-01', 'alpha_01', 'A1B2'])
def test_validate_run_id_accepts_allowed_characters(run_id: str) -> None:
    assert validate_preflight_run_id(run_id) == run_id
    assert validate_closeout_run_id(run_id) == run_id


def test_validate_run_id_max_length_is_consistent() -> None:
    assert PREFLIGHT_RUN_ID_MAX_LEN == CLOSEOUT_RUN_ID_MAX_LEN == 128


@pytest.mark.parametrize('run_id', ['a' * PREFLIGHT_RUN_ID_MAX_LEN, 'A' * CLOSEOUT_RUN_ID_MAX_LEN])
def test_validate_run_id_accepts_max_length_boundary(run_id: str) -> None:
    assert validate_preflight_run_id(run_id) == run_id
    assert validate_closeout_run_id(run_id) == run_id


def test_validate_run_id_rejects_length_over_max_boundary() -> None:
    run_id = 'a' * (PREFLIGHT_RUN_ID_MAX_LEN + 1)
    expected_message = f'run-id exceeds max length ({PREFLIGHT_RUN_ID_MAX_LEN})'

    with pytest.raises(ValueError) as exc_info:
        validate_preflight_run_id(run_id)
    assert expected_message in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        validate_closeout_run_id(run_id)
    assert expected_message in str(exc_info.value)


@pytest.mark.parametrize(
    'run_id, expected_message',
    [
        ('alpha/beta', "run-id contains '/'"),
        ('alpha\\beta', "run-id contains '\\'"),
        ('alpha..beta', "run-id contains '..'"),
    ],
)
def test_validate_run_id_rejects_path_traversal(
    run_id: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_preflight_run_id(run_id)
    assert expected_message in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        validate_closeout_run_id(run_id)
    assert expected_message in str(exc_info.value)


def test_validate_run_id_rejects_other_characters() -> None:
    with pytest.raises(ValueError, match=r'run-id must match only \[A-Za-z0-9_\-\]'):
        validate_preflight_run_id('alpha.b')

    with pytest.raises(ValueError, match=r'run-id must match only \[A-Za-z0-9_\-\]'):
        validate_closeout_run_id('alpha.b')


def test_build_report_rejects_invalid_run_id(tmp_path: Path) -> None:
    benchmark = tmp_path / 'benchmark.json'
    utility = tmp_path / 'utility.json'
    lane_hygiene = tmp_path / 'lane_hygiene.json'

    benchmark.write_text(json.dumps({'bicameral_aggregate': {}}), encoding='utf-8')
    utility.write_text(json.dumps({'aggregate': {}}), encoding='utf-8')
    lane_hygiene.write_text(json.dumps({'unresolved_lanes': []}), encoding='utf-8')

    with pytest.raises(ValueError, match="run-id contains '/'"):
        build_report(
            run_id='abc/def',
            benchmark_path=benchmark,
            utility_path=utility,
            lane_hygiene_path=lane_hygiene,
            pr_a_sha='abc',
            overlay_manifest_ref='overlay-manifest.json',
        )


def test_build_report_accepts_valid_run_id(tmp_path: Path) -> None:
    benchmark = tmp_path / 'benchmark.json'
    utility = tmp_path / 'utility.json'
    lane_hygiene = tmp_path / 'lane_hygiene.json'

    benchmark.write_text(json.dumps({'bicameral_aggregate': {}}), encoding='utf-8')
    utility.write_text(json.dumps({'aggregate': {}}), encoding='utf-8')
    lane_hygiene.write_text(json.dumps({'unresolved_lanes': []}), encoding='utf-8')

    run_id = 'run_01'
    report = build_report(
        run_id=run_id,
        benchmark_path=benchmark,
        utility_path=utility,
        lane_hygiene_path=lane_hygiene,
        pr_a_sha='abc',
        overlay_manifest_ref='overlay-manifest.json',
    )

    assert f'OM Closeout Report ({run_id})' in report


def test_owned_paths_scope_matches_current_pr140_branch_files() -> None:
    required_paths = {
        str(PREFLIGHT_SCRIPT.relative_to(REPO_ROOT)),
        str(CLOSEOUT_SCRIPT.relative_to(REPO_ROOT)),
        str(Path(__file__).resolve().relative_to(REPO_ROOT)),
    }

    assert required_paths.issubset(OWNED_PATHS)
    assert all(not Path(path).is_absolute() for path in OWNED_PATHS)
    assert all('..' not in Path(path).parts for path in OWNED_PATHS)
    assert all(path.startswith(('docs/', 'mcp_server/', 'scripts/', 'tests/')) for path in OWNED_PATHS)


def test_owned_paths_snapshot_fingerprint() -> None:
    fingerprint = hashlib.sha256('\n'.join(sorted(OWNED_PATHS)).encode('utf-8')).hexdigest()
    assert fingerprint == OWNED_PATHS_SNAPSHOT_SHA256


def test_cli_rejects_invalid_run_id_for_owned_paths_preflight(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(PREFLIGHT_SCRIPT), '--run-id', 'bad/run'],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert 'Invalid --run-id:' in result.stderr
    assert "run-id contains '/'" in result.stderr


def test_cli_rejects_invalid_run_id_for_closeout_report(tmp_path: Path) -> None:
    benchmark = tmp_path / 'benchmark.json'
    utility = tmp_path / 'utility.json'
    lane_hygiene = tmp_path / 'lane_hygiene.json'
    out_path = tmp_path / 'closeout.md'

    benchmark.write_text(json.dumps({'bicameral_aggregate': {}}), encoding='utf-8')
    utility.write_text(json.dumps({'aggregate': {}}), encoding='utf-8')
    lane_hygiene.write_text(json.dumps({'unresolved_lanes': []}), encoding='utf-8')

    result = subprocess.run(
        [
            sys.executable,
            str(CLOSEOUT_SCRIPT),
            '--run-id',
            'bad/run',
            '--benchmark',
            str(benchmark),
            '--utility',
            str(utility),
            '--lane-hygiene',
            str(lane_hygiene),
            '--out',
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert 'Invalid --run-id:' in result.stderr
    assert "run-id contains '/'" in result.stderr


def test_closeout_cli_non_run_id_value_error_not_mislabeled(tmp_path: Path) -> None:
    benchmark = tmp_path / 'benchmark.json'
    utility = tmp_path / 'utility.json'
    lane_hygiene = tmp_path / 'lane_hygiene.json'
    out_path = tmp_path / 'closeout.md'

    benchmark.write_text('[]', encoding='utf-8')
    utility.write_text(json.dumps({'aggregate': {}}), encoding='utf-8')
    lane_hygiene.write_text(json.dumps({'unresolved_lanes': []}), encoding='utf-8')

    result = subprocess.run(
        [
            sys.executable,
            str(CLOSEOUT_SCRIPT),
            '--run-id',
            'good_run_id',
            '--benchmark',
            str(benchmark),
            '--utility',
            str(utility),
            '--lane-hygiene',
            str(lane_hygiene),
            '--out',
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert 'artifact must be JSON object' in result.stderr
    assert 'Invalid --run-id' not in result.stderr

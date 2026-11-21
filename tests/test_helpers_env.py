"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from graphiti_core.helpers import _env_float, _env_int

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('value', ['', '   '])
def test_an_empty_setting_keeps_the_default(monkeypatch, value):
    # `.env.example` spells a setting that keeps its default as an empty value.
    monkeypatch.setenv('GRAPHITI_TEST_INT', value)
    monkeypatch.setenv('GRAPHITI_TEST_FLOAT', value)

    assert _env_int('GRAPHITI_TEST_INT', 20) == 20
    assert _env_float('GRAPHITI_TEST_FLOAT', 0.15) == 0.15


def test_a_set_setting_keeps_its_value(monkeypatch):
    monkeypatch.setenv('GRAPHITI_TEST_INT', '7')
    monkeypatch.setenv('GRAPHITI_TEST_FLOAT', '0.5')

    assert _env_int('GRAPHITI_TEST_INT', 20) == 7
    assert _env_float('GRAPHITI_TEST_FLOAT', 0.15) == 0.5


def test_a_setting_that_is_not_a_number_keeps_the_default(monkeypatch, caplog):
    monkeypatch.setenv('GRAPHITI_TEST_INT', 'auto')

    with caplog.at_level(logging.WARNING, logger='graphiti_core.helpers'):
        assert _env_int('GRAPHITI_TEST_INT', 20) == 20

    assert 'GRAPHITI_TEST_INT' in caplog.text


def test_an_empty_numeric_setting_does_not_stop_the_import(tmp_path):
    """The shipped `.env.example` must not make `import graphiti_core` fail."""
    env = dict(os.environ)
    env.update(
        {
            'SEMAPHORE_LIMIT': '',
            'CHUNK_TOKEN_SIZE': '',
            'CHUNK_OVERLAP_TOKENS': '',
            'CHUNK_MIN_TOKENS': '',
            'CHUNK_DENSITY_THRESHOLD': '',
            'EMBEDDING_DIM': '',
            'GRAPHITI_TELEMETRY_ENABLED': 'false',
            'PYTHONPATH': str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', ''),
        }
    )
    code = (
        'import graphiti_core\n'
        'from graphiti_core import helpers\n'
        'from graphiti_core.embedder.client import EMBEDDING_DIM\n'
        'assert helpers.SEMAPHORE_LIMIT == helpers.DEFAULT_SEMAPHORE_LIMIT\n'
        'assert helpers.CHUNK_TOKEN_SIZE == 3000\n'
        'assert helpers.CHUNK_OVERLAP_TOKENS == 200\n'
        'assert helpers.CHUNK_MIN_TOKENS == 1000\n'
        'assert helpers.CHUNK_DENSITY_THRESHOLD == 0.15\n'
        'assert EMBEDDING_DIM == 1024\n'
        "print('ok')\n"
    )

    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'ok'

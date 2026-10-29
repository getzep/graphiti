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

import importlib.util
import sys
from pathlib import Path

import httpx as real_httpx
import pytest

import graphiti_core.llm_client.client as client_module

# Regression tests for https://github.com/getzep/graphiti/issues/1893:
# `import graphiti_core` failed on a fresh install because openai>=3 depends on httpx2
# and no longer pulls in httpx, which client.py imported unconditionally.

CLIENT_PATH = Path(client_module.__file__)


def _load_fresh_client_module():
    """Execute client.py as a throwaway module so its import-time behaviour can be tested.

    This is used instead of importlib.reload() because reloading the real module would rebind
    LLMClient to a new class object and break isinstance checks in other tests.
    """
    spec = importlib.util.spec_from_file_location(
        'graphiti_core.llm_client._client_httpx_fallback_test', CLIENT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_client_imports_when_httpx_is_missing(monkeypatch):
    # Setting a sys.modules entry to None makes `import httpx` raise ImportError.
    monkeypatch.setitem(sys.modules, 'httpx', None)
    # The locked dev environment does not ship httpx2, so stand in with the real httpx module.
    monkeypatch.setitem(sys.modules, 'httpx2', real_httpx)

    module = _load_fresh_client_module()

    assert module.httpx is real_httpx


def test_client_imports_with_real_httpx2(monkeypatch):
    httpx2 = pytest.importorskip('httpx2')
    monkeypatch.setitem(sys.modules, 'httpx', None)

    module = _load_fresh_client_module()

    assert module.httpx is httpx2
    assert issubclass(module.httpx.HTTPStatusError, Exception)

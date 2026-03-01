"""Phase C — Slice 3: Endpoint Split Safety tests.

Validates that:
1. resolve_llm_base_url and resolve_embedder_base_url are importable
2. Priority order is correct for each resolver
3. LLM_BASE_URL is consulted before OPENAI_BASE_URL for LLM routing
4. EMBEDDER_BASE_URL takes precedence over OPENAI_BASE_URL for embeddings
5. SSRF hardening: link-local addresses always blocked
6. LLM resolver blocks loopback/private by default (unless OM_ALLOW_LOCAL_LLM=1)
7. Embedder resolver allows loopback/private (Ollama use-case)
8. EndpointResolutionError is raised on invalid URLs
9. All three scripts use the shared utility (import path check)
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from graphiti_core.utils.env_utils import (
    EndpointResolutionError,
    resolve_embedder_base_url,
    resolve_llm_base_url,
)


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_resolve_llm_base_url_callable(self):
        assert callable(resolve_llm_base_url)

    def test_resolve_embedder_base_url_callable(self):
        assert callable(resolve_embedder_base_url)

    def test_endpoint_resolution_error_is_valueerror(self):
        assert issubclass(EndpointResolutionError, ValueError)

    def test_all_exported(self):
        from graphiti_core.utils.env_utils import __all__
        assert 'resolve_llm_base_url' in __all__
        assert 'resolve_embedder_base_url' in __all__
        assert 'EndpointResolutionError' in __all__


# ---------------------------------------------------------------------------
# 2. LLM base URL priority
# ---------------------------------------------------------------------------

class TestLLMBaseUrlPriority:
    @pytest.fixture(autouse=True)
    def _clean_env(self):
        """Ensure relevant env vars are unset before each test."""
        keys = [
            'OM_COMPRESSOR_LLM_BASE_URL', 'LLM_BASE_URL',
            'OPENAI_BASE_URL', 'OM_ALLOW_LOCAL_LLM',
        ]
        with patch.dict(os.environ, {k: '' for k in keys}, clear=False):
            for k in keys:
                os.environ.pop(k, None)
            yield

    def test_default_is_openai(self):
        url = resolve_llm_base_url()
        assert url == 'https://api.openai.com/v1'

    def test_openai_base_url_used_as_fallback(self):
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://openrouter.ai/api/v1'}):
            url = resolve_llm_base_url()
        assert url == 'https://openrouter.ai/api/v1'

    def test_llm_base_url_takes_priority_over_openai(self):
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'https://llm.example.com/v1',
            'OPENAI_BASE_URL': 'https://openrouter.ai/api/v1',
        }):
            url = resolve_llm_base_url()
        assert url == 'https://llm.example.com/v1'

    def test_script_override_takes_highest_priority(self):
        with patch.dict(os.environ, {
            'OM_COMPRESSOR_LLM_BASE_URL': 'https://override.example.com/v1',
            'LLM_BASE_URL': 'https://llm.example.com/v1',
            'OPENAI_BASE_URL': 'https://openrouter.ai/api/v1',
        }):
            url = resolve_llm_base_url(script_override_env='OM_COMPRESSOR_LLM_BASE_URL')
        assert url == 'https://override.example.com/v1'

    def test_trailing_slash_stripped(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'https://llm.example.com/v1/'}):
            url = resolve_llm_base_url()
        assert not url.endswith('/')


# ---------------------------------------------------------------------------
# 3. Embedder base URL priority
# ---------------------------------------------------------------------------

class TestEmbedderBaseUrlPriority:
    @pytest.fixture(autouse=True)
    def _clean_env(self):
        keys = ['EMBEDDER_BASE_URL', 'OPENAI_BASE_URL']
        with patch.dict(os.environ, {k: '' for k in keys}, clear=False):
            for k in keys:
                os.environ.pop(k, None)
            yield

    def test_default_is_local_ollama(self):
        url = resolve_embedder_base_url()
        assert url == 'http://localhost:11434/v1'

    def test_embedder_base_url_takes_priority(self):
        with patch.dict(os.environ, {
            'EMBEDDER_BASE_URL': 'http://embedder.example.com/v1',
            'OPENAI_BASE_URL': 'https://openrouter.ai/api/v1',
        }):
            url = resolve_embedder_base_url()
        assert url == 'http://embedder.example.com/v1'

    def test_openai_base_url_fallback_for_embedder(self):
        """OPENAI_BASE_URL is only used if EMBEDDER_BASE_URL is not set."""
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://openrouter.ai/api/v1'}):
            url = resolve_embedder_base_url()
        assert url == 'https://openrouter.ai/api/v1'

    def test_custom_default(self):
        url = resolve_embedder_base_url(default='http://myollama:11434/v1')
        assert url == 'http://myollama:11434/v1'

    def test_trailing_slash_stripped(self):
        with patch.dict(os.environ, {'EMBEDDER_BASE_URL': 'http://localhost:11434/v1/'}):
            url = resolve_embedder_base_url()
        assert not url.endswith('/')


# ---------------------------------------------------------------------------
# 4. SSRF hardening
# ---------------------------------------------------------------------------

class TestSSRFHardening:
    @pytest.fixture(autouse=True)
    def _clean_env(self):
        keys = ['OM_COMPRESSOR_LLM_BASE_URL', 'LLM_BASE_URL', 'OPENAI_BASE_URL',
                'EMBEDDER_BASE_URL', 'OM_ALLOW_LOCAL_LLM']
        with patch.dict(os.environ, {k: '' for k in keys}, clear=False):
            for k in keys:
                os.environ.pop(k, None)
            yield

    def test_llm_link_local_always_blocked(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'http://169.254.169.254/v1'}):
            with pytest.raises(EndpointResolutionError, match='link-local'):
                resolve_llm_base_url()

    def test_llm_ipv6_link_local_blocked(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'http://[fe80::1]/v1'}):
            with pytest.raises(EndpointResolutionError, match='link-local'):
                resolve_llm_base_url()

    def test_llm_localhost_blocked_by_default(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'http://localhost:11434/v1'}):
            with pytest.raises(EndpointResolutionError, match='private/loopback'):
                resolve_llm_base_url()

    def test_llm_localhost_allowed_with_override(self):
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://localhost:11434/v1',
            'OM_ALLOW_LOCAL_LLM': '1',
        }):
            url = resolve_llm_base_url()
        assert 'localhost' in url

    def test_embedder_localhost_allowed(self):
        """Embedder MUST allow localhost (local Ollama is a primary use-case)."""
        with patch.dict(os.environ, {'EMBEDDER_BASE_URL': 'http://localhost:11434/v1'}):
            url = resolve_embedder_base_url()
        assert url == 'http://localhost:11434/v1'

    def test_embedder_link_local_blocked(self):
        """Even embedder must block link-local / cloud-metadata."""
        with patch.dict(os.environ, {'EMBEDDER_BASE_URL': 'http://169.254.169.254/v1'}):
            with pytest.raises(EndpointResolutionError, match='link-local'):
                resolve_embedder_base_url()

    def test_invalid_url_scheme_rejected(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'ftp://example.com/v1'}):
            with pytest.raises(EndpointResolutionError):
                resolve_llm_base_url()

    def test_credentials_in_url_rejected(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'https://user:pass@example.com/v1'}):
            with pytest.raises(EndpointResolutionError, match='credentials'):
                resolve_llm_base_url()

    def test_query_string_rejected(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'https://example.com/v1?key=secret'}):
            with pytest.raises(EndpointResolutionError, match='query string'):
                resolve_llm_base_url()

    def test_private_rfc1918_blocked_for_llm_by_default(self):
        with patch.dict(os.environ, {'LLM_BASE_URL': 'http://192.168.1.10:8080/v1'}):
            with pytest.raises(EndpointResolutionError, match='private/loopback'):
                resolve_llm_base_url()

    def test_private_rfc1918_allowed_for_embedder(self):
        with patch.dict(os.environ, {'EMBEDDER_BASE_URL': 'http://192.168.1.10:11434/v1'}):
            url = resolve_embedder_base_url()
        assert '192.168.1.10' in url


# ---------------------------------------------------------------------------
# 5. Scripts use the shared utility (import-path verification)
# ---------------------------------------------------------------------------

class TestScriptsUseSharedUtility:
    def _read_script(self, name: str) -> str:
        from pathlib import Path
        p = Path(__file__).parents[1] / 'scripts' / name
        return p.read_text(encoding='utf-8')

    def test_om_compressor_imports_env_utils(self):
        src = self._read_script('om_compressor.py')
        assert 'resolve_embedder_base_url' in src or 'env_utils' in src

    def test_om_fast_write_imports_env_utils(self):
        src = self._read_script('om_fast_write.py')
        assert 'env_utils' in src or 'resolve_embedder_base_url' in src

    def test_import_transcripts_imports_env_utils(self):
        src = self._read_script('import_transcripts_to_neo4j.py')
        assert 'env_utils' in src or 'resolve_embedder_base_url' in src

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
10. Fallback paths in om_compressor.py enforce the same SSRF checks as the primary
    env_utils path (no silent weakening when graphiti_core is not importable)
11. Fallback paths in om_fast_write.py reject query/fragment URLs (parity regression)
12. Fallback paths in import_transcripts_to_neo4j.py reject query/fragment URLs (parity)
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


# ---------------------------------------------------------------------------
# 6. Fallback path parity — om_compressor.py SSRF enforcement without
#    graphiti_core on PYTHONPATH
# ---------------------------------------------------------------------------

class TestFallbackPathSSRFParity:
    """Verify that the om_compressor.py fallback validation paths enforce the
    same SSRF rules as the primary env_utils path.

    Strategy: directly exercise _fallback_ssrf_validate — the shared helper
    that both fallback paths delegate to.  No import-blocking needed; this
    makes the test suite deterministic and avoids sys.modules side-effects.
    """

    @pytest.fixture(autouse=True)
    def _import_helpers(self):
        """Import the compressor's standalone SSRF helper and error type."""
        import importlib
        import sys
        from pathlib import Path

        mod_name = '_om_compressor_fallback_test'
        spec = importlib.util.spec_from_file_location(
            mod_name,
            Path(__file__).parents[1] / 'scripts' / 'om_compressor.py',
        )
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so @dataclass can resolve cls.__module__
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(mod_name, None)
        self.validate = mod._fallback_ssrf_validate
        self.Error = mod.OMCompressorError

    @pytest.fixture(autouse=True)
    def _clean_env(self, _import_helpers):
        keys = ['OM_ALLOW_LOCAL_LLM']
        with patch.dict(os.environ, {k: '' for k in keys}, clear=False):
            for k in keys:
                os.environ.pop(k, None)
            yield

    # ---- Shared (both paths) ----

    def test_link_local_always_blocked(self):
        """169.254.x.x must be blocked regardless of allow_private flag."""
        with pytest.raises(self.Error, match='link-local'):
            self.validate('http://169.254.169.254/v1', 'Test', allow_private=True)

    def test_ipv6_link_local_blocked(self):
        """fe80:: addresses must be blocked."""
        with pytest.raises(self.Error, match='link-local'):
            self.validate('http://[fe80::1]/v1', 'Test', allow_private=False)

    def test_credentials_rejected(self):
        with pytest.raises(self.Error, match='credentials'):
            self.validate('https://user:pass@host.com/v1', 'Test', allow_private=True)

    def test_query_string_rejected(self):
        with pytest.raises(self.Error, match='query string'):
            self.validate('https://host.com/v1?key=secret', 'Test', allow_private=True)

    def test_fragment_rejected(self):
        with pytest.raises(self.Error, match='fragment'):
            self.validate('https://host.com/v1#section', 'Test', allow_private=True)

    def test_non_http_scheme_rejected(self):
        with pytest.raises(self.Error):
            self.validate('ftp://host.com/v1', 'Test', allow_private=True)

    def test_trailing_slash_stripped(self):
        url = self.validate('https://api.openai.com/v1/', 'Test', allow_private=False)
        assert not url.endswith('/')

    # ---- Embedder fallback (allow_private=True) ----

    def test_embedder_fallback_allows_localhost(self):
        """Embedder path must permit localhost (local Ollama use-case)."""
        url = self.validate('http://localhost:11434/v1', 'Embedder', allow_private=True)
        assert 'localhost' in url

    def test_embedder_fallback_allows_rfc1918(self):
        """Embedder path must permit RFC-1918 addresses."""
        url = self.validate('http://192.168.1.10:11434/v1', 'Embedder', allow_private=True)
        assert '192.168.1.10' in url

    def test_embedder_fallback_still_blocks_link_local(self):
        """Even allow_private=True must not permit link-local addresses."""
        with pytest.raises(self.Error, match='link-local'):
            self.validate('http://169.254.169.254/v1', 'Embedder', allow_private=True)

    # ---- LLM fallback (allow_private=False) ----

    def test_llm_fallback_blocks_localhost_by_default(self):
        """LLM path must block localhost unless OM_ALLOW_LOCAL_LLM=1."""
        with pytest.raises(self.Error, match='private/loopback'):
            self.validate(
                'http://localhost:11434/v1', 'LLM chat',
                allow_private=False, allow_local_override_env='OM_ALLOW_LOCAL_LLM',
            )

    def test_llm_fallback_allows_localhost_with_env_override(self):
        """LLM path must allow localhost when OM_ALLOW_LOCAL_LLM=1."""
        with patch.dict(os.environ, {'OM_ALLOW_LOCAL_LLM': '1'}):
            url = self.validate(
                'http://localhost:11434/v1', 'LLM chat',
                allow_private=False, allow_local_override_env='OM_ALLOW_LOCAL_LLM',
            )
        assert 'localhost' in url

    def test_llm_fallback_blocks_private_rfc1918(self):
        """LLM path must block RFC-1918 addresses by default."""
        with pytest.raises(self.Error, match='private/loopback'):
            self.validate(
                'http://192.168.1.10:8080/v1', 'LLM chat',
                allow_private=False, allow_local_override_env='OM_ALLOW_LOCAL_LLM',
            )

    def test_llm_fallback_blocks_link_local_always(self):
        """Link-local is blocked even with OM_ALLOW_LOCAL_LLM=1."""
        with patch.dict(os.environ, {'OM_ALLOW_LOCAL_LLM': '1'}):
            with pytest.raises(self.Error, match='link-local'):
                self.validate(
                    'http://169.254.169.254/v1', 'LLM chat',
                    allow_private=False, allow_local_override_env='OM_ALLOW_LOCAL_LLM',
                )

    def test_llm_fallback_accepts_public_url(self):
        """LLM path must accept valid public URLs."""
        url = self.validate(
            'https://api.openai.com/v1', 'LLM chat', allow_private=False
        )
        assert url == 'https://api.openai.com/v1'


# ---------------------------------------------------------------------------
# 7. Fallback parity — om_fast_write.py query/fragment rejection
#    Exercises the ImportError fallback in _validated_embedding_base_url()
#    by blocking graphiti_core in sys.modules for the duration of each call.
# ---------------------------------------------------------------------------

class TestOmFastWriteFallbackURLParity:
    """om_fast_write.py fallback validator must reject query/fragment (parity with primary path).

    Strategy: temporarily shadow graphiti_core.utils.env_utils in sys.modules to
    force the ImportError path, then assert the expected RuntimeError is raised.
    Private/loopback must still be ALLOWED (local Ollama use-case).
    """

    @pytest.fixture(autouse=True)
    def _load_module(self):
        import importlib.util
        import sys as _sys
        from pathlib import Path
        mod_name = '_om_fast_write_fallback_test'
        spec = importlib.util.spec_from_file_location(
            mod_name,
            Path(__file__).parents[1] / 'scripts' / 'om_fast_write.py',
        )
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            _sys.modules.pop(mod_name, None)
        self.mod = mod

    def _call_fallback(self, url: str) -> str:
        """Force the ImportError path and call _validated_embedding_base_url."""
        import sys as _sys
        blocked = {
            'graphiti_core': None,
            'graphiti_core.utils': None,
            'graphiti_core.utils.env_utils': None,
        }
        with patch.dict(_sys.modules, blocked):
            with patch.dict(os.environ, {'EMBEDDER_BASE_URL': url}, clear=False):
                return self.mod._validated_embedding_base_url()

    def test_query_string_rejected(self):
        with pytest.raises(RuntimeError, match='query'):
            self._call_fallback('https://embedder.example.com/v1?key=secret')

    def test_fragment_rejected(self):
        with pytest.raises(RuntimeError, match='fragment'):
            self._call_fallback('https://embedder.example.com/v1#anchor')

    def test_query_and_fragment_both_rejected(self):
        # query is checked first; fragment alone must also fail separately
        with pytest.raises(RuntimeError, match='query|fragment'):
            self._call_fallback('https://embedder.example.com/v1?k=v#frag')

    def test_link_local_blocked(self):
        with pytest.raises(RuntimeError, match='link-local'):
            self._call_fallback('http://169.254.169.254/v1')

    def test_ipv6_link_local_blocked(self):
        with pytest.raises(RuntimeError, match='link-local'):
            self._call_fallback('http://[fe80::1]/v1')

    def test_localhost_allowed(self):
        """Embedder fallback must permit localhost (local Ollama)."""
        url = self._call_fallback('http://localhost:11434/v1')
        assert 'localhost' in url

    def test_private_rfc1918_allowed(self):
        """Embedder fallback must permit RFC-1918 (LAN Ollama)."""
        url = self._call_fallback('http://192.168.1.10:11434/v1')
        assert '192.168.1.10' in url

    def test_credentials_rejected(self):
        with pytest.raises(RuntimeError, match='credentials'):
            self._call_fallback('https://user:pass@embedder.example.com/v1')

    def test_valid_public_url_accepted(self):
        url = self._call_fallback('https://embedder.example.com/v1')
        assert url == 'https://embedder.example.com/v1'

    def test_trailing_slash_stripped(self):
        url = self._call_fallback('https://embedder.example.com/v1/')
        assert not url.endswith('/')


# ---------------------------------------------------------------------------
# 8. Fallback parity — import_transcripts_to_neo4j.py query/fragment rejection
# ---------------------------------------------------------------------------

class TestImportTranscriptsFallbackURLParity:
    """import_transcripts_to_neo4j.py fallback validator must reject query/fragment."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        import importlib.util
        import sys as _sys
        from pathlib import Path
        mod_name = '_import_transcripts_fallback_test'
        spec = importlib.util.spec_from_file_location(
            mod_name,
            Path(__file__).parents[1] / 'scripts' / 'import_transcripts_to_neo4j.py',
        )
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            _sys.modules.pop(mod_name, None)
        self.mod = mod

    def _call_fallback(self, url: str) -> str:
        import sys as _sys
        blocked = {
            'graphiti_core': None,
            'graphiti_core.utils': None,
            'graphiti_core.utils.env_utils': None,
        }
        with patch.dict(_sys.modules, blocked):
            with patch.dict(os.environ, {'EMBEDDER_BASE_URL': url}, clear=False):
                return self.mod._validated_embedding_base_url()

    def test_query_string_rejected(self):
        with pytest.raises(RuntimeError, match='query'):
            self._call_fallback('https://embedder.example.com/v1?token=abc')

    def test_fragment_rejected(self):
        with pytest.raises(RuntimeError, match='fragment'):
            self._call_fallback('https://embedder.example.com/v1#section')

    def test_link_local_blocked(self):
        with pytest.raises(RuntimeError, match='link-local'):
            self._call_fallback('http://169.254.169.254/v1')

    def test_ipv6_link_local_blocked(self):
        with pytest.raises(RuntimeError, match='link-local'):
            self._call_fallback('http://[fe80::1]/v1')

    def test_localhost_allowed(self):
        url = self._call_fallback('http://localhost:11434/v1')
        assert 'localhost' in url

    def test_private_rfc1918_allowed(self):
        url = self._call_fallback('http://10.0.0.5:11434/v1')
        assert '10.0.0.5' in url

    def test_credentials_rejected(self):
        with pytest.raises(RuntimeError, match='credentials'):
            self._call_fallback('https://u:p@embedder.example.com/v1')

    def test_valid_public_url_accepted(self):
        url = self._call_fallback('https://embedder.example.com/v1')
        assert url == 'https://embedder.example.com/v1'

    def test_trailing_slash_stripped(self):
        url = self._call_fallback('https://embedder.example.com/v1/')
        assert not url.endswith('/')

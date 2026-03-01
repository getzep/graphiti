"""Environment variable resolution utilities for Bicameral ingestion paths.

Centralises the EMBEDDER_BASE_URL / LLM_BASE_URL endpoint-split logic that
was previously duplicated across ``scripts/om_compressor.py``,
``scripts/om_fast_write.py``, and ``scripts/import_transcripts_to_neo4j.py``.

Design contract
---------------
**LLM (chat-completions) endpoint**

Priority order:
1. ``OM_COMPRESSOR_LLM_BASE_URL`` — explicit per-script override
2. ``LLM_BASE_URL`` — shared LLM base URL (replaces the overloaded OPENAI_BASE_URL)
3. ``OPENAI_BASE_URL`` — legacy shared base URL (kept for backward compat)
4. ``https://api.openai.com/v1`` — default

**Embedding endpoint**

Priority order:
1. ``EMBEDDER_BASE_URL`` — explicit embedder override (e.g. local Ollama)
2. ``OPENAI_BASE_URL`` — legacy shared base URL (backward compat)
3. ``http://localhost:11434/v1`` — default (local Ollama)

The key invariant: if a user sets ``OPENAI_BASE_URL=https://openrouter.ai/api/v1``
to route LLM calls through OpenRouter, embedding calls must NOT also go to
OpenRouter unless ``EMBEDDER_BASE_URL`` is unset *and* the user explicitly
wants that.  Prefer ``EMBEDDER_BASE_URL`` for embedding-specific overrides.

SSRF hardening
--------------
Both resolvers validate the URL structure and block cloud-metadata/link-local
addresses (169.254.x.x, fe80::).  LLM endpoints additionally block loopback
and RFC-1918 ranges by default (set ``OM_ALLOW_LOCAL_LLM=1`` for dev use with
local models).  Embedding endpoints allow loopback/RFC-1918 since local Ollama
is a primary production use-case.

Usage
-----
>>> from graphiti_core.utils.env_utils import resolve_llm_base_url, resolve_embedder_base_url
>>> llm_url = resolve_llm_base_url()
>>> emb_url = resolve_embedder_base_url()
"""

from __future__ import annotations

import ipaddress
import os
import urllib.parse
from typing import Optional

__all__ = [
    'resolve_llm_base_url',
    'resolve_embedder_base_url',
    'EndpointResolutionError',
]

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class EndpointResolutionError(ValueError):
    """Raised when an endpoint URL cannot be resolved or fails validation."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRUTHY_ENV: frozenset[str] = frozenset({'1', 'true', 'yes', 'on'})


def _is_link_local(host: str) -> bool:
    """Return True if *host* is a link-local (cloud-metadata) address."""
    try:
        return ipaddress.ip_address(host).is_link_local
    except ValueError:
        return False  # hostname — not numeric, skip check


def _is_private_or_loopback(host: str) -> bool:
    """Return True if *host* is loopback or RFC-1918 private."""
    if host.lower() in {'localhost', 'ip6-localhost', 'ip6-loopback'}:
        return True
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_loopback or addr.is_private
    except ValueError:
        return False  # hostname


def _parse_host(netloc: str) -> str:
    """Extract bare hostname from netloc (handles IPv6 brackets)."""
    return (urllib.parse.urlparse(f'//{netloc}').hostname or '').strip()


def _validate_base_url(
    url: str,
    label: str,
    *,
    allow_private: bool = True,
    allow_local_override_env: Optional[str] = None,
) -> str:
    """Validate an HTTP(S) base URL and return it stripped of trailing slash.

    Parameters
    ----------
    url:
        The URL string to validate.
    label:
        Human-readable label used in error messages (e.g. 'LLM chat').
    allow_private:
        If True, loopback and RFC-1918 addresses are accepted.
    allow_local_override_env:
        If provided, check this env var name; if truthy, allow private addrs
        even when allow_private=False (used for OM_ALLOW_LOCAL_LLM).

    Returns
    -------
    str
        The validated URL with no trailing slash.

    Raises
    ------
    EndpointResolutionError
        On any validation failure.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise EndpointResolutionError(
            f'{label} base URL must be an absolute http(s) URL with a host, got: {url!r}'
        )
    if parsed.username or parsed.password:
        raise EndpointResolutionError(
            f'{label} base URL must not include embedded credentials: {url!r}'
        )
    if parsed.query:
        raise EndpointResolutionError(
            f'{label} base URL must not include a query string: {url!r}'
        )
    if parsed.fragment:
        raise EndpointResolutionError(
            f'{label} base URL must not include a fragment: {url!r}'
        )

    host = _parse_host(parsed.netloc)

    # Cloud-metadata / link-local: ALWAYS blocked
    if _is_link_local(host):
        raise EndpointResolutionError(
            f'{label} base URL {url!r} targets a link-local/cloud-metadata address. '
            'This is always blocked.'
        )

    if not allow_private:
        # Check env override first
        override_key = allow_local_override_env or ''
        allowed_by_env = (
            os.environ.get(override_key, '').strip().lower() in _TRUTHY_ENV
        ) if override_key else False

        if not allowed_by_env and _is_private_or_loopback(host):
            raise EndpointResolutionError(
                f'{label} base URL {url!r} targets a private/loopback address. '
                f'Set {override_key}=1 to allow local model endpoints (dev only).'
                if override_key else
                f'{label} base URL {url!r} targets a private/loopback address.'
            )

    return url.rstrip('/')


# ---------------------------------------------------------------------------
# Public resolvers
# ---------------------------------------------------------------------------


def resolve_llm_base_url(
    *,
    script_override_env: str = 'OM_COMPRESSOR_LLM_BASE_URL',
) -> str:
    """Resolve the LLM chat-completions base URL from environment variables.

    Priority:
    1. ``script_override_env`` (default: ``OM_COMPRESSOR_LLM_BASE_URL``)
    2. ``LLM_BASE_URL``
    3. ``OPENAI_BASE_URL`` (legacy / shared)
    4. ``https://api.openai.com/v1``

    Returns
    -------
    str
        Validated URL, no trailing slash.

    Raises
    ------
    EndpointResolutionError
        If the resolved URL fails validation.
    """
    base = (
        os.environ.get(script_override_env, '').strip()
        or os.environ.get('LLM_BASE_URL', '').strip()
        or os.environ.get('OPENAI_BASE_URL', '').strip()
        or 'https://api.openai.com/v1'
    )
    return _validate_base_url(
        base,
        label='LLM chat',
        allow_private=False,
        allow_local_override_env='OM_ALLOW_LOCAL_LLM',
    )


def resolve_embedder_base_url(
    *,
    default: str = 'http://localhost:11434/v1',
) -> str:
    """Resolve the embedding API base URL from environment variables.

    Priority:
    1. ``EMBEDDER_BASE_URL`` — explicit embedder override
    2. ``OPENAI_BASE_URL`` — legacy / shared (used only if EMBEDDER_BASE_URL unset)
    3. *default* (``http://localhost:11434/v1`` — local Ollama)

    Note: OPENAI_BASE_URL is consulted as a *fallback* only.  If users route
    LLM traffic through a provider like OpenRouter via OPENAI_BASE_URL, they
    must set EMBEDDER_BASE_URL explicitly to point to their local embedder
    rather than accidentally sending embeddings to OpenRouter.

    Returns
    -------
    str
        Validated URL, no trailing slash.

    Raises
    ------
    EndpointResolutionError
        If the resolved URL fails validation.
    """
    base = (
        os.environ.get('EMBEDDER_BASE_URL', '').strip()
        or os.environ.get('OPENAI_BASE_URL', '').strip()
        or default
    )
    # Embedder allows local/private addresses (Ollama is typically localhost)
    return _validate_base_url(base, label='Embedder', allow_private=True)

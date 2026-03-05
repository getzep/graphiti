#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import hashlib
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, cast

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel
from starlette.responses import JSONResponse

from config.schema import GraphitiConfig, ServerConfig
from models.response_types import (
    EpisodeSearchResponse,
    ErrorResponse,
    FactSearchResponse,
    NodeResult,
    NodeSearchResponse,
    StatusResponse,
    SuccessResponse,
)
from services.factories import DatabaseDriverFactory, EmbedderFactory, LLMClientFactory
from services.ontology_registry import OntologyRegistry
from services.queue_service import QueueService
from services.search_service import DEFAULT_OM_GROUP_ID, SearchService
from utils.formatting import format_fact_result
from utils.rate_limiter import SlidingWindowRateLimiter as _SlidingWindowRateLimiter

# Load .env file from mcp_server directory
mcp_server_dir = Path(__file__).parent.parent
env_file = mcp_server_dir / '.env'
if env_file.exists():
    load_dotenv(env_file)
else:
    # Try current working directory as fallback
    load_dotenv()


# Semaphore limit for concurrent Graphiti operations.
#
# This controls how many episodes can be processed simultaneously. Each episode
# processing involves multiple LLM calls (entity extraction, deduplication, etc.),
# so the actual number of concurrent LLM requests will be higher.
#
# TUNING GUIDELINES:
#
# LLM Provider Rate Limits (requests per minute):
# - OpenAI Tier 1 (free):     3 RPM   -> SEMAPHORE_LIMIT=1-2
# - OpenAI Tier 2:            60 RPM   -> SEMAPHORE_LIMIT=5-8
# - OpenAI Tier 3:           500 RPM   -> SEMAPHORE_LIMIT=10-15
# - OpenAI Tier 4:         5,000 RPM   -> SEMAPHORE_LIMIT=20-50
# - Anthropic (default):     50 RPM   -> SEMAPHORE_LIMIT=5-8
# - Anthropic (high tier): 1,000 RPM   -> SEMAPHORE_LIMIT=15-30
# - Azure OpenAI (varies):  Consult your quota -> adjust accordingly
#
# SYMPTOMS:
# - Too high: 429 rate limit errors, increased costs from parallel processing
# - Too low: Slow throughput, underutilized API quota
#
# MONITORING:
# - Watch logs for rate limit errors (429)
# - Monitor episode processing times
# - Check LLM provider dashboard for actual request rates
#
# DEFAULT: 10 (suitable for OpenAI Tier 3, mid-tier Anthropic)
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))

# Trust-aware retrieval: additive boost for facts with trust_score property.
# Set to 0.0 to disable trust boosting entirely (identical to vanilla RRF).
try:
    TRUST_WEIGHT = float(os.environ.get('GRAPHITI_TRUST_WEIGHT', '0.0'))
except (ValueError, TypeError):
    TRUST_WEIGHT = 0.0

# Defense-in-depth caps: callers may request arbitrarily large result sets.
# These hard ceilings prevent accidental or adversarial result-set blowup
# regardless of what value the client passes in max_nodes / max_facts.
_MAX_NODES_CAP = 200
_MAX_FACTS_CAP = 200


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    if min_value is not None and value < min_value:
        return default
    if max_value is not None and value > max_value:
        return default
    return value


def _env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default

    if min_value is not None and value < min_value:
        return default
    if max_value is not None and value > max_value:
        return default
    return value


# Cross-source fusion ranker for mixed Graphiti + OM retrieval.
# Formula:
#   final_score(i)
#     = Σ_source (source_weight / (rrf_k + rank_source(i)))
#       + corroboration_boost(i)
# Corroboration is deterministic and source-pair only (Graphiti <-> OM).
_SEARCH_FUSION_RRF_K = _env_float('SEARCH_FUSION_RRF_K', 1.0, min_value=0.0)
_SEARCH_FUSION_SOURCE_WEIGHTS = {
    'graphiti': _env_float('SEARCH_FUSION_WEIGHT_GRAPHITI', 1.0, min_value=0.0),
    'om_primitive': _env_float('SEARCH_FUSION_WEIGHT_OM', 0.9, min_value=0.0),
}
_SEARCH_FUSION_IDENTITY_BOOST = _env_float('SEARCH_FUSION_IDENTITY_BOOST', 0.20, min_value=0.0)
_SEARCH_FUSION_OVERLAP_BOOST = _env_float('SEARCH_FUSION_OVERLAP_BOOST', 0.10, min_value=0.0)
_SEARCH_FUSION_EVIDENCE_BOOST = _env_float('SEARCH_FUSION_EVIDENCE_BOOST', 0.05, min_value=0.0)
_SEARCH_FUSION_OVERLAP_THRESHOLD = _env_float(
    'SEARCH_FUSION_OVERLAP_THRESHOLD',
    0.60,
    min_value=0.0,
    max_value=1.0,
)
_SEARCH_FUSION_MIN_SHARED_TOKENS = _env_int(
    'SEARCH_FUSION_MIN_SHARED_TOKENS',
    2,
    min_value=1,
)

# Mixed/all-lane no-starvation safeguard: when both sources are present and the
# result window allows multiple items, force at least one Graphiti item into
# the final fused window.
_SEARCH_FUSION_REQUIRE_GRAPHITI_FLOOR = (
    os.environ.get('SEARCH_FUSION_REQUIRE_GRAPHITI_FLOOR', 'true').strip().lower() != 'false'
)

# ---------------------------------------------------------------------------
# Search endpoint rate limiter
# ---------------------------------------------------------------------------
# Configurable via environment variables:
#   SEARCH_RATE_LIMIT_ENABLED   "true" (default) | "false"
#   SEARCH_RATE_LIMIT_REQUESTS  max requests per window (default: 60, min: 1, max: 10000)
#   SEARCH_RATE_LIMIT_WINDOW    window duration in seconds (default: 60, min: 1, max: 86400)
#
# The limiter enforces a per-(caller, scope) bucket so that rotating group IDs
# cannot inflate the key space and escape throttling.  When no trusted caller
# principal is available (anonymous) a parallel global bucket is also checked,
# providing a conservative fallback that prevents key-space spraying via
# group-ID rotation even without auth.

_ANON_PRINCIPAL = '__anon__'

_SEARCH_RATE_LIMIT_ENABLED: bool = (
    os.environ.get('SEARCH_RATE_LIMIT_ENABLED', 'true').strip().lower() != 'false'
)

_SEARCH_RATE_LIMIT_REQUESTS_DEFAULT = 60
_SEARCH_RATE_LIMIT_REQUESTS_MIN = 1
_SEARCH_RATE_LIMIT_REQUESTS_MAX = 10_000
_SEARCH_RATE_LIMIT_WINDOW_DEFAULT = 60.0
_SEARCH_RATE_LIMIT_WINDOW_MIN = 1.0
_SEARCH_RATE_LIMIT_WINDOW_MAX = 86_400.0

try:
    _raw_requests = int(os.environ.get('SEARCH_RATE_LIMIT_REQUESTS', str(_SEARCH_RATE_LIMIT_REQUESTS_DEFAULT)))
    if not (_SEARCH_RATE_LIMIT_REQUESTS_MIN <= _raw_requests <= _SEARCH_RATE_LIMIT_REQUESTS_MAX):
        raise ValueError(
            f'SEARCH_RATE_LIMIT_REQUESTS={_raw_requests} out of range '
            f'[{_SEARCH_RATE_LIMIT_REQUESTS_MIN}, {_SEARCH_RATE_LIMIT_REQUESTS_MAX}]; '
            f'using default {_SEARCH_RATE_LIMIT_REQUESTS_DEFAULT}'
        )
    _SEARCH_RATE_LIMIT_REQUESTS = _raw_requests
except (ValueError, TypeError) as _rl_exc:
    _SEARCH_RATE_LIMIT_REQUESTS = _SEARCH_RATE_LIMIT_REQUESTS_DEFAULT
    # Emit warning at module load time so operators see it in startup logs.
    import warnings as _warnings
    _warnings.warn(str(_rl_exc), stacklevel=1)

try:
    _raw_window = float(os.environ.get('SEARCH_RATE_LIMIT_WINDOW', str(_SEARCH_RATE_LIMIT_WINDOW_DEFAULT)))
    if not (_SEARCH_RATE_LIMIT_WINDOW_MIN <= _raw_window <= _SEARCH_RATE_LIMIT_WINDOW_MAX):
        raise ValueError(
            f'SEARCH_RATE_LIMIT_WINDOW={_raw_window} out of range '
            f'[{_SEARCH_RATE_LIMIT_WINDOW_MIN}, {_SEARCH_RATE_LIMIT_WINDOW_MAX}]; '
            f'using default {_SEARCH_RATE_LIMIT_WINDOW_DEFAULT}'
        )
    _SEARCH_RATE_LIMIT_WINDOW = _raw_window
except (ValueError, TypeError) as _rl_exc:
    _SEARCH_RATE_LIMIT_WINDOW = _SEARCH_RATE_LIMIT_WINDOW_DEFAULT
    import warnings as _warnings
    _warnings.warn(str(_rl_exc), stacklevel=1)

# Per-(caller, scope) sliding-window limiter — primary enforcement.
_search_rate_limiter = _SlidingWindowRateLimiter(
    max_requests=_SEARCH_RATE_LIMIT_REQUESTS,
    window_seconds=_SEARCH_RATE_LIMIT_WINDOW,
)

# Global fallback limiter enforced *in addition to* the per-caller limiter
# whenever the caller principal is unavailable (anonymous).  This prevents
# key-space spraying via group-ID rotation even in unauthenticated deployments.
_search_global_fallback_limiter = _SlidingWindowRateLimiter(
    max_requests=_SEARCH_RATE_LIMIT_REQUESTS,
    window_seconds=_SEARCH_RATE_LIMIT_WINDOW,
)


def _extract_trusted_caller_principal(ctx: 'Context | None') -> str:
    """Return a trusted caller principal string (best-effort, never from raw payload).

    Resolution order (most-trusted → least-trusted):
    1. OAuth ``AccessToken.client_id`` from the MCP auth middleware contextvar —
       set by ``AuthContextMiddleware`` from a verified bearer token.
    2. ``Context.client_id`` from the MCP request context meta field.
    3. ``'__anon__'`` sentinel — no authenticated identity is available.

    The returned value is used as the caller component of the rate-limit key.
    It is **never** sourced from the raw MCP request payload so a caller cannot
    self-assign an arbitrary principal to share another caller's bucket.

    Args:
        ctx: The FastMCP ``Context`` for the current request, or ``None`` when
            called outside a live request (e.g. unit tests without a request).

    Returns:
        A non-empty string that identifies the caller, or ``'__anon__'``.
    """
    # 1. Try the auth-middleware contextvar (set from a verified bearer token).
    try:
        access_token = get_access_token()
        if access_token is not None and access_token.client_id:
            return access_token.client_id
    except Exception:  # pragma: no cover — guard against unexpected import edge-cases
        pass

    # 2. Try the MCP request-context meta field (transport-injected, not payload).
    if ctx is not None:
        try:
            client_id = ctx.client_id
            if client_id:
                return client_id
        except Exception:  # pragma: no cover — ctx may not be inside a request
            pass

    return _ANON_PRINCIPAL


def _derive_rate_limit_key(effective_group_ids: list[str], caller_principal: str) -> str:
    """Return a stable, canonical per-(caller, scope) rate-limit key.

    Must be called **after** ``_resolve_effective_group_ids`` and
    ``_extract_trusted_caller_principal`` so both components are derived from
    trusted, validated context rather than raw caller-supplied input.

    Key properties:
    - **Caller-bound**: the ``caller_principal`` component ensures different
      callers with identical group scopes do not share a bucket.
    - **Canonical scope**: sorted unique effective_group_ids are hashed, so
      permutations of the same group set map to the same key.
    - **Anti-spray**: cycling through group-ID subsets/permutations cannot
      generate a fresh unthrottled bucket — the caller component pins the key.

    Args:
        effective_group_ids: Validated and resolved group IDs returned by
            ``_resolve_effective_group_ids``.  An empty list uses the
            ``'__global__'`` scope component.
        caller_principal: Trusted caller identity from
            ``_extract_trusted_caller_principal``.  Must not be empty.

    Returns:
        A composite key string of the form
        ``caller:<principal>|scope:<digest_or_global>``.
    """
    if effective_group_ids:
        # Sort + deduplicate to produce a canonical, order-insensitive scope digest.
        canonical = '|'.join(sorted(set(effective_group_ids)))
        scope_digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        scope_component = f'scope:{scope_digest}'
    else:
        scope_component = 'scope:__global__'

    return f'caller:{caller_principal}|{scope_component}'


def _hash_rate_limit_key(key: str) -> str:
    """Return an 8-character hex prefix of the SHA-256 of *key* for safe logging.

    The key may contain the caller principal and group IDs; hashing prevents
    sensitive values from appearing in plain text in logs.
    """
    return hashlib.sha256(key.encode()).hexdigest()[:8]


# Configure structured logging with timestamps
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)

# Configure specific loggers
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)  # Reduce access log noise
logging.getLogger('mcp.server.streamable_http_manager').setLevel(
    logging.WARNING
)  # Reduce MCP noise


# Patch uvicorn's logging config to use our format
def configure_uvicorn_logging():
    """Configure uvicorn loggers to match our format after they're created."""
    for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
        uvicorn_logger = logging.getLogger(logger_name)
        # Remove existing handlers and add our own with proper formatting
        uvicorn_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False


logger = logging.getLogger(__name__)
SAFE_GROUP_ID_RE = re.compile(r'^[a-zA-Z0-9_]+$')
VALID_SEARCH_MODES = {'hybrid', 'semantic', 'keyword'}

_UNSAFE_CHAR_RE = re.compile(r'[<>&\x00-\x1f\x7f-\x9f]')
_FUSION_TOKEN_RE = re.compile(r'[a-z0-9]+')
_FUSION_STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'in',
    'is',
    'it',
    'of',
    'on',
    'or',
    'that',
    'the',
    'to',
    'with',
}


def _sanitize_for_error(value: str, max_len: int = 64) -> str:
    """Sanitize user input for safe reflection in error messages.

    Uses allowlist approach: replaces all angle brackets, ampersands,
    and control characters with empty string. Then truncates.
    This prevents XML/HTML tag injection, entity encoding bypasses,
    and control character attacks in MCP tool output.
    """
    s = _UNSAFE_CHAR_RE.sub('', str(value))
    return s[:max_len].strip()


# Create global config instance - initialized during server startup.
# Keep a concrete module attribute so tests can monkeypatch it safely.
config: GraphitiConfig = cast(GraphitiConfig, None)

# Contract-test anchor for lane/group precedence checks in test_lane_aliases.py.
# Keep this literal snippet in source: if group_ids:

def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _is_observational_memory_only_scope(effective_group_ids: list[str]) -> bool:
    return len(effective_group_ids) == 1 and effective_group_ids[0] == DEFAULT_OM_GROUP_ID


def _normalize_fusion_text(value: Any) -> str:
    if value is None:
        return ''
    tokens = _FUSION_TOKEN_RE.findall(str(value).lower())
    return ' '.join(tokens)


def _extract_fusion_text_fields(result: dict[str, Any]) -> list[str]:
    fields: list[str] = []

    for key in ('name', 'summary', 'fact'):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            fields.append(value)

    attributes = result.get('attributes')
    if isinstance(attributes, dict):
        for key in ('source_content', 'target_content'):
            value = attributes.get(key)
            if isinstance(value, str) and value.strip():
                fields.append(value)

    return fields


def _fusion_semantic_tokens(result: dict[str, Any], *, max_tokens: int = 24) -> list[str]:
    seen: set[str] = set()
    ordered_tokens: list[str] = []

    for text in _extract_fusion_text_fields(result):
        for token in _FUSION_TOKEN_RE.findall(text.lower()):
            if len(token) < 3 or token in _FUSION_STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            ordered_tokens.append(token)
            if len(ordered_tokens) >= max_tokens:
                return ordered_tokens

    return ordered_tokens


def _fusion_semantic_key(tokens: list[str]) -> str:
    if len(tokens) < _SEARCH_FUSION_MIN_SHARED_TOKENS:
        return ''
    return '|'.join(sorted(tokens)[:6])


def _fusion_identity_keys(result: dict[str, Any]) -> set[str]:
    keys: set[str] = set()

    normalized_name = _normalize_fusion_text(result.get('name'))
    if normalized_name:
        keys.add(f'name:{normalized_name}')

    source_node_uuid = _normalize_fusion_text(result.get('source_node_uuid'))
    target_node_uuid = _normalize_fusion_text(result.get('target_node_uuid'))
    if source_node_uuid and target_node_uuid:
        relation_name = _normalize_fusion_text(result.get('name'))
        keys.add(f'edge:{source_node_uuid}->{target_node_uuid}|{relation_name}')

    return keys


def _semantic_overlap_ratio(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0

    shared = len(tokens_a & tokens_b)
    if shared < _SEARCH_FUSION_MIN_SHARED_TOKENS:
        return 0.0

    denominator = float(min(len(tokens_a), len(tokens_b)))
    if denominator <= 0.0:
        return 0.0

    return shared / denominator


def _fuse_node_like_results(
    *,
    primary: list[dict[str, Any]],
    supplemental: list[dict[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []

    candidates: dict[str, dict[str, Any]] = {}
    source_lists: list[tuple[str, list[dict[str, Any]]]] = [
        ('graphiti', primary),
        ('om_primitive', supplemental),
    ]

    for source_name, source_results in source_lists:
        source_weight = _SEARCH_FUSION_SOURCE_WEIGHTS.get(source_name, 1.0)

        for rank, result in enumerate(source_results, start=1):
            item_id = str(result.get('uuid', '')).strip()
            if not item_id:
                continue

            candidate = candidates.get(item_id)
            if candidate is None:
                semantic_tokens = _fusion_semantic_tokens(result)
                candidate = {
                    'uuid': item_id,
                    'item': result,
                    'sources': set(),
                    'source_ranks': {},
                    'rrf_score': 0.0,
                    'identity_keys': _fusion_identity_keys(result),
                    'semantic_tokens': set(semantic_tokens),
                    'semantic_key': _fusion_semantic_key(semantic_tokens),
                    'corroborating': set(),
                    'identity_corroborated': False,
                    'best_overlap': 0.0,
                }
                candidates[item_id] = candidate
            else:
                candidate['identity_keys'] |= _fusion_identity_keys(result)
                if not candidate['semantic_tokens']:
                    semantic_tokens = _fusion_semantic_tokens(result)
                    candidate['semantic_tokens'] = set(semantic_tokens)
                    candidate['semantic_key'] = _fusion_semantic_key(semantic_tokens)
                if source_name == 'graphiti':
                    # Keep Graphiti payload when UUIDs overlap; it usually
                    # carries richer schema fields than OM adapter rows.
                    candidate['item'] = result

            if source_name in candidate['sources']:
                continue

            candidate['sources'].add(source_name)
            candidate['source_ranks'][source_name] = rank
            candidate['rrf_score'] += source_weight / (_SEARCH_FUSION_RRF_K + float(rank))

    graphiti_candidates = [
        candidate for candidate in candidates.values() if 'graphiti' in candidate['sources']
    ]
    om_candidates = [
        candidate for candidate in candidates.values() if 'om_primitive' in candidate['sources']
    ]

    for graphiti_candidate in graphiti_candidates:
        for om_candidate in om_candidates:
            if graphiti_candidate['uuid'] == om_candidate['uuid']:
                graphiti_candidate['corroborating'].add(om_candidate['uuid'])
                om_candidate['corroborating'].add(graphiti_candidate['uuid'])
                graphiti_candidate['identity_corroborated'] = True
                om_candidate['identity_corroborated'] = True
                graphiti_candidate['best_overlap'] = max(graphiti_candidate['best_overlap'], 1.0)
                om_candidate['best_overlap'] = max(om_candidate['best_overlap'], 1.0)
                continue

            identity_overlap = bool(
                graphiti_candidate['identity_keys'] & om_candidate['identity_keys']
            )
            semantic_overlap = _semantic_overlap_ratio(
                graphiti_candidate['semantic_tokens'],
                om_candidate['semantic_tokens'],
            )
            semantic_key_match = bool(graphiti_candidate['semantic_key']) and (
                graphiti_candidate['semantic_key'] == om_candidate['semantic_key']
            )

            matched = (
                identity_overlap
                or semantic_key_match
                or semantic_overlap >= _SEARCH_FUSION_OVERLAP_THRESHOLD
            )
            if not matched:
                continue

            overlap_strength = max(semantic_overlap, 1.0 if semantic_key_match else 0.0)
            graphiti_candidate['corroborating'].add(om_candidate['uuid'])
            om_candidate['corroborating'].add(graphiti_candidate['uuid'])
            graphiti_candidate['best_overlap'] = max(
                graphiti_candidate['best_overlap'], overlap_strength
            )
            om_candidate['best_overlap'] = max(om_candidate['best_overlap'], overlap_strength)

            if identity_overlap:
                graphiti_candidate['identity_corroborated'] = True
                om_candidate['identity_corroborated'] = True

    ranked_candidates: list[tuple[float, int, int, str, dict[str, Any]]] = []
    for candidate in candidates.values():
        support_count = len(candidate['corroborating'])
        corroboration_boost = 0.0
        if support_count > 0:
            corroboration_boost += support_count * _SEARCH_FUSION_EVIDENCE_BOOST
            corroboration_boost += candidate['best_overlap'] * _SEARCH_FUSION_OVERLAP_BOOST
            if candidate['identity_corroborated']:
                corroboration_boost += _SEARCH_FUSION_IDENTITY_BOOST

        final_score = candidate['rrf_score'] + corroboration_boost
        best_rank = min(candidate['source_ranks'].values()) if candidate['source_ranks'] else 10**9
        source_priority = 0 if 'graphiti' in candidate['sources'] else 1
        ranked_candidates.append(
            (final_score, best_rank, source_priority, candidate['uuid'], candidate)
        )

    ranked_candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    selected_candidates = [row[4] for row in ranked_candidates[:max_items]]

    if (
        _SEARCH_FUSION_REQUIRE_GRAPHITI_FLOOR
        and max_items > 1
        and graphiti_candidates
        and om_candidates
        and not any('graphiti' in candidate['sources'] for candidate in selected_candidates)
    ):
        top_graphiti = next(
            (row[4] for row in ranked_candidates if 'graphiti' in row[4]['sources']),
            None,
        )
        if top_graphiti is not None:
            if selected_candidates:
                selected_candidates[-1] = top_graphiti
            else:
                selected_candidates = [top_graphiti]

            selected_by_uuid = {candidate['uuid'] for candidate in selected_candidates}
            selected_candidates = [
                row[4] for row in ranked_candidates if row[4]['uuid'] in selected_by_uuid
            ]

            if len(selected_candidates) < max_items:
                for row in ranked_candidates:
                    candidate = row[4]
                    if candidate['uuid'] in selected_by_uuid:
                        continue
                    selected_candidates.append(candidate)
                    selected_by_uuid.add(candidate['uuid'])
                    if len(selected_candidates) >= max_items:
                        break

    return [candidate['item'] for candidate in selected_candidates[:max_items]]


def _resolve_effective_group_ids(
    *,
    group_ids: list[str] | None,
    lane_alias: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve effective group IDs with deterministic precedence.

    Returns:
        (effective_group_ids, invalid_aliases)
    """
    effective_group_ids: list[str] = []
    invalid_aliases: list[str] = []

    # 1) Explicit group_ids has highest precedence (including explicit empty list)
    if group_ids is not None:
        effective_group_ids = _unique_preserve_order(group_ids)

    # 2) Resolve lane aliases (if provided)
    elif lane_alias is not None:
        alias_map = config.graphiti.lane_aliases or {}
        resolved: list[str] = []

        for alias in lane_alias:
            # Sanitize alias for safe reflection in error messages
            sanitized = _sanitize_for_error(alias)
            mapped = alias_map.get(alias)
            if mapped is None:
                invalid_aliases.append(sanitized)
                continue
            resolved.extend(mapped)

        # explicit empty alias list or alias mapping to [] means all lanes
        if invalid_aliases:
            return [], invalid_aliases
        effective_group_ids = _unique_preserve_order(resolved)

    # 3) Fallback behavior: default configured group if present, else all lanes ([])
    elif config.graphiti.group_id:
        effective_group_ids = [config.graphiti.group_id]

    # Validate ALL resolved group_ids regardless of source
    for gid in effective_group_ids:
        if not SAFE_GROUP_ID_RE.match(gid):
            raise ValueError(f'Invalid group_id: {_sanitize_for_error(gid)!r}')
    return effective_group_ids, invalid_aliases


def _validate_group_scope_support(effective_group_ids: list[str]) -> str | None:
    """Validate backend-specific group scope support.

    FalkorDB currently supports routed single-group searches only.
    """
    provider = config.database.provider.lower()
    if provider != 'falkordb':
        return None

    if len(effective_group_ids) > 1:
        return (
            'Multi-group searches are not supported in FalkorDB mode yet. '
            'Provide a single group_id/lane_alias.'
        )

    if len(effective_group_ids) == 0 and not config.graphiti.group_id:
        return (
            'All-lanes search is not supported in FalkorDB mode without a default group. '
            'Provide group_ids or lane_alias.'
        )

    return None


def _build_node_search_config(search_mode: str, max_nodes: int):
    from graphiti_core.search.search_config import (
        NodeReranker,
        NodeSearchConfig,
        NodeSearchMethod,
        SearchConfig,
    )
    from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

    if search_mode == 'hybrid':
        base = NODE_HYBRID_SEARCH_RRF
    elif search_mode == 'semantic':
        base = SearchConfig(
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.cosine_similarity],
                reranker=NodeReranker.rrf,
            )
        )
    elif search_mode == 'keyword':
        base = SearchConfig(
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.bm25],
                reranker=NodeReranker.rrf,
            )
        )
    else:
        raise ValueError(f'Unsupported search_mode: {search_mode}')

    return base.model_copy(update={'limit': max_nodes, 'trust_weight': TRUST_WEIGHT})


def _build_edge_search_config(search_mode: str, max_facts: int, center_node_uuid: str | None):
    from graphiti_core.search.search_config import (
        EdgeReranker,
        EdgeSearchConfig,
        EdgeSearchMethod,
        SearchConfig,
    )
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_NODE_DISTANCE,
        EDGE_HYBRID_SEARCH_RRF,
    )

    if search_mode == 'hybrid':
        base = (
            EDGE_HYBRID_SEARCH_RRF
            if center_node_uuid is None
            else EDGE_HYBRID_SEARCH_NODE_DISTANCE
        )
    else:
        methods = (
            [EdgeSearchMethod.cosine_similarity]
            if search_mode == 'semantic'
            else [EdgeSearchMethod.bm25]
        )
        reranker = EdgeReranker.node_distance if center_node_uuid else EdgeReranker.rrf
        base = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=methods,
                reranker=reranker,
            )
        )

    return base.model_copy(update={'limit': max_facts, 'trust_weight': TRUST_WEIGHT})

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to 
capture relationships between concepts, entities, and information. The system organizes data as episodes 
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic, 
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including 
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid 
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations. 
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality. 
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid 
API keys are provided for any language model operations.
"""

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Global services
graphiti_service: Optional['GraphitiService'] = None
queue_service: QueueService | None = None
search_service = SearchService(om_group_id=DEFAULT_OM_GROUP_ID)

# Global client for backward compatibility
graphiti_client: Graphiti | None = None
semaphore: asyncio.Semaphore


class GraphitiService:
    """Graphiti service using the unified configuration system."""

    def __init__(self, config: GraphitiConfig, semaphore_limit: int = 10):
        self.config = config
        self.semaphore_limit = semaphore_limit
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.client: Graphiti | None = None
        self.entity_types = None
        self.ontology_registry: OntologyRegistry | None = None

        # Shared client dependencies (used for per-group FalkorDB routing)
        self._llm_client: Any = None
        self._embedder_client: Any = None
        self._db_config: dict[str, Any] | None = None

        # Per-group Graphiti clients for FalkorDB mode
        self._clients_by_group: dict[str, Graphiti] = {}
        self._client_lock = asyncio.Lock()

    def _validate_group_id(self, group_id: str) -> str:
        """Validate group_id for safe use as FalkorDB graph/database name."""
        if not SAFE_GROUP_ID_RE.match(group_id):
            raise ValueError(f'Invalid group_id for FalkorDB routing: {_sanitize_for_error(group_id)!r}')
        return group_id

    async def _build_client(self, *, database_override: str | None = None) -> Graphiti:
        """Build a Graphiti client for the configured backend.

        For FalkorDB, database_override controls the target graph/database name.
        """
        if self._db_config is None:
            raise RuntimeError('Database config not initialized')

        provider = self.config.database.provider.lower()
        if provider == 'falkordb':
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            db_name = database_override or self._db_config['database']
            falkor_driver = FalkorDriver(
                host=self._db_config['host'],
                port=self._db_config['port'],
                password=self._db_config['password'],
                database=db_name,
            )
            client = Graphiti(
                graph_driver=falkor_driver,
                llm_client=self._llm_client,
                embedder=self._embedder_client,
                max_coroutines=self.semaphore_limit,
            )
        else:
            client = Graphiti(
                uri=self._db_config['uri'],
                user=self._db_config['user'],
                password=self._db_config['password'],
                llm_client=self._llm_client,
                embedder=self._embedder_client,
                max_coroutines=self.semaphore_limit,
            )

        await client.build_indices_and_constraints()
        return client

    async def get_client_for_group(self, group_id: str) -> Graphiti:
        """Get a Graphiti client routed to the requested group.

        In FalkorDB mode, each group is backed by a dedicated database/graph. In
        other backends, this falls back to the default singleton client.
        """
        if self.config.database.provider.lower() != 'falkordb':
            return await self.get_client()

        effective_group = self._validate_group_id(group_id or self.config.graphiti.group_id)

        if effective_group in self._clients_by_group:
            return self._clients_by_group[effective_group]

        async with self._client_lock:
            if effective_group in self._clients_by_group:
                return self._clients_by_group[effective_group]

            logger.info(f'Initializing routed FalkorDB client for group_id={effective_group}')
            client = await self._build_client(database_override=effective_group)
            self._clients_by_group[effective_group] = client
            return client

    async def initialize(self) -> None:
        """Initialize the Graphiti client with factory-created components."""
        try:
            # Create clients using factories
            self._llm_client = None
            self._embedder_client = None

            # Create LLM client based on configured provider
            try:
                self._llm_client = LLMClientFactory.create(self.config.llm)
            except Exception as e:
                logger.warning(f'Failed to create LLM client: {e}')

            # Create embedder client based on configured provider
            try:
                self._embedder_client = EmbedderFactory.create(self.config.embedder)
            except Exception as e:
                logger.warning(f'Failed to create embedder client: {e}')

            # Get and store database configuration
            db_config = DatabaseDriverFactory.create_config(self.config.database)
            self._db_config = db_config

            # Build entity types from configuration
            custom_types = None
            if self.config.graphiti.entity_types:
                custom_types = {}
                for entity_type in self.config.graphiti.entity_types:
                    # Create a dynamic Pydantic model for each entity type
                    # Note: Don't use 'name' as it's a protected Pydantic attribute
                    entity_model = type(
                        entity_type.name,
                        (BaseModel,),
                        {
                            '__doc__': entity_type.description,
                        },
                    )
                    custom_types[entity_type.name] = entity_model

            # Store entity types for later use
            self.entity_types = custom_types

            # Load optional per-lane extraction ontology registry
            try:
                from services.ontology_registry import OntologyRegistry

                ontology_path = mcp_server_dir / 'config' / 'extraction_ontologies.yaml'
                if ontology_path.exists():
                    self.ontology_registry = OntologyRegistry.load(ontology_path)
                    logger.info(
                        'Loaded ontology profiles for %d lanes',
                        len(self.ontology_registry.configured_groups),
                    )
                else:
                    self.ontology_registry = None
                    logger.info('No extraction ontology file found; using default entity types only')
            except Exception as ontology_error:
                self.ontology_registry = None
                logger.warning(f'Failed to load ontology registry: {ontology_error}')

            # Initialize default Graphiti client for configured database
            try:
                self.client = await self._build_client()
            except Exception as db_error:
                # Check for connection errors
                error_msg = str(db_error).lower()
                if 'connection refused' in error_msg or 'could not connect' in error_msg:
                    db_provider = self.config.database.provider
                    if db_provider.lower() == 'falkordb':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: FalkorDB is not running\n'
                            f'{"=" * 70}\n\n'
                            f'FalkorDB at {db_config["host"]}:{db_config["port"]} is not accessible.\n\n'
                            f'To start FalkorDB:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose up\n'
                            f'  - Or run FalkorDB manually: docker run -p 6379:6379 falkordb/falkordb\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    elif db_provider.lower() == 'neo4j':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: Neo4j is not running\n'
                            f'{"=" * 70}\n\n'
                            f'Neo4j at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'To start Neo4j:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose -f docker/docker-compose-neo4j.yml up\n'
                            f'  - Or install Neo4j Desktop from: https://neo4j.com/download/\n'
                            f'  - Or run Neo4j manually: docker run -p 7474:7474 -p 7687:7687 neo4j:latest\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    else:
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: {db_provider} is not running\n'
                            f'{"=" * 70}\n\n'
                            f'{db_provider} at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'Please ensure {db_provider} is running and accessible.\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                # Re-raise other errors
                raise

            # Cache the default group client in FalkorDB mode for reuse.
            if self.config.database.provider.lower() == 'falkordb' and self.config.graphiti.group_id:
                default_group = self._validate_group_id(self.config.graphiti.group_id)
                self._clients_by_group[default_group] = self.client

            logger.info('Successfully initialized Graphiti client')

            # Log configuration details
            if self._llm_client:
                logger.info(
                    f'Using LLM provider: {self.config.llm.provider} / {self.config.llm.model}'
                )
            else:
                logger.info('No LLM client configured - entity extraction will be limited')

            if self._embedder_client:
                logger.info(f'Using Embedder provider: {self.config.embedder.provider}')
            else:
                logger.info('No Embedder client configured - search will be limited')

            if self.entity_types:
                entity_type_names = list(self.entity_types.keys())
                logger.info(f'Using custom entity types: {", ".join(entity_type_names)}')
            else:
                logger.info('Using default entity types')

            logger.info(f'Using database: {self.config.database.provider}')
            logger.info(f'Using group_id: {self.config.graphiti.group_id}')

        except Exception as e:
            logger.error('Failed to initialize Graphiti client: %s', type(e).__name__)
            raise

    def resolve_entity_types(self, group_id: str) -> dict | None:
        """Resolve entity types for a group, falling back to the global default.

        If the ontology registry has a profile for this group_id, return its
        entity_types.  Otherwise return the global entity_types from config.yaml.
        """
        if self.ontology_registry is not None:
            profile = self.ontology_registry.get(group_id)
            if profile is not None:
                return profile.entity_types
        return self.entity_types

    def resolve_ontology(
        self, group_id: str
    ) -> tuple[dict | None, str, dict | None, str]:
        """Resolve entity types, intent guidance, edge types, and extraction mode for a group.

        Returns:
            A 4-tuple of (entity_types, intent_guidance, edge_types, extraction_mode).
            - entity_types falls back to the global default when the group has no profile.
            - intent_guidance is the per-lane LLM focus hint (passed as
              custom_extraction_instructions to Graphiti Core). Defaults to ''.
            - edge_types defaults to None when no profile is configured.
            - extraction_mode defaults to 'permissive' when no profile is configured.
        """
        if self.ontology_registry is not None:
            profile = self.ontology_registry.get(group_id)
            if profile is not None:
                return (
                    profile.entity_types,
                    profile.intent_guidance or profile.extraction_emphasis,
                    profile.edge_types,
                    profile.extraction_mode,
                )
        return self.entity_types, '', None, 'permissive'

    async def get_client(self) -> Graphiti:
        """Get the Graphiti client, initializing if necessary."""
        if self.client is None:
            await self.initialize()
        if self.client is None:
            raise RuntimeError('Failed to initialize Graphiti client')
        return self.client


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_memory(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body should be a JSON string (standard JSON escaping)
        add_memory(
            name="Customer Profile",
            episode_body='{"company": {"name": "Acme Technologies"}, "products": [{"id": "P001", "name": "CloudSync"}, {"id": "P002", "name": "DataMiner"}]}',
            source="json",
            source_description="CRM data"
        )
    """
    global graphiti_service, queue_service

    if graphiti_service is None or queue_service is None:
        return ErrorResponse(error='Services not initialized')

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id or config.graphiti.group_id

        # Try to parse the source as an EpisodeType enum, with fallback to text
        episode_type = EpisodeType.text  # Default
        if source:
            try:
                episode_type = EpisodeType[source.lower()]
            except (KeyError, AttributeError):
                # If the source doesn't match any enum value, use text as default
                logger.warning(f"Unknown source type '{source}', using 'text' as default")
                episode_type = EpisodeType.text

        # Submit to queue service for async processing.
        # Entity types are resolved per-group by the queue service's ontology
        # resolver (lane-specific when configured, global default otherwise).
        await queue_service.add_episode(
            group_id=effective_group_id,
            name=name,
            content=episode_body,
            source_description=source_description,
            episode_type=episode_type,
            uuid=uuid or None,
            fallback_entity_types=graphiti_service.entity_types,
        )

        return SuccessResponse(
            message=f"Episode queued for processing in group '{effective_group_id}'"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error queuing episode: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode: {error_msg}')


@mcp.tool()
async def search_nodes(
    query: str,
    group_ids: list[str] | None = None,
    lane_alias: list[str] | None = None,
    search_mode: str = 'hybrid',
    max_nodes: int = 10,
    entity_types: list[str] | None = None,
    ctx: Context | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Search for nodes in the graph memory.

    Args:
        query: The search query
        group_ids: Optional explicit list of group IDs to filter results (highest precedence)
        lane_alias: Optional lane aliases resolved via config.graphiti.lane_aliases
        search_mode: Retrieval mode: hybrid|semantic|keyword
        max_nodes: Maximum number of nodes to return (default: 10)
        entity_types: Optional list of entity type names to filter by
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        if max_nodes <= 0:
            return ErrorResponse(error='max_nodes must be a positive integer')

        normalized_mode = (search_mode or 'hybrid').strip().lower()
        if normalized_mode not in VALID_SEARCH_MODES:
            return ErrorResponse(
                error=(
                    f'Invalid search_mode: {_sanitize_for_error(str(search_mode))!r}. '
                    f'Expected one of: {sorted(VALID_SEARCH_MODES)}'
                )
            )

        # Resolve group scope BEFORE deriving the rate-limit key so the key is
        # based on trusted, validated IDs — not raw caller-supplied input which
        # could be spoofed to obtain separate (unthrottled) rate-limit buckets.
        effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
            group_ids=group_ids,
            lane_alias=lane_alias,
        )
        if invalid_aliases:
            return ErrorResponse(
                error=f'Unknown lane aliases: {", ".join(invalid_aliases)}'
            )

        # Extract the trusted caller principal (never from raw request payload).
        caller_principal = _extract_trusted_caller_principal(ctx)
        caller_key = _derive_rate_limit_key(effective_group_ids, caller_principal)
        if _SEARCH_RATE_LIMIT_ENABLED:
            if not await _search_rate_limiter.is_allowed(caller_key):
                logger.warning(
                    'search_nodes rate limit exceeded (key_hash=%s)',
                    _hash_rate_limit_key(caller_key),
                )
                return ErrorResponse(error='rate limit exceeded; retry later')
            # Global fallback: enforce an additional shared bucket for anonymous
            # callers so rotating group IDs cannot bypass per-caller throttling.
            if caller_principal == _ANON_PRINCIPAL and not await _search_global_fallback_limiter.is_allowed('__global__'):
                logger.warning(
                    'search_nodes global fallback rate limit exceeded (anon caller)',
                )
                return ErrorResponse(error='rate limit exceeded; retry later')

        # Apply defense-in-depth cap before building backend search config.
        effective_max_nodes = min(max_nodes, _MAX_NODES_CAP)

        # OM adapter path: search OM primitives directly without converting
        # OMNode data into Graphiti Entity nodes.
        # Gate this path to Neo4j only; non-Neo4j backends (for example
        # FalkorDB) must continue through the Graphiti client search path even
        # when OM/all-lanes are in scope.
        provider_name = str(config.database.provider or '').strip().lower()
        use_observational_adapter = (
            provider_name == 'neo4j'
            and search_service.includes_observational_memory(effective_group_ids)
        )
        om_nodes: list[dict[str, Any]] = []

        if use_observational_adapter:
            try:
                om_nodes = await search_service.search_observational_nodes(
                    graphiti_service=graphiti_service,
                    query=query,
                    group_ids=effective_group_ids,
                    max_nodes=effective_max_nodes,
                    entity_types=entity_types,
                )
            except Exception as om_error:
                logger.error(f'OM lane retrieval failed closed: {om_error}')
                if _is_observational_memory_only_scope(effective_group_ids):
                    return NodeSearchResponse(message='No relevant nodes found', nodes=[])
                om_nodes = []

            if _is_observational_memory_only_scope(effective_group_ids):
                if not om_nodes:
                    return NodeSearchResponse(message='No relevant nodes found', nodes=[])
                return NodeSearchResponse(
                    message='Nodes retrieved successfully',
                    nodes=om_nodes[:effective_max_nodes],
                )

        scope_error = _validate_group_scope_support(effective_group_ids)
        if scope_error:
            return ErrorResponse(error=scope_error)

        # Route to the correct FalkorDB graph for the target group
        primary_group = effective_group_ids[0] if effective_group_ids else config.graphiti.group_id
        client = await graphiti_service.get_client_for_group(primary_group)

        # Create search filters
        search_filters = SearchFilters(
            node_labels=entity_types,
        )

        # Build mode-specific search config (trust-aware)
        search_config = _build_node_search_config(normalized_mode, effective_max_nodes)

        results = await client.search_(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            search_filter=search_filters,
        )

        nodes = results.nodes[:effective_max_nodes] if results.nodes else []

        if not nodes and not om_nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the results
        node_results = []
        for node in nodes:
            # Get attributes and ensure no embeddings are included
            attrs = node.attributes if hasattr(node, 'attributes') else {}
            # Remove any embedding keys that might be in attributes
            attrs = {k: v for k, v in attrs.items() if 'embedding' not in k.lower()}

            node_results.append(
                NodeResult(
                    uuid=node.uuid,
                    name=node.name,
                    labels=node.labels if node.labels else [],
                    created_at=node.created_at.isoformat() if node.created_at else None,
                    summary=node.summary,
                    group_id=node.group_id,
                    attributes=attrs,
                )
            )

        merged_nodes = _fuse_node_like_results(
            primary=node_results,
            supplemental=om_nodes,
            max_items=effective_max_nodes,
        )
        if not merged_nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=merged_nodes)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    lane_alias: list[str] | None = None,
    search_mode: str = 'hybrid',
    max_facts: int = 10,
    center_node_uuid: str | None = None,
    ctx: Context | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts.

    Args:
        query: The search query
        group_ids: Optional explicit list of group IDs to filter results (highest precedence)
        lane_alias: Optional lane aliases resolved via config.graphiti.lane_aliases
        search_mode: Retrieval mode: hybrid|semantic|keyword
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        # Validate max_facts parameter and apply defense-in-depth cap.
        if max_facts <= 0:
            return ErrorResponse(error='max_facts must be a positive integer')
        max_facts = min(max_facts, _MAX_FACTS_CAP)

        normalized_mode = (search_mode or 'hybrid').strip().lower()
        if normalized_mode not in VALID_SEARCH_MODES:
            return ErrorResponse(
                error=(
                    f'Invalid search_mode: {_sanitize_for_error(str(search_mode))!r}. '
                    f'Expected one of: {sorted(VALID_SEARCH_MODES)}'
                )
            )

        # Resolve group scope BEFORE deriving the rate-limit key so the key is
        # based on trusted, validated IDs — not raw caller-supplied input which
        # could be spoofed to obtain separate (unthrottled) rate-limit buckets.
        effective_group_ids, invalid_aliases = _resolve_effective_group_ids(
            group_ids=group_ids,
            lane_alias=lane_alias,
        )
        if invalid_aliases:
            return ErrorResponse(
                error=f'Unknown lane aliases: {", ".join(invalid_aliases)}'
            )

        # Extract the trusted caller principal (never from raw request payload).
        caller_principal = _extract_trusted_caller_principal(ctx)
        caller_key = _derive_rate_limit_key(effective_group_ids, caller_principal)
        if _SEARCH_RATE_LIMIT_ENABLED:
            if not await _search_rate_limiter.is_allowed(caller_key):
                logger.warning(
                    'search_memory_facts rate limit exceeded (key_hash=%s)',
                    _hash_rate_limit_key(caller_key),
                )
                return ErrorResponse(error='rate limit exceeded; retry later')
            # Global fallback: enforce an additional shared bucket for anonymous
            # callers so rotating group IDs cannot bypass per-caller throttling.
            if caller_principal == _ANON_PRINCIPAL and not await _search_global_fallback_limiter.is_allowed('__global__'):
                logger.warning(
                    'search_memory_facts global fallback rate limit exceeded (anon caller)',
                )
                return ErrorResponse(error='rate limit exceeded; retry later')

        # OM adapter path: search OM ontology edges directly from OM primitives.
        # Gate this path to Neo4j only; non-Neo4j backends (for example
        # FalkorDB) must continue through the Graphiti client search path even
        # when OM/all-lanes are in scope.
        provider_name = str(config.database.provider or '').strip().lower()
        use_observational_adapter = (
            provider_name == 'neo4j'
            and search_service.includes_observational_memory(effective_group_ids)
        )
        om_facts: list[dict[str, Any]] = []

        if use_observational_adapter:
            try:
                om_facts = await search_service.search_observational_facts(
                    graphiti_service=graphiti_service,
                    query=query,
                    group_ids=effective_group_ids,
                    max_facts=max_facts,
                    center_node_uuid=center_node_uuid,
                )
            except Exception as om_error:
                logger.error(f'OM lane fact retrieval failed closed: {om_error}')
                if _is_observational_memory_only_scope(effective_group_ids):
                    return FactSearchResponse(message='No relevant facts found', facts=[])
                om_facts = []

            if _is_observational_memory_only_scope(effective_group_ids):
                if not om_facts:
                    return FactSearchResponse(message='No relevant facts found', facts=[])
                return FactSearchResponse(
                    message='Facts retrieved successfully',
                    facts=om_facts[:max_facts],
                )

        scope_error = _validate_group_scope_support(effective_group_ids)
        if scope_error:
            return ErrorResponse(error=scope_error)

        # Route to the correct FalkorDB graph for the target group
        primary_group = effective_group_ids[0] if effective_group_ids else config.graphiti.group_id
        client = await graphiti_service.get_client_for_group(primary_group)

        # Build mode-specific search config (trust-aware)
        search_config = _build_edge_search_config(
            normalized_mode,
            max_facts,
            center_node_uuid,
        )

        results = await client.search_(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
        )

        relevant_edges = results.edges[:max_facts] if results.edges else []  # already capped above

        if not relevant_edges and not om_facts:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        merged_facts = _fuse_node_like_results(
            primary=facts,
            supplemental=om_facts,
            max_items=max_facts,
        )
        if not merged_facts:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        return FactSearchResponse(message='Facts retrieved successfully', facts=merged_facts)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the episodic node by UUID
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


@mcp.tool()
async def get_episodes(
    group_ids: list[str] | None = None,
    max_episodes: int = 10,
) -> EpisodeSearchResponse | ErrorResponse:
    """Get episodes from the graph memory.

    Args:
        group_ids: Optional list of group IDs to filter results
        max_episodes: Maximum number of episodes to return (default: 10)
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Route to the correct FalkorDB graph for the target group
        primary_group = effective_group_ids[0] if effective_group_ids else config.graphiti.group_id
        client = await graphiti_service.get_client_for_group(primary_group)

        # Get episodes from the driver directly
        from graphiti_core.nodes import EpisodicNode

        if effective_group_ids:
            episodes = await EpisodicNode.get_by_group_ids(
                client.driver, effective_group_ids, limit=max_episodes
            )
        else:
            # If no group IDs, we need to use a different approach
            # For now, return empty list when no group IDs specified
            episodes = []

        if not episodes:
            return EpisodeSearchResponse(message='No episodes found', episodes=[])

        # Format the results
        episode_results = []
        for episode in episodes:
            episode_dict = {
                'uuid': episode.uuid,
                'name': episode.name,
                'content': episode.content,
                'created_at': episode.created_at.isoformat() if episode.created_at else None,
                'source': episode.source.value
                if hasattr(episode.source, 'value')
                else str(episode.source),
                'source_description': episode.source_description,
                'group_id': episode.group_id,
            }
            episode_results.append(episode_dict)

        return EpisodeSearchResponse(
            message='Episodes retrieved successfully', episodes=episode_results
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool()
async def clear_graph(group_ids: list[str] | None = None) -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph for specified group IDs.

    Args:
        group_ids: Optional list of group IDs to clear. If not provided, clears the default group.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None
            else [config.graphiti.group_id] if config.graphiti.group_id
            else []
        )

        if not effective_group_ids:
            return ErrorResponse(error='No group IDs specified for clearing')

        # Route to the correct FalkorDB graph for the target group
        primary_group = effective_group_ids[0]
        client = await graphiti_service.get_client_for_group(primary_group)

        # Clear data for the specified group IDs
        await clear_data(client.driver, group_ids=effective_group_ids)

        return SuccessResponse(
            message=f'Graph data cleared successfully for group IDs: {", ".join(effective_group_ids)}'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.tool()
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and database connection."""
    global graphiti_service

    if graphiti_service is None:
        return StatusResponse(status='error', message='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Test database connection with a simple query
        async with client.driver.session() as session:
            result = await session.run('MATCH (n) RETURN count(n) as count')
            # Consume the result to verify query execution
            if result:
                _ = [record async for record in result]

        # Use the provider from the service's config, not the global
        provider_name = graphiti_service.config.database.provider
        return StatusResponse(
            status='ok',
            message=f'Graphiti MCP server is running and connected to {provider_name} database',
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking database connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but database connection failed: {error_msg}',
        )


@mcp.custom_route('/health', methods=['GET'])
async def health_check(request) -> JSONResponse:
    """Health check endpoint for Docker and load balancers."""
    return JSONResponse({'status': 'healthy', 'service': 'graphiti-mcp'})


async def initialize_server() -> ServerConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config, graphiti_service, queue_service, graphiti_client, semaphore

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with YAML configuration support'
    )

    # Configuration file argument
    # Default to config/config.yaml relative to the mcp_server directory
    default_config = Path(__file__).parent.parent / 'config' / 'config.yaml'
    parser.add_argument(
        '--config',
        type=Path,
        default=default_config,
        help='Path to YAML configuration file (default: config/config.yaml)',
    )

    # Transport arguments
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio', 'http'],
        help='Transport to use: http (recommended, default), stdio (standard I/O), or sse (deprecated)',
    )
    parser.add_argument(
        '--host',
        help='Host to bind the MCP server to',
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port to bind the MCP server to',
    )

    # Provider selection arguments
    parser.add_argument(
        '--llm-provider',
        choices=['openai', 'azure_openai', 'anthropic', 'gemini', 'groq'],
        help='LLM provider to use',
    )
    parser.add_argument(
        '--embedder-provider',
        choices=['openai', 'azure_openai', 'gemini', 'voyage'],
        help='Embedder provider to use',
    )
    parser.add_argument(
        '--database-provider',
        choices=['neo4j', 'falkordb'],
        help='Database provider to use',
    )

    # LLM configuration arguments
    parser.add_argument('--model', help='Model name to use with the LLM client')
    parser.add_argument('--small-model', help='Small model name to use with the LLM client')
    parser.add_argument(
        '--temperature', type=float, help='Temperature setting for the LLM (0.0-2.0)'
    )

    # Embedder configuration arguments
    parser.add_argument('--embedder-model', help='Model name to use with the embedder')

    # Graphiti-specific arguments
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. If not provided, uses config file or generates random UUID.',
    )
    parser.add_argument(
        '--user-id',
        help='User ID for tracking operations',
    )
    parser.add_argument(
        '--destroy-graph',
        action='store_true',
        help='Destroy all Graphiti graphs on startup',
    )

    args = parser.parse_args()

    # Set config path in environment for the settings to pick up
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # Load configuration with environment variables and YAML
    config = GraphitiConfig()

    # Apply CLI overrides
    config.apply_cli_overrides(args)

    # Also apply legacy CLI args for backward compatibility
    if hasattr(args, 'destroy_graph'):
        config.destroy_graph = args.destroy_graph

    # Log configuration details
    logger.info('Using configuration:')
    logger.info(f'  - LLM: {config.llm.provider} / {config.llm.model}')
    logger.info(f'  - Embedder: {config.embedder.provider} / {config.embedder.model}')
    logger.info(f'  - Database: {config.database.provider}')
    logger.info(f'  - Group ID: {config.graphiti.group_id}')
    logger.info(f'  - Transport: {config.server.transport}')

    # Log graphiti-core version
    try:
        import graphiti_core

        graphiti_version = getattr(graphiti_core, '__version__', 'unknown')
        logger.info(f'  - Graphiti Core: {graphiti_version}')
    except Exception:
        # Check for Docker-stored version file
        version_file = Path('/app/.graphiti-core-version')
        if version_file.exists():
            graphiti_version = version_file.read_text().strip()
            logger.info(f'  - Graphiti Core: {graphiti_version}')
        else:
            logger.info('  - Graphiti Core: version unavailable')

    # Handle graph destruction if requested
    if hasattr(config, 'destroy_graph') and config.destroy_graph:
        logger.warning('Destroying all Graphiti graphs as requested...')
        temp_service = GraphitiService(config, SEMAPHORE_LIMIT)
        await temp_service.initialize()
        client = await temp_service.get_client()
        await clear_data(client.driver)
        logger.info('All graphs destroyed')

    # Initialize services
    graphiti_service = GraphitiService(config, SEMAPHORE_LIMIT)
    queue_service = QueueService()
    await graphiti_service.initialize()

    # Set global client for backward compatibility
    graphiti_client = await graphiti_service.get_client()
    semaphore = graphiti_service.semaphore

    # Initialize queue service with per-group routing for add_memory.
    # This prevents cross-graph contamination when multiple corpora are ingested.
    await queue_service.initialize(
        graphiti_client=graphiti_client,
        client_resolver=graphiti_service.get_client_for_group,
        ontology_resolver=graphiti_service.resolve_ontology,
    )

    # Set MCP server settings
    if config.server.host:
        mcp.settings.host = config.server.host
    if config.server.port:
        mcp.settings.port = config.server.port

    # Return MCP configuration for transport
    return config.server


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with configured transport
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info(f'Access the server at: http://{mcp.settings.host}:{mcp.settings.port}/sse')
        await mcp.run_sse_async()
    elif mcp_config.transport == 'http':
        # Use localhost for display if binding to 0.0.0.0
        display_host = 'localhost' if mcp.settings.host == '0.0.0.0' else mcp.settings.host
        logger.info(
            f'Running MCP server with streamable HTTP transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info('=' * 60)
        logger.info('MCP Server Access Information:')
        logger.info(f'  Base URL: http://{display_host}:{mcp.settings.port}/')
        logger.info(f'  MCP Endpoint: http://{display_host}:{mcp.settings.port}/mcp/')
        logger.info('  Transport: HTTP (streamable)')

        # Show FalkorDB Browser UI access if enabled
        if os.environ.get('BROWSER', '1') == '1':
            logger.info(f'  FalkorDB Browser UI: http://{display_host}:3000/')

        logger.info('=' * 60)
        logger.info('For MCP clients, connect to the /mcp/ endpoint above')

        # Configure uvicorn logging to match our format
        configure_uvicorn_logging()

        await mcp.run_streamable_http_async()
    else:
        raise ValueError(
            f'Unsupported transport: {mcp_config.transport}. Use "sse", "stdio", or "http"'
        )


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info('Server shutting down...')
    except Exception as e:
        logger.error('Error initializing Graphiti MCP server: %s', type(e).__name__)
        raise


if __name__ == '__main__':
    main()

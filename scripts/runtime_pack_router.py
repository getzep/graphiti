from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from string import Template

ALLOWED_SCOPE: dict[str, int] = {
    'public': 0,
    'group-safe': 1,
    'private': 2,
}
ALLOWED_CHATGPT_MODES = ('global', 'scoped', 'off')

DEFAULT_REGISTRY_CANDIDATES = (
    'config/runtime_pack_registry.json',
    'config/runtime_pack_registry.yaml',
)
DEFAULT_PROFILES_CANDIDATES = (
    'config/runtime_consumer_profiles.json',
    'config/runtime_consumer_profiles.yaml',
)

ENGINEERING_LOOP_FILES = (
    'state/engineering/loops/clr_learnings.latest.jsonl',
    'state/engineering/loops/antfarm_learnings.latest.jsonl',
)

REQUIRED_PROFILE_KEYS = (
    'consumer',
    'workflow_id',
    'step_id',
    'scope',
    'schema_version',
    'task',
    'injection_text',
    'pack_ids',
)
REQUIRED_PACK_KEYS = (
    'pack_id',
    'path',
    'scope',
)
REQUIRED_PLAN_KEYS = (
    'consumer',
    'workflow_id',
    'step_id',
    'scope',
    'schema_version',
    'task',
    'normalized_query',
    'query_hash',
    'index_health',
    'vector_errors',
    'injection_text',
    'packs',
    'selected_packs',
    'dropped_packs',
    'decision_path',
    'budget_summary',
)

DEFAULT_TIER_C_FIXED_TOKENS = 3000
DEFAULT_OUTPUT_RESERVE_TOKENS = 2500
PINNED_TIER_C_PROFILES: dict[str, int] = {
    'main_session_dining_recs': 3000,
    'main_session_content_tweet': 3500,
    'main_session_vc_memo': 6500,
    'main_session_vc_deal_brief': 6000,
    'main_session_vc_diligence_questions': 5500,
    'main_session_vc_ic_prep': 6500,
    'main_session_content_long_form': 7500,
}

QUERY_TEXT_NORMALIZATION_V1 = 'unicode_nfkc_trim_collapse_ws_lower'
RRF_K = 60
VECTOR_EMBEDDING_DIM = 768
INDEX_SCHEMA_VERSION = 1
INDEX_BM25_SCHEMA_VERSION = 1
INDEX_VECTOR_SCHEMA_VERSION = 1
MATERIALIZE_DEFAULT_MAX_ITEMS = 10
MATERIALIZE_DEFAULT_TIMEOUT_SEC = 0.8
MATERIALIZE_MIN_COVERAGE_ITEMS = 2
MATERIALIZE_DEFAULT_MAX_FACT_CHARS = 240
MATERIALIZE_MAX_FACT_CHARS = 800
MATERIALIZE_DEFAULT_MAX_BLOCK_TOKENS = 320
MATERIALIZE_MAX_BLOCK_TOKENS = 1200
GRAPHITI_DEFAULT_BASE_URL = 'http://localhost:8000'
GRAPHITI_SEARCH_PATH = '/search'
MATERIALIZE_SOURCE_CONTENT_VOICE_STYLE = 'graphiti_content_voice_style'
MATERIALIZE_SOURCE_CONTENT_WRITING_SAMPLES = 'graphiti_content_writing_samples'
MATERIALIZE_SOURCE_CONTENT_LONG_FORM_ARTIFACTS = 'graphiti_content_long_form_artifacts'


class OMIndexMismatchError(ValueError):
    """Raised when retrieval index metadata does not match canonical schema."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ensure_dict(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be an object')
    return value


def _ensure_list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f'{context} must be a list')
    return value


def _ensure_string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{context} must be a string')
    return value


def _ensure_non_empty_string(value: object, *, context: str) -> str:
    text = _ensure_string(value, context=context).strip()
    if not text:
        raise ValueError(f'{context} must be a non-empty string')
    return text


def _ensure_int(value: object, *, context: str, min_value: int | None = None) -> int:
    if not isinstance(value, int):
        raise ValueError(f'{context} must be an integer')
    if min_value is not None and value < min_value:
        raise ValueError(f'{context} must be >= {min_value}')
    return value


def _ensure_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{context} must be a boolean')
    return value


def _ensure_string_list(value: object, *, context: str) -> list[str]:
    raw = _ensure_list(value, context=context)
    parsed: list[str] = []
    for idx, item in enumerate(raw):
        parsed.append(_ensure_non_empty_string(item, context=f'{context}[{idx}]'))
    return parsed


def _load_file(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding='utf-8')

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml
        except ModuleNotFoundError as fallback:
            raise ValueError(f'{path} must be JSON (or install PyYAML to parse YAML)') from fallback

        payload = yaml.safe_load(raw)  # type: ignore[attr-defined]

    return _ensure_dict(payload, context=str(path))


def _resolve_config_path(repo_root: Path, candidates: tuple[str, ...], label: str) -> Path:
    for rel in candidates:
        p = repo_root / rel
        if p.exists():
            return p
    joined = ', '.join(candidates)
    raise ValueError(f'Config file missing for {label}. Tried: {joined}')


def _normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _normalize_query_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text)
    normalized = _normalize_whitespace(normalized)
    return normalized.lower()


def _query_hash(normalized_query: str) -> str:
    return hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _tokenize_for_rank(text: str) -> list[str]:
    return re.findall(r'[0-9a-z]+', _normalize_query_text(text))


def _normalize_secret_for_match(secret: str) -> str:
    return _normalize_whitespace(secret)


def _redact_output_text(text: str, *secrets: str | None) -> str:
    redacted = text
    for secret in secrets:
        if secret is None:
            continue
        normalized_secret = _normalize_secret_for_match(secret)
        if not normalized_secret:
            continue
        pattern = re.compile(r'\b' + r'\s+'.join(map(re.escape, normalized_secret.split())) + r'\b')
        redacted = pattern.sub('[REDACTED]', redacted)
    return redacted


def _normalize_scope(scope: str) -> str:
    token = scope.strip().lower().replace('_', '-')
    if token not in ALLOWED_SCOPE:
        raise ValueError(f'scope must be one of: {sorted(ALLOWED_SCOPE)}')
    return token


def _load_registry(path: Path) -> dict[str, dict[str, object]]:
    data = _load_file(path)
    _ensure_int(data.get('schema_version'), context=f'{path}.schema_version', min_value=1)
    packs_raw = _ensure_list(data.get('packs'), context=f'{path}.packs')

    registry: dict[str, dict[str, object]] = {}
    for index, item in enumerate(packs_raw):
        entry = _ensure_dict(item, context=f'{path}.packs[{index}]')
        for key in REQUIRED_PACK_KEYS:
            if key not in entry:
                raise ValueError(f'{path}.packs[{index}] missing {key}')

        pack_id = _ensure_non_empty_string(entry['pack_id'], context=f'{path}.packs[{index}].pack_id')
        if pack_id in registry:
            raise ValueError(f'{path}.packs[{index}] duplicates pack_id={pack_id}')

        path_value = _ensure_non_empty_string(entry['path'], context=f'{path}.packs[{index}].path')
        scope = _normalize_scope(_ensure_non_empty_string(entry['scope'], context=f'{path}.packs[{index}].scope'))

        query_template = _ensure_non_empty_string(
            entry.get('query_template', '${path}'),
            context=f'{path}.packs[{index}].query_template',
        )

        normalized: dict[str, object] = {
            'pack_id': pack_id,
            'path': path_value,
            'scope': scope,
            'query_template': query_template,
            'required': bool(entry.get('required', True)),
        }

        if 'retrieval' in entry and entry['retrieval'] is not None:
            normalized['retrieval'] = _ensure_dict(
                entry['retrieval'],
                context=f'{path}.packs[{index}].retrieval',
            )

        if 'materialization' in entry and entry['materialization'] is not None:
            normalized['materialization'] = _ensure_dict(
                entry['materialization'],
                context=f'{path}.packs[{index}].materialization',
            )

        registry[pack_id] = normalized

    return registry


def _load_profiles(path: Path) -> list[dict[str, object]]:
    data = _load_file(path)
    _ensure_int(data.get('schema_version'), context=f'{path}.schema_version', min_value=1)
    profiles_raw = _ensure_list(data.get('profiles'), context=f'{path}.profiles')

    profiles: list[dict[str, object]] = []
    for index, item in enumerate(profiles_raw):
        entry = _ensure_dict(item, context=f'{path}.profiles[{index}]')
        for key in REQUIRED_PROFILE_KEYS:
            if key not in entry:
                raise ValueError(f'{path}.profiles[{index}] missing {key}')

        entry['consumer'] = _ensure_non_empty_string(entry['consumer'], context=f'{path}.profiles[{index}].consumer')
        entry['workflow_id'] = _ensure_non_empty_string(entry['workflow_id'], context=f'{path}.profiles[{index}].workflow_id')
        entry['step_id'] = _ensure_non_empty_string(entry['step_id'], context=f'{path}.profiles[{index}].step_id')
        entry['scope'] = _normalize_scope(_ensure_non_empty_string(entry['scope'], context=f'{path}.profiles[{index}].scope'))
        _ensure_int(entry['schema_version'], context=f'{path}.profiles[{index}].schema_version', min_value=1)

        entry['task'] = _ensure_string(entry['task'], context=f'{path}.profiles[{index}].task')
        entry['injection_text'] = _ensure_string(entry['injection_text'], context=f'{path}.profiles[{index}].injection_text')
        entry['pack_ids'] = _ensure_string_list(entry['pack_ids'], context=f'{path}.profiles[{index}].pack_ids')

        chatgpt_mode = str(entry.get('chatgpt_mode', 'scoped')).strip().lower()
        if chatgpt_mode not in ALLOWED_CHATGPT_MODES:
            raise ValueError(
                f'{path}.profiles[{index}].chatgpt_mode must be one of {ALLOWED_CHATGPT_MODES}'
            )
        entry['chatgpt_mode'] = chatgpt_mode

        if 'pack_modes' in entry and entry['pack_modes'] is not None:
            pack_modes = _ensure_dict(entry['pack_modes'], context=f'{path}.profiles[{index}].pack_modes')
            normalized_pack_modes: dict[str, str] = {}
            for k, v in pack_modes.items():
                key = _ensure_non_empty_string(k, context=f'{path}.profiles[{index}].pack_modes key')
                val = _ensure_non_empty_string(v, context=f'{path}.profiles[{index}].pack_modes[{key}]')
                normalized_pack_modes[key] = val
            entry['pack_modes'] = normalized_pack_modes
        else:
            entry['pack_modes'] = {}

        if 'tier_c_fixed_tokens' in entry and entry['tier_c_fixed_tokens'] is not None:
            entry['tier_c_fixed_tokens'] = _ensure_int(
                entry['tier_c_fixed_tokens'],
                context=f'{path}.profiles[{index}].tier_c_fixed_tokens',
                min_value=0,
            )

        if 'output_reserve_tokens' in entry and entry['output_reserve_tokens'] is not None:
            entry['output_reserve_tokens'] = _ensure_int(
                entry['output_reserve_tokens'],
                context=f'{path}.profiles[{index}].output_reserve_tokens',
                min_value=0,
            )

        if 'model_context_limit' in entry and entry['model_context_limit'] is not None:
            entry['model_context_limit'] = _ensure_int(
                entry['model_context_limit'],
                context=f'{path}.profiles[{index}].model_context_limit',
                min_value=1,
            )

        profiles.append(entry)

    return profiles


def _build_warning(event: str, **payload: object) -> dict[str, object]:
    body: dict[str, object] = {'event': event}
    body.update(payload)
    return body


def _validate_tier_c_profile_pins(
    profiles: list[dict[str, object]],
) -> tuple[dict[str, int], list[dict[str, object]]]:
    by_consumer: dict[str, dict[str, object]] = {}
    for profile in profiles:
        consumer = profile.get('consumer')
        if isinstance(consumer, str) and consumer.strip():
            by_consumer[consumer.strip()] = profile

    forced_fallback: dict[str, int] = {}
    warnings: list[dict[str, object]] = []

    for consumer, expected in sorted(PINNED_TIER_C_PROFILES.items()):
        profile = by_consumer.get(consumer)
        if profile is None:
            forced_fallback[consumer] = DEFAULT_TIER_C_FIXED_TOKENS
            warnings.append(
                _build_warning(
                    'TIER_C_PROFILE_MISMATCH',
                    consumer=consumer,
                    expected=expected,
                    actual='missing',
                )
            )
            continue

        actual = profile.get('tier_c_fixed_tokens')
        if not isinstance(actual, int) or actual != expected:
            forced_fallback[consumer] = DEFAULT_TIER_C_FIXED_TOKENS
            warnings.append(
                _build_warning(
                    'TIER_C_PROFILE_MISMATCH',
                    consumer=consumer,
                    expected=expected,
                    actual=actual,
                )
            )

    return forced_fallback, warnings


def _resolve_tier_c_fixed_tokens(
    profile: dict[str, object],
    *,
    consumer: str,
    forced_fallback: dict[str, int],
) -> tuple[int, list[dict[str, object]]]:
    warnings: list[dict[str, object]] = []

    if consumer in forced_fallback:
        return forced_fallback[consumer], warnings

    value = profile.get('tier_c_fixed_tokens')
    if value is None:
        warnings.append(
            _build_warning(
                'TIER_C_DEFAULT_FALLBACK_USED',
                consumer=consumer,
                value=DEFAULT_TIER_C_FIXED_TOKENS,
            )
        )
        return DEFAULT_TIER_C_FIXED_TOKENS, warnings

    if not isinstance(value, int) or value < 0:
        warnings.append(
            _build_warning(
                'TIER_C_DEFAULT_FALLBACK_USED',
                consumer=consumer,
                value=DEFAULT_TIER_C_FIXED_TOKENS,
                reason='invalid_tier_c_fixed_tokens',
            )
        )
        return DEFAULT_TIER_C_FIXED_TOKENS, warnings

    return int(value), warnings


def _resolve_output_reserve_tokens(profile: dict[str, object]) -> int:
    value = profile.get('output_reserve_tokens')
    if isinstance(value, int) and value >= 0:
        return value
    return DEFAULT_OUTPUT_RESERVE_TOKENS


def _build_query(profile: dict[str, object], registry_entry: dict[str, object], *, repo_path: Path) -> str:
    template = _ensure_non_empty_string(
        registry_entry['query_template'],
        context='registry_entry.query_template',
    )

    mapping = {
        'repo_path': str(repo_path),
        'path': _ensure_non_empty_string(registry_entry['path'], context='registry_entry.path'),
        'pack_id': _ensure_non_empty_string(registry_entry['pack_id'], context='registry_entry.pack_id'),
        'consumer': _ensure_non_empty_string(profile['consumer'], context='profile.consumer'),
        'workflow_id': _ensure_non_empty_string(profile['workflow_id'], context='profile.workflow_id'),
        'step_id': _ensure_non_empty_string(profile['step_id'], context='profile.step_id'),
    }

    try:
        query = Template(template).substitute(mapping)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(f'query_template references unknown key {missing!r}') from exc

    query_path = (repo_path / query).resolve()
    if not _is_relative_to(query_path, repo_path.resolve()):
        raise ValueError(f'Pack path escapes repo root: {query}')
    if not query_path.exists():
        raise ValueError(f'Pack path not found: {query}')
    return query


def _select_profile(
    profiles: list[dict[str, object]],
    *,
    consumer: str,
    workflow_id: str,
    step_id: str,
) -> dict[str, object]:
    matches: list[dict[str, object]] = [
        profile
        for profile in profiles
        if profile['consumer'] == consumer
        and profile['workflow_id'] == workflow_id
        and profile['step_id'] == step_id
    ]
    if len(matches) == 0:
        raise ValueError(f'consumer_not_found: {consumer}/{workflow_id}/{step_id}')
    if len(matches) > 1:
        raise ValueError(
            f'consumer profile ambiguous: {consumer}/{workflow_id}/{step_id} matched {len(matches)} entries',
        )
    return matches[0]


def _validate_pack_scope(profile_scope: str, pack_scope: str) -> None:
    if ALLOWED_SCOPE[pack_scope] > ALLOWED_SCOPE[profile_scope]:
        raise ValueError(
            f'pack scope {pack_scope} exceeds profile scope {profile_scope} for consumer profile'
        )


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _resolve_pack_mode(profile: dict[str, object], pack_id: str) -> str:
    pack_modes = profile.get('pack_modes')
    if isinstance(pack_modes, dict):
        val = pack_modes.get(pack_id)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return 'default'


def _resolve_materialize_max_items(materialization: dict[str, object], mode: str) -> int:
    mode_key = f"max_items_{mode.strip().lower().replace('-', '_')}"
    mode_value = materialization.get(mode_key)
    if isinstance(mode_value, int) and mode_value > 0:
        return mode_value

    default_value = materialization.get('max_items')
    if isinstance(default_value, int) and default_value > 0:
        return default_value

    return MATERIALIZE_DEFAULT_MAX_ITEMS


def _resolve_group_ids(
    *,
    retrieval_cfg: dict[str, object] | None,
    mode: str,
    chatgpt_mode: str,
) -> tuple[list[str], bool, str | None]:
    if retrieval_cfg is None:
        return ([], False, None)

    group_ids_by_mode = retrieval_cfg.get('group_ids_by_mode')
    selected_groups: list[str] = []
    if isinstance(group_ids_by_mode, dict):
        raw_for_mode = group_ids_by_mode.get(mode)
        raw_default = group_ids_by_mode.get('default')
        raw_values = raw_for_mode if raw_for_mode is not None else raw_default
        if isinstance(raw_values, list):
            for item in raw_values:
                if isinstance(item, str) and item.strip():
                    selected_groups.append(item.strip())

    chatgpt_cfg = retrieval_cfg.get('chatgpt_lane')
    chatgpt_group: str | None = None
    include_chatgpt = False
    if isinstance(chatgpt_cfg, dict):
        raw_gid = chatgpt_cfg.get('group_id')
        if isinstance(raw_gid, str) and raw_gid.strip():
            chatgpt_group = raw_gid.strip()

        allow_scoped = bool(chatgpt_cfg.get('allow_scoped', False))
        allow_global = bool(chatgpt_cfg.get('allow_global', False))

        if (
            (chatgpt_mode == 'global' and allow_global)
            or (chatgpt_mode == 'scoped' and allow_scoped)
        ):
            include_chatgpt = chatgpt_group is not None

    if include_chatgpt and chatgpt_group is not None:
        selected_groups.append(chatgpt_group)

    return (_dedupe_ordered(selected_groups), include_chatgpt, chatgpt_group)


def _parse_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists() or not path.is_file():
        return rows

    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _learning_text(row: dict[str, object]) -> str:
    desc = row.get('description')
    if isinstance(desc, str) and desc.strip():
        return desc.strip()

    payload = row.get('payload')
    if isinstance(payload, dict):
        signal = payload.get('signal')
        if isinstance(signal, str) and signal.strip():
            return f'signal={signal.strip()}'

    kind = row.get('kind')
    if isinstance(kind, str) and kind.strip():
        return f'kind={kind.strip()}'

    return 'engineering learning captured'


def _materialize_engineering(
    *,
    repo_root: Path,
    pack_entry: dict[str, object],
    mode: str,
) -> tuple[str, int, list[str]]:
    mat_cfg = pack_entry.get('materialization')
    mat = mat_cfg if isinstance(mat_cfg, dict) else {}

    max_items = 8 if mode == 'short' else 16
    if mode == 'short' and isinstance(mat.get('max_items_short'), int):
        max_items = max(1, int(mat['max_items_short']))
    if mode == 'long' and isinstance(mat.get('max_items_long'), int):
        max_items = max(1, int(mat['max_items_long']))

    rows: list[dict[str, object]] = []
    used_sources: list[str] = []
    for rel in ENGINEERING_LOOP_FILES:
        p = repo_root / rel
        parsed = _parse_jsonl(p)
        if parsed:
            rows.extend(parsed)
            used_sources.append(rel)

    if not rows:
        return (
            'No engineering learnings materialized yet (loop artifacts missing or empty).',
            0,
            used_sources,
        )

    # Stable deterministic ordering with recency bias by file read order + row order.
    rows = rows[-(max_items * 4) :]

    seen: set[str] = set()
    bullets: list[str] = []
    for row in reversed(rows):
        key = ''
        fp = row.get('fingerprint')
        if isinstance(fp, str) and fp.strip():
            key = fp.strip()
        if not key:
            key = _learning_text(row)
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f'- {_learning_text(row)}')
        if len(bullets) >= max_items:
            break

    bullets.reverse()
    header = f'Engineering learnings ({len(bullets)} items, mode={mode})'
    return ('\n'.join([header, *bullets]), len(bullets), used_sources)


def _read_pack_excerpt(path: Path, *, limit_chars: int = 1000) -> str:
    text = path.read_text(encoding='utf-8', errors='replace').strip()
    if len(text) <= limit_chars:
        return text
    return text[:limit_chars].rstrip() + '\n…(truncated)…'


def _load_yaml_domain_context(*, repo_root: Path, pack_yaml_path: str) -> str:
    if not pack_yaml_path:
        return ''

    candidate = Path(pack_yaml_path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    if not _is_relative_to(resolved, repo_root.resolve()):
        raise ValueError(f'pack_yaml path escapes repo root: {pack_yaml_path}')

    if not resolved.exists() or not resolved.is_file():
        return ''

    try:
        data = _load_file(resolved)
    except Exception:
        return ''

    domain_context = data.get('domain_context')
    if isinstance(domain_context, str):
        return domain_context.strip()
    return ''


def _materialize_timeout_seconds(materialization: dict[str, object] | None) -> float:
    if isinstance(materialization, dict):
        value = materialization.get('timeout_seconds')
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return MATERIALIZE_DEFAULT_TIMEOUT_SEC


def _materialize_max_block_tokens(materialization: dict[str, object] | None) -> int:
    if isinstance(materialization, dict):
        value = materialization.get('max_block_tokens')
        if isinstance(value, int) and value > 0:
            return min(value, MATERIALIZE_MAX_BLOCK_TOKENS)
    return MATERIALIZE_DEFAULT_MAX_BLOCK_TOKENS


def _materialize_max_fact_chars(materialization: dict[str, object] | None) -> int:
    if isinstance(materialization, dict):
        value = materialization.get('max_fact_chars')
        if isinstance(value, int) and value > 0:
            return min(value, MATERIALIZE_MAX_FACT_CHARS)
    return MATERIALIZE_DEFAULT_MAX_FACT_CHARS


def _materialize_min_coverage_items(materialization: dict[str, object] | None) -> int:
    if isinstance(materialization, dict):
        value = materialization.get('min_coverage_items')
        if isinstance(value, int) and value > 0:
            return value
    return MATERIALIZE_MIN_COVERAGE_ITEMS


def _normalize_fact_text(text: str, *, max_fact_chars: int) -> tuple[str, bool]:
    collapsed = _normalize_whitespace(text)
    if len(collapsed) <= max_fact_chars:
        return collapsed, False
    return collapsed[:max_fact_chars].rstrip() + '…', True


def _graphiti_search_facts(
    *,
    query: str,
    group_ids: list[str],
    max_items: int,
    timeout_seconds: float,
    max_fact_chars: int,
) -> tuple[list[str], dict[str, object]]:
    base_url = os.environ.get('GRAPHITI_BASE_URL', GRAPHITI_DEFAULT_BASE_URL).rstrip('/')
    endpoint = os.environ.get('GRAPHITI_SEARCH_PATH', GRAPHITI_SEARCH_PATH)
    if not endpoint.startswith('/'):
        endpoint = f'/{endpoint}'

    request_body = {
        'query': query,
        'group_ids': group_ids if group_ids else None,
        'max_facts': max(1, max_items),
    }
    data = json.dumps(request_body).encode('utf-8')

    headers = {
        'Content-Type': 'application/json',
    }
    api_key = os.environ.get('GRAPHITI_API_KEY')
    auth_header = os.environ.get('GRAPHITI_AUTH_HEADER', 'Authorization')
    if api_key and auth_header:
        headers[auth_header] = f'Bearer {api_key}'

    req = urllib.request.Request(
        f'{base_url}{endpoint}',
        data=data,
        headers=headers,
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            payload_raw = response.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'graphiti_http_{exc.code}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'graphiti_unreachable:{exc.reason}') from exc

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError('graphiti_invalid_json') from exc

    if not isinstance(payload, dict):
        raise RuntimeError('graphiti_invalid_payload')

    facts_raw = payload.get('facts')
    if not isinstance(facts_raw, list):
        raise RuntimeError('graphiti_invalid_facts')

    normalized: list[str] = []
    seen: set[str] = set()
    fact_char_truncations = 0
    duplicates_skipped = 0
    fact_candidates = 0

    for item in facts_raw:
        if not isinstance(item, dict):
            continue
        fact = item.get('fact')
        if not isinstance(fact, str):
            continue
        fact_candidates += 1
        cleaned, was_truncated = _normalize_fact_text(fact, max_fact_chars=max_fact_chars)
        if was_truncated:
            fact_char_truncations += 1
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            duplicates_skipped += 1
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= max(1, max_items):
            break

    stats: dict[str, object] = {
        'requested_max_items': max(1, max_items),
        'facts_received': len(facts_raw),
        'fact_candidates': fact_candidates,
        'facts_selected': len(normalized),
        'duplicates_skipped': duplicates_skipped,
        'fact_char_cap': max_fact_chars,
        'fact_char_truncations': fact_char_truncations,
    }

    return normalized, stats


def _cap_block_by_token_budget(text: str, max_tokens: int) -> tuple[str, bool, int]:
    cap_chars = max(120, max_tokens * 4)
    if len(text) <= cap_chars:
        return text, False, cap_chars
    return text[:cap_chars].rstrip() + '\n…(truncated)…', True, cap_chars


def _materialize_content_pack(
    *,
    source: str,
    query: str,
    group_ids: list[str],
    mode: str,
    max_items: int,
    materialization: dict[str, object] | None,
) -> tuple[str, dict[str, object]]:
    timeout_seconds = _materialize_timeout_seconds(materialization)
    min_coverage = _materialize_min_coverage_items(materialization)
    max_block_tokens = _materialize_max_block_tokens(materialization)
    max_fact_chars = _materialize_max_fact_chars(materialization)

    lane_label = ', '.join(group_ids) if group_ids else 'default lane'
    facts, fact_stats = _graphiti_search_facts(
        query=query,
        group_ids=group_ids,
        max_items=max_items,
        timeout_seconds=timeout_seconds,
        max_fact_chars=max_fact_chars,
    )

    stats: dict[str, object] = {
        'source': source,
        'mode': mode,
        'lane_count': len(group_ids),
        'lane_label': lane_label,
        'min_coverage_items': min_coverage,
        'max_block_tokens': max_block_tokens,
        'max_fact_chars': max_fact_chars,
        **fact_stats,
    }

    if len(facts) < min_coverage:
        stats['coverage_met'] = False
        stats['coverage_shortfall'] = int(min_coverage - len(facts))
        return '', stats

    if source == MATERIALIZE_SOURCE_CONTENT_VOICE_STYLE:
        header = f'Live voice-style signals (mode={mode}, lanes={lane_label})'
    elif source == MATERIALIZE_SOURCE_CONTENT_WRITING_SAMPLES:
        header = f'Live writing-sample signals (mode={mode}, lanes={lane_label})'
    elif source == MATERIALIZE_SOURCE_CONTENT_LONG_FORM_ARTIFACTS:
        header = f'Live long-form artifact signals (mode={mode}, lanes={lane_label})'
    else:
        stats['coverage_met'] = False
        stats['error'] = 'unsupported_source'
        return '', stats

    bullets = [f'- {fact}' for fact in facts]
    block = '\n'.join([header, *bullets]).strip()
    capped_block, block_truncated, block_cap_chars = _cap_block_by_token_budget(block, max_block_tokens)

    stats['coverage_met'] = True
    stats['block_chars_raw'] = len(block)
    stats['block_chars_final'] = len(capped_block)
    stats['block_cap_chars'] = block_cap_chars
    stats['block_truncated'] = bool(block_truncated)

    return capped_block, stats


def materialize_with_stats(
    source: str,
    pack_yaml_path: str,
    *,
    repo_root: Path,
    mode: str = 'default',
    max_items: int = MATERIALIZE_DEFAULT_MAX_ITEMS,
    query: str = '',
    group_ids: list[str] | None = None,
    materialization: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    normalized_source = source.strip().lower()
    effective_groups = [gid for gid in (group_ids or []) if isinstance(gid, str) and gid.strip()]

    stats: dict[str, object] = {
        'source': normalized_source,
        'mode': mode,
        'requested_max_items': max_items,
        'requested_group_count': len(effective_groups),
        'used_dynamic': False,
        'fallback': 'none',
    }

    if normalized_source in {
        MATERIALIZE_SOURCE_CONTENT_VOICE_STYLE,
        MATERIALIZE_SOURCE_CONTENT_WRITING_SAMPLES,
        MATERIALIZE_SOURCE_CONTENT_LONG_FORM_ARTIFACTS,
    }:
        try:
            dynamic, dynamic_stats = _materialize_content_pack(
                source=normalized_source,
                query=query,
                group_ids=effective_groups,
                mode=mode,
                max_items=max_items,
                materialization=materialization,
            )
            stats['dynamic'] = dynamic_stats
        except Exception as exc:
            stats['dynamic_error'] = f'{type(exc).__name__}:{str(exc).strip()}'
            dynamic = ''
        if dynamic:
            stats['used_dynamic'] = True
            stats['dynamic_chars'] = len(dynamic)
            return dynamic, stats

    static = _load_yaml_domain_context(repo_root=repo_root, pack_yaml_path=pack_yaml_path)
    if static:
        stats['fallback'] = 'domain_context'
        stats['fallback_chars'] = len(static)
        return static, stats
    # Return empty string so callers can gracefully fall back to excerpt/query
    # handling instead of injecting debug NOTE text into model context.
    return '', stats


def materialize(
    source: str,
    pack_yaml_path: str,
    *,
    repo_root: Path,
    mode: str = 'default',
    max_items: int = MATERIALIZE_DEFAULT_MAX_ITEMS,
    query: str = '',
    group_ids: list[str] | None = None,
    materialization: dict[str, object] | None = None,
) -> str:
    content, _stats = materialize_with_stats(
        source,
        pack_yaml_path,
        repo_root=repo_root,
        mode=mode,
        max_items=max_items,
        query=query,
        group_ids=group_ids,
        materialization=materialization,
    )
    return content


def _build_selected_pack(
    *,
    pack_id: str,
    profile: dict[str, object],
    pack_entry: dict[str, object],
    repo_root: Path,
    query: str,
    task: str,
    materialize_requested: bool,
    scope: str,
    decision_path: list[str],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    pack_scope = _ensure_non_empty_string(pack_entry['scope'], context='pack.scope')
    required = bool(pack_entry.get('required', True))

    if ALLOWED_SCOPE[pack_scope] > ALLOWED_SCOPE[scope]:
        return (
            None,
            {
                'pack_id': pack_id,
                'required': required,
                'reason_code': 'scope_exceeded',
                'reason': f'pack scope {pack_scope} exceeds requested scope {scope}',
            },
        )

    mode = _resolve_pack_mode(profile, pack_id)
    retrieval_cfg = pack_entry.get('retrieval') if isinstance(pack_entry.get('retrieval'), dict) else None
    profile_chatgpt_mode = _ensure_non_empty_string(profile.get('chatgpt_mode', 'scoped'), context='profile.chatgpt_mode')

    group_ids, chatgpt_included, chatgpt_group = _resolve_group_ids(
        retrieval_cfg=retrieval_cfg,
        mode=mode,
        chatgpt_mode=profile_chatgpt_mode,
    )

    query_abs = (repo_root / query).resolve()
    if not _is_relative_to(query_abs, repo_root.resolve()):
        raise ValueError(f'Pack query escapes repo root: {query}')

    materialized_excerpt = ''
    materialized_items = 0
    materialized_sources: list[str] = []
    materialization_stats: dict[str, object] = {}
    content = ''

    materialization = pack_entry.get('materialization') if isinstance(pack_entry.get('materialization'), dict) else None
    mat_source = None
    if isinstance(materialization, dict):
        src = materialization.get('source')
        if isinstance(src, str):
            mat_source = src.strip()

    if materialize_requested:
        if mat_source:
            pack_yaml_path = _ensure_non_empty_string(pack_entry.get('path', ''), context=f'pack[{pack_id}].path')
            max_items = _resolve_materialize_max_items(materialization, mode) if materialization else MATERIALIZE_DEFAULT_MAX_ITEMS
            try:
                content, materialization_stats = materialize_with_stats(
                    mat_source,
                    pack_yaml_path=pack_yaml_path,
                    repo_root=repo_root,
                    mode=mode,
                    max_items=max_items,
                    query=task,
                    group_ids=group_ids,
                    materialization=materialization,
                )
                content = content.strip()
            except Exception as exc:
                decision_path.append(
                    f'pack:{pack_id}:materialize_failed:{type(exc).__name__}:{str(exc).strip()}'
                )
            else:
                if content:
                    decision_path.append(f'pack:{pack_id}:materialized_content')
                    if materialized_items == 0:
                        materialized_items = 1
                if materialization_stats:
                    dynamic_stats = materialization_stats.get('dynamic')
                    if isinstance(dynamic_stats, dict):
                        decision_path.append(
                            f"pack:{pack_id}:materialize_stats "
                            f"received={dynamic_stats.get('facts_received', 0)} "
                            f"selected={dynamic_stats.get('facts_selected', 0)} "
                            f"fact_trunc={dynamic_stats.get('fact_char_truncations', 0)} "
                            f"block_trunc={1 if dynamic_stats.get('block_truncated') else 0}"
                        )

        if pack_id == 'engineering_learnings' or mat_source == 'engineering_loops_latest':
            materialized_excerpt, materialized_items, materialized_sources = _materialize_engineering(
                repo_root=repo_root,
                pack_entry=pack_entry,
                mode=mode,
            )
            decision_path.append(
                f'pack:{pack_id}:materialized_engineering items={materialized_items} sources={len(materialized_sources)}'
            )
        else:
            materialized_excerpt = _read_pack_excerpt(query_abs)
            materialized_items = 1
            materialized_sources = [query]
            decision_path.append(f'pack:{pack_id}:materialized_file chars={len(materialized_excerpt)}')

    selected = {
        'pack_id': pack_id,
        'query': query,
        'scope': pack_scope,
        'required': required,
        'mode': mode,
        'group_ids': group_ids,
        'chatgpt_mode': profile_chatgpt_mode,
        'chatgpt_lane_included': chatgpt_included,
        'chatgpt_group_id': chatgpt_group,
        'materialized_items': materialized_items,
        'materialized_sources': materialized_sources,
        'materialized_excerpt': materialized_excerpt,
        'materialization_stats': materialization_stats,
    }
    if content:
        selected['content'] = content
    return (selected, None)


def _build_injection_text(profile: dict[str, object], selected_packs: list[dict[str, object]]) -> str:
    header = _ensure_non_empty_string(profile.get('injection_text', ''), context='profile.injection_text')
    lines = [header]

    for pack in selected_packs:
        pack_id = _ensure_non_empty_string(pack.get('pack_id'), context='selected.pack_id')
        mode = _ensure_non_empty_string(pack.get('mode', 'default'), context=f'{pack_id}.mode')
        groups = pack.get('group_ids') if isinstance(pack.get('group_ids'), list) else []
        group_text = ', '.join(str(g) for g in groups) if groups else '(none declared)'
        lines.append('')
        lines.append(f'[{pack_id}] mode={mode} groups={group_text}')

        content = pack.get('content')
        if isinstance(content, str) and content.strip():
            lines.append(content.strip())
            continue

        excerpt = pack.get('materialized_excerpt')
        if isinstance(excerpt, str) and excerpt.strip():
            lines.append(excerpt.strip())
        else:
            query = _ensure_non_empty_string(pack.get('query', ''), context=f'{pack_id}.query')
            lines.append(f'query={query}')

    return '\n'.join(lines).strip()


def _canonical_index_spec() -> dict[str, object]:
    return {
        'schema_version': INDEX_SCHEMA_VERSION,
        'bm25': {
            'schema_version': INDEX_BM25_SCHEMA_VERSION,
            'normalization': QUERY_TEXT_NORMALIZATION_V1,
        },
        'vector': {
            'schema_version': INDEX_VECTOR_SCHEMA_VERSION,
            'embedding_dim': VECTOR_EMBEDDING_DIM,
            'normalization': 'l2',
        },
    }


def _validate_index_schema(*, pack_id: str, retrieval_cfg: dict[str, object] | None) -> None:
    if not isinstance(retrieval_cfg, dict):
        return

    index_cfg = retrieval_cfg.get('index') if isinstance(retrieval_cfg.get('index'), dict) else None
    if index_cfg is None:
        return

    schema_version = index_cfg.get('schema_version')
    bm25_schema = index_cfg.get('bm25_schema_version')
    vector_schema = index_cfg.get('vector_schema_version')
    vector_dim = index_cfg.get('vector_dim')

    bm25_nested = index_cfg.get('bm25') if isinstance(index_cfg.get('bm25'), dict) else None
    if bm25_schema is None and isinstance(bm25_nested, dict):
        bm25_schema = bm25_nested.get('schema_version')

    vector_nested = index_cfg.get('vector') if isinstance(index_cfg.get('vector'), dict) else None
    if vector_schema is None and isinstance(vector_nested, dict):
        vector_schema = vector_nested.get('schema_version')
    if vector_dim is None and isinstance(vector_nested, dict):
        vector_dim = vector_nested.get('embedding_dim')

    checks = (
        ('schema_version', schema_version, INDEX_SCHEMA_VERSION),
        ('bm25_schema_version', bm25_schema, INDEX_BM25_SCHEMA_VERSION),
        ('vector_schema_version', vector_schema, INDEX_VECTOR_SCHEMA_VERSION),
        ('vector_dim', vector_dim, VECTOR_EMBEDDING_DIM),
    )
    for field_name, actual, expected in checks:
        if actual is None:
            continue
        if not isinstance(actual, int) or actual != expected:
            raise OMIndexMismatchError(
                f'pack_id={pack_id} {field_name} mismatch expected={expected} actual={actual!r}'
            )


def _bootstrap_or_validate_indexes(
    *,
    selected_packs: list[dict[str, object]],
    registry: dict[str, dict[str, object]],
) -> dict[str, object]:
    validated_pack_ids: list[str] = []
    for pack in selected_packs:
        pack_id = _ensure_non_empty_string(pack.get('pack_id'), context='selected.pack_id')
        registry_entry = registry.get(pack_id)
        if registry_entry is None:
            raise ValueError(f'pack_id not found in registry during index validation: {pack_id}')
        retrieval_cfg = registry_entry.get('retrieval') if isinstance(registry_entry.get('retrieval'), dict) else None
        _validate_index_schema(pack_id=pack_id, retrieval_cfg=retrieval_cfg)
        validated_pack_ids.append(pack_id)

    canonical = _canonical_index_spec()
    canonical_bm25 = _ensure_dict(canonical.get('bm25'), context='index.canonical.bm25')
    canonical_vector = _ensure_dict(canonical.get('vector'), context='index.canonical.vector')
    return {
        'status': 'ok',
        'validated_pack_ids': sorted(validated_pack_ids),
        'canonical': canonical,
        'bm25': {
            'status': 'ready',
            'schema_version': canonical_bm25['schema_version'],
        },
        'vector': {
            'status': 'ready',
            'schema_version': canonical_vector['schema_version'],
            'embedding_dim': canonical_vector['embedding_dim'],
        },
    }


def _term_freq(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _bm25_scores(*, query_tokens: list[str], docs: dict[str, list[str]]) -> dict[str, float]:
    if not docs:
        return {}

    doc_lengths = {pack_id: len(tokens) for pack_id, tokens in docs.items()}
    avg_doc_len = sum(doc_lengths.values()) / max(1, len(doc_lengths))

    doc_freq: dict[str, int] = {}
    for tokens in docs.values():
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    query_counts = _term_freq(query_tokens)
    n_docs = len(docs)
    k1 = 1.5
    b = 0.75

    out: dict[str, float] = {}
    for pack_id, tokens in docs.items():
        tf = _term_freq(tokens)
        score = 0.0
        doc_len = max(1, len(tokens))
        for term, qf in query_counts.items():
            df = doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
            term_tf = tf.get(term, 0)
            if term_tf == 0:
                continue
            numer = term_tf * (k1 + 1.0)
            denom = term_tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
            score += idf * (numer / denom) * (1.0 + 0.25 * max(0, qf - 1))
        out[pack_id] = score

    return out


def _rank_from_scores(scores: dict[str, float], *, pack_ids: list[str]) -> dict[str, int]:
    ordered = sorted(pack_ids, key=lambda pid: (-float(scores.get(pid, 0.0)), pid))
    return {pack_id: idx + 1 for idx, pack_id in enumerate(ordered)}


def _deterministic_embedding(text: str, *, dim: int = VECTOR_EMBEDDING_DIM) -> list[float]:
    tokens = _tokenize_for_rank(text)
    if not tokens:
        return [0.0] * dim

    vector = [0.0] * dim
    for token in tokens:
        digest = hashlib.sha256(token.encode('utf-8')).digest()
        seed = int.from_bytes(digest[:2], byteorder='big', signed=False)
        for i, byte in enumerate(digest):
            idx = (seed + i) % dim
            vector[idx] += byte / 255.0

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def _generate_query_embedding(normalized_query: str) -> list[float]:
    if os.environ.get('OM_VECTOR_EMBEDDING_FORCE_FAIL') == '1':
        raise RuntimeError('forced embedding failure (OM_VECTOR_EMBEDDING_FORCE_FAIL=1)')
    return _deterministic_embedding(normalized_query)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=False))


def _rank_selected_packs(
    *,
    selected_packs: list[dict[str, object]],
    normalized_query: str,
    query_hash: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not selected_packs:
        return (selected_packs, [])

    docs: dict[str, list[str]] = {}
    pack_ids: list[str] = []
    for pack in selected_packs:
        pack_id = _ensure_non_empty_string(pack.get('pack_id'), context='selected.pack_id')
        query = _ensure_non_empty_string(pack.get('query', ''), context=f'{pack_id}.query')
        docs[pack_id] = _tokenize_for_rank(f'{pack_id} {query}')
        pack_ids.append(pack_id)

    query_tokens = _tokenize_for_rank(normalized_query)
    bm25_scores = _bm25_scores(query_tokens=query_tokens, docs=docs)
    bm25_ranks = _rank_from_scores(bm25_scores, pack_ids=pack_ids)

    vector_errors: list[dict[str, object]] = []
    vector_scores: dict[str, float]
    vector_ranks: dict[str, int]

    try:
        query_embedding = _generate_query_embedding(normalized_query)
        vector_scores = {
            pack_id: _cosine_similarity(query_embedding, _deterministic_embedding(' '.join(docs[pack_id])))
            for pack_id in pack_ids
        }
        vector_ranks = _rank_from_scores(vector_scores, pack_ids=pack_ids)
    except Exception as exc:
        vector_errors.append(
            {
                'event': 'OM_VECTOR_QUERY_EMBEDDING_FAILED',
                'query_hash': query_hash,
                'error_message': str(exc),
                'timestamp': _utc_timestamp(),
            }
        )
        vector_scores = dict(bm25_scores)
        vector_ranks = dict(bm25_ranks)

    enriched: list[dict[str, object]] = []
    for pack in selected_packs:
        pack_id = _ensure_non_empty_string(pack.get('pack_id'), context='selected.pack_id')
        rank_bm25 = int(bm25_ranks[pack_id])
        rank_vector = int(vector_ranks[pack_id])
        fusion_rrf = (1.0 / (RRF_K + rank_bm25)) + (1.0 / (RRF_K + rank_vector))

        updated = dict(pack)
        updated['rank_bm25'] = rank_bm25
        updated['rank_vector'] = rank_vector
        updated['score_rrf'] = round(fusion_rrf, 12)
        updated['score_bm25'] = round(float(bm25_scores.get(pack_id, 0.0)), 12)
        updated['score_vector'] = round(float(vector_scores.get(pack_id, 0.0)), 12)
        enriched.append(updated)

    enriched.sort(
        key=lambda item: (
            -float(item.get('score_rrf', 0.0)),
            int(item.get('rank_bm25', 0)),
            int(item.get('rank_vector', 0)),
            _ensure_non_empty_string(item.get('pack_id'), context='selected.pack_id'),
        )
    )

    return enriched, vector_errors


def _validate_alignment(
    registry: dict[str, dict[str, object]],
    profiles: list[dict[str, object]],
) -> None:
    referenced = {pack_id for profile in profiles for pack_id in profile['pack_ids']}
    unused = sorted(set(registry) - set(referenced))
    if unused:
        raise ValueError(f'dangling pack keys in registry: {unused}')


def _validate_plan(plan: dict[str, object], *, repo_root: Path) -> None:
    for key in REQUIRED_PLAN_KEYS:
        if key not in plan:
            raise ValueError(f'router plan missing key: {key}')

    for key in ('consumer', 'workflow_id', 'step_id', 'scope', 'task', 'injection_text'):
        _ensure_non_empty_string(plan[key], context=f'plan.{key}')

    _ensure_int(plan['schema_version'], context='plan.schema_version', min_value=1)

    packs = _ensure_list(plan['packs'], context='plan.packs')
    selected = _ensure_list(plan['selected_packs'], context='plan.selected_packs')
    dropped = _ensure_list(plan['dropped_packs'], context='plan.dropped_packs')
    _ensure_list(plan['decision_path'], context='plan.decision_path')
    _ensure_dict(plan['budget_summary'], context='plan.budget_summary')

    for index, item in enumerate(packs):
        item_dict = _ensure_dict(item, context=f'plan.packs[{index}]')
        for pack_key in ('pack_id', 'query'):
            item_dict[pack_key] = _ensure_non_empty_string(
                item_dict.get(pack_key),
                context=f'plan.packs[{index}].{pack_key}',
            )

        query_raw = _ensure_non_empty_string(item_dict['query'], context=f'plan.packs[{index}].query')
        query_path = (repo_root / query_raw).resolve()
        if not _is_relative_to(query_path, repo_root.resolve()):
            raise ValueError(f'plan.packs[{index}].query escapes repo root: {query_raw}')
        if not query_path.exists():
            raise ValueError(f'plan.packs[{index}].query does not exist: {query_path}')

    normalized_query = _ensure_non_empty_string(plan.get('normalized_query'), context='plan.normalized_query')
    query_hash = _ensure_non_empty_string(plan.get('query_hash'), context='plan.query_hash')
    if query_hash != _query_hash(normalized_query):
        raise ValueError('plan.query_hash does not match plan.normalized_query')

    _ensure_dict(plan.get('index_health'), context='plan.index_health')
    vector_errors = _ensure_list(plan.get('vector_errors', []), context='plan.vector_errors')
    for index, event in enumerate(vector_errors):
        event_dict = _ensure_dict(event, context=f'plan.vector_errors[{index}]')
        _ensure_non_empty_string(event_dict.get('event'), context=f'plan.vector_errors[{index}].event')
        _ensure_non_empty_string(event_dict.get('query_hash'), context=f'plan.vector_errors[{index}].query_hash')
        _ensure_non_empty_string(event_dict.get('error_message'), context=f'plan.vector_errors[{index}].error_message')
        _ensure_non_empty_string(event_dict.get('timestamp'), context=f'plan.vector_errors[{index}].timestamp')

    for index, item in enumerate(selected):
        item_dict = _ensure_dict(item, context=f'plan.selected_packs[{index}]')
        _ensure_non_empty_string(item_dict.get('pack_id'), context=f'plan.selected_packs[{index}].pack_id')
        _ensure_int(item_dict.get('rank_bm25'), context=f'plan.selected_packs[{index}].rank_bm25', min_value=1)
        _ensure_int(item_dict.get('rank_vector'), context=f'plan.selected_packs[{index}].rank_vector', min_value=1)

    for index, item in enumerate(dropped):
        item_dict = _ensure_dict(item, context=f'plan.dropped_packs[{index}]')
        _ensure_non_empty_string(item_dict.get('pack_id'), context=f'plan.dropped_packs[{index}].pack_id')
        _ensure_non_empty_string(item_dict.get('reason_code'), context=f'plan.dropped_packs[{index}].reason_code')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Route workflow execution through runtime packs.')
    parser.add_argument('--consumer', required=True)
    parser.add_argument('--workflow-id', required=True)
    parser.add_argument('--step-id', required=True)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--task', required=True)
    parser.add_argument('--scope', default=None, help='Optional run scope override: public|group-safe|private')
    parser.add_argument('--materialize', action='store_true', help='Materialize pack excerpts for runtime injection')
    parser.add_argument('--injection-text', default='')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--out')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve()

    try:
        registry_path = _resolve_config_path(repo_root, DEFAULT_REGISTRY_CANDIDATES, 'registry')
        profiles_path = _resolve_config_path(repo_root, DEFAULT_PROFILES_CANDIDATES, 'profiles')
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    try:
        registry = _load_registry(registry_path)
        profiles = _load_profiles(profiles_path)
        _validate_alignment(registry, profiles)
        forced_tier_c_fallback, startup_warnings = _validate_tier_c_profile_pins(profiles)

        consumer = args.consumer
        workflow_id = args.workflow_id
        step_id = args.step_id

        profile = _select_profile(
            profiles,
            consumer=consumer,
            workflow_id=workflow_id,
            step_id=step_id,
        )

        profile_scope = _ensure_non_empty_string(profile['scope'], context='profile.scope')
        run_scope = _normalize_scope(args.scope) if args.scope else profile_scope

        # Run scope may narrow profile scope, but cannot widen it.
        if ALLOWED_SCOPE[run_scope] > ALLOWED_SCOPE[profile_scope]:
            raise ValueError(
                f'run scope {run_scope} exceeds profile scope {profile_scope} for consumer profile'
            )

        tier_c_fixed_tokens, tier_c_warnings = _resolve_tier_c_fixed_tokens(
            profile,
            consumer=consumer,
            forced_fallback=forced_tier_c_fallback,
        )
        output_reserve_tokens = _resolve_output_reserve_tokens(profile)
        model_context_limit = profile.get('model_context_limit')
        budget_warnings = [*startup_warnings, *tier_c_warnings]

        if (
            isinstance(model_context_limit, int)
            and model_context_limit > 0
            and tier_c_fixed_tokens > int(model_context_limit * 0.40)
        ):
            budget_warnings.append(
                _build_warning(
                    'TIER_C_OVERSIZED',
                    consumer=consumer,
                    tier_c_fixed_tokens=tier_c_fixed_tokens,
                    model_context_limit=model_context_limit,
                )
            )

        selected_pack_ids = _ensure_string_list(profile['pack_ids'], context='profile.pack_ids')
        seen: set[str] = set()
        selected_packs: list[dict[str, object]] = []
        dropped_packs: list[dict[str, object]] = []
        decision_path: list[str] = [
            f'consumer={consumer}',
            f'workflow_id={workflow_id}',
            f'step_id={step_id}',
            f'profile_scope={profile_scope}',
            f'run_scope={run_scope}',
            f'materialize={bool(args.materialize)}',
        ]

        for pack_id in selected_pack_ids:
            if pack_id in seen:
                raise ValueError(f'duplicate pack_id in profile.pack_ids: {pack_id}')
            seen.add(pack_id)

            if pack_id not in registry:
                raise ValueError(f'pack_id not found in registry: {pack_id}')

            pack_entry = registry[pack_id]
            pack_scope = _ensure_non_empty_string(pack_entry['scope'], context='pack.scope')
            _validate_pack_scope(profile_scope, pack_scope)

            query = _build_query(profile, pack_entry, repo_path=repo_root)
            selected, dropped = _build_selected_pack(
                pack_id=pack_id,
                profile=profile,
                pack_entry=pack_entry,
                repo_root=repo_root,
                query=query,
                task=args.task,
                materialize_requested=bool(args.materialize),
                scope=run_scope,
                decision_path=decision_path,
            )

            if selected is not None:
                selected_packs.append(selected)
                decision_path.append(f'pack:{pack_id}:selected')
            elif dropped is not None:
                dropped_packs.append(dropped)
                decision_path.append(f'pack:{pack_id}:dropped:{dropped.get("reason_code")}')

        dropped_packs.sort(key=lambda item: str(item.get('pack_id', '')))

        normalized_query = _normalize_query_text(args.task)
        query_hash = _query_hash(normalized_query)
        decision_path.append(f'query_normalization={QUERY_TEXT_NORMALIZATION_V1}')

        index_health = _bootstrap_or_validate_indexes(selected_packs=selected_packs, registry=registry)
        decision_path.append('index_health=ok')

        selected_packs, vector_errors = _rank_selected_packs(
            selected_packs=selected_packs,
            normalized_query=normalized_query,
            query_hash=query_hash,
        )
        decision_path.append(f'ranking=rrf(k={RRF_K})')
        if vector_errors:
            decision_path.append('vector_errors=present')

        plan: dict[str, object] = {
            'consumer': consumer,
            'workflow_id': workflow_id,
            'step_id': step_id,
            'scope': run_scope,
            'schema_version': _ensure_int(profile['schema_version'], context='profile.schema_version', min_value=1),
            'task': _ensure_non_empty_string(profile.get('task', ''), context='profile.task'),
            'normalized_query': normalized_query,
            'query_hash': query_hash,
            'query_normalization_contract': QUERY_TEXT_NORMALIZATION_V1,
            'index_health': index_health,
            'vector_errors': vector_errors,
            'injection_text': '',
            'packs': [
                {
                    'pack_id': _ensure_non_empty_string(pack['pack_id'], context='selected.pack_id'),
                    'query': _ensure_non_empty_string(pack['query'], context='selected.query'),
                    **(
                        {
                            'content': _ensure_non_empty_string(
                                pack.get('content', ''),
                                context='selected.content',
                            )
                        }
                        if isinstance(pack.get('content'), str) and str(pack.get('content')).strip()
                        else {}
                    ),
                }
                for pack in selected_packs
            ],
            'selected_packs': selected_packs,
            'dropped_packs': dropped_packs,
            'decision_path': decision_path,
            'budget_summary': {
                'selected_count': len(selected_packs),
                'dropped_count': len(dropped_packs),
                'materialized_items_total': sum(
                    int(p.get('materialized_items', 0))
                    for p in selected_packs
                    if isinstance(p.get('materialized_items', 0), int)
                ),
                'tier_c_fixed_tokens': tier_c_fixed_tokens,
                'output_reserve_tokens': output_reserve_tokens,
                'warning_count': len(budget_warnings),
                'warnings': budget_warnings,
            },
            'config_paths': {
                'registry': str(registry_path.relative_to(repo_root)),
                'profiles': str(profiles_path.relative_to(repo_root)),
            },
        }

        plan['injection_text'] = _build_injection_text(profile, selected_packs)

        if args.validate:
            _validate_plan(plan, repo_root=repo_root)

        redacted = _redact_output_text(
            json.dumps(plan, indent=2, sort_keys=True),
            args.task,
            _ensure_string(profile['task'], context='profile.task'),
            _ensure_string(profile['injection_text'], context='profile.injection_text'),
            args.injection_text,
        )

        if args.out:
            Path(args.out).write_text(f'{redacted}\n', encoding='utf-8')
        else:
            print(redacted)

        required_drops = [d for d in dropped_packs if bool(d.get('required'))]
        return 1 if required_drops else 0

    except OMIndexMismatchError as exc:
        print(f'ERROR: OMIndexMismatchError: {exc}', file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

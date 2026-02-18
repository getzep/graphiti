from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from string import Template
from typing import Any


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
    'injection_text',
    'packs',
    'selected_packs',
    'dropped_packs',
    'decision_path',
    'budget_summary',
)


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

        profiles.append(entry)

    return profiles


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

        if chatgpt_mode == 'global' and allow_global:
            include_chatgpt = chatgpt_group is not None
        elif chatgpt_mode == 'scoped' and allow_scoped:
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


def _build_selected_pack(
    *,
    pack_id: str,
    profile: dict[str, object],
    pack_entry: dict[str, object],
    repo_root: Path,
    query: str,
    materialize: bool,
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

    if materialize:
        materialization = pack_entry.get('materialization')
        mat_source = None
        if isinstance(materialization, dict):
            src = materialization.get('source')
            if isinstance(src, str):
                mat_source = src.strip()

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
    }
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

        excerpt = pack.get('materialized_excerpt')
        if isinstance(excerpt, str) and excerpt.strip():
            lines.append(excerpt.strip())
        else:
            query = _ensure_non_empty_string(pack.get('query', ''), context=f'{pack_id}.query')
            lines.append(f'query={query}')

    return '\n'.join(lines).strip()


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

    for index, item in enumerate(selected):
        item_dict = _ensure_dict(item, context=f'plan.selected_packs[{index}]')
        _ensure_non_empty_string(item_dict.get('pack_id'), context=f'plan.selected_packs[{index}].pack_id')

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
                materialize=bool(args.materialize),
                scope=run_scope,
                decision_path=decision_path,
            )

            if selected is not None:
                selected_packs.append(selected)
                decision_path.append(f'pack:{pack_id}:selected')
            elif dropped is not None:
                dropped_packs.append(dropped)
                decision_path.append(f'pack:{pack_id}:dropped:{dropped.get("reason_code")}')

        selected_packs.sort(key=lambda item: str(item.get('pack_id', '')))
        dropped_packs.sort(key=lambda item: str(item.get('pack_id', '')))

        plan: dict[str, object] = {
            'consumer': consumer,
            'workflow_id': workflow_id,
            'step_id': step_id,
            'scope': run_scope,
            'schema_version': _ensure_int(profile['schema_version'], context='profile.schema_version', min_value=1),
            'task': _ensure_non_empty_string(profile.get('task', ''), context='profile.task'),
            'injection_text': '',
            'packs': [
                {
                    'pack_id': _ensure_non_empty_string(pack['pack_id'], context='selected.pack_id'),
                    'query': _ensure_non_empty_string(pack['query'], context='selected.query'),
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

    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

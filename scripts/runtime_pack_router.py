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

DEFAULT_REGISTRY = 'config/runtime_pack_registry.yaml'
DEFAULT_PROFILES = 'config/runtime_consumer_profiles.yaml'
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
)


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
    except json.JSONDecodeError as exc:
        try:
            import yaml
        except ModuleNotFoundError as fallback:
            raise ValueError(
                f'{path} must be JSON (or install PyYAML to parse YAML)'
            ) from fallback

        payload = yaml.safe_load(raw)  # type: ignore[attr-defined]

    return _ensure_dict(payload, context=str(path))


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
        scope = _ensure_non_empty_string(entry['scope'], context=f'{path}.packs[{index}].scope')
        if scope not in ALLOWED_SCOPE:
            raise ValueError(f'{path}.packs[{index}].scope must be one of: {sorted(ALLOWED_SCOPE)}')

        query_template = _ensure_non_empty_string(
            entry.get('query_template', '${path}'),
            context=f'{path}.packs[{index}].query_template',
        )

        registry[pack_id] = {
            'pack_id': pack_id,
            'path': path_value,
            'scope': scope,
            'query_template': query_template,
        }

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

        entry['consumer'] = _ensure_non_empty_string(
            entry['consumer'], context=f'{path}.profiles[{index}].consumer'
        )
        entry['workflow_id'] = _ensure_non_empty_string(
            entry['workflow_id'], context=f'{path}.profiles[{index}].workflow_id'
        )
        entry['step_id'] = _ensure_non_empty_string(
            entry['step_id'], context=f'{path}.profiles[{index}].step_id'
        )
        entry['scope'] = _ensure_non_empty_string(
            entry['scope'], context=f'{path}.profiles[{index}].scope'
        )
        if entry['scope'] not in ALLOWED_SCOPE:
            raise ValueError(
                f'{path}.profiles[{index}].scope must be one of: {sorted(ALLOWED_SCOPE)}'
            )

        _ensure_int(entry['schema_version'], context=f'{path}.profiles[{index}].schema_version', min_value=1)

        # P0/P1-safe type checks to keep behavior explicit.
        entry['task'] = _ensure_string(
            entry['task'], context=f'{path}.profiles[{index}].task'
        )
        entry['injection_text'] = _ensure_string(
            entry['injection_text'], context=f'{path}.profiles[{index}].injection_text'
        )
        entry['pack_ids'] = _ensure_string_list(
            entry['pack_ids'], context=f'{path}.profiles[{index}].pack_ids'
        )

        profiles.append(entry)

    return profiles


def _build_query(
    profile: dict[str, object],
    registry_entry: dict[str, object],
    *,
    repo_path: Path,
) -> str:
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
        raise ValueError(
            f'query_template references unknown key {missing!r}'
        ) from exc

    query_path = repo_path / query
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
            f'pack scope {pack_scope} exceeds profile scope {profile_scope} '
            f'for consumer profile'
        )


def _materialize_packs(
    profile: dict[str, object],
    registry: dict[str, dict[str, object]],
    *,
    repo_root: Path,
) -> list[dict[str, str]]:
    profile_scope = _ensure_non_empty_string(profile['scope'], context='profile.scope')
    selected_pack_ids = _ensure_string_list(profile['pack_ids'], context='profile.pack_ids')
    seen: set[str] = set()
    selected: list[dict[str, str]] = []

    for pack_id in selected_pack_ids:
        if pack_id in seen:
            raise ValueError(f'duplicate pack_id in profile.pack_ids: {pack_id}')
        seen.add(pack_id)

        if pack_id not in registry:
            raise ValueError(f'pack_id not found in registry: {pack_id}')

        pack_entry = registry[pack_id]
        _validate_pack_scope(profile_scope, _ensure_non_empty_string(pack_entry['scope'], context='pack.scope'))

        query = _build_query(profile, pack_entry, repo_path=repo_root)
        selected.append(
            {
                'pack_id': pack_id,
                'query': query,
            },
        )

    selected.sort(key=lambda item: item['pack_id'])
    return selected


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
    for index, item in enumerate(packs):
        item_dict = _ensure_dict(item, context=f'plan.packs[{index}]')
        for pack_key in ('pack_id', 'query'):
            item_dict[pack_key] = _ensure_non_empty_string(
                item_dict.get(pack_key),
                context=f'plan.packs[{index}].{pack_key}',
            )

        query_path = repo_root / _ensure_non_empty_string(item_dict['query'], context=f'plan.packs[{index}].query')
        if not query_path.exists():
            raise ValueError(f'plan.packs[{index}].query does not exist: {query_path}')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Route workflow execution through runtime packs.')
    parser.add_argument('--consumer', required=True)
    parser.add_argument('--workflow-id', required=True)
    parser.add_argument('--step-id', required=True)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--task', required=True)
    parser.add_argument('--injection-text', default='')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--out')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve()
    registry_path = repo_root / DEFAULT_REGISTRY
    profiles_path = repo_root / DEFAULT_PROFILES

    if not registry_path.exists():
        print(f'Config file missing: {registry_path}', file=sys.stderr)
        return 1
    if not profiles_path.exists():
        print(f'Config file missing: {profiles_path}', file=sys.stderr)
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

        packs = _materialize_packs(profile, registry, repo_root=repo_root)
        plan: dict[str, object] = {
            'consumer': consumer,
            'workflow_id': workflow_id,
            'step_id': step_id,
            'scope': _ensure_non_empty_string(profile['scope'], context='profile.scope'),
            'schema_version': _ensure_int(profile['schema_version'], context='profile.schema_version', min_value=1),
            'task': _ensure_non_empty_string(profile.get('task', ''), context='profile.task'),
            'injection_text': _ensure_non_empty_string(profile.get('injection_text', ''), context='profile.injection_text'),
            'packs': packs,
        }

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

        return 0
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

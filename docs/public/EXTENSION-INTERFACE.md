# Extension Interface Contract v1

This repository supports **optional extensions** through a declarative manifest contract.
Core must remain usable with **zero extensions installed**.

## Design goals

- Explicit versioning (`api_version`) for compatibility checks.
- Declarative registration via manifest files (no hardcoded private hooks).
- Fail-safe loading: one bad extension must not break all extension discovery.
- Actionable diagnostics for CI and local debugging.

## Manifest location

Each extension lives under:

- `extensions/<extension-id>/manifest.json`

The loader discovers direct child folders of `extensions/`.

## Contract fields (v1)

Required fields:

- `name` (string; must normalize to a non-empty slug)
- `version` (string)
- `api_version` (integer, currently `1`)
- `capabilities` (non-empty list of unique strings)
- `entrypoints` (non-empty object, key -> repo-relative file path)

Optional fields:

- `description` (string)
- `command_contract`:
  - `version` (must match `api_version`)
  - `namespace` (must equal normalized extension name)
- `commands` (object: `<namespace>/<command>` -> repo-relative file path)

### `api_version` semantics

- `api_version=1` is the current supported contract.
- Unknown versions are rejected.
- Legacy manifests without `api_version` are tolerated for compatibility and emit warnings.
  New extensions should always set `api_version` explicitly.

## Minimal extension template

```json
{
  "name": "sample-extension",
  "version": "0.1.0",
  "api_version": 1,
  "description": "Example extension contract.",
  "capabilities": ["sync"],
  "entrypoints": {
    "doctor": "scripts/tool.py"
  },
  "command_contract": {
    "version": 1,
    "namespace": "sample-extension"
  },
  "commands": {
    "sample-extension/doctor": "scripts/tool.py"
  }
}
```

## Loader behavior

- Extension loading is **optional**.
- Missing `extensions/` directory is non-fatal (warning only).
- Each extension is validated independently.
- Invalid extensions are isolated and reported; valid extensions continue loading.
- Duplicate extension names (after normalization) or duplicate command IDs are rejected.

## Contract checker

Run strict checker:

```bash
python3 scripts/extension_contract_check.py --strict
```

Strict mode fails on compatibility **errors**; warnings are still printed for cleanup guidance.

## Layering rules

1. Core runtime must not depend on any specific extension.
2. Extension behavior is discovered only from manifest declarations.
3. Extension loading/validation should stay side-effect free (no import-time execution of extension code).
4. All extension paths must remain repo-relative and traversal-safe.

## Anti-patterns

- Hardcoding a specific extension name or entrypoint in core logic.
- Using private/internal hooks as registration side channels.
- Global mutable side effects during extension import/validation.
- Non-namespaced commands (for example `doctor` instead of `my-pack/doctor`).
- Unsafe paths (`../`, absolute paths, or dot-path traversal segments).

# Graphiti OpenClaw Runtime Injection Plugin (v1 Scaffold)

This plugin injects Graphiti recall and workflow pack context on every agent turn using OpenClaw's `before_agent_start` and `agent_end` hooks.

## What It Does

- Always returns `prependContext` from `before_agent_start`.
- Injects `<graphiti-context>` with recall facts when available.
- Injects `<pack-context>` when intent routing selects a pack.
- Strips injected context before capture and ingests the clean turn to Graphiti.
- Enforces group-chat scope isolation for private packs.
- Falls back gracefully on errors or timeouts.

## Installation

This repo only provides the plugin scaffold. Install via your OpenClaw runtime using a local path:

```bash
openclaw plugins install /path/to/graphiti-openclaw/plugin
```

## Configuration

Configuration is passed as JSON through `GRAPHITI_PLUGIN_CONFIG` or OpenClaw's plugin config system. Example JSON:

```json
{
  "graphitiBaseUrl": "http://localhost:8000",
  "intentRulesPath": "config/example_intent_rules.json",
  "compositionRulesPath": "config/example_pack_composition.json",
  "packRegistryPath": "config/example_pack_registry.json",
  "packRouterCommand": "python3 scripts/runtime_pack_router.py",
  "packRouterRepoRoot": ".",
  "configPathRoots": ["."],
  "debug": true
}
```

Notes:
- The scaffold loader expects JSON content.
- `packRouterCommand` is optional. If omitted, the pack registry is used directly.
- For command paths with spaces, wrap paths in quotes or use an array form:
  - `"packRouterCommand": "python3 \"scripts/pack router.py\""`
  - `"packRouterCommand": ["python3", "scripts/pack router.py"]`
- `configPathRoots` is an allowlist. Config files outside these roots are rejected. Defaults to the current working directory.

## Intent Rules

Intent rules map prompts to workflow packs using deterministic keyword routing plus entity boosts.

- `keywords` drive the initial match.
- `entityBoosts` apply additive score changes when Graphiti facts match `summaryPattern` or `factPattern`.
- `minConfidence` enforces safe defaults.
- Ties resolve to **no pack**.
- `scope` enforces group-chat isolation (`private` packs never injected in group chat).

Example: `config/example_intent_rules.json`.

## Composition Rules

Composition rules declare secondary packs to inject alongside the primary intent pack.

- `primary_intent` maps to an intent rule ID.
- `inject_additional` lists extra packs to load.
- `required: false` keeps injection optional if the pack is missing.

Example: `config/example_pack_composition.json`.

## Pack Registry

The pack registry maps pack types to pack files and scopes.

Example: `config/example_pack_registry.json`.

## Hooks

- `before_agent_start`: recall + intent routing + pack composition. Always returns `{ prependContext }`.
- `agent_end`: strips injected blocks and ingests the clean turn into Graphiti.

## Correctness Guarantees (Milestone 1)

- Safe default (no keywords → no pack).
- Deterministic routing (same input → same output).
- Ambiguity tie → no pack.
- Monotonic entity boosts (matches only add to score).
- Scope isolation in group chat for private packs.
- Graceful fallback on errors/timeouts.

## Tests

Initial correctness tests live in `plugin/tests/correctness.test.ts`.

From repo root:

```bash
node --experimental-strip-types --test plugin/tests/correctness.test.ts
node --experimental-strip-types --check plugin/index.ts
```

From `plugin/` directory:

```bash
npm run test
npm run check
```

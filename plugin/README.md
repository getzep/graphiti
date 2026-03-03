# Bicameral OpenClaw Runtime Injection Plugin (v1 Scaffold)

This plugin injects Graphiti recall and workflow pack context on every agent turn using OpenClaw hooks (`before_model_resolve`, `before_prompt_build`) with lifecycle scaffolding for optional context-map anchoring, a legacy `before_agent_start` compatibility shim, and `agent_end` capture.

## What It Does

- Uses `before_model_resolve` for deterministic model/provider overrides (optional via config).
- Uses `before_prompt_build` for Graphiti recall + intent routing + pack composition context injection.
- Includes a disabled-by-default lifecycle scaffold for optional `<context-map-anchor>` injection.
- Keeps `before_agent_start` as a compatibility shim for older runtime paths.
- Injects `<graphiti-context>` with recall facts when available.
- Injects `<pack-context>` when intent routing selects a pack.
- Strips injected context before capture and ingests the clean turn to Graphiti.
- Enforces group-chat scope isolation for private packs.
- Falls back gracefully on errors or timeouts.

## Installation

This repo only provides the plugin scaffold. Install via your OpenClaw runtime using a local path:

```bash
openclaw plugins install /path/to/bicameral/plugin
```

## Configuration

> **⚠️ IMPORTANT: Do not add an `"id"` key to this configuration.** OpenClaw's plugin config schema is extremely strict (`additionalProperties: false`). Adding an `id` key under `.plugins.entries.<name>` in your `openclaw.json` will cause the gateway to crash hard and loop.

Configuration is passed as JSON through `BICAMERAL_PLUGIN_CONFIG` (legacy fallback: `GRAPHITI_PLUGIN_CONFIG`) or OpenClaw's plugin config system. Example JSON:

```json
{
  "graphitiBaseUrl": "http://localhost:8000",
  "allowModelRoutingOverride": true,
  "providerOverride": "openai",
  "modelOverride": "gpt-5.2",
  "allowedProviderOverrides": ["openai", "anthropic"],
  "allowedModelOverrides": ["gpt-5.2", "claude-sonnet-4-6"],
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

### Single-tenant multi-lane recall (`memoryGroupIds`)

For deployments with a single logical tenant (e.g. a personal assistant), recall can fan out across multiple named Graphiti group lanes simultaneously — sessions history, observational memory, self-audit learnings, etc.

```json
{
  "singleTenant": true,
  "memoryGroupIds": [
    "s1_sessions_main",
    "s1_observational_memory",
    "learning_self_audit"
  ]
}
```

Precedence (highest to lowest):
1. `memoryGroupIds` (non-empty, `singleTenant: true`) — multi-lane fan-out
2. `memoryGroupId` (non-empty, `singleTenant: true`) — single-lane pin
3. `messageProvider.groupId` — per-provider lane
4. `sessionKey` (hashed) — per-session derived lane

Safety rules:
- **Both `memoryGroupIds` and `memoryGroupId` require `singleTenant: true`.** Without it, multi-tenant isolation applies and these fields are silently ignored.
- Duplicate entries and blank strings are stripped during normalization; insertion order is preserved.
- `memoryGroupIds` takes precedence over `memoryGroupId` when both are set.

## Context-map anchor scaffold (public, generic)

A minimal lifecycle scaffold is available for injecting a lightweight `<context-map-anchor>` block that points the runtime at your context-map artifacts.

- **Purpose:** provide an integration point for context-map-aware plugins/workflows without baking in any private paths or operator assumptions.
- **Default behavior:** disabled (`enableContextMapAnchor: false`). No anchor is injected unless explicitly enabled.
- **Lifecycle integration points:** `session_start`, `after_compaction`, `before_reset`, and `before_prompt_build`.
- **How to enable:** set `enableContextMapAnchor: true` and configure one or both file paths (`contextMapPath`, `contextMapMetaPath`), plus optional custom `contextMapAnchorText`.
- **Change detection note:** hash-change detection is handled in plugin logic by fingerprinting configured files; there is no native hook event that directly emits “context map changed”.

Example:

```json
{
  "enableContextMapAnchor": true,
  "contextMapPath": "state/context-map.md",
  "contextMapMetaPath": "state/context-map.meta.json",
  "contextMapAnchorText": "Use the current context map and metadata when planning this turn."
}
```

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

- `session_start`: lifecycle trigger for optional context-map anchor scaffold (disabled by default).
- `after_compaction`: lifecycle trigger for optional context-map anchor scaffold hash re-check.
- `before_reset`: clears context-map anchor scaffold state.
- `before_model_resolve`: deterministic model/provider override surface (`modelOverride`, `providerOverride`).
  - Secure-by-default: requires `allowModelRoutingOverride: true`.
  - Overrides must be explicitly allowlisted (`allowedProviderOverrides`, `allowedModelOverrides`).
- `before_prompt_build`: recall + intent routing + pack composition, plus optional context-map anchor injection when scaffold gates pass. Returns `{ prependContext }`.
- `before_agent_start` (legacy): compatibility shim; only delegates when a **non-empty** `messages` list is present and prompt-build injection has not already run for the same turn.
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

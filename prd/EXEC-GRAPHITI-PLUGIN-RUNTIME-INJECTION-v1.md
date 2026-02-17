# PRD: Graphiti OpenClaw Plugin — Runtime Context Injection v1

## PRD Metadata
- Type: Execution
- Kanban Task: (pending)
- Parent Epic: graphiti-openclaw publicization
- Depends On: PR #34 + PR #7 merged (hardening complete)
- Preferred Engine: Codex (TypeScript plugin) + Claude (review)
- Owned Paths (PUBLIC repo — `graphiti-openclaw`):
  - `prd/EXEC-GRAPHITI-PLUGIN-RUNTIME-INJECTION-v1.md`
  - `plugin/openclaw.plugin.json`
  - `plugin/index.ts`
  - `plugin/config.ts`
  - `plugin/hooks/recall.ts`
  - `plugin/hooks/capture.ts`
  - `plugin/hooks/pack-injector.ts`
  - `plugin/client.ts`
  - `plugin/tools/search.ts`
  - `plugin/tools/ingest.ts`
  - `plugin/commands/cli.ts`
  - `plugin/commands/slash.ts`
  - `plugin/package.json`
  - `plugin/tsconfig.json`
  - `plugin/README.md`
  - `plugin/tests/`
- Owned Paths (PRIVATE repo — `graphiti-openclaw-private`):
  - `config/runtime_pack_registry.yaml` (add new packs)
  - `config/runtime_consumer_profiles.yaml` (add new profiles)
  - `config/plugin_intent_rules.yaml` (NEW — private intent routing rules)
  - `config/pack_composition_rules.yaml` (NEW — cross-pack injection rules)
  - `workflows/dining_recs.pack.yaml` (NEW)
  - `workflows/content_tweet.pack.yaml` (NEW)
  - `workflows/content_long_form.pack.yaml` (NEW)

---

## Overview

Build an OpenClaw plugin that **guarantees** Graphiti knowledge graph retrieval, context injection, and content/workflow pack selection on every agent turn — replacing the current optional `memory_search` tool-call pattern with infrastructure-level injection that the model cannot skip.

**Core insight:** OpenClaw's Plugin SDK exposes `before_agent_start` and `agent_end` hooks that fire on every turn with full access to the user's prompt and conversation history. The `before_agent_start` handler can return `{ prependContext }` which OpenClaw mechanically prepends to the prompt before the LLM sees it. This is the same pattern used by the Supermemory plugin (`@supermemory/openclaw-supermemory`) — proven in production.

---

## Public vs Private Split

The plugin architecture follows the same public/private overlay pattern as the rest of graphiti-openclaw.

### PUBLIC (`graphiti-openclaw` repo) — Generic Framework

| Component | Purpose |
|-----------|---------|
| Plugin scaffold (`plugin/`) | TypeScript plugin: manifest, index, config schema |
| Recall hook (`plugin/hooks/recall.ts`) | Generic Graphiti query → `prependContext` injection |
| Capture hook (`plugin/hooks/capture.ts`) | Generic conversation → Graphiti ingest |
| Pack injector (`plugin/hooks/pack-injector.ts`) | Generic intent-detection → pack-routing → context-injection engine |
| Graphiti client (`plugin/client.ts`) | HTTP client for Graphiti API |
| Pack composition engine | Load composition rules, resolve cross-pack dependencies |
| Intent detection engine | Keyword matching + entity-type boost (configurable rules) |
| Example intent rules | `config/runtime_pack_registry.yaml` + `config/runtime_consumer_profiles.yaml` (example packs) |
| Tests + eval integration | Unit tests, shadow-compare integration hooks |

### PRIVATE (`graphiti-openclaw-private` repo) — Our Specific Wiring

| Component | Purpose |
|-----------|---------|
| `config/plugin_intent_rules.yaml` | Our specific intent → consumer profile mappings (dining, VC, content) |
| `config/pack_composition_rules.yaml` | Cross-pack injection rules (voice/writing → VC memo, etc.) |
| `config/runtime_pack_registry.yaml` | Real pack registry (VC + dining + content packs) |
| `config/runtime_consumer_profiles.yaml` | Real consumer profiles |
| `workflows/*.pack.yaml` | Real workflow packs (VC memo, deal brief, IC prep, dining, content) |
| Content voice/writing context packs | Yuan's voice model, writing samples, content strategy |
| Compliance rules | SEC/RIA hard gates for content workflows |

**Principle:** The public plugin knows *how* to detect intent, route to packs, compose cross-pack context, and inject it. The private overlay knows *which* intents map to *which* packs with *which* composition rules. A different user could install the public plugin and wire their own packs/intents without touching our private config.

---

## Problem Statement

Currently, Graphiti retrieval is exposed as optional tool calls (`memory_search`, `memory_get`) that the model may choose not to invoke. This means:
- The model can "forget" to search for relevant context
- Quality depends on prompt engineering ("always run memory_search first")
- No guarantee of retrieval on every turn
- Content/workflow pack injection is also model-discretionary — the agent may not load the right domain context
- Cross-pack composition (e.g., injecting voice/writing packs into VC memo flows) doesn't happen automatically

After QMD inverts to Graphiti as the primary knowledge backend, these guarantees become critical.

---

## Solution: OpenClaw Plugin with Guaranteed Hooks

### Architecture

```
User message arrives
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│  before_agent_start (GUARANTEED, every turn)              │
│                                                           │
│  1. Extract query from event.prompt                       │
│  2. Query Graphiti (semantic search)                      │
│     → On failure: inject QMD fallback note                │
│  3. Intent detection (keyword + entity-type boost)        │
│     → Match against intent rules (private config)         │
│  4. Resolve primary pack via runtime_pack_router           │
│  5. Resolve composition rules → inject dependent packs    │
│     (e.g., VC memo intent → also inject voice/writing)    │
│  6. Load all matched pack YAMLs + format as context       │
│  7. Return { prependContext }                             │
└───────────────────────────────────────────────────────────┘
    │
    ▼  prompt = <graphiti-context> + <pack-context> + original_prompt
    │
┌───────────────────────────────────────────────────────────┐
│  Agent runs (with injected context + pack knowledge)      │
│  QMD memory_search/memory_get still available as          │
│  fallback tools for targeted manual queries               │
└───────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│  agent_end (GUARANTEED)                                   │
│                                                           │
│  1. Extract conversation turn                             │
│  2. Strip <graphiti-context> + <pack-context> blocks      │
│  3. Ingest into Graphiti (async, non-blocking)            │
└───────────────────────────────────────────────────────────┘
```

### Plugin SDK Hooks Used

| Hook | Event | Direction | Purpose |
|------|-------|-----------|---------|
| `before_agent_start` | Every turn, pre-LLM | Inject | Graphiti recall + intent detection + pack composition → prepend context |
| `agent_end` | Every turn, post-LLM | Capture | Ingest conversation turn into Graphiti |
| `message_received` | Every inbound message | Observe | Optional: telemetry |

### Key Design Decisions

1. **Plugin, not hook directory.** The directory-based hooks system (`agent:bootstrap`) lacks access to the user message. The Plugin SDK's `before_agent_start` receives `event.prompt` + `event.messages` — exactly what's needed for semantic retrieval and intent detection.

2. **`prependContext` injection is mechanical.** The return value is prepended to the prompt at `attempt.ts:866`. No model discretion. No tool call to skip.

3. **QMD fallback on Graphiti failure.** `memory_search` and `memory_get` are core agent tools in `agents/tools/memory-tool.ts`, gated by `group:memory` in tool policy — **independent of the memory plugin slot**. Taking the slot does NOT remove QMD tools. Both coexist.

4. **Pack injection uses the existing `runtime_pack_router.py`.** The router, registry, and consumer profiles already exist. The plugin bridges intent detection → profile selection → pack materialization → context injection.

5. **Cross-pack composition is config-driven.** Composition rules (e.g., "when VC memo intent fires, also inject voice/writing packs") live in private config, not hardcoded in the plugin. The public engine just reads rules and resolves dependencies.

---

## Intent Detection — Guaranteeing Correct Pack Selection

Intent detection must be **fast** (adds to every-turn latency), **deterministic** (same input → same packs), and **correct** (wrong pack is worse than no pack). Here's the multi-layer approach:

### Layer 1 — Keyword Matching (deterministic, ~0ms)

Each intent rule has a `keywords` array. Case-insensitive substring match against `event.prompt`. If any keyword hits, the rule is a candidate.

**Problem:** Keywords alone have false positives. "I had dinner with the Series A founder" matches both `dining` and `vc_deal`. Multiple matches need disambiguation.

### Layer 2 — Graphiti Entity-Type Boost (semantic, from graph results)

The Graphiti search results from step 2 include entity types. These boost or suppress intent candidates:

| Entity Type | Boosts | Suppresses |
|-------------|--------|------------|
| `company`, `deal`, `round` | `vc_deal`, `vc_memo`, `vc_diligence`, `vc_ic` | `dining` |
| `restaurant`, `venue`, `bar` | `dining` | `vc_*` |
| `person` + relationship `investor` | `vc_*` | — |
| `person` + relationship `chef`/`sommelier` | `dining` | — |
| `article`, `content`, `tweet` | `content_*` | — |

### Layer 3 — Confidence Threshold + Tiebreaking

Each candidate gets a score: `keyword_hits * keyword_weight + entity_boosts * entity_weight`. Rules:
- Score < `minConfidence` (configurable, default 0.3) → no pack injected (safe default)
- Single candidate above threshold → inject that pack
- Multiple candidates above threshold → inject the **highest-scoring** one only
- Tie → inject **none** (ambiguous intent; let the agent handle it naturally)

### Layer 4 — Conversation-Context Sticky (optional)

If the previous turn already had a pack injected and the current prompt is a follow-up (e.g., "what about their Series B?"), re-inject the same pack even without fresh keyword hits. Detected via:
- Short prompt (<20 words) + previous turn had a pack → re-inject
- Explicit continuation signals ("also", "and what about", "continue")

This prevents pack context from dropping on follow-up questions in the same conversation thread.

### Verification / Eval

Intent detection accuracy is verified using:
1. **Existing shadow-compare eval cases** (`evals/cases/synthetic.workflow_pack_shadow_compare.json`) — maps queries to expected `pack_type` categories
2. **New intent-detection unit tests** with:
   - True positive cases (clear intent → correct pack)
   - True negative cases (ambiguous → no pack)
   - False positive regression tests (keyword overlap → correct disambiguation)
   - Entity-boost integration tests (graph results alter pack selection)

---

## Cross-Pack Composition

Some workflows require context from **multiple pack types**. This is driven by composition rules.

### The Problem

The VC memo workflow needs Yuan's voice/writing style to produce memos that sound like him, not generic output. The content creation epic (`EXEC-CONTENT-CREATION-EPIC-v0`) defines context packs (`content_voice_style`, `content_writing_samples`, `content_strategy`) that capture Yuan's voice profile. These should **optionally inject** into VC memo flows when writing quality matters.

Similarly, the content tweet workflow might want `dining_preferences` context if the tweet is about a restaurant experience, or `work_company_deal` context if the tweet is about a deal.

### Composition Rules (private config)

`config/pack_composition_rules.yaml`:

```yaml
schema_version: 1
rules:
  # When VC memo intent fires, also inject voice/writing packs
  - primary_intent: "vc_memo"
    inject_additional:
      - pack_type: "content_voice_style"
        mode: "formal"
        required: false  # don't block memo if voice pack unavailable
        condition: "always"
      - pack_type: "content_writing_samples"
        mode: "formal"
        required: false
        condition: "always"

  # When VC deal brief intent fires, inject voice (formal) if available
  - primary_intent: "vc_deal"
    inject_additional:
      - pack_type: "content_voice_style"
        mode: "formal"
        required: false
        condition: "always"

  # When content article intent fires, inject topic context from graph
  - primary_intent: "content_article"
    inject_additional:
      - pack_type: "content_voice_style"
        mode: "formal"
        required: true  # articles need voice
        condition: "always"
      - pack_type: "content_writing_samples"
        mode: "formal"
        required: true
        condition: "always"
      - pack_type: "content_strategy"
        required: true
        condition: "always"

  # When content tweet intent fires, inject voice (casual) + strategy
  - primary_intent: "content_tweet"
    inject_additional:
      - pack_type: "content_voice_style"
        mode: "casual"
        required: true
        condition: "always"
      - pack_type: "content_writing_samples"
        mode: "casual"
        required: true
        condition: "always"
      - pack_type: "content_strategy"
        required: true
        condition: "always"

  # Dining intent — standalone, no composition needed
  - primary_intent: "dining"
    inject_additional: []
```

### Composition Engine (public plugin code)

The pack-injector reads composition rules and for each matched primary intent:
1. Loads the primary pack (via `runtime_pack_router.py`)
2. Iterates `inject_additional` rules
3. For each additional pack: checks availability (does the pack exist? is it in scope?), loads it, appends to context
4. If `required: true` and the additional pack is unavailable → log warning, optionally skip the entire primary pack injection
5. If `required: false` and unavailable → proceed without it

### Injected Context Format (with composition)

```xml
<pack-context intent="vc_memo" primary-pack="vc_memo_drafting" scope="private">
## Active Workflow: VC Memo Drafting

[Primary pack YAML content / instructions here]

### Composition: Voice Profile (formal)
[content_voice_style pack content — Yuan's formal writing voice model]

### Composition: Writing Samples (formal)
[content_writing_samples excerpts — formal register examples]
</pack-context>
```

---

## QMD Coexistence — Why It Works

**Critical finding:** `memory_search` and `memory_get` are **core agent tools** defined in `agents/tools/memory-tool.ts`, gated by `group:memory` in tool policy. They are **not** controlled by `plugins.slots.memory`. Taking the slot with `graphiti-openclaw` does NOT remove QMD-backed tools.

**Fallback flow:**
1. Graphiti recall succeeds → rich context injected automatically
2. Graphiti recall fails (timeout/error) → inject `<graphiti-fallback>` note suggesting QMD
3. Agent can still call `memory_search` / `memory_get` as tools (QMD) for manual retrieval
4. Both paths coexist — no configuration change needed

---

## Component Specification

### 1. Recall Hook (`plugin/hooks/recall.ts`)

**Fires:** `before_agent_start` (every turn)
**Input:** `event.prompt`, `event.messages`, `ctx.sessionKey`, `ctx.messageProvider`
**Output:** `{ prependContext: string }` or `void`

1. Skip if prompt is too short
2. Query Graphiti semantic search with `event.prompt` (hard timeout: `recallTimeoutMs`)
3. On failure → log, inject QMD fallback note, return
4. Run intent detection (§Intent Detection above)
5. If intent matched → run pack injector with composition (§Cross-Pack Composition)
6. Format Graphiti results + pack context into combined `prependContext`
7. Return `{ prependContext }`

### 2. Capture Hook (`plugin/hooks/capture.ts`)

**Fires:** `agent_end` (every turn)
1. Skip if `!event.success`
2. Extract last user-assistant turn pair
3. Strip `<graphiti-context>` and `<pack-context>` blocks
4. POST to Graphiti ingest (async, non-blocking, `captureTimeoutMs`)

### 3. Pack Injector (`plugin/hooks/pack-injector.ts`)

Called from recall hook. This is the **generic engine** (public code):
1. Load intent rules from config (private overlay provides the actual rules)
2. Run intent detection (keyword + entity-type + conversation-sticky)
3. Load composition rules from config
4. Call `runtime_pack_router.py` for primary pack
5. Resolve composition dependencies → load additional packs
6. Format all packs into `<pack-context>` XML
7. Enforce scope (group chat → no private packs)

### 4. Graphiti Client (`plugin/client.ts`)

Thin HTTP client: `search()`, `getEntityFacts()`, `ingestEpisode()`, `getProfile()`. Timeout-aware, `fetch`-based.

### 5. Optional Tools

| Tool | Description |
|------|-------------|
| `graphiti_search` | Manual semantic search beyond auto-injection |
| `graphiti_ingest` | Manual ingest |

Coexist with `memory_search`/`memory_get` (QMD).

### 6. Slash Commands

| Command | Description |
|---------|-------------|
| `/recall <query>` | Manual search |
| `/graphiti status` | Connection + recall/capture/pack stats |
| `/graphiti debug` | Toggle debug logging |

---

## Rollout / QMD Inversion Strategy

| Phase | State | Memory Slot | QMD Tools | Graphiti Plugin | Pack Injection |
|-------|-------|-------------|-----------|-----------------|----------------|
| **0 (now)** | QMD primary | `memory-core` | ✅ | ❌ not installed | ❌ |
| **1 (dual-write)** | Both active | `graphiti-openclaw` | ✅ | ✅ recall + capture | ✅ |
| **2 (graphiti primary)** | Graphiti primary | `graphiti-openclaw` | ⚠️ fallback only | ✅ full | ✅ |
| **3 (QMD retired)** | Graphiti only | `graphiti-openclaw` | ❌ removed | ✅ full | ✅ |

---

## Testing Strategy

1. **Unit tests:** Mock Graphiti client, verify recall/capture hook behavior
2. **Intent detection tests:**
   - True positives (clear dining query → dining pack)
   - True negatives (ambiguous → no pack)
   - Entity-type boost (Graphiti returns company entity → VC pack wins over dining)
   - Conversation-sticky (follow-up retains pack)
3. **Composition tests:**
   - VC memo intent → also loads voice + writing packs
   - Missing optional pack → proceeds without
   - Missing required pack → skips or logs warning
4. **Scope enforcement:** Group chat sessions → no private packs injected
5. **Timeout/fallback:** Graphiti down → QMD fallback note, agent runs normally
6. **Shadow compare integration:** Existing `synthetic.workflow_pack_shadow_compare.json` eval cases validate pack selection matches expected categories
7. **Context format:** XML blocks well-formed, stripped on capture (no recursive loops)

---

## Definition of Done

### Public Plugin
- [ ] Plugin installs via `openclaw plugins install` (or local path)
- [ ] `before_agent_start` fires every turn → Graphiti context injected
- [ ] `agent_end` fires every turn → conversation ingested
- [ ] Pack injector loads intent rules + composition rules from config
- [ ] Intent detection: keyword + entity-type + conversation-sticky
- [ ] Composition engine resolves cross-pack dependencies
- [ ] Scope enforcement (no private packs in group chats)
- [ ] QMD fallback on Graphiti failure
- [ ] `/graphiti status` shows stats
- [ ] Tests pass (unit + intent + composition + scope + shadow compare)
- [ ] README documents: setup, config schema, pack authoring, QMD inversion phases

### Private Wiring
- [ ] Intent rules for dining, VC (4 workflows), content (tweet + article)
- [ ] Composition rules: voice/writing → VC memo, content article, content tweet
- [ ] New packs: `dining_recs`, `content_tweet`, `content_long_form`
- [ ] New consumer profiles for dining + content marketing
- [ ] Updated pack registry with all new pack IDs
- [ ] Content voice/writing/strategy packs (blocked on Yuan's writing samples — see Content Creation Epic)

---

## Estimated Effort

| Component | Hours |
|-----------|-------|
| Plugin scaffold + manifest | 1 |
| Recall hook + QMD fallback | 2–3 |
| Capture hook | 1–2 |
| Graphiti client | 1–2 |
| Intent detection engine (keyword + entity + sticky) | 3–4 |
| Pack injector + composition engine | 3–4 |
| Private intent/composition config | 1–2 |
| New pack definitions (dining, content) | 2–3 |
| Tools + commands | 1 |
| Tests (intent + composition + shadow compare) | 3–4 |
| Documentation | 1 |
| **Total** | **~20–26 hours** |

---

## Resolved Questions

1. **Memory slot exclusivity:** ✅ Taking the slot does NOT disable `memory_search`/`memory_get`. They're core tools in `agents/tools/memory-tool.ts`, gated by `group:memory` tool policy — independent of the plugin slot.
2. **Pack routing infrastructure:** ✅ `runtime_pack_router.py` + registry + profiles already exist. Plugin bridges intent detection to this system.
3. **Cross-pack composition:** ✅ Voice/writing packs inject into VC memo flows via config-driven composition rules. The engine is generic (public); the rules are specific (private).
4. **Graphiti API auth:** ✅ The Graphiti MCP server runs as a local HTTP server (`transport: http`, `host: 0.0.0.0`, `port: 8000`) with no auth layer on the search/ingest endpoints. It authenticates internally to Neo4j and LLM providers, but the HTTP API itself is open. Plugin hits `localhost:8000` directly. `apiKey` kept as optional config for future reverse-proxy scenarios but not required for local deployment. Source: `mcp_server/src/config/schema.py` — `ServerConfig` has no auth fields.
5. **Capture granularity:** ✅ Every-turn capture, async fire-and-forget. Graphiti's ingest pipeline handles deduplication and entity resolution internally. Batching would add buffer management complexity for no meaningful benefit. No latency impact since capture is non-blocking.
6. **Entity-type vocabulary:** ✅ Graphiti's `EntityNode` uses a `labels: list[str]` field + `attributes: dict` + `summary: str` — no fixed type enum. Labels are whatever the LLM assigns during episode ingestion. For intent boosting, key off `node.summary` text and `edge.fact` context (keyword matching against summaries), not a hardcoded entity-type map. This makes the boost layer data-driven rather than schema-dependent. Source: `graphiti_core/nodes.py:484` (`EntityNode`), `graphiti_core/search/search_config.py:121` (`SearchResults` — returns nodes with summaries and edges with facts).
7. **Pack YAML enrichment:** ✅ Yes — enrich pack YAMLs with `description`, `domain_context`, and optional `compliance_rules` fields. Current pack YAMLs are bare skeletons (name + steps only). The `domain_context` field is what the plugin actually injects as `<pack-context>` content. Without it, pack injection is a label with no guidance. Update the workflow pack schema in the public repo to support these fields; author rich content in the private overlay packs. Example enriched structure:
   ```yaml
   name: vc_memo_drafting
   workflow_id: vc_memo_drafting
   description: "VC investment memo drafting with Aleks Larsen quality framework"
   domain_context: |
     Standard memo structure: Executive Summary, Team, Market, Product,
     Competitive Landscape, Traction, GTM, Risks/Mitigants, Financing
     History, What We Have To Believe. Bind claims to evidence. Flag
     unknowns explicitly. Stage-aware expectations (seed/A/B+).
   compliance_rules: |
     No fund performance numbers. No live/trading token mentions.
     No language readable as investment advice. Draft-only.
   steps:
     - id: draft
     - id: review
   ```
   Source: legacy PRDs `EPIC-VC-MEMO-POLICY-BINDING-v1.md` (memo structure + compliance gates) and `EXEC-WORKFLOW-PACK-AUTHORING-SPEC-v0.md` (pack schema contract).
8. **Composition rule conditions:** ✅ Keep `condition: "always"` for v1. Voice/writing packs are lightweight (~200–500 tokens injected). False-positive injection is low-cost. More granular conditions (`if_entity_type_present`, `if_prompt_mentions`) add engine complexity for marginal token savings. Revisit in v2 if token budget becomes tight or injection noise is observed.
9. **Content voice/writing/strategy packs:** ✅ Blocked on Yuan's writing samples (Content Creation Epic inputs #1–#3: casual writing samples, formal/BCAP writing samples, Twitter algo/content marketing materials). The pack injection engine ships independently — voice/writing/strategy packs plug in once samples are provided. The composition rules reference these packs with `required: false` for VC workflows (proceed without voice if unavailable) and `required: true` for content workflows (articles/tweets need voice to function). Source: `EXEC-CONTENT-CREATION-EPIC-v0.md` §Blocked Inputs.

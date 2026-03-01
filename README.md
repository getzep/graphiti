# Bicameral — Dual-Brain Memory Runtime for Agents

This repository is a **production delta layer** on top of [upstream Graphiti](https://github.com/getzep/graphiti). It turns Graphiti from a graph-memory library into a **dual-brain, policy-governed memory runtime** for OpenClaw agents.

The key insight: **A single LLM-powered brain can't be trusted with your memories.** We added a second brain—a strict, append-only Fact Ledger—that acts as a thermostat to keep the semantic engine honest.

If you're looking for the core Graphiti framework docs:
- Upstream repo: <https://github.com/getzep/graphiti>
- Upstream docs: <https://help.getzep.com/graphiti>

---

## Why This Exists: The Dual-Brain Philosophy

Plain vector search (RAG) is good at recall ("find documents about X") but terrible at **truth management**. And even Graphiti—a breakthrough temporal knowledge graph—relies 100% on an LLM to decide what's true. We call this "Brain 1 only."

**Brain 1 alone is unreliable for high-stakes decisions.** When your AI manages your calendar, drafts your deals, and handles your relationships, you can't trust an LLM to silently invalidate facts or decide what supersedes what. There's no audit trail. There's no rollback. If the LLM hallucinates over your metadata, you'll never know.

**We built Brain 2: A Fact Ledger.**

Brain 1 (Neo4j) holds all the semantic richness—messy, probabilistic, non-deterministic. Brain 2 (SQLite) holds deterministic truth—append-only, auditable, hash-chained. At retrieval time, a trust multiplier combines them: `final_score = semantic_relevance + (trust_score × trust_weight)`.

Promoted facts get a `trust_score = 1.0`. Hallucinations don't. You get deterministic truth out of a non-deterministic graph.

This architecture solves three problems vanilla Graphiti can't:

- **No silent truth mutations.** Every promotion, supersession, and invalidation is logged in Brain 2 with full provenance.
- **No LLM hallucinations becoming doctrine.** Strangers' claims stay quarantined until you approve them.
- **Agents that learn.** After every coding run, learnings are extracted and promoted to Brain 2 with `trust_score = 1.0`. The next run gets them injected as context. Knowledge compounds instead of evaporating.

For the full architecture deep-dive, see [The Dual-Brain Architecture](docs/DUAL-BRAIN-ARCHITECTURE.md).

---

## How It Works: Three Layers

This fork solves these problems by adding three layers on top of Graphiti's graph engine:

### The Three-Layer Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  L3 — Workflow Packs                                         │
│  Multi-step orchestration (e.g., "Draft VC Memo", "Book      │
│  Dinner"). Declares which context packs it needs, what       │
│  tools it can use, and what approvals are required.          │
├──────────────────────────────────────────────────────────────┤
│  L2 — Context Packs                                          │
│  Scoped read-assembly bundles. Each pack queries specific    │
│  graph lanes, applies retrieval policy, and formats output   │
│  for the consumer agent. Supports private vs group-safe      │
│  scoping, token budgets, and provenance tagging.             │
├──────────────────────────────────────────────────────────────┤
│  L1 — Truth Substrate (Graphiti + Fact Ledger)               │
│  Append-only Fact Ledger is the canonical truth.             │
│  The graph DB (FalkorDB/Neo4j) is a derived query index.    │
│  Markdown exports are derived audit views.                   │
│  If canonical and derived disagree, canonical wins.          │
└──────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### The Fact Ledger (Canonical Truth)

The graph database is **not** the source of truth. The **Fact Ledger** is — an append-only, hash-chained log that records every promotion, supersession, and invalidation with full provenance. The graph and markdown exports are both derived and rebuildable from the ledger.

Each domain has its own ledger:
- **Personal truth** (`state/fact_ledger.db`) — preferences, identity, relationships, style
- **Engineering learnings** (`state/fact_ledger_engineering.db`) — tool behaviors, failure patterns, architecture decisions
- **Operational learnings** (`state/fact_ledger_learning.db`) — self-audit findings, preference misses

Facts enter via a quarantine queue (`candidates.db`) and must pass through a **Promotion Policy** before being written to the ledger. This means you can audit every fact back to its source evidence, replay history, or roll back to any prior state.

### Promotion Policy (Truth Firewall)

Not everything ingested becomes truth. The Promotion Policy governs how candidate facts move from "observed" to "promoted":

- **Trust boundary:** Only owner-authored evidence is eligible for auto-promotion. Non-owner content can create entities but facts stay quarantined.
- **Assertion gating:** Questions, hypotheticals, and quotes are blocked. Only decisions, preferences, and factual assertions can promote.
- **Risk tiers:** Low-risk facts (preferences) can auto-promote at high confidence (>0.90). Medium and high-risk facts require human approval.
- **Conflict detection:** Contradictions are flagged. The system never silently overwrites existing truth.
- **Corroboration:** Independent evidence from multiple sources carries more weight than repeated mentions from the same source. Same-lineage evidence (e.g., a curated summary derived from a session) gets bounded boost caps to prevent confidence inflation.

### Candidate Generation Policy

Not all graph groups generate promotion candidates. Groups are classified by their role in the truth pipeline:

| Role | Groups | Behavior |
|---|---|---|
| **Candidate-generating** | Sessions, ChatGPT history, curated refs, bootstrap memory | RELATES_TO edges from anchor entities (owner, agent, org) are imported into `candidates.db` for promotion evaluation |
| **Corroboration-only** | Content packs (inspiration, writing samples, content strategy) | Provide Lane B evidence for boosting candidate confidence. Never generate their own candidates. Custom ontologies extract craft patterns (RhetoricalMove, HookPattern), not personal facts. |
| **Separate domain** | Engineering learnings, self-audit | Have their own ledger pipelines with dedicated ingest scripts. Different trust semantics and consumer profiles. |

### Trust-Aware Retrieval

Promotion status feeds back into retrieval quality. Entities and relationships in the graph carry an optional `trust_score` property that biases search result ranking:

| Status | `trust_score` | Effect |
|---|---|---|
| Promoted core truth (in fact ledger) | 1.0 | Strongest boost — surfaces verified facts when relevance is comparable |
| Corroborated candidate (multiple independent sources) | 0.6 | Moderate boost — rewards evidence convergence |
| Standard candidate (single source) | 0.25 | Minimal boost — "we've seen this" signal |
| Not in candidates pipeline | NULL | **Neutral baseline — no boost, no penalty.** Content packs, engineering entities, and other non-candidate nodes rank purely on relevance. |

The boost is **post-RRF additive**: `final_score = rrf_score + (trust_score × trust_weight)`. Default `trust_weight` is 0.15, configurable via `GRAPHITI_TRUST_WEIGHT` env var. Setting it to 0 disables trust boosting entirely (identical to vanilla RRF).

For implementation details, see [Retrieval Trust Scoring](docs/retrieval-trust-scoring.md).

### Dual-Lane Retrieval

Memory retrieval and fact verification are separate concerns:

- **Lane A (Runtime Retrieval):** Fast graph queries to answer user requests in real-time. Sources: sessions, curated references, ChatGPT history (query-gated). Trust-boosted reranking surfaces promoted facts.
- **Lane B (Corroboration):** Background verification that checks new candidate facts against the ledger and historical evidence before promotion. Sources: everything including derived memory logs.

The lanes never cross: runtime retrieval is never used for promotion decisions, and corroboration is never used for real-time answers.

### Content & Workflow Pack Injection

Content and workflow packs inject **parallel** to normal retrieval, not through it. The OpenClaw Pack Injector plugin fires on `before_agent_start`, detects intent, resolves which packs to load, and injects assembled context as `<pack-context>` blocks into the agent's prompt. Normal retrieval (agent-initiated MCP search) still runs separately.

Pack retrieval queries the same MCP search endpoints, but since content pack entities have NULL `trust_score`, trust boosting has zero effect on pack results — they rank purely on relevance. This is correct by design: content packs should surface the most relevant craft patterns, not filter by personal truth approval status.

### Graph Lanes & Custom Ontologies

Each domain of knowledge lives in its own isolated graph (identified by `group_id`). Data never leaks between lanes — entities, relationships, and episodes are fully isolated per graph.

Each lane can define its own **extraction ontology** — domain-specific entity types and relationship types that tell the LLM extractor what to look for. A content-inspiration lane extracts `RhetoricalMove`, `HookPattern`, and `VoiceQuality` entities; an engineering lane extracts `FailurePattern`, `ToolApiBehavior`, and `ArchitectureDecision`.

Ontologies are defined in YAML config (`config/extraction_ontologies.yaml`). Adding a new lane requires only a YAML block — zero code changes. Unconfigured lanes fall through to Graphiti's generic extraction.

For the full schema and examples, see [Custom Ontologies](docs/custom-ontologies.md).

### Content & Workflow Packs

A **Content Pack** (L2) is a bundle that defines:
- Which graph lanes to query (retrieval matrix by mode: `default`, `short`, `long`, `casual`, `formal`)
- ChatGPT history inclusion policy (`off`, `scoped`, `global`)
- Token budget tier (A: 600 tokens/10 items, B: 1200/20 default, C: 2400/40)
- Scope policy (`private` in DMs, `group_safe` in group chats — auto-selected by channel context)
- Provenance requirements

A **Workflow Pack** (L3) is a multi-step orchestration that:
- Declares which content packs it depends on
- Defines tool steps with explicit permission boundaries
- Supports two execution modes: `draft_only` (output only) or `execute_with_approval` (external writes after approval gate)
- Includes self-check and provenance output sections
- Ingestion supports **dual-lane** mode: some content enters as both **artifacts** (few-shot examples, retrieval-eligible) and **extracted claims** (policy-gated, may promote to truth)

Packs are defined declaratively in YAML/JSON config. The **Runtime Pack Router** (`scripts/runtime_pack_router.py`) resolves which packs are active for a given agent + intent, applies policy, and returns formatted context for the agent's context window.

---

## How Packs Work with Agents (OpenClaw Integration)

Agents don't query the graph directly. They use the **Runtime Pack Router** as a high-level hook.

### The Routing Flow

```
Agent receives task (e.g., "draft a deal brief for Acme")
    │
    ▼
Intent Classification
    │  "vc_deal_brief" matched via keyword/entity boost rules
    ▼
Consumer Profile Lookup (config/runtime_consumer_profiles.json)
    │  Maps agent + intent → pack_ids, modes, chatgpt_mode, scope
    ▼
Pack Resolution (config/runtime_pack_registry.json)
    │  Resolves each pack_id → retrieval matrix, query template
    ▼
Graph Query (per graph lane in retrieval matrix)
    │  Queries s1_sessions_main, s1_curated_refs, etc.
    ▼
Policy Enforcement
    │  Applies scope (private vs group_safe), ChatGPT lane gating
    ▼
Context Assembly
    │  Formats retrieved facts + provenance into injection text
    ▼
Agent receives structured context in its context window
```

### Materialization

For engineering learnings, the router supports `--materialize` which reads the latest structured JSONL artifacts directly (bypassing graph query) when the engineering graph is still being populated. This allows CLR/Antfarm agents to benefit from learnings even during the graph bootstrap period.

---

## Observational Memory (OM)

OM is the runtime synthesis/control ("metabolism") loop for the Dual Brain stack.
It is not a separate sovereign brain or truth authority. Instead, it coordinates
high-throughput transcript intake in Brain 1 with governed promotion pathways into
Brain 2.

### The Problem OM Solves

The Dual Brain governs *what to trust*. But the MCP server's ingestion pipeline has
non-trivial latency (LLM extraction per episode). In a high-throughput agent runtime,
messages accumulate faster than the pipeline can drain. OM closes this gap with a
lightweight control loop that runs in parallel with Graphiti.

### How OM Works

```
Live transcript
      │
      ▼
om_fast_write.py          ← Embeds message content + writes Message/Episode to Neo4j
      │                      Fail-closed (no embedding = no write)
      ▼
om_compressor.py          ← Background: drains Message backlog into OMNode observations
      │                      Trigger: backlog ≥ 50 OR oldest ≥ 48h
      │                      Chunks: N = min(50, backlog), max 10 chunks/run
      ▼
om_convergence.py         ← Drives OMNode lifecycle state machine
      │                      States: OPEN → MONITORING → CLOSED / ABANDONED / REOPENED
      │                      Watermark: max 500 nodes/pass, cursor-based resumption
      ▼
promotion_policy_v3.py    ← Promotes corroborated nodes to CoreMemory (retained by GC)
```

### Quick Commands

```bash
# Fast-write a message
python3 scripts/om_fast_write.py write \
  --session-id "<session_id>" --role user \
  --content "<text>" --created-at "2026-02-26T12:00:00Z"

# Run the compressor (respects trigger threshold)
python3 scripts/om_compressor.py

# Run convergence (state machine + dead-letter sync)
PYTHONPATH=. python3 scripts/om_convergence.py

# GC dry-run (90-day TTL, no deletions)
PYTHONPATH=. python3 scripts/om_convergence.py --run-gc --gc-dry-run

# Promote a corroborated candidate to CoreMemory
PYTHONPATH=. python3 truth/promotion_policy_v3.py --candidate-id <id>

# Exact-dedupe OMNodes (dry-run first, then --apply)
uv run python scripts/om_dedupe.py --dry-run
uv run python scripts/om_dedupe.py --apply

# Backfill timeline timestamps on pre-Phase-B OMNodes
uv run python scripts/om_backfill_timestamps.py --dry-run
uv run python scripts/om_backfill_timestamps.py --apply

# Normalize edge names to SCREAMING_SNAKE_CASE (dry-run first)
python scripts/normalize_edge_names.py
python scripts/normalize_edge_names.py --apply

# Run closure semantics pass (RESOLVES/SUPERSEDES auto-invalidation)
python scripts/apply_closure_semantics.py
python scripts/apply_closure_semantics.py --apply

# Cross-lane contamination check (exit 0 = clean, 1 = contamination)
python scripts/contamination_sentinel.py --json
```

### Key Numbers

| Parameter | Value |
|---|---|
| Compressor trigger (backlog) | ≥ 50 messages |
| Compressor trigger (age) | ≥ 48 hours |
| Messages per chunk | min(50, backlog) |
| Max chunks per run | 10 (configurable) |
| Dead-letter threshold | 3 failed attempts |
| Convergence pass limit | 500 nodes |
| GC TTL (default) | 90 days |
| GC retention gate | EVIDENCE\_FOR active OMNode OR SUPPORTS\_CORE active CoreMemory |

### Phase B: Dedupe & Timeline Semantics

- **Exact dedupe** (`scripts/om_dedupe.py`): Detects and merges OMNodes sharing the same `node_type + normalize(content)` key across semantic domains. Canonical node = earliest `created_at`; metadata merged (union provenance, max urgency, most-active status). Dry-run default.
- **Timeline semantics**: OMNodes carry `first_observed_at` and `last_observed_at` derived from source message event time (not wall-clock extraction time). Convergence age-decay and GC eligibility use event time. Pre-existing nodes: `scripts/om_backfill_timestamps.py`.

### Phase C: Graph Maintenance & Guardrails

- **Edge normalization** (`scripts/normalize_edge_names.py`): Universal SCREAMING\_SNAKE\_CASE normalization, preventing case-variant dedup collisions. Also exported as `graphiti_core.utils.maintenance.normalize_relation_type` for inline use.
- **Closure semantics** (`scripts/apply_closure_semantics.py`): RESOLVES/SUPERSEDES edges auto-invalidate the target entity's active facts. Pure graph pass — no LLM calls. Idempotent, dry-run default.
- **Endpoint split** (`graphiti_core/utils/env_utils.py`): Separate `LLM_BASE_URL` and `EMBEDDER_BASE_URL` resolution to prevent accidental embedding-to-OpenRouter routing. See `.env.example` for priority chain.
- **Contamination sentinel** (`scripts/contamination_sentinel.py`): Read-only cross-lane integrity check. `--json` for CI. Exit 0 = clean.
- **Recall gate** (`--recall-gate 0.75 --recall-baseline ...` on `run_retrieval_benchmark.py`): CI-friendly quality gate.
- **Scope policy** ([`docs/scope-policy.md`](docs/scope-policy.md)): Frozen as of Phase C. Messages only by default; `toolResult` opt-in via `TOOL_RESULT_ALLOWLIST`.

For the full operations guide including lock ordering, split/isolate failure recovery,
and convergence state machine details, see [OM Operations Runbook](docs/runbooks/om-operations.md).

---

## Public/Private Split Model

This fork operates on an **Engine/Fuel** split:

- **Public repo (this one):** The runtime architecture, routing logic, policy enforcement, ontology framework, and tooling. It knows *how* to route requests to packs but contains no private data.
- **Private overlay repo (yours):** Your actual `runtime_pack_registry.json`, `runtime_consumer_profiles.json`, `plugin_intent_rules.json`, `workflows/*.pack.yaml`, extraction ontology config, and graph state.

**To deploy:** Clone the public repo, overlay your private config, and run the runtime from the merged working tree.

```bash
# Apply private overlay into your runtime checkout
./path/to/private-repo/scripts/apply-overlay.sh /path/to/runtime-checkout
```

---

## Installation & Setup

### Prerequisites
- Python 3.13+
- Neo4j (default) or FalkorDB (legacy) — graph database backend
- OpenAI API key (for LLM extraction + embeddings)
- **Endpoint split (recommended):** set `LLM_BASE_URL` for LLM routing and `EMBEDDER_BASE_URL` for embedding routing separately. This prevents accidental embedding traffic to OpenRouter when LLM routing is redirected. See `.env.example` for the full priority chain.

### Quick Start

```bash
# Clone
git clone https://github.com/yhl999/bicameral.git
cd bicameral

# Install dependencies
uv sync
# OR: pip install -e ".[neo4j]"

# Configure (copy and edit)
cp config/config.example.yaml config/config.yaml
# Edit config.yaml: set your Neo4j/FalkorDB connection, OpenAI key, etc.

# Verify delta tooling
python3 scripts/delta_tool.py list-commands

# Verify runtime pack configuration
python3 scripts/runtime_pack_router.py --verify-only

# Start the MCP server (HTTP transport)
python3 mcp_server/main.py
```

### Ingesting Data

Data flows through a 7-stage pipeline: Source Material → Evidence (deterministic chunking) → Ingest Registry (watermarks + dedup) → Graphiti MCP (LLM extraction) → Candidates DB (quarantine) → Promotion Policy (gates) → Fact Ledger (canonical truth).

```bash
# 1. Bootstrap Neo4j with historical session transcripts (first-time setup)
python3 scripts/import_transcripts_to_neo4j.py \
  --sessions-dir path/to/session_transcripts/ --dry-run
python3 scripts/import_transcripts_to_neo4j.py \
  --sessions-dir path/to/session_transcripts/

# 2. Ingest sessions — Neo4j source mode (default, production path)
# NOTE: If Neo4j has no Message nodes (empty graph), a BOOTSTRAP_REQUIRED guard
# fires and the script exits non-zero. Run step 1 first to populate Neo4j.
python3 scripts/mcp_ingest_sessions.py \
  --group-id s1_sessions_main \
  --source-mode neo4j \
  --mcp-url http://localhost:8000/mcp

# 2b. Rollback — evidence source mode (reads from disk, bypasses Neo4j)
python3 scripts/mcp_ingest_sessions.py \
  --group-id s1_sessions_main \
  --source-mode evidence \
  --evidence path/to/sessions_evidence/

# 3. Check ingestion status (watermarks, queue depth, last success/failure)
python3 scripts/registry_status.py

# 4. Verify adapter contract compliance (INGEST_ADAPTER_CONTRACT_V1)
python3 scripts/ingest_adapter_contract_check.py --strict
```

Ingestion is idempotent (content-hash dedup), incremental (delta since last watermark), and supports sub-chunking for large evidence (>10k chars). All adapters must conform to `INGEST_ADAPTER_CONTRACT_V1` — see `ingest/contracts.py` and `docs/runbooks/adding-data-sources.md`.

---

## Keeping Up with Upstream

This fork tracks `getzep/graphiti` via a deterministic PR-based sync lane using an explicit patch stack for `graphiti_core` hotfixes.

- **Default cadence:** Weekly (Monday) via GitHub Action.
- **Conflict policy:** Upstream wins. If conflicts arise in `graphiti_core/**`, accept upstream's version and re-apply our local patch stack located in `patches/graphiti_core/`.
- **Core Guardrail:** CI automatically enforces that no undocumented `graphiti_core` files drift from upstream.

For the full sync and patch application procedure, see the [Upstream Sync Runbook](docs/runbooks/upstream-sync-openclaw.md) and [`HOTFIXES.md`](HOTFIXES.md).

---

## Documentation Index

### Architecture & Concepts
- [Custom Ontologies](docs/custom-ontologies.md) — defining per-lane extraction entity types
- [Retrieval Trust Scoring](docs/retrieval-trust-scoring.md) — how promoted/corroborated facts boost search ranking
- [Memory Runtime Wiring](docs/MEMORY-RUNTIME-WIRING.md) — Graphiti-primary retrieval + QMD failover contract
- [Scope Policy](docs/scope-policy.md) — ingestion scope freeze: message-only default, toolResult opt-in allowlist, change process
- [Runtime Pack Overlay](docs/runbooks/runtime-pack-overlay.md) — how private packs map to agents

### Operations
- [OM Operations](docs/runbooks/om-operations.md) — Observational Memory: fast-write, compressor, convergence, GC, promotion, dedupe, timeline semantics
- [Sessions Ingestion](docs/runbooks/sessions-ingestion.md) — architecture, batch & steady-state config, high-throughput tuning, sub-chunking, retrieval benchmark, recall gate, post-processing, troubleshooting
- [Adding Data Sources](docs/runbooks/adding-data-sources.md) — onboarding new content: group_id, ontology design, adapter patterns, cron setup
- [Upstream Sync Runbook](docs/runbooks/upstream-sync-openclaw.md)
- [State Migration Runbook](docs/runbooks/state-migration.md)
- [Publicization & Backup](docs/runbooks/publicization-backup-cutover.md)
- [OpenClaw Plugin Troubleshooting](docs/runbooks/openclaw-plugin-troubleshooting.md) — handling strict schema validation crashes and gateway port exhaustion

### Technical Contracts
- [Boundary Contract](docs/public/BOUNDARY-CONTRACT.md)
- [Migration Sync Toolkit](docs/public/MIGRATION-SYNC-TOOLKIT.md)
- [Release Checklist](docs/public/RELEASE-CHECKLIST.md)

### Delta Tooling
- `scripts/delta_tool.py` — unified CLI for delta operations
- `scripts/runtime_pack_router.py` — the pack routing engine
- `scripts/upstream_sync_doctor.py` — sync safety checks

---

## Current Status

- Publicization execution lanes completed: adapter wiring, backup wiring, cron cutover.
- Integration gate: **GO** (`reports/publicization/integration-report.md`).
- Boundary policy: **ALLOW=370 / BLOCK=0 / AMBIGUOUS=0**.
- **Truth pipeline: operational, with deterministic migration closeout policy.** Fact ledger + trust-aware retrieval are live, and curated-facts migration now uses deterministic disposition-aware validation (rather than ad-hoc/manual overrides) for unresolved legacy mappings. Canonical curated migration closeout is complete.
- **Flip readiness caveat:** do not treat this status as automatic “Graphiti-primary GO.” First confirm extraction freshness / queue-drain reliability and pass a fresh shadow-compare window; otherwise keep Graphiti in governed shadow mode with QMD failover semantics.

## CI Policy

Canonical PR gates:
- `.github/workflows/ci.yml`
- `.github/workflows/migration-sync-tooling.yml`

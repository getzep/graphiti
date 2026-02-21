# Graphiti OpenClaw — Dual-Brain Memory Runtime for Agents

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

### Quick Start

```bash
# Clone
git clone https://github.com/yhl999/graphiti-openclaw.git
cd graphiti-openclaw

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
# Parse session transcripts into evidence JSON
python3 ingest/parse_sessions_v1.py --agent main

# Ingest sessions into graph (incremental, idempotent)
python3 scripts/mcp_ingest_sessions.py --group-id s1_sessions_main --incremental

# Check ingestion status (watermarks, queue depth, last success/failure)
python3 scripts/registry_status.py
```

Ingestion is idempotent (content-hash dedup), incremental (delta since last watermark), and supports sub-chunking for large evidence (>10k chars).

---

## Keeping Up with Upstream

This fork tracks `getzep/graphiti` via a deterministic PR-based sync lane.

- **Default cadence:** Weekly (Monday) via GitHub Action.
- **Manual:** `workflow_dispatch` for on-demand sync.
- **Conflict policy:** Clean merges auto-PR. Conflicts fail fast — resolve locally.

For the full sync procedure, see [Upstream Sync Runbook](docs/runbooks/upstream-sync-openclaw.md).

---

## Documentation Index

### Architecture & Concepts
- [Custom Ontologies](docs/custom-ontologies.md) — defining per-lane extraction entity types
- [Retrieval Trust Scoring](docs/retrieval-trust-scoring.md) — how promoted/corroborated facts boost search ranking
- [Memory Runtime Wiring](docs/MEMORY-RUNTIME-WIRING.md) — Graphiti-primary retrieval + QMD failover contract
- [Runtime Pack Overlay](docs/runbooks/runtime-pack-overlay.md) — how private packs map to agents

### Operations
- [Sessions Ingestion](docs/runbooks/sessions-ingestion.md) — architecture, batch & steady-state config, high-throughput tuning, sub-chunking, post-processing, troubleshooting
- [Adding Data Sources](docs/runbooks/adding-data-sources.md) — onboarding new content: group_id, ontology design, adapter patterns, cron setup
- [Upstream Sync Runbook](docs/runbooks/upstream-sync-openclaw.md)
- [State Migration Runbook](docs/runbooks/state-migration.md)
- [Publicization & Backup](docs/runbooks/publicization-backup-cutover.md)

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
- **Truth pipeline: operational, with deterministic migration closeout policy.** Fact ledger + trust-aware retrieval are live, and curated-facts migration now uses deterministic disposition-aware validation (rather than ad-hoc/manual overrides) for unresolved legacy mappings. Content packs remain isolated (NULL trust_score, zero distortion).

## CI Policy

Canonical PR gates:
- `.github/workflows/ci.yml`
- `.github/workflows/migration-sync-tooling.yml`

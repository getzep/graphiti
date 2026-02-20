# Graphiti OpenClaw — Memory Runtime for Agents

This repository is a **production delta layer** on top of [upstream Graphiti](https://github.com/getzep/graphiti). It turns Graphiti from a graph-memory library into a policy-governed, pack-driven memory runtime for OpenClaw agents.

If you're looking for the core Graphiti framework docs:
- Upstream repo: <https://github.com/getzep/graphiti>
- Upstream docs: <https://help.getzep.com/graphiti>

---

## Why This Exists

Plain vector search (RAG) is good at recall ("find documents about X") but terrible at **truth management**:

- **No temporal reasoning.** "I prefer coffee" and "I stopped drinking coffee" both match a query about coffee. Which is current?
- **No supersession.** When a fact changes, old versions linger as equal-confidence results.
- **No relationships.** "Who is the CEO of the company Yuan met last week?" requires traversal, not keyword matching.
- **No promotion gates.** Anything ingested becomes "truth" with no audit trail.

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

The graph database is **not** the source of truth. The **Fact Ledger** is — an append-only log that records every promotion, supersession, and invalidation with full provenance. The graph and markdown exports are both derived and rebuildable from the ledger.

This means you can audit every fact back to its source evidence, replay history, or roll back to any prior state.

### Promotion Policy (Truth Firewall)

Not everything ingested becomes truth. The Promotion Policy governs how candidate facts move from "observed" to "promoted":

- **Trust boundary:** Only owner-authored evidence is eligible for auto-promotion.
- **Assertion gating:** Questions, hypotheticals, and quotes are blocked. Only decisions, preferences, and factual assertions can promote.
- **Risk tiers:** Low-risk facts (preferences) can auto-promote at high confidence (>0.90). Medium and high-risk facts require human approval.
- **Conflict detection:** Contradictions are flagged. The system never silently overwrites existing truth.
- **Corroboration:** Independent evidence from multiple sources carries more weight than repeated mentions from the same source. Same-lineage evidence (e.g., a curated summary derived from a session) gets bounded boost caps to prevent confidence inflation.

### Dual-Lane Retrieval

Memory retrieval and fact verification are separate concerns:

- **Lane A (Runtime Retrieval):** Fast graph queries to answer user requests in real-time. Sources: sessions, curated references, ChatGPT history (query-gated).
- **Lane B (Corroboration):** Background verification that checks new candidate facts against the ledger and historical evidence before promotion. Sources: everything including derived memory logs.

The lanes never cross: runtime retrieval is never used for promotion decisions, and corroboration is never used for real-time answers.

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
- [Memory Runtime Wiring](docs/MEMORY-RUNTIME-WIRING.md) — Graphiti-primary retrieval + QMD failover contract
- [Runtime Pack Overlay](docs/runbooks/runtime-pack-overlay.md) — how private packs map to agents

### Operations
- [Upstream Sync Runbook](docs/runbooks/upstream-sync-openclaw.md)
- [State Migration Runbook](docs/runbooks/state-migration.md)
- [High-Throughput Extraction](docs/runbooks/high-throughput-extraction.md)
- [Sessions Sub-Chunking](docs/runbooks/sessions-subchunking.md)
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

## CI Policy

Canonical PR gates:
- `.github/workflows/ci.yml`
- `.github/workflows/migration-sync-tooling.yml`

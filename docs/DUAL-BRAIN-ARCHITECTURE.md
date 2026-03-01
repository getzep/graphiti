# The Dual-Brain Architecture: Why Two Systems Are Better Than One

## The Problem With "Magic" AI Memory

Most AI memory systems rely on a single brain: an LLM that reads new information, looks at old information, and decides what's true. Graphiti (the upstream library) does this. It uses a prompt to ask an LLM: "Here's what we knew. Here's what we just learned. Does the new thing contradict the old thing?"

This works for a demo. But for a Chief-of-Staff AI managing your calendar, drafting your emails, and handling your deals, it breaks down.

**The LLM has no concept of authority.** If someone in a Telegram group jokingly says "Yuan loves eating glass," the LLM might look at your "loves steak" fact, decide the joke contradicts it, and silently delete your actual diet preference. There is no audit log. There is no rollback. Just an LLM hallucinating over your life's metadata, creating an artifact you'll never know was deleted.

**The LLM struggles with context limits.** To auto-invalidate facts, the system has to load the old facts into the prompt. If you have thousands of facts about yourself, the LLM can't see them all. It misses contradictions. You end up with a graph that simultaneously asserts you're 25, 26, and live in both NYC and SF.

**There's no trace of truth decisions.** When an old fact is invalidated, it's gone. No history of why. No way to understand the reasoning. No way to revert if the decision was wrong.

This is what happens when you have only **Brain 1** — the semantic, LLM-powered, non-deterministic brain.

---

## The Dual-Brain Solution

We built **Brain 2** — a strict, append-only Fact Ledger backed by SQLite. This is a production-grade governance database.

Here's how it works:

### Brain 1: The Semantic Engine (Neo4j)
- **Purpose:** Holds all the semantic richness. Every entity, every relationship, every nuance of meaning.
- **Characteristics:** Messy, non-deterministic, probabilistic. The LLM extracts freely. Relationships are not perfectly consistent.
- **What it's good for:** Vector search. Finding things that *feel* relevant to a query.
- **What it's bad at:** Knowing what's true.

### Brain 2: The Logic Ledger (SQLite)
- **Purpose:** Holds the deterministic, auditable record of what you actually believe.
- **Characteristics:** Strict, append-only, hash-chained, fully normalized. Every fact has provenance.
- **What it's good for:** Knowing what's true. Audit trails. Rollback. Conflict resolution.
- **What it's bad at:** Semantic understanding.

### The Bridge: Trust as a Thermostat

When you (or a high-confidence policy) promote a fact in Brain 2, the system reaches down into Brain 1 and stamps that edge with `trust_score = 1.0`.

At retrieval time, we use **Reciprocal Rank Fusion (RRF)** to combine semantic relevance with trust. The formula is simple:

```
final_score = rrf_semantic_score + (trust_score × trust_weight)
```

The "eating glass" joke gets ranked by semantic relevance (probably low). Your actual preference gets a `trust_score = 1.0` boost. At retrieval time, the truth crushes the noise.

You get deterministic truth out of a non-deterministic graph.

---

## Why Not Just Use One Brain?

**Option A: Brain 1 only (vanilla Graphiti approach)**
- Pros: Simpler. No extra database.
- Cons: Unreliable truth. Hallucinating LLM. No audit trail. Silent mutations. When your AI manages high-stakes decisions (calendar, deals, relationships), this is unacceptable.

**Option B: Brain 2 only (pure SQLite ledger)**
- Pros: Perfect auditability. Perfect truth.
- Cons: No semantic understanding. Can't answer "who is related to this person?" Can't do vector search. No relationship traversal. Slow and brittle.

**Option C: Dual Brain**
- Pros: Semantic richness of Brain 1 + deterministic truth of Brain 2. Fast retrieval + auditable truth. The LLM can hallucinate freely; the ledger keeps it honest.
- Cons: Operational complexity. You have to maintain two systems and sync them.

We chose C because the alternative is an AI that eventually believes it's a die-hard vegan and has been living in Portland for six months.

---

## How the Dual Brain Actually Works

### The Flow

1. **Extraction (Brain 1):** Transcript comes in. LLM extracts entities and relationships. They go into Neo4j with no approval required. This happens fast, messily, non-deterministically.

2. **Candidate Generation (Bridge):** A subset of extractions are marked as "promotion candidates" — potential facts worthy of truth status. They get logged into `candidates.db` (SQLite) along with their source, confidence, and a UUID pointing back to the Neo4j edge.

3. **Promotion Decision (Brain 2):** Candidates wait for approval. Only owner-authored facts auto-promote (if confidence is high). Third-party facts stay quarantined forever unless you approve them. Risky facts require human review.

4. **Conflict Detection (Brain 2):** Before a new fact is promoted, the system checks: does this contradict an existing promoted fact? If yes, flag it. If the new fact is more recent and reliable, supersede the old one with an explicit ledger entry (not a silent deletion).

5. **Trust Sync (Bridge):** After facts are promoted in Brain 2, a sync script reaches into Brain 1 and sets `trust_score = 1.0` on the corresponding edges. Unpromoted edges get `trust_score = 0.25` (a "we've seen this" signal) or NULL (no boost).

6. **Retrieval (Bridge):** When the AI needs context, it queries Neo4j with RRF. Trust scores act as a reranking multiplier. Promoted facts surface higher (all else equal). Hallucinated noise sinks.

### Supersession Without Silent Deletion

When a fact changes (e.g., "Yuan is 25" → "Yuan is 26"), Brain 2 records this explicitly:

```json
{
  "id": "fact-uuid-26",
  "content": "Yuan is 26",
  "valid_from": "2026-02-21",
  "supersedes": "fact-uuid-25",
  "ledger_hash": "abc123..."
}
```

The old fact is not deleted. It's marked as superseded, with a timestamp and a pointer to what replaced it. You can audit the entire history. You can revert if needed.

Brain 1 now supports a **closure semantics pass** (`scripts/apply_closure_semantics.py`) that handles this at the graph level: when `RESOLVES` or `SUPERSEDES` edges exist, the target entity's active facts are automatically invalidated at the closure event's timestamp — no LLM reasoning required. This is idempotent, runs offline, and can be scheduled as a maintenance job. It doesn't replace Brain 2's governance; it ensures Brain 1's graph state is kept consistent with the semantic edges already present.

Brain 2 handles the *policy* decisions cleanly. The closure pass handles the *graph housekeeping* deterministically.

---

## Why This Matters for Agents

The Dual Brain unlocks a capability that vanilla Graphiti can't offer: **agents that learn and improve over time**.

After every coding run, the system extracts learnings into the engineering graph: "Neo4j rejects heap configs exceeding physical RAM." This gets promoted into Brain 2 of the engineering ledger. On the next run, the agent wakes up, gets that fact injected as `trust_score = 1.0` context, and it just... knows.

Each run deposits knowledge. Each future run withdraws it. The agents get smarter. The mistakes shrink.

This doesn't happen in vanilla Graphiti because there's no deterministic record of "things we've learned," no way to say "this specific fact earned a 1.0 trust score because it was extracted from 6 independent agent runs," and no way to prioritize high-confidence learnings during retrieval.

---

## The Tradeoff

The Dual Brain is operationally complex. You have to:

- Maintain two systems and keep them synced
- Write promotion policies
- Monitor the candidates queue
- Roll back or override decisions when the system is wrong

Vanilla Graphiti is simpler. You just dump transcripts in and let the LLM figure it out.

But if your AI is managing anything high-stakes — calendar, deals, relationships, code deployments — "letting the LLM figure it out" is a recipe for operational failure.

We bet the complexity was worth it.

---

## Observational Memory (OM): Synthesis + Control Layer

After shipping, we added **Observational Memory (OM)** as a synthesis/control
("metabolism") layer for the Dual Brain runtime.

OM is **not** an independent third authority on truth. Its job is to coordinate flow:
rapid intake into Brain 1, structured observation lifecycle, and governed promotion
signals into Brain 2.

The Dual Brain addresses *governance* — which facts to trust. But it doesn't address
*velocity*: raw transcript messages accumulate faster than the MCP server's full
extraction pipeline can drain them. OM closes that gap:

### Stage 1 — Fast-Write (`scripts/om_fast_write.py`)

Transcript messages are written directly into Neo4j as `Message` and `Episode` nodes
**without** going through the MCP extraction queue. This is a lightweight, fail-closed
path: embed the content, write the node, done. Latency is bounded by the embedding
endpoint (~20s timeout), not by LLM extraction.

Each new `Message` node is stamped with extraction lifecycle defaults:
- `om_extracted = false` — queued for OM compressor
- `om_dead_letter = false` — not yet failed
- `graphiti_extracted_at = NULL` — not yet processed by Graphiti/MCP

### Stage 2 — Compressor (`scripts/om_compressor.py`)

A background process drains the unextracted `Message` backlog into `OMNode`
observations. Trigger: backlog ≥ 50 messages **or** oldest message ≥ 48 hours old.
Each run processes up to 10 chunks of 50 messages each (500 messages max per run).

Extracted `OMNode` types: `WorldState`, `Judgment`, `OperationalRule`, `Commitment`,
`Friction`. Nodes are linked back to their source messages via `EVIDENCE_FOR` edges.

### Stage 3 — Convergence (`scripts/om_convergence.py`)

The convergence engine runs the OMNode lifecycle state machine
(`OPEN → MONITORING → CLOSED/ABANDONED, REOPENED` paths), reconciles the dead-letter
queue between Neo4j and the candidates SQLite DB, and optionally runs GC (90-day TTL
with retention gates for `EVIDENCE_FOR` and `SUPPORTS_CORE` edges).

### Stage 4 — CoreMemory Promotion (`truth/promotion_policy_v3.py`)

Corroborated OMNode candidates are promoted to `CoreMemory` nodes, which carry
`retention_status = 'active'` and are linked to their source `Message` evidence via
`SUPPORTS_CORE` edges. CoreMemory nodes are exempt from GC.

### Why This Matters

OM closes a gap in the Dual Brain: Brain 1 (Neo4j) was populated via MCP ingestion,
which requires full LLM extraction. OM allows Brain 1 to populate from **live
transcript stream** without MCP latency. Observations arrive in near-real-time, are
governed by the same convergence policy, and — when corroborated — feed Brain 2's
fact ledger via CoreMemory promotion.

The Dual Brain with an OM control loop:
- **Brain 1 (Neo4j):** Semantic engine — LLM-extracted entities and relationships
- **Brain 2 (SQLite Fact Ledger):** Logic ledger — deterministic, auditable truth
- **OM (synthesis/control layer):** Fast-write + observation lifecycle + convergence to coordinate how evidence moves from transcript stream into Brain 2 promotion paths

---

## See Also

- [Retrieval Trust Scoring](retrieval-trust-scoring.md) — How the trust multiplier actually works in code
- [Custom Ontologies](custom-ontologies.md) — How to teach each graph lane what to extract
- [Promotion Policy](promotion-policy.md) — How to decide what becomes truth
- [Fact Ledger](fact-ledger-schema.md) — The SQLite schema and hash-chain design
- [OM Operations Runbook](runbooks/om-operations.md) — Trigger math, lock ordering, GC invariants, convergence state machine

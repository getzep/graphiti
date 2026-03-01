# OM Operations Runbook

Observational Memory (OM) is the pipeline that converts raw transcript messages into
structured `OMNode` observations in Neo4j, manages their lifecycle through a
deterministic state machine, and promotes high-confidence nodes to `CoreMemory`.

---

## ⚠️ CRITICAL: OM Extraction Path

**OM extraction in production MUST use `om_compressor`, NOT `mcp_ingest_sessions.py`.**

Using `mcp_ingest_sessions.py --group-id s1_observational_memory ...` routes through
Graphiti MCP's `add_memory` tool, which:
- Creates `Entity` nodes (not `OMNode`) — wrong schema
- Bypasses OM ontology constraints (MOTIVATES/GENERATES/SUPERSEDES/ADDRESSES/RESOLVES)
- Skips OM node deduplication and provenance tracking
- Ignores `om_extractor` schema version pinning
- Does not integrate with the dead-letter isolation flow

The `mcp_ingest_sessions.py` script includes a runtime guardrail (`_check_om_path_guard`)
that prints a prominent warning when targeting OM namespaces (group-id starts with
`s1_observational_memory`). Set `OM_PATH_GUARD=strict` in the environment to make this
an abort instead of a warning.

### Correct OM extraction command:

```bash
# Standard OM extraction run
uv run python scripts/om_compressor.py --force --max-chunks-per-run 10

# Dry run (preview only — no writes)
uv run python scripts/om_compressor.py --dry-run --max-chunks-per-run 10

# Pilot with bounded sample (FR-11 style)
uv run python scripts/om_compressor.py --force --max-chunks-per-run 50 --mode backfill

# Pin model explicitly (model also set via OM_COMPRESSOR_MODEL env var)
OM_COMPRESSOR_MODEL=gpt-5.1-codex-mini uv run python scripts/om_compressor.py --force
```

### Wrong command (do NOT use for OM):

```bash
# ❌ WRONG — routes through add_memory, creates Entity nodes, bypasses OM
uv run python scripts/mcp_ingest_sessions.py --group-id s1_observational_memory ...
```

---

## Components

| Script | Purpose |
|---|---|
| `scripts/om_fast_write.py` | Append Episode/Message nodes directly to Neo4j (bypass MCP) |
| `scripts/om_compressor.py` | Extract OMNodes from unprocessed Message backlog |
| `scripts/om_convergence.py` | Drive OMNode lifecycle transitions + GC + dead-letter reconciliation |
| `truth/candidates.py` | SQLite candidates DB — dead-letter queue, verification, promotion scaffolding |
| `truth/promotion_policy_v3.py` | Promote corroborated OMNodes to CoreMemory |

---

## Prerequisites

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="<password>"
# Optional: NEO4J_DATABASE (defaults to "neo4j")

# Embedding endpoint (defaults to Ollama at localhost:11434/v1)
export EMBEDDER_BASE_URL="http://localhost:11434/v1"
# OM_EMBEDDING_MODEL default: embeddinggemma
# OM_EMBEDDING_DIM   default: 768

# Ontology config (default: mcp_server/config/extraction_ontologies.yaml)
# Override: OM_ONTOLOGY_CONFIG_PATH
```

Dev environments auto-load credentials from `~/.clawdbot/credentials/neo4j.env`.
In production/staging, set `NEO4J_PASSWORD` directly; the file fallback is disabled unless
`OM_NEO4J_ENV_FALLBACK_NON_DEV=1`.

---

## 1. Fast-Write Pipeline

`scripts/om_fast_write.py` writes one transcript message at a time directly into Neo4j
as `Message` and `Episode` nodes with embeddings. Use this when you need sub-second
writes without going through the MCP server's full ingestion queue.

### Commands

```bash
# Write a single message
python3 scripts/om_fast_write.py write \
  --session-id <session_id> \
  --role user \
  --content "The Neo4j heap should be capped at 70% of available RAM." \
  --created-at 2026-02-26T12:00:00Z

# Write from a JSON payload file
python3 scripts/om_fast_write.py write \
  --payload-file /path/to/payload.json

# Enable fast-write for a runtime repo
python3 scripts/om_fast_write.py set-state \
  --runtime-repo /path/to/runtime-repo \
  --enabled \
  --reason "pipeline_wired"

# Disable fast-write
python3 scripts/om_fast_write.py set-state \
  --runtime-repo /path/to/runtime-repo \
  --disabled \
  --reason "maintenance"
```

### State File

State is persisted at `state/om_fast_write_state.json` inside the runtime repo.
This file is gitignored automatically. The `set-state` command emits
`FAST_WRITE_ENABLED` or `FAST_WRITE_DISABLED` JSON events to stdout.

### Embedding Config

| Env | Default | Description |
|---|---|---|
| `OM_EMBEDDING_MODEL` | `embeddinggemma` | Model name sent to embeddings endpoint |
| `OM_EMBEDDING_DIM` | `768` | Expected vector dimension (mismatch = error) |
| `OM_EMBED_TIMEOUT_SECONDS` | `20` | HTTP timeout for embedding calls |
| `EMBEDDER_BASE_URL` / `OPENAI_BASE_URL` | `http://localhost:11434/v1` | Base URL for OpenAI-compatible embeddings API |

Fast-write is **fail-closed**: if embedding fails, the message is not written.

### Message Schema Written to Neo4j

New `Message` nodes are created with extraction lifecycle defaults:

```
m.om_extracted       = false
m.om_extract_attempts = 0
m.om_dead_letter     = false
m.graphiti_extracted_at = NULL
```

---

## 2. Compressor

`scripts/om_compressor.py` processes the unextracted `Message` backlog into `OMNode`
observations. Run it periodically (e.g., every 15–30 minutes via cron).

### Trigger Math

The compressor checks two conditions before each chunk:

```
trigger = (backlog_count >= 50) OR (oldest_backlog_age_hours >= 48.0)
```

Where:
- `backlog_count` = count of `Message` nodes where `om_extracted=false` AND `om_dead_letter=false`
- `oldest_backlog_age_hours` = age of the oldest such message in hours

Use `--force` to bypass the trigger check (useful for testing or manual drains).

### Chunk Semantics

```
N = min(50, backlog_count)   # messages per chunk (MAX_PARENT_CHUNK_SIZE = 50)
MAX_CHUNKS_PER_RUN = 10      # default; override with --max-chunks-per-run
```

Each run processes up to `MAX_CHUNKS_PER_RUN` chunks. Each chunk contains up to 50
messages ordered by `created_at ASC`. A single run processes at most 500 messages.

### Lock Ordering

The compressor acquires an exclusive process lock before any Neo4j writes:

```python
# Resolved via _resolve_lock_path():
# 1. OM_COMPRESSOR_LOCK_PATH env override
# 2. $XDG_RUNTIME_DIR/bicameral/om_graph_write.lock
# 3. state/locks/om_graph_write.lock  (repo-local, preferred)
# 4. ~/.cache/bicameral/locks/om_graph_write.lock
# 5. /tmp/bicameral/locks/om_graph_write.lock  (final fallback)
fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # blocking, exclusive
```

The convergence runner uses a fixed path: `/tmp/om_graph_write.lock`.

> **Production alignment:** set `OM_COMPRESSOR_LOCK_PATH=/tmp/om_graph_write.lock`
> to ensure compressor and convergence share the same lock file.

### Running the Compressor

```bash
# Normal run (respects trigger threshold)
python3 scripts/om_compressor.py

# Force-process regardless of trigger
python3 scripts/om_compressor.py --force

# Custom chunk limit
python3 scripts/om_compressor.py --max-chunks-per-run 5

# Custom ontology config
python3 scripts/om_compressor.py --config /path/to/extraction_ontologies.yaml
```

### High-throughput backfill (manifest + claim mode)

FR-11 backfill throughput uses a two-phase flow:

1. Build a deterministic chunk manifest from current pending OM messages.
2. Run one or more shard workers in claim mode against that manifest.

```bash
# Phase A: build manifest + seed claim-state DB (<manifest>.claims.db)
python3 scripts/om_compressor.py \
  --mode backfill \
  --build-manifest state/om_backfill_manifest.jsonl

# Phase B: shard workers (run in parallel processes/hosts)
python3 scripts/om_compressor.py \
  --mode backfill \
  --claim-mode \
  --build-manifest state/om_backfill_manifest.jsonl \
  --shards 4 \
  --shard-index 0 \
  --max-chunks-per-run 100
```

Claim semantics:
- Claim state is persisted in SQLite (`<manifest>.claims.db`, table `chunk_claims`).
- Rows move `pending -> claimed -> done` (or `failed` on hard error).
- Claimed rows include a lease (`lease_expires_at`), so stale/abandoned claims are reclaimable.
- Done-confirm is enforced: claim row is marked `done` **only after** successful OM extraction/write and
  a Neo4j confirmation query verifies all chunk messages are marked extracted for that chunk ID.
- Shard isolation uses deterministic `chunk_id` hashing (`claim_shard % shards == shard_index`) to prevent
  overlapping execution across workers.

Optional lease override:

```bash
OM_CLAIM_LEASE_SECONDS=1800 python3 scripts/om_compressor.py --mode backfill --claim-mode ...
```

### Ontology Config

Default: `mcp_server/config/extraction_ontologies.yaml`

The compressor reads `schema_version` and the `om_extractor` block:

```yaml
schema_version: "2026-02-17"

om_extractor:
  model_id: "gpt-5.1-codex-mini"
  prompt_template: "<extraction instructions>"
```

Missing `schema_version` raises `SchemaVersionMissingError` and exits with code 1.

### OMNode Types Extracted

The fallback (rules-based) extractor classifies messages into:

| Type | Detection heuristic |
|---|---|
| `Judgment` | content contains "because" or "decision" |
| `OperationalRule` | content contains "rule" or "always" |
| `Commitment` | content contains "commit" or "promise" |
| `Friction` | content contains "problem", "friction", or "blocked" |
| `WorldState` | default fallback |

Allowed edge types (enforced via allowlist + regex — no free-form interpolation):
`MOTIVATES`, `GENERATES`, `SUPERSEDES`, `ADDRESSES`, `RESOLVES`

### Output Events (stdout JSONL)

| Event | When |
|---|---|
| `OM_TRIGGER_NOT_MET` | Backlog below threshold, nothing processed |
| `OM_MANIFEST_BUILT` | Backfill manifest + claim DB seeded (`--build-manifest`) |
| `OM_CHUNK_PROCESSED` | Successful chunk (includes `messages`, `nodes`, `edges` counts) |
| `OM_CHUNK_DONE_CONFIRM_FAILED` | Extraction returned but Neo4j done-confirm check failed; claim not marked done |
| `OM_CHUNK_FAILED` | Chunk extraction/claim processing error |
| `OM_NODE_CONTENT_MISMATCH` | Existing OMNode has different content hash — exits code 1 |
| `OM_DEAD_LETTER` | Message promoted to dead-letter queue |

---

## 3. Dead-Letter Lifecycle

Messages that repeatedly fail extraction enter the dead-letter queue.

### Threshold

```
DEAD_LETTER_ATTEMPTS = 3
```

After 3 failed extraction attempts (tracked per `Message` node as `om_extract_attempts`):
1. `m.om_dead_letter = true` is set on the Neo4j `Message` node.
2. A row is upserted into `state/candidates.db` (table `om_dead_letter_queue`).
3. `OM_DEAD_LETTER` event is emitted to stdout.

Dead-letter messages are permanently excluded from compressor processing.

### Inspecting the Dead-Letter Queue

```python
from truth import candidates as candidates_store

conn = candidates_store.connect(candidates_store.DB_PATH_DEFAULT)
dead = candidates_store.list_om_dead_letters(conn)
for row in dead:
    print(row["message_id"], row["attempts"], row["last_error"])
conn.close()
```

### Removing a Dead-Letter Entry (manual recovery)

```python
from truth import candidates as candidates_store

conn = candidates_store.connect(candidates_store.DB_PATH_DEFAULT)
candidates_store.remove_om_dead_letter(conn, "<message_id>")
conn.close()

# Then reset the Neo4j flag to allow reprocessing:
# MATCH (m:Message {message_id: "<message_id>"})
# SET m.om_dead_letter = false, m.om_extract_attempts = 0
```

### Split/Isolate Failure Recovery

When a parent chunk (≤50 messages) fails extraction:

1. **First failure:** `OMChunkFailure` node created, `attempts=1`.
2. **Second failure:** If `attempts >= 2` and `split_status='none'`:
   - Chunk is split into child chunks (`OMChunkChild` nodes) with `split_status='pending'`.
   - Child chunks are sized at `MAX_CHILD_CHUNK_SIZE = 10` messages each.
   - If the parent has ≤10 messages, `split_status='isolate'` (single-message children).
3. Each subsequent compressor run processes one child chunk at a time.
4. If a child chunk (multi-message) fails twice, it falls back to single-message isolation.
5. When all children are terminal, parent `split_status` advances to `'completed'`.

**Checking split state:**

```cypher
MATCH (p:OMChunkFailure)
WHERE p.split_status IN ['pending', 'isolate']
RETURN p.chunk_id, p.attempts, p.split_status, p.next_subchunk_index, size(p.child_chunk_ids) AS children
ORDER BY p.first_failed_at ASC
```

**Resetting a corrupt split:**

If `OM_SPLIT_STATE_CORRUPT` is emitted with a `violation_code`, the parent state is
inconsistent. Manual recovery:

```cypher
-- Option 1: Archive the parent (it will be skipped)
MATCH (p:OMChunkFailure {chunk_id: "<chunk_id>"})
SET p.split_status = 'completed', p.archived_at = datetime()

-- Option 2: Delete and allow reprocessing (messages stay unextracted)
MATCH (p:OMChunkFailure {chunk_id: "<chunk_id>"})
DETACH DELETE p
```

---

## 4. Convergence Engine

`scripts/om_convergence.py` drives the OMNode lifecycle state machine, reconciles the
dead-letter queue between Neo4j and the candidates SQLite DB, and optionally runs GC.

### Running the Convergence Engine

```bash
# Normal convergence pass (state transitions + dead-letter sync)
PYTHONPATH=. python3 scripts/om_convergence.py

# With GC (90-day TTL, default)
PYTHONPATH=. python3 scripts/om_convergence.py --run-gc

# GC with custom TTL
PYTHONPATH=. python3 scripts/om_convergence.py --run-gc --gc-days 60

# Dry-run GC (computes eligible counts without deleting)
PYTHONPATH=. python3 scripts/om_convergence.py --run-gc --gc-dry-run
```

### Lock

Convergence uses a fixed exclusive lock at `/tmp/om_graph_write.lock`:

```python
LOCK_PATH = Path("/tmp/om_graph_write.lock")
fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # blocking, exclusive
```

Never run compressor and convergence simultaneously without ensuring they use the same lock path.

### Convergence State Machine

OMNodes transition through these states:

```
OPEN ──────────────────────────────────────► MONITORING ──► CLOSED
  │   (addresses link or fresh addresses)         │              │
  │                                               │ (14d no      │
  │ (30d no mentions)                             │  reappear)   │
  ▼                                               ▼              │
ABANDONED ◄──────────────────────────────── ABANDONED           │
                                                                 │
     REOPENED ◄───────────────────────────────────────────────── ┘
       (reappearance in convergence window for any non-OPEN status)
```

**Transition precedence (highest first):**

1. `MONITORING | CLOSED | ABANDONED → REOPENED` — if a related `Friction` or `Commitment`
   event reappears in the convergence window.
2. `OPEN | REOPENED → MONITORING` — if a `Judgment` node with an `ADDRESSES` edge exists
   (or fresh `ADDRESSES` since last reopen).
3. `MONITORING → ABANDONED` — if node has been in monitoring ≥ 14 days with zero mentions.
4. `MONITORING → CLOSED` — if monitoring ≥ 14 days with mentions but no reappearance.
5. `OPEN → ABANDONED` — if node is ≥ 30 days old with zero mentions.
6. `REOPENED → ABANDONED` — if ≥ 30 days since reopen with no mentions or fresh addresses.

At most one transition per node per pass.

When a node transitions to `closed` and its type is `Friction` or `Commitment`, a
`RESOLVES` edge is automatically written from any addressing `Judgment` node.

### Watermark Mechanics (Bounded Pass)

Each convergence run processes at most `MAX_NODES_PER_CONVERGENCE_PASS = 500` nodes.

State is tracked in a singleton `OMConvergenceState` node:

```
state_id = "singleton"
last_convergence_at: ISO timestamp of last completed cycle
next_node_cursor:    node_id cursor for resuming mid-cycle
cycle_started_at:    timestamp when current cycle began
```

Nodes are iterated in `node_id ASC` order. After each pass:
- If more nodes remain: `next_node_cursor` is set to the last processed `node_id`.
- If all nodes processed: `next_node_cursor = NULL`, `last_convergence_at` updated.

Subsequent runs continue from the cursor until the cycle completes, then reset.

**Checking watermark state:**

```cypher
MATCH (s:OMConvergenceState {state_id: 'singleton'})
RETURN s.last_convergence_at, s.next_node_cursor, s.cycle_started_at
```

**Resetting a stuck cycle** (e.g., after bulk node deletion):

```cypher
MATCH (s:OMConvergenceState {state_id: 'singleton'})
SET s.next_node_cursor = NULL, s.cycle_started_at = NULL
```

### Output Events (stdout JSONL)

| Event | When |
|---|---|
| `OM_CONVERGENCE_WINDOW` | Start of each pass (includes cursor state) |
| `OM_CONVERGENCE_TRANSITION` | Each node state transition (includes from/to/reason) |
| `OM_CONVERGENCE_DONE` | End of pass (includes counts, GC summary, cycle_complete flag) |
| `OM_CONVERGENCE_FAILED` | Unhandled error (exits code 1) |

---

## 5. GC Invariants

The GC pass (triggered by `--run-gc`) deletes old `Message` and empty `Episode` nodes
from Neo4j. It respects strict retention gates:

### Eligibility (Message must satisfy ALL of the following)

```
m.graphiti_extracted_at IS NOT NULL        -- Graphiti processing completed
m.om_extracted = true                       -- OM extraction completed
m.om_dead_letter = false                    -- Not in dead-letter queue
m.created_at < cutoff_iso                   -- Older than TTL (default: 90 days)
NOT EXISTS { (m)-[:EVIDENCE_FOR]->(n:OMNode) WHERE n.status IN ['open','monitoring','reopened'] }
NOT EXISTS { (m)-[:SUPPORTS_CORE]->(c:CoreMemory) WHERE c.retention_status = 'active' }
```

**Retention gates:**
- `EVIDENCE_FOR` edge to an active-status OMNode → retained (node is still live)
- `SUPPORTS_CORE` edge to an active CoreMemory → retained (promoted evidence)

Empty `Episode` nodes (no remaining `HAS_MESSAGE` edges) older than the cutoff are also
eligible for deletion.

GC deletes at most 2,000 messages and 2,000 episodes per run. Run multiple times for
large backlogs.

### Default TTL

```
--gc-days 90   (default)
```

Always run `--gc-dry-run` first to verify eligible counts before deleting:

```bash
PYTHONPATH=. python3 scripts/om_convergence.py --run-gc --gc-dry-run
```

---

## 6. CoreMemory Promotion

`truth/promotion_policy_v3.py` promotes corroborated OMNode candidates to `CoreMemory`
nodes in Neo4j.

```bash
PYTHONPATH=. python3 truth/promotion_policy_v3.py \
  --candidate-id <candidate_id> \
  [--candidates-db /path/to/candidates.db]
```

### Promotion Contract

1. Reads verification record from `candidates.db`.
2. **Blocks if** `verification_status != 'corroborated'`.
3. Runs hard-block checks via allowlisted policy modules (fail-closed: if unavailable,
   blocks promotion).
4. Derives `core_memory_id = SHA256("core|{candidate_id}")` — stable and idempotent.
5. MERGEs a `CoreMemory` node linked to the source `OMNode`.
6. Writes `SUPPORTS_CORE` edges from each `evidence_source_ids` `Message` to the new
   `CoreMemory` node.

Hard-block check import order (first available wins):
`truth.policy_scanner`, `truth.security_policy`, `truth.promotion_policy`

Override via `OM_HARD_BLOCK_IMPORT=truth.policy_scanner:hard_block_check`.

### Candidates DB

Default path: `state/candidates.db`

The candidates SQLite DB tracks OM dead letters and candidate verification:

```bash
# List dead letters
python3 -c "
from truth import candidates as c
conn = c.connect(c.DB_PATH_DEFAULT)
for r in c.list_om_dead_letters(conn): print(r)
"

# Upsert a verification record
python3 -c "
from truth import candidates as c
conn = c.connect(c.DB_PATH_DEFAULT)
c.upsert_candidate_verification(conn,
  candidate_id='<id>',
  verification_status='corroborated',
  evidence_source_ids=['msg_id_1', 'msg_id_2'],
  verifier_version='v1.0')
"
```

---

## 7. Neo4j Constraints and Indexes

The compressor auto-creates these on first run (idempotent):

```cypher
CONSTRAINT message_message_id       UNIQUE ON Message.message_id
CONSTRAINT omnode_node_id           UNIQUE ON OMNode.node_id
CONSTRAINT corememory_core_memory_id UNIQUE ON CoreMemory.core_memory_id
CONSTRAINT omchunkfailure_chunk_id  UNIQUE ON OMChunkFailure.chunk_id
CONSTRAINT omchunkchild_child_id    UNIQUE ON OMChunkChild.child_id
CONSTRAINT omextractionevent_event_id UNIQUE ON OMExtractionEvent.event_id
CONSTRAINT omconvergence_state_id   UNIQUE ON OMConvergenceState.state_id
INDEX om_extraction_event_emitted_at ON OMExtractionEvent.emitted_at
INDEX om_extraction_event_semantic_domain ON OMExtractionEvent.semantic_domain
INDEX om_extraction_event_node_id   ON OMExtractionEvent.node_id
```

---

## 8. Common Failure Modes

### Compressor exits code 1 with `OM_NODE_CONTENT_MISMATCH`

Two different message batches produced an OMNode with the same `node_id` but different
content. This indicates a hash collision in the deterministic ID derivation or duplicate
ingestion of slightly mutated content.

**Recovery:** Inspect the conflicting node. If content is effectively the same, set
`OM_REWRITE_EMBEDDINGS=1` to allow the compressor to rewrite the embedding.

```bash
OM_REWRITE_EMBEDDINGS=1 python3 scripts/om_compressor.py --force
```

### `OM_SPLIT_STATE_CORRUPT` emitted, compressor exits

The `OMChunkFailure` split state failed an integrity check (`check_1` through `check_7`).

**Recovery:** See [Split/Isolate Failure Recovery](#splitisolate-failure-recovery) above.

### Convergence cycle stuck (cursor never resets)

If the OMNode set is large, multiple convergence runs may be needed to complete a cycle.
Check `OMConvergenceState.cycle_started_at` — if it's very old and `next_node_cursor`
is non-null, the cycle is progressing but slow.

If the cursor advances but never resets after full iteration, check for nodes being added
faster than the cycle completes. Consider increasing cron frequency.

### Dead-letter queue growing continuously

Large dead-letter growth usually indicates an embedding endpoint problem (model
unavailable, dimension mismatch) or malformed message content.

1. Check `OM_DEAD_LETTER` events for `last_error`.
2. Verify the embedding endpoint is responding: `curl $EMBEDDER_BASE_URL/embeddings`
3. After fixing root cause, reset affected messages (see [Dead-Letter Lifecycle](#3-dead-letter-lifecycle)).

### `SchemaVersionMissingError` on compressor start

The `schema_version` key is missing from `mcp_server/config/extraction_ontologies.yaml`.

Ensure the file contains:
```yaml
schema_version: "2026-02-17"
```

### Lock contention (compressor hangs)

If a previous compressor or convergence run crashed while holding the lock:

```bash
# Check if any process holds the lock
fuser /tmp/om_graph_write.lock 2>/dev/null
# or (repo-local lock):
fuser state/locks/om_graph_write.lock 2>/dev/null

# If no process holds it, the lock is stale. Restart the stalled script
# or delete the lock file (fcntl locks are released on process exit).
```

---

## 9. Recommended Cron Schedule

```cron
# Compressor: every 20 minutes
*/20 * * * * cd /path/to/bicameral && python3 scripts/om_compressor.py >> logs/om-compressor.log 2>&1

# Convergence: every hour (includes dead-letter reconciliation)
0 * * * * cd /path/to/bicameral && PYTHONPATH=. python3 scripts/om_convergence.py >> logs/om-convergence.log 2>&1

# GC: daily at 03:00 (dry-run on weekdays, real on Sunday)
0 3 * * 0 cd /path/to/bicameral && PYTHONPATH=. python3 scripts/om_convergence.py --run-gc >> logs/om-gc.log 2>&1
0 3 * * 1-6 cd /path/to/bicameral && PYTHONPATH=. python3 scripts/om_convergence.py --run-gc --gc-dry-run >> logs/om-gc-dry.log 2>&1
```

---

## See Also

- [Custom Ontologies](../custom-ontologies.md) — OM lane ontology config (`s1_observational_memory`)
- [Dual-Brain Architecture](../DUAL-BRAIN-ARCHITECTURE.md) — OM as the Dual Brain synthesis/control layer
- [Memory Runtime Wiring](../MEMORY-RUNTIME-WIRING.md) — fast-write integration paths
- `scripts/om_fast_write.py --help`
- `scripts/om_compressor.py --help`
- `scripts/om_convergence.py --help` (run with `PYTHONPATH=.`)

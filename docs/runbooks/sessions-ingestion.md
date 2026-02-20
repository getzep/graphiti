# Sessions Ingestion Runbook

## Overview

This runbook covers the full lifecycle of Graphiti extraction: batch ingestion for initial/re-extraction, steady-state configuration for ongoing operations, high-throughput tuning, sub-chunking, and the post-processing pipeline that keeps the graph clean.

**Default backend: Neo4j.** FalkorDB is legacy-only. Neo4j is strongly recommended for high-throughput extraction — FalkorDB's single-threaded write model saturates at low concurrency.

---

## Table of Contents

1. [Architecture: How Extraction Works](#architecture-how-extraction-works)
2. [Graph Groups & Ontologies](#graph-groups--ontologies)
3. [Batch Ingestion (Initial / Re-Extraction)](#batch-ingestion)
4. [High-Throughput Tuning](#high-throughput-tuning)
5. [Sub-Chunking](#sub-chunking)
6. [Steady-State Configuration](#steady-state-configuration)
7. [Post-Processing Pipeline](#post-processing-pipeline)
8. [Adding New Data Sources](adding-data-sources.md)
9. [Troubleshooting & Failure Modes](#troubleshooting--failure-modes)
10. [Operational Learnings](#operational-learnings)

---

## Architecture: How Extraction Works

```
Evidence (JSON files)
    ↓  enqueue via ingest scripts
MCP Server (add_memory endpoint)
    ↓  per-group async queue (QueueService)
Graphiti Core (add_episode)
    ↓  LLM extraction → entity resolution → embedding → graph write
Neo4j (nodes, relationships, episodic timeline)
```

### Key Components

| Component | Role | Config |
|-----------|------|--------|
| **Ingest scripts** | Parse source data → evidence JSON → enqueue to MCP | `scripts/mcp_ingest_sessions.py`, `scripts/ingest_content_groups.py`, etc. |
| **MCP Server** | Hosts `add_memory` tool, manages per-group queues | `mcp_server/main.py`, port 8000 (default) |
| **QueueService** | Async episode processing with per-group serial queues | `mcp_server/src/services/queue_service.py` |
| **OntologyRegistry** | Per-group entity type resolution | `mcp_server/config/extraction_ontologies.yaml` |
| **Graphiti Core** | LLM-powered entity/relationship extraction + graph writes | `graphiti_core/` |
| **Neo4j** | Graph database (single DB, group_id property scoping) | `bolt://localhost:7687` |

### Per-Group Queue Isolation

The MCP server maintains **separate async queues per `group_id`**. This means:

- Episodes within the same group are processed according to `GRAPHITI_QUEUE_CONCURRENCY` (serial by default)
- Different groups process in parallel (bounded by `SEMAPHORE_LIMIT`)
- Ontology profiles are resolved per-call, not per-shard — a single MCP instance handles all groups correctly
- No risk of cross-group contamination: `group_id` is stamped on every node/edge at write time

### Evidence → Episode Flow

1. Ingest script reads evidence JSON (or generates from source DB/files)
2. Large evidence is **sub-chunked** at enqueue time (>10k chars → split into `:p0`, `:p1`, ... parts)
3. Each chunk is sent to MCP `add_memory` with: `name`, `episode_body`, `group_id`, `source_description`
4. MCP queues it in the group's async queue
5. Graphiti Core processes: LLM entity extraction → entity resolution (dedup against existing graph) → embedding → Neo4j write
6. Each episode creates: 1 Episodic node + N Entity nodes + M Relationship edges + episodic edges

### Important: Workers vs MCP Drain

Ingest worker processes exit 0 when **enqueuing** is done, NOT when extraction is complete. The MCP server drains queued episodes asynchronously. To know when extraction is truly finished, **poll the Neo4j episode count** — not the worker exit state.

---

## Graph Groups & Ontologies

### Defining Graph Groups

Each data source should be assigned a unique `group_id` — an isolated namespace in Neo4j. Convention: `s1_<domain>` (snake_case).

Example groups for a typical deployment:

| Group ID | Source | Ontology | Ingest Script |
|----------|--------|----------|---------------|
| `s1_sessions` | Agent session transcripts | Custom | `mcp_ingest_sessions.py` |
| `s1_documents` | Document corpus | Custom | `ingest_content_groups.py` |
| `s1_notes` | Markdown notes | Default | `mcp_ingest_sessions.py` |
| `engineering_learnings` | Engineering compound notes | Custom | `ingest_compound_notes.py` |

### Ontology Configuration

Custom entity types are defined in `mcp_server/config/extraction_ontologies.yaml`. Groups without an explicit ontology entry fall back to the global default entity types from `config.yaml`.

The ontology resolver is called **per-episode** at extraction time (not per-shard), so a single MCP instance correctly handles all groups with their respective ontologies.

---

## Batch Ingestion

Use batch mode for initial extraction, re-extraction after provider/embedding model changes, or disaster recovery.

### When You Need Batch Mode

- Switching embedding models (dimension change makes old vectors incompatible → full re-extract)
- Switching LLM providers (different extraction quality → may want fresh graph)
- Neo4j database wipe/recovery
- First-time setup

### Pre-Flight Checklist

1. **Wipe Neo4j** (if re-extracting): `MATCH (n) DETACH DELETE n`
2. **Reset ingest registry**: `DELETE FROM extraction_tracking WHERE group_id = '<group>'`
3. **Reset cursors** (for cursor-based scripts) in your registry DB
4. **Verify evidence files exist** for all groups
5. **Verify LLM provider** has sufficient credits/quota
6. **Verify embedding service** is running with the correct model loaded

### Batch Configuration: Maximize Throughput

```bash
# Launch multiple MCP shards for parallel extraction
SEMAPHORE_LIMIT=20              # 20 concurrent extractions per shard
GRAPHITI_QUEUE_CONCURRENCY=20   # parallel within each group (batch only!)
GRAPHITI_MAX_EPISODE_BODY_CHARS=12000

for port in 8000 8001 8002 8003; do
  SEMAPHORE_LIMIT=20 python mcp_server/main.py --port $port &
done
```

### Launching Batch Workers

```bash
# Example: sessions with 20 workers across 4 shards
for shard in $(seq 0 19); do
  port=$((8000 + shard / 5))
  nohup python scripts/mcp_ingest_sessions.py \
    --mcp-url http://localhost:${port}/mcp \
    --group-id s1_sessions \
    --shards 20 --shard-index $shard \
    --force \
    > logs/worker-${shard}.log 2>&1 &
done

# Content groups (sequential with drain-waiting)
nohup python scripts/ingest_content_groups.py \
  --backend neo4j --mcp-url http://localhost:8001/mcp \
  --force --sleep 0.1 --poll 15 --stable-checks 15 --max-wait 7200 \
  > logs/content-groups.log 2>&1 &
```

### Monitoring Batch Progress

```bash
# Episode counts by group (Neo4j)
cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (e:Episodic) RETURN e.group_id AS gid, count(e) AS cnt ORDER BY cnt DESC"

# Or via the extraction monitor script
python scripts/extraction_monitor.py --backend neo4j
```

### Batch Throughput Reference

Throughput varies by hardware, LLM provider, and graph size. Approximate ranges:

| Configuration | Throughput | Notes |
|---------------|-----------|-------|
| 4 shards × SEMAPHORE=20 (small graph) | ~2,000-2,500 ep/hr | Peak, <2k nodes |
| 4 shards × SEMAPHORE=20 (large graph) | ~1,200-1,500 ep/hr | 10k+ nodes, entity resolution scaling |
| Single shard × SEMAPHORE=10 | ~400-600 ep/hr | Conservative, good dedup quality |

**Why throughput degrades with graph size:** Entity resolution during `add_episode` performs similarity searches against existing nodes. More nodes = more comparisons per episode.

---

## High-Throughput Tuning

Extraction speed at scale is bounded by three limits, encountered in this order:

1. **Database query queue depth** — Neo4j and FalkorDB both cap concurrent inflight queries
2. **`SEMAPHORE_LIMIT`** — The MCP server's asyncio semaphore capping concurrent `add_episode` coroutines
3. **Single-instance throughput ceiling** — One MCP instance can only sustain `SEMAPHORE_LIMIT` concurrent extractions; sharding multiplies capacity

### Database Concurrency Limits

#### Neo4j: Connection Pool

The bottleneck in Neo4j is bolt thread pool exhaustion, surfacing as `neo4j.exceptions.ServiceUnavailable` or connection pool timeout errors.

| Setting | Neo4j 4.x | Neo4j 5.x |
|---|---|---|
| Server-side thread pool | `dbms.connector.bolt.thread_pool_max_size` | `server.bolt.thread_pool_max_size` |
| Default | 400 | 400 |
| Config file | `neo4j.conf` | `neo4j.conf` |

On the client side (neo4j Python driver):

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
    max_connection_pool_size=100,  # default: 100
)
```

Raise `max_connection_pool_size` to match your expected concurrency: `num_instances * SEMAPHORE_LIMIT * 3`.

#### FalkorDB: MAX_QUEUED_QUERIES (legacy only)

FalkorDB queues incoming graph queries up to a configurable maximum. When full, new queries are rejected with `Max pending queries exceeded`.

```bash
# Check current value
redis-cli -p 6379 GRAPH.CONFIG GET MAX_QUEUED_QUERIES

# Set at runtime (no restart)
redis-cli -p 6379 GRAPH.CONFIG SET MAX_QUEUED_QUERIES 2500
```

**Rule of thumb:** `MIN_MAX_QUEUED_QUERIES = num_mcp_instances * SEMAPHORE_LIMIT * 3`

### Graphiti Semaphore (SEMAPHORE_LIMIT)

The MCP server caps concurrent `add_episode` coroutines with an `asyncio.Semaphore`. This is the primary throughput knob.

| Env var | Default |
|---|---|
| `SEMAPHORE_LIMIT` | 10 |

#### Guidance by LLM Rate-Limit Tier

| Provider / Tier | Approx RPM | Recommended SEMAPHORE_LIMIT |
|---|---|---|
| OpenAI Tier 1 (free) | 3 | 1-2 |
| OpenAI Tier 2 | 60 | 5-8 |
| OpenAI Tier 3 | 500 | 10-15 |
| OpenAI Tier 4 | 5,000 | 20-50 |
| Anthropic default | 50 | 5-8 |
| Anthropic high tier | 1,000 | 15-30 |

**Key rule:** Always raise the database concurrency limit **before** raising `SEMAPHORE_LIMIT`.

### Sharding Across Multiple MCP Instances

When single-instance throughput is insufficient for a large backlog (roughly >500 episodes), shard across multiple MCP instances:

1. Launch N MCP instances on different ports, all pointing at the same Neo4j DB
2. Partition episodes across instances by shard index (modulo N)
3. Pass `--mcp-url http://localhost:<PORT>/mcp` per shard

```bash
# Example: 3 shards for a graph group
for port in 8001 8002 8003; do
  SEMAPHORE_LIMIT=20 python mcp_server/main.py --port $port &
done

# Run ingestion in parallel across shards
python scripts/ingest_content_groups.py --shards 3 --shard-index 0 --mcp-url http://localhost:8001/mcp &
python scripts/ingest_content_groups.py --shards 3 --shard-index 1 --mcp-url http://localhost:8002/mcp &
python scripts/ingest_content_groups.py --shards 3 --shard-index 2 --mcp-url http://localhost:8003/mcp &
```

**Warning:** Do NOT pass `--force` on restart. That flag is only for the first clean-slate run.

### Sizing Formula

```
total_concurrent_slots = num_instances * SEMAPHORE_LIMIT
min_db_queue = total_concurrent_slots * 3
```

| Instances | SEMAPHORE_LIMIT | Concurrent Slots | Min DB Queue |
|---|---|---|---|
| 1 | 10 | 10 | 30 |
| 3 | 15 | 45 | 135 |
| 6 | 15 | 90 | 270 |
| 6 | 30 | 180 | 540 |

---

## Sub-Chunking

Large evidence (sessions can be 25k+ chars) will exceed the LLM context window during extraction. The ingestion scripts **deterministically sub-chunk** large evidence at enqueue time.

### How It Works

1. **Size check**: If evidence exceeds `--subchunk-size` (default: 10,000 chars), it is split
2. **Deterministic splitting**: Split on paragraph boundaries (double newlines), falling back to single newlines, then hard splits — same input always produces the same sub-chunks
3. **Stable keys**: Each sub-chunk gets a `:p0`, `:p1`, `:p2`, ... suffix appended to the original chunk key (e.g., `session:2026-02-19:c3:p0`, `session:2026-02-19:c3:p1`)
4. **Registry dedup**: Each sub-chunk has its own content hash and registry entry — re-runs skip already-ingested sub-chunks (idempotent)
5. **Episode count**: Sub-chunking increases total episode count — this is expected and acceptable

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--subchunk-size` | 10,000 | Max characters per sub-chunk |

### When to Adjust

- Extraction still hits context limits → reduce to 5,000
- Episodes are too fragmented → increase to 15,000
- Default of 10,000 chars leaves comfortable headroom for most LLM context windows

### Design Rationale

Previous approaches used runtime hacks (truncation, retry-with-shrink, reduced previous-episode window). These were fragile: truncation lost information, retry-with-shrink was non-deterministic and wasted API calls, and reduced prev-episode window degraded quality for all sessions. Sub-chunking at enqueue time is deterministic, lossless, and idempotent.

---

## Steady-State Configuration

After batch ingestion completes, switch to a configuration that prioritizes **extraction quality over throughput**.

### Why Quality > Speed for Steady-State

- **Serial per-group processing** means each episode sees the latest graph state → entity resolution catches duplicates in real-time
- **Timeline integrity** is maintained when episodes are processed in chronological order within each group
- **Incremental volume is low** — serial handles typical daily volumes in minutes
- **Multiple groups process in parallel** — the semaphore allows N groups to extract simultaneously

### Steady-State MCP Config

```bash
# Single MCP shard (port 8000)
SEMAPHORE_LIMIT=8              # max 8 groups extracting simultaneously
GRAPHITI_QUEUE_CONCURRENCY=1   # SERIAL within each group (key for dedup + timeline)
GRAPHITI_MAX_EPISODE_BODY_CHARS=12000
```

**Why `CONCURRENCY=1`:** Each episode's entity resolution compares against the current graph state. With concurrency=1, episode N+1 sees episode N's entities. With concurrency=20, episodes N+1 through N+20 all see the same stale snapshot → duplicated entities.

**Why `SEMAPHORE_LIMIT=8`:** Total concurrent extraction slots across all groups. Adjust based on your number of active groups.

### Steady-State Cron Schedule (Example)

| Cron | Script | Frequency | Notes |
|------|--------|-----------|-------|
| sessions | `mcp_ingest_sessions.py --incremental` | Every 30 min | Delta since last watermark |
| content groups | `ingest_content_groups.py` | After new content | Triggered by new data |
| dedupe_nodes | `dedupe_nodes.py --backend neo4j` | Daily | Merge duplicate entities |
| repair_timeline | `repair_timeline.py --backend neo4j` | Daily (after dedup) | Rebuild NEXT_EPISODE chains |

### Switching from Batch to Steady-State

1. **Verify batch is complete** — poll Neo4j episode counts against expected totals
2. **Stop all batch shards and workers**
3. **Run post-processing** (see next section)
4. **Start single production shard** with steady-state config
5. **Enable cron jobs** for incremental ingest + daily maintenance

---

## Post-Processing Pipeline

Run these **after batch ingestion completes** (or periodically during steady-state).

### 1. Deduplicate Entities

High-concurrency batch extraction creates duplicate entity nodes.

```bash
# Dry run first
python scripts/dedupe_nodes.py --backend neo4j --group-id s1_sessions --dry-run

# Execute (destructive — merges duplicate nodes)
python scripts/dedupe_nodes.py --backend neo4j --group-id s1_sessions --confirm-destructive
```

### 2. Repair Timeline

Batch extraction creates isolated episodes (0% NEXT_EPISODE linkage). The repair script rebuilds chronological chains.

```bash
python scripts/repair_timeline.py --backend neo4j --group-id s1_sessions --confirm-destructive
```

### 3. Graph Health Check

```cypher
-- Contamination check (should be zero)
MATCH (a)-[r:RELATES_TO]->(b) WHERE a.group_id <> b.group_id RETURN count(r)

-- Duplicate count per group
MATCH (n:Entity) WITH n.group_id as gid, toLower(n.name) as name, count(n) as cnt
WHERE cnt > 1 RETURN gid, count(name) as duped_names, sum(cnt) as total_dupes

-- Timeline coverage
MATCH (e:Episodic)-[:NEXT_EPISODE]->() RETURN e.group_id as gid, count(*) as linked
```

---

## Troubleshooting & Failure Modes

### LLM Provider Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `402 Insufficient credits` | Provider balance depleted | Top up credits |
| `429 Rate limited` | Provider rate limit exceeded | Reduce `SEMAPHORE_LIMIT` |
| `context_length_exceeded` | Episode too large for LLM context | Reduce `--subchunk-size` or `GRAPHITI_MAX_EPISODE_BODY_CHARS` |

### Database Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `Max pending queries exceeded` | FalkorDB queue full | Raise `MAX_QUEUED_QUERIES` or reduce concurrency |
| `ServiceUnavailable` | Neo4j connection pool exhausted | Raise `max_connection_pool_size` in driver config |
| `episode_body too large` | Content exceeds `MAX_EPISODE_BODY_CHARS` | Sub-chunk upstream or raise the limit |

### Embedding Issues

| Error | Cause | Fix |
|-------|-------|-----|
| Dimension mismatch | Old embeddings mixed with new model | Full re-extraction required — wipe Neo4j + registry |

**Critical: Embedding model changes require full re-extraction.** You cannot mix embeddings from different models in the same graph.

### Stale Cursors / Registry State

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Done: 0/0 ingested` | Cursor thinks everything is already processed | Reset cursor in registry DB |
| Workers skip all chunks | Registry marks chunks as already queued | Delete from `extraction_tracking` or use `--force` |

### Monitoring During Extraction

```bash
# Episode count (Neo4j)
cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (e:Episodic) WHERE e.group_id = '<group_id>' RETURN count(e)"

# Backend-agnostic CLI
python scripts/extraction_monitor.py --backend neo4j
```

Watch MCP server logs for `Max pending queries exceeded` (FalkorDB) or HTTP 429 (LLM rate limits). If either appears, reduce `SEMAPHORE_LIMIT`.

---

## Operational Learnings

### Throughput vs Quality Tradeoff

**Batch mode** (high concurrency): Optimized for speed. Creates significant duplicates and zero timeline links. Requires post-processing (dedup + timeline repair). Use for initial/re-extraction only.

**Steady-state** (serial per-group): Optimized for quality. Entity resolution sees latest graph state → minimal duplicates. Timeline maintained naturally by chronological processing order. Use for ongoing operations.

**Never run batch-mode concurrency in steady-state.** The cleanup cost exceeds the time saved.

### FalkorDB → Neo4j Migration

FalkorDB's single-threaded write model can't handle concurrent extraction pipelines without constant timeouts. Neo4j handles high concurrency without issue. Migration requires full re-extraction if embedding dimensions differ.

### Sub-Chunking is Essential

Without sub-chunking, large evidence causes `context_length_exceeded` errors, retry-with-shrink loops that waste API calls, and truncated content that loses information. Sub-chunking at enqueue time (default 10k chars) is deterministic, lossless, and idempotent. **Enable it for all groups.**

### Cross-Contamination Prevention

Neo4j uses a single database with `group_id` property scoping. Ingest scripts always pass `group_id` explicitly per-call. Cross-contamination is architecturally prevented as long as ingest scripts set `group_id` correctly. Verify with the contamination check query above.

### Graph Size Impacts Throughput

Entity resolution during `add_episode` performs similarity searches against existing nodes. Throughput degrades as graph size grows — this is expected. Post-extraction dedup (which merges duplicates) reduces node count and improves subsequent extraction performance.

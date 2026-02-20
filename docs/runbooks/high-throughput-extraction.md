# High-Throughput Extraction

## Overview

**Default backend (2026-02-19): Neo4j.** FalkorDB is legacy-only.
Use Neo4j for any high-throughput ingest to avoid Redis single-thread bottlenecks.

Graphiti extraction speed and reliability at scale are bounded by three limits, encountered in this order:

1. **Database query queue depth** — FalkorDB and Neo4j both cap concurrent inflight queries. Exceed the cap and the database rejects work.
2. **`SEMAPHORE_LIMIT`** — The MCP server's asyncio semaphore that caps concurrent `add_episode` coroutines. Too low wastes capacity; too high without a matching DB queue bump causes rejections.
3. **Single-instance throughput ceiling** — One MCP server on one port can only sustain `SEMAPHORE_LIMIT` concurrent extractions. For large backlogs you shard episodes across multiple MCP instances.

Each section below covers one limit: what it is, how to detect it, how to raise it safely.

## 1. Database Concurrency Limits

### FalkorDB: MAX_QUEUED_QUERIES (legacy only)

FalkorDB queues incoming graph queries up to a configurable maximum. When the queue is full, new queries are rejected immediately with:

```
Max pending queries exceeded
```

| Setting | Value |
|---|---|
| Config key | `MAX_QUEUED_QUERIES` |
| Default | 500 |
| Scope | Per redis-server process |

**Check the current value:**

```bash
redis-cli -p 6379 GRAPH.CONFIG GET MAX_QUEUED_QUERIES
```

**Set at runtime (no restart):**

```bash
redis-cli -p 6379 GRAPH.CONFIG SET MAX_QUEUED_QUERIES 2500
```

**Set at startup** — add to redis-server args:

```bash
redis-server --loadmodule /path/to/falkordb.so MAX_QUEUED_QUERIES 2500
```

**Rule of thumb:**

```
MIN_MAX_QUEUED_QUERIES = num_mcp_instances * SEMAPHORE_LIMIT * 3
```

The 3x headroom factor accounts for the fact that a single `add_episode` call fans out into multiple graph queries (entity extraction, deduplication, edge resolution, etc.).

### Neo4j: Connection Pool

The equivalent bottleneck in Neo4j is bolt thread pool exhaustion, surfacing as `neo4j.exceptions.ServiceUnavailable` or connection pool timeout errors.

| Setting | Neo4j 4.x | Neo4j 5.x |
|---|---|---|
| Server-side thread pool | `dbms.connector.bolt.thread_pool_max_size` | `server.bolt.thread_pool_max_size` |
| Default | 400 | 400 |
| Config file | `neo4j.conf` | `neo4j.conf` |

On the client side (neo4j Python driver), also check:

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
    max_connection_pool_size=100,  # default: 100
)
```

Raise `max_connection_pool_size` to match your expected concurrency. The same sizing formula applies: `num_instances * SEMAPHORE_LIMIT * 3`.

## 2. Graphiti Semaphore (SEMAPHORE_LIMIT)

The MCP server caps concurrent `add_episode` coroutines with an `asyncio.Semaphore`. This is the primary throughput knob.

| Env var | `SEMAPHORE_LIMIT` |
|---|---|
| Default | 10 |
| Set by | Environment variable on MCP server process |

### Guidance by LLM Rate-Limit Tier

| Provider / Tier | Approx RPM | Recommended SEMAPHORE_LIMIT |
|---|---|---|
| OpenAI Tier 1 (free) | 3 | 1-2 |
| OpenAI Tier 2 | 60 | 5-8 |
| OpenAI Tier 3 | 500 | 10-15 |
| OpenAI Tier 4 | 5,000 | 20-50 |
| Anthropic default | 50 | 5-8 |
| Anthropic high tier | 1,000 | 15-30 |

### Key sequencing rule

Always raise the database queue limit **before** raising `SEMAPHORE_LIMIT`. If you bump the semaphore without raising `MAX_QUEUED_QUERIES` (FalkorDB) or the connection pool (Neo4j), you will hit database rejections under load.

## 3. Sharding Across Multiple MCP Instances

### When to shard

Single-instance throughput is insufficient for a large backlog (roughly >500 episodes on a time-sensitive extraction). One MCP instance can sustain at most `SEMAPHORE_LIMIT` concurrent extractions; sharding multiplies that by the number of instances.

### How to shard

1. Launch N MCP instances on different ports.
   - **Neo4j (default):** all instances point at the same Neo4j DB (`NEO4J_URI`), and use `GRAPHITI_GROUP_ID` to namespace data within the single DB.
   - **FalkorDB (legacy):** set `FALKORDB_DATABASE` per graph.
2. Partition episodes across instances by shard index (modulo N on episode index).
3. Pass `--mcp-url http://localhost:<PORT>/mcp` to the ingestion script for each shard.

### Example: 3 instances for `my_graph` (Neo4j default)

```bash
SEMAPHORE_LIMIT=15 GRAPHITI_GROUP_ID=my_graph \
  NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=... \
  python mcp_server/main.py --database-provider neo4j --port 8001 &

SEMAPHORE_LIMIT=15 GRAPHITI_GROUP_ID=my_graph \
  NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=... \
  python mcp_server/main.py --database-provider neo4j --port 8002 &

SEMAPHORE_LIMIT=15 GRAPHITI_GROUP_ID=my_graph \
  NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=... \
  python mcp_server/main.py --database-provider neo4j --port 8003 &
```

**FalkorDB (legacy) example:** replace the `NEO4J_*` vars with `FALKORDB_DATABASE=my_graph` and set `--database-provider falkordb`.

Then run the ingestion script three times in parallel, each targeting one shard:

```bash
python scripts/ingest_content_groups.py --shards 3 --shard-index 0 --mcp-url http://localhost:8001/mcp &
python scripts/ingest_content_groups.py --shards 3 --shard-index 1 --mcp-url http://localhost:8002/mcp &
python scripts/ingest_content_groups.py --shards 3 --shard-index 2 --mcp-url http://localhost:8003/mcp &
```

The `--shards` / `--shard-index` flags in `scripts/ingest_content_groups.py` modulo-partition the episode list so no episode is processed twice.

### Warning: do NOT pass `--force` on restart

If a shard partially completes and you restart it, do **not** pass `--force`. That flag is only for the first clean-slate run. Re-running with `--force` creates duplicate episodes.

## 4. Putting It All Together: Sizing Formula

```
total_concurrent_slots = num_instances * SEMAPHORE_LIMIT
min_MAX_QUEUED_QUERIES = total_concurrent_slots * 3
```

**Worked example:** 6 instances x 15 semaphore = 90 concurrent slots. Minimum `MAX_QUEUED_QUERIES` = 270. The default of 500 was sufficient at this scale.

**Scaling table:**

| Instances | SEMAPHORE_LIMIT | Concurrent Slots | Min MAX_QUEUED_QUERIES |
|---|---|---|---|
| 1 | 10 | 10 | 30 |
| 3 | 15 | 45 | 135 |
| 6 | 15 | 90 | 270 |
| 6 | 30 | 180 | 540 |
| 10 | 50 | 500 | 1500 |

## 5. Monitoring During Extraction

### Episode count (progress indicator)

```bash
# Neo4j (default)
cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (e:Episodic) WHERE e.group_id = '<group_id>' RETURN count(e)"

# FalkorDB (legacy)
redis-cli -p 6379 GRAPH.QUERY <db> "MATCH (e:Episodic) RETURN count(e)"
```

Replace `<group_id>` and `<db>` with your target lane.

### FalkorDB queue pressure

Watch MCP server logs for:

```
Max pending queries exceeded
```

If these appear, reduce `SEMAPHORE_LIMIT` or raise `MAX_QUEUED_QUERIES`.

### LLM rate limits

Watch for HTTP 429 responses in MCP server logs. If they appear, reduce `SEMAPHORE_LIMIT` to stay within your provider's rate limit tier.

### Workers vs. MCP drain

Ingestion worker processes exit 0 when **queuing** is done, not when extraction is complete. The MCP server drains queued episodes asynchronously. To know when extraction is truly finished, poll the episode count — not the worker process exit state.

---

## FalkorDB-only scripts (legacy)

These scripts are FalkorDB-specific and should **not** be used with Neo4j:

- `scripts/cleanup_misplacements_all_graphs.py`
- `scripts/scan_misplacements_all_graphs.py`
- `scripts/extraction_monitor.py`
- `scripts/ingest_content_groups.py`

(Neo4j equivalents require different queries and group_id scoping.)

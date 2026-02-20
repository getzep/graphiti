# Adding New Data Sources to the Knowledge Graph

This runbook covers everything needed to add a new data source to the Graphiti extraction pipeline. Each source becomes a **graph group** — an isolated namespace in Neo4j with its own ontology, ingest script, and cron schedule.

**Prerequisites:** Read [sessions-ingestion.md](sessions-ingestion.md) for architecture context. Your MCP server should be running in steady-state mode (`CONCURRENCY=1`).

---

## The Checklist

Every new data source needs exactly four things:

1. **A `group_id`** — namespace identifier for all nodes/edges from this source
2. **An ontology** — custom entity/relationship types (optional but strongly recommended)
3. **An ingest adapter script** — reads source data, formats episodes, sends to MCP
4. **A cron entry** — determines how often new data is ingested

---

## Step 1: Choose a `group_id`

Convention: `s1_<domain>_<source>` (snake_case).

Examples:
- `s1_sessions` — agent session transcripts
- `s1_documents` — document corpus
- `s1_notes` — markdown notes
- `s1_articles` — article/blog content
- `engineering_learnings` — engineering compound notes

**Rules:**
- Must be unique across all groups (check `extraction_ontologies.yaml`)
- Keep it short but descriptive — this appears in every node/edge in Neo4j
- The `s1_` prefix is conventional but not enforced

---

## Step 2: Define an Ontology

Add a block to `mcp_server/config/extraction_ontologies.yaml`:

```yaml
s1_my_source:
  extraction_emphasis: >-
    Focus on the key concepts, relationships, and facts relevant to
    this domain. Extract what matters for downstream retrieval.
  entity_types:
    - name: MyEntityType
      description: "Description of what this entity represents"
    - name: AnotherType
      description: "Description of this entity type"
  relationship_types:
    - name: RELATES_TO
    - name: BELONGS_TO
```

### Ontology Design Principles

- **Be specific.** Generic entity types ("Thing", "Concept") produce noisy graphs. Domain-specific types extract targeted, useful knowledge.
- **`extraction_emphasis` matters.** This is injected into the LLM prompt. Tell it what to focus on and what to ignore. Be opinionated.
- **Keep entity types to 5-10.** More types = more extraction cost + more fragmented graph. You can always add types later.
- **Relationship types are optional but valuable.** Without them, Graphiti uses generic relationships. With them, you get structured edges that enable meaningful graph traversals.
- **Groups without an explicit ontology fall back to global defaults.** This works fine for general-purpose content but produces poor results for domain-specific data.

### How Ontology Resolution Works

The `OntologyRegistry` resolves ontologies **per-episode at extraction time** (not per-shard or per-startup). This means:

1. A single MCP instance handles all groups correctly
2. You can add new ontology entries without restarting MCP (the YAML is re-read on each call)
3. Multiple groups with different ontologies can extract concurrently

### Verifying Your Ontology

After adding the YAML block:

```bash
# Verify YAML parses correctly
python3 -c "import yaml; yaml.safe_load(open('mcp_server/config/extraction_ontologies.yaml'))"

# Check that your group_id resolves
python3 -c "
from mcp_server.src.services.ontology_registry import OntologyRegistry
reg = OntologyRegistry()
profile = reg.resolve('s1_my_source')
print(f'Entity types: {[e.name for e in profile.entity_types]}')
print(f'Relationship types: {[r.name for r in profile.relationship_types]}')
print(f'Emphasis: {profile.extraction_emphasis[:80]}...')
"
```

---

## Step 3: Write an Ingest Adapter Script

### Pattern A: SQLite Database Source

For structured data in SQLite:

```python
#!/usr/bin/env python3
"""Ingest <source> into Graphiti via MCP add_memory.

Usage:
    python3 scripts/ingest_my_source.py [--mcp-url URL] [--dry-run]
"""

import argparse
import json
import sqlite3
import urllib.request
from uuid import uuid5, NAMESPACE_URL

GROUP_ID = "s1_my_source"
DEFAULT_MCP_URL = "http://localhost:8000/mcp"
MAX_BODY_CHARS = 10_000  # sub-chunk above this


def add_memory(mcp_url: str, name: str, body: str, group_id: str, source_desc: str) -> None:
    """Send one episode to MCP add_memory."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "add_memory",
            "arguments": {
                "name": name,
                "episode_body": body,
                "group_id": group_id,
                "source_description": source_desc,
            },
        },
    }
    req = urllib.request.Request(
        mcp_url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
        if "error" in result:
            raise RuntimeError(f"MCP error: {result['error']}")


def sub_chunk(text: str, max_chars: int = MAX_BODY_CHARS) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    current = ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    ap.add_argument("--db", required=True, help="Path to SQLite database")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT id, title, content FROM my_table").fetchall()

    print(f"Found {len(rows)} records to ingest")

    for row in rows:
        body = f"Title: {row['title']}\nContent: {row['content']}"

        # Deterministic UUID from source ID → idempotent re-runs
        chunk_key = str(uuid5(NAMESPACE_URL, f"my_source:{row['id']}"))

        chunks = sub_chunk(body)
        for i, chunk in enumerate(chunks):
            name = row["title"]
            if len(chunks) > 1:
                name = f"{name} (part {i + 1}/{len(chunks)})"

            if args.dry_run:
                print(f"  [dry-run] {name} ({len(chunk)} chars)")
                continue

            add_memory(
                args.mcp_url,
                name=name,
                body=chunk,
                group_id=GROUP_ID,
                source_desc=f"my_source:{row['id']}",
            )
            print(f"  Queued: {name}")

    print(f"Done: {len(rows)} records {'would be' if args.dry_run else ''} ingested")


if __name__ == "__main__":
    main()
```

### Pattern B: JSONL / Evidence File Source

For pre-parsed evidence files:

```python
# Use mcp_ingest_sessions.py with:
#   --group-id <your_group_id>
#   --evidence <path/to/evidence.json>
#   --force  (for initial load)

# Evidence JSON format:
# [
#   {
#     "name": "Episode title",
#     "episode_body": "Content to extract from...",
#     "source_description": "source:identifier",
#     "reference_time": "2026-02-20T00:00:00Z"
#   },
#   ...
# ]
```

### Pattern C: Compound Notes / Markdown Source

For markdown files that need sectional chunking:

```python
# Use ingest_compound_notes.py as a template.
# Key: split at H2 headers, deterministic chunk keys from file path + section index.
```

### Adapter Design Principles

| Principle | Why |
|-----------|-----|
| **Deterministic chunk keys** from source IDs | Idempotent re-runs — same input never creates duplicates in the registry |
| **Sub-chunk large content** (>10k chars) | Prevents `context_length_exceeded` from the LLM |
| **Track cursor/watermark** for incremental ingest | Don't re-process the entire source every run |
| **Validate body size** against `MAX_EPISODE_BODY_CHARS` (12k default) | Episodes exceeding this are silently truncated by the MCP server |
| **Include `source_description`** | Provenance tracking — know where every episode came from |
| **Handle errors gracefully** | Network timeouts to MCP are normal under load; retry with backoff |

---

## Step 4: Set Up Cron Schedule

Choose frequency based on how often the source data changes:

| Source Type | Recommended Frequency | Rationale |
|-------------|----------------------|-----------|
| High-churn data (sessions, tasks) | Every 30 min | Time-sensitive, delta ingestion |
| Medium-churn data (notes, CRM) | Hourly or daily | Changes accumulate gradually |
| Low-churn data (documents, snapshots) | Daily or on-change | Hash-gated; only runs when files change |
| One-time imports | Manual | Run once with `--force`, then disable |

---

## Step 5: Verify the Integration

After the first ingest run:

### Check Episode Counts

```bash
cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (e:Episodic) WHERE e.group_id = 's1_my_source' RETURN count(e) AS cnt"
```

### Check Entity Extraction Quality

```cypher
-- Sample entities created by your ontology
MATCH (n:Entity {group_id: 's1_my_source'})
RETURN n.name, labels(n), n.entity_type
ORDER BY n.name LIMIT 20
```

### Contamination Check

```cypher
-- Should return 0 — no cross-group edges
MATCH (a)-[r:RELATES_TO]->(b)
WHERE a.group_id = 's1_my_source' AND b.group_id <> 's1_my_source'
RETURN count(r)
```

### Post-Verification

After confirming the integration works:

1. **Add to monitoring** — add your group to the extraction monitor's group list with expected target count
2. **Add to maintenance scripts** — add your group_id to dedup and timeline repair scripts
3. **Update documentation** — add a row to the Graph Groups table in `sessions-ingestion.md`

---

## Reference: Ingest Script Patterns

| Script | Source Type | Usage |
|--------|-----------|-------|
| `mcp_ingest_sessions.py` | Evidence JSON (sessions, transcripts) | General-purpose evidence ingestion |
| `ingest_content_groups.py` | Content batch evidence (sequential with drain) | For content that needs ordered processing |
| `ingest_compound_notes.py` | Markdown compound notes | For markdown files split by section |
| `curated_snapshot_ingest.py` | Curated markdown snapshots (hash-gated) | For files that change infrequently |

When writing a new adapter, start from the script closest to your source type. Copy the patterns (error handling, sub-chunking, cursor tracking) rather than writing from scratch.

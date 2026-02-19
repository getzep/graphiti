# Sessions Sub-Chunking

## Overview

Session transcript evidence (particularly `s1_sessions_main`) can produce large evidence
chunks that exceed the LLM context window during Graphiti extraction. Rather than truncating
content (which loses information) or relying on runtime retry-with-shrink hacks, the
ingestion script now **deterministically sub-chunks** large evidence at enqueue time.

## How It Works

When `mcp_ingest_sessions.py` processes evidence for a group ID in the sub-chunking set
(currently `s1_sessions_main`):

1. **Size check**: If evidence content exceeds `--subchunk-size` (default: 10,000 chars),
   it is split into multiple sub-chunks.

2. **Deterministic splitting**: Sub-chunks are created by splitting on paragraph boundaries
   (double newlines) when possible, falling back to single newlines, then hard splits.
   This ensures the same input always produces the same sub-chunks.

3. **Stable keys**: Each sub-chunk receives a `:p0`, `:p1`, `:p2`, ... suffix appended to
   the original chunk key. For example:
   - Original: `session:main:2026-02-19T12:00:c3`
   - Sub-chunks: `session:main:2026-02-19T12:00:c3:p0`, `session:main:2026-02-19T12:00:c3:p1`

4. **Registry dedup**: Each sub-chunk has its own content hash and registry entry, so
   re-running the script skips already-ingested sub-chunks (idempotent).

5. **Episode count**: Sub-chunking increases the total episode count — this is expected and
   acceptable. More, smaller episodes extract more reliably than fewer oversized ones.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--subchunk-size` | 10,000 | Max characters per sub-chunk |
| `--group-id` | (required) | Only groups in `_SUBCHUNK_GROUP_IDS` get sub-chunked |

## When to Adjust

- If extraction still hits context limits, reduce `--subchunk-size` (e.g., 5000).
- If episodes are too fragmented, increase `--subchunk-size` (e.g., 15000).
- The default of 10,000 chars leaves comfortable headroom for gpt-4o-mini's context window
  after accounting for system prompts, previous episodes, and extraction instructions.

## Design Rationale

Previous approaches used runtime hacks in `queue_service.py` (truncation, retry-with-shrink)
and `graphiti_core/graphiti.py` (reduced previous-episode window). These were fragile:

- **Truncation** lost information (head+tail kept, middle discarded)
- **Retry-with-shrink** was non-deterministic and wasteful (failed attempts still consumed API calls)
- **Reduced prev-episode window** degraded extraction quality for all sessions, not just oversized ones

Sub-chunking at enqueue time is deterministic, lossless, and idempotent. The runtime
(queue_service, graphiti_core) no longer needs special-case handling for large evidence.

# Mission

## What this is

Graphiti is a Python framework for building and querying **temporal context graphs** for AI agents. It turns a stream of episodes (text, JSON, or chat messages) into a graph of entities and bi-temporal facts — each fact carries a validity window (when it became true, when it was superseded) and a provenance link back to the episode that produced it. The library handles entity/edge extraction, deduplication, invalidation of contradicted facts, community detection, and hybrid retrieval (semantic + BM25 + graph traversal) on top of pluggable graph backends (Neo4j, FalkorDB, Kuzu, Amazon Neptune).

## Who it serves

Primary user: developers building production AI agents that need persistent, queryable, time-aware memory — and who are willing to operate the surrounding infrastructure themselves. Graphiti is the open-source core of Zep's managed context platform, so the audience is teams that chose self-hosting over the managed service. Secondary user: researchers and practitioners experimenting with agent-memory architectures and ontology design.

## What it is NOT

- Not a managed service. Hosting, observability, governance, SLAs, and user/thread management are out of scope — those live in Zep.
- Not a document-RAG library. Graphiti is built for evolving, contradiction-prone real-world data, not static document summarization.
- Not a general-purpose graph database. It is a context-graph engine that sits on top of one (Neo4j / FalkorDB / Kuzu / Neptune).
- Not a dashboard or visualization product. There is no UI; the REST server and MCP server expose ingestion and retrieval only.
- Not reliable with LLMs that lack structured-output support. Smaller or non-conforming models will produce schema failures during extraction and deduplication.
- Not a deletion-based store. Superseded facts are invalidated (`expired_at` / `invalid_at`), never silently dropped — history is preserved by design.

## Success signal

An agent can ask Graphiti "what is true now about X" and "what was true about X at time T" and get the right answer after arbitrarily many ingest events, without batch recomputation and with sub-second retrieval latency.

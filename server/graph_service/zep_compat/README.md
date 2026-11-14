# Zep Cloud v2 compatibility layer

A drop-in replacement for the subset of the Zep Cloud API that
[MiroFish](https://github.com/uxe-security-solutions/MiroFish) calls, implemented on
top of `graphiti_core` — the same engine Zep Cloud is built on. It exists so
MiroFish can run with no hosted service: Zep Community Edition is discontinued and
there is no self-hostable Zep server.

The `zep-cloud` Python SDK talks to this unmodified. MiroFish only sets
`ZEP_BASE_URL`.

## Running it

```bash
cd server
uv sync --extra dev
uvicorn graph_service.zep_compat.app:app --host 127.0.0.1 --port 8088
```

Deliberately a separate ASGI app from `graph_service.main`, whose settings require
`OPENAI_API_KEY` and which builds its own Graphiti instance. Keeping them apart
means upstream changes to `main.py` never conflict here.

### Configuration

| Variable | Default | Notes |
|---|---|---|
| `ZEP_COMPAT_API_PREFIX` | `/api/v2` | MiroFish pins the SDK `base_url` to `<host>/api/v2` |
| `ZEP_COMPAT_DB_PATH` | `./data/zep_compat.sqlite3` | graph registry, ontologies, batch state |
| `ZEP_COMPAT_BATCH_CONCURRENCY` | `4` | episodes ingested at once; each fans out into several LLM calls |
| `GRAPHITI_LLM_BASE_URL` | `http://localhost:8000/v1` | any OpenAI-compatible endpoint |
| `GRAPHITI_EMBEDDER_BASE_URL` | `http://localhost:8081/v1` | required — Zep Cloud embedded server-side, Graphiti does not |
| `EMBEDDING_DIM` | `1024` | **one-way door**: the vector index dimension is fixed at creation |
| `GRAPHITI_RERANKER` | `none` | `none` = RRF fusion, no reranker call. `bge` loads a local cross-encoder |
| `GRAPHITI_DB_BACKEND` | `falkordb` | or `neo4j` |
| `GRAPHITI_TELEMETRY_ENABLED` | forced `false` | Graphiti otherwise reports to PostHog on init |

Never point `GRAPHITI_RERANKER` at Graphiti's OpenAI reranker: it scores with
`logit_bias={'6432': 1, '7983': 1}`, hard-coded OpenAI tokenizer IDs for
True/False that are meaningless to a Qwen or Llama tokenizer.

## Layout

| File | Role |
|---|---|
| `WIRE_SPEC.md` | The contract, derived mechanically from the `zep-cloud==3.25.0` wheel. Read this first. |
| `models.py` | Wire-exact request/response models |
| `router.py` | The 17 endpoints |
| `runtime.py` | Graphiti construction, the per-graph pool, the batch worker |
| `store.py` | SQLite: graph registry, ontologies, batch lifecycle, UUID→graph index |
| `ontology.py` | Zep `EntityType`/`EdgeType` → Graphiti Pydantic models |
| `paging.py` | `SKIP`/`LIMIT` paging, because `uuid_cursor` is broken on FalkorDB |
| `app.py` | ASGI entrypoint |

## Four things that will bite you

Each of these was found the hard way and is covered by a test.

**1. `uuid` on the wire, `uuid_` in Python.** Fern declares
`uuid_: Annotated[str, FieldMetadata(alias="uuid")]`. Emitting `uuid_` fails
client-side with a `ValidationError`. `models.py` uses the wire names directly so
the mistake is unavailable.

**2. `group_id` is the database name.** In `graphiti_core` 0.29.3, `add_episode`
does:

```python
if group_id != self.driver._database:
    self.driver = self.driver.clone(database=group_id)
```

So each graph lives in its own database, *and* that assignment mutates the shared
instance — two concurrent `add_episode` calls for different graphs would write into
the wrong database. Hence `GraphitiPool`: one instance per graph, driver already
pointed at a database named exactly `graph_id`. The pool asserts that invariant,
because if the names differ the clone fires and pre-saved episodes vanish.

Consequence: a node or episode UUID is only resolvable inside its own graph, but
`GET graph/node/{uuid}` and `GET graph/episodes/{uuid}` carry no `graph_id`. The
store keeps a UUID→graph index, populated whenever a UUID is handed out.

**3. `add_episode(uuid=...)` means "process the existing episode", not "create with
this UUID"** — it calls `EpisodicNode.get_by_uuid` and raises if absent. But Zep's
batch API promises the client an episode UUID at `batch.add` time. So
`ingest_episode` persists the node first, then hands its UUID to `add_episode`,
keeping one UUID end to end.

**4. Empty results are not uniform.** `EntityNode.get_by_group_ids` returns `[]`;
`EntityEdge.get_by_group_ids` *raises* `GroupsEdgesNotFoundError`. A fresh graph has
no edges, so an unguarded read 500s on a completely normal state — and MiroFish
retries a 500 three times before failing the whole drain.

And one upstream bug worth knowing: on FalkorDB the `WHERE n.uuid < $uuid` cursor
clause is silently not applied — verified against a live FalkorDB, where even a
literal comparison returns every row while `ORDER BY` and `SKIP`/`LIMIT` are exact.
`paging.py` uses offsets instead.

## Tests

```bash
cd server
uv run --extra dev pytest tests/ -q --asyncio-mode=auto
```

No GPU, no network, no database. `test_zep_compat_e2e.py` drives a real `AsyncZep`
client over an ASGI transport, so paths, payloads, error mapping and the
header-based pagination cursor are all exercised against the SDK MiroFish uses.

Against a real FalkorDB:

```bash
docker run -d --name falkordb-it -p 6399:6379 falkordb/falkordb:latest
ZEP_COMPAT_IT=1 FALKORDB_IT_PORT=6399 uv run --extra dev pytest tests/test_zep_compat_integration.py -v
```

## Scope

Only the 17 endpoints MiroFish calls: graph create/get/delete/add/search,
`entity-types`, node get / node edges / node listing, edge listing, episode get /
mentions, and the six `batches` endpoints. Users, threads and fact-triples are not
implemented.

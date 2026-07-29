# Graphiti on Render

Deploy [Graphiti](https://github.com/getzep/graphiti) on Render in one click. Get a hosted
temporal knowledge-graph API your agents can write memories to and search over time.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/render-examples/graphiti)

https://github.com/user-attachments/assets/da4041b7-f59e-4fb3-929b-fae2d6b949b0

## What you get

Graphiti turns a stream of conversations or events into a knowledge graph, and keeps track of
_when_ each fact was true. When something changes — someone switches teams, a deadline moves —
it doesn't overwrite the old fact, it invalidates it and records the new one. So an agent can
ask what's true now, and also what was true last quarter.

This template deploys the backend for that: a REST API and the graph store behind it. **There
is no web UI.** The deliverable is a live URL like `https://graphiti-api.onrender.com` that you
call from your own agent code, a script, or a framework like LangGraph.

### Architecture

```
                       ┌───────────────────────────┐
   your agent /        │   graphiti-api (docker)   │        ┌──────────────┐
   script / app  ─────▶│   FastAPI · public HTTPS  │───────▶│  OpenAI API  │
      HTTPS            │   /messages  /search      │  facts └──────────────┘
                       └─────────────┬─────────────┘  entities
                                     │
                         Redis proto │ private network only
                                     ▼
                       ┌───────────────────────────┐
                       │ graphiti-falkordb (image) │
                       │ private service · :6379   │
                       │ 10 GB disk, append-only   │
                       └───────────────────────────┘
```

`graphiti-falkordb` is a **private service**: it has no public address and accepts traffic only
from `graphiti-api` over Render's private network.

## Deploy

1. Click **Deploy to Render** above. Render reads [`render.yaml`](render.yaml) and creates a
   `graphiti` project containing both services.
2. Render prompts for one secret on `graphiti-api`:

   | Variable         | What it's for                                                                                       | Where to get it                                                      |
   | ---------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
   | `OPENAI_API_KEY` | Graphiti calls an LLM to pull entities and facts out of each episode, and to embed them for search. | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

   If you use a **restricted** OpenAI key, this deployment needs exactly two of the
   permissions under **Model capabilities**, both write:

   | Permission     | Used for                                                                           |
   | -------------- | ---------------------------------------------------------------------------------- |
   | **Responses**  | Write access on `/v1/responses` — extracting entities and facts from each episode. |
   | **Embeddings** | Write access on `/v1/embeddings` — embedding nodes, edges, and search queries.     |

   Everything else is set for you in `render.yaml`. The ones worth knowing about:

   | Variable          | Default   | Notes                                                                                                                                                                                                             |
   | ----------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `MODEL_NAME`      | `gpt-5.5` | Any OpenAI model id.                                                                                                                                                                                              |
   | `SEMAPHORE_LIMIT` | `10`      | Concurrent LLM calls during ingestion. Deliberately below graphiti-core's default of 20, so a burst of episodes doesn't trip the rate limits on a fresh OpenAI key. Raise it once you know your account's limits. |

   The rest is wiring you shouldn't need to touch. `DB_BACKEND` selects the FalkorDB code
   path; `FALKORDB_HOST` is filled in from the private service's hostname, with
   `FALKORDB_PORT` and `FALKORDB_DATABASE` fixed to match it. `INSTALL_FALKORDB` is a build
   arg that pulls in the `graphiti-core[falkordb]` extra, and `BROWSER=0` turns off the
   FalkorDB Browser UI so `6379` is the only port the graph store listens on.

   The `graphiti-core` version is pinned in one place: the `GRAPHITI_VERSION` build arg
   default in [`Dockerfile`](Dockerfile). Every fork therefore deploys the same library,
   and local `docker compose` builds match Render without a second pin to keep in step.
   Bump it there, and verify a write-then-search round trip before you do — `0.29.2`, for
   one, writes episodes on FalkorDB but reads them back empty. To override it for a single
   Render service without editing the Dockerfile, add `GRAPHITI_VERSION` as an env var on
   that service; Render passes env vars to the Docker build as build args.

3. Wait for both services to go live. `graphiti-api` passes its health check at `/healthcheck`.

> **This API has no authentication.** Anyone who knows your URL can write to your graph and
> spend your OpenAI key. Before you point real traffic at it, put it behind your own auth,
> an API gateway, or a private network. In the meantime, use a dedicated OpenAI key you can
> revoke and set a monthly spend cap in the OpenAI console — that cap is your backstop.

### Using the app

Ingestion is asynchronous — `/messages` queues the episode and returns immediately, then
Graphiti extracts entities and facts in the background. Give it 10–30 seconds before searching.

Set your URL once:

```bash
export GRAPHITI_URL=https://graphiti-api.onrender.com
```

1. **Check it's up.**

   ```bash
   curl $GRAPHITI_URL/healthcheck
   # {"status":"healthy"}
   ```

2. **Ingest the sample episode** that ships in this repo
   ([`examples/render/sample-episode.json`](examples/render/sample-episode.json)). It's a short
   conversation where Alex leads the payments team in May and hands it to Priya in July. The
   messages carry explicit `timestamp` fields — that spread is what gives Graphiti something to
   be temporal about. Omit them and every message is stamped with the time it arrived, so the
   facts all land at once and nothing supersedes anything:

   ```bash
   curl -X POST $GRAPHITI_URL/messages \
     -H 'Content-Type: application/json' \
     -d @examples/render/sample-episode.json
   # {"message":"Messages added to processing queue","success":true}
   ```

3. **Watch the episodes land.**

   ```bash
   curl "$GRAPHITI_URL/episodes/demo?last_n=5"
   ```

4. **Search the facts Graphiti extracted.**

   ```bash
   curl -X POST $GRAPHITI_URL/search \
     -H 'Content-Type: application/json' \
     -d '{"group_ids": ["demo"], "query": "who leads the payments team?", "max_facts": 10}'
   ```

   Among the results you'll find both sides of the handover, each stamped with when it
   became true:

   ```
   Alex leads the payments team at Acme.
       valid_at: 2026-05-04T14:00:00+00:00   invalid_at: None
   Priya now leads the payments team at Acme.
       valid_at: 2026-07-13T10:15:00+00:00   invalid_at: None
   ```

   That's the point of the graph. Alex's fact wasn't overwritten when Priya took over — both
   are stored, each anchored to the message that asserted it, so you can ask who led the team
   in June and get an answer. The `valid_at` values come straight from the message timestamps
   and are reliable.

   `invalid_at` is the other half: when Graphiti recognizes that a new fact contradicts an
   older one, it closes the old one off at that moment, and you'd see
   `invalid_at: 2026-07-13T10:15:00+00:00` on Alex's fact. Whether that fires is an LLM
   judgment call made during ingestion, and on this three-message sample it happens on some
   runs and not others. Treat it as opportunistic: real workloads give the model far more
   signal than three messages do. The extracted wording, the capitalization, and the result
   ordering vary between runs too.

5. **Clean up** when you're done experimenting. On FalkorDB each `group_id` is a separate
   graph, so deleting the group means deleting that graph. Open a shell on
   `graphiti-falkordb` from the Render Dashboard and run:

   ```bash
   redis-cli GRAPH.DELETE demo
   ```

   > The HTTP endpoints `DELETE /group/{group_id}` and `POST /clear` do **not** work on this
   > deployment. Both return `{"success": true}` and delete nothing: they query the default
   > graph, while the data lives in a graph named after the `group_id`. This is a
   > [graphiti-core](https://github.com/getzep/graphiti) driver issue, not a misconfiguration
   > here. Don't rely on them to erase a tenant's data.

Full endpoint list at `$GRAPHITI_URL/docs` (FastAPI's generated OpenAPI docs).

`group_id` partitions the graph. Use one per user, per tenant, or per agent, and search stays
scoped to it. On FalkorDB each one is a distinct graph held in memory, so the instance plan —
not the disk — is what bounds how many you can keep.

## Configuration notes

**FalkorDB has no password.** It's a private service with no public address, so isolation comes
from the network. To add one anyway, set `REDIS_ARGS` to `--appendonly yes --requirepass <your
password>` on `graphiti-falkordb`, and set `FALKORDB_PASSWORD` to the same value on
`graphiti-api`. Render can't interpolate one env var into another, so this is a manual step.

**`graphiti-falkordb` logs a security warning while it starts.** For the first few minutes
you'll see `Possible SECURITY ATTACK detected ... Connection from 127.0.0.1 aborted` about
once a minute. That's Render's port scanner sending an HTTP probe to `6379` to work out which
port the service listens on; FalkorDB speaks the Redis protocol, so it reports the HTTP
request as a cross-protocol attempt and drops it. It stops once the port is detected, and
nothing reaches the graph. The startup line about Redis having no authentication is expected
too — see the note above on why the private network is what isolates it.

**Using Neo4j instead.** Set `DB_BACKEND=neo4j` and supply `NEO4J_URI`, `NEO4J_USER`, and
`NEO4J_PASSWORD` (e.g. from Neo4j Aura), then drop the `graphiti-falkordb` service from
`render.yaml`.

**Local development.** Copy [`.env.example`](.env.example) to `.env`, fill in `OPENAI_API_KEY`,
and run `docker compose --profile falkordb up`. That mirrors this Blueprint — the API on
`http://localhost:8001`, FalkorDB beside it. A plain `docker compose up` runs the repo's other
pairing, API plus Neo4j, on port 8000.

## Learn more

This repo is a fork of [getzep/graphiti](https://github.com/getzep/graphiti) with a Render
Blueprint added. For how Graphiti works — custom entity types, search strategies, the MCP
server, the Python library — see the
[upstream README](https://github.com/getzep/graphiti#readme) and
[the paper](https://arxiv.org/abs/2501.13956).

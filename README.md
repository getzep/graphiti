# Graphiti on Render

Deploy [Graphiti](https://github.com/getzep/graphiti) on Render in one click. Get a hosted
temporal knowledge-graph API your agents can write memories to and search over time.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/render-examples/graphiti-render)

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

4. **Copy your API key.** Every endpoint except `/healthcheck` requires it, and there's
   nothing to paste in: Render generated the key when it created the service. Open
   `graphiti-api` → **Environment** in the Dashboard and copy `GRAPHITI_API_KEY`. Send it as
   a bearer token:

   ```
   Authorization: Bearer <GRAPHITI_API_KEY>
   ```

   Auth is mandatory — the service won't start without a key, so a fork can't go live open by
   accident. The generated value persists across deploys, so rotating it is up to you: edit it
   in the Dashboard and Render restarts the service with the new value. The replacement has to
   be at least 16 printable-ASCII characters, as the generated one is. Length because `dev` is
   not a key — the service limits failed attempts, but that can't help against one that's
   guessed on the first try; printable ASCII because HTTP sends header values as latin-1 and
   clients disagree about how to encode anything outside it, so a key with an accent or an emoji
   in it would authenticate for some clients and not others. The service refuses to start on
   either rather than leaving you to discover it as a 401, so a bad rotation fails the deploy and
   Render keeps serving the previous version.

   A wrong or missing key gets `401` with `WWW-Authenticate: Bearer`. After 10 rejections in a
   minute the service answers `429` with `Retry-After` instead, so the key can't be worked
   through and your URL isn't a free target for scanners. Requests carrying the *right* key are
   never rate limited, so a stranger hammering your service can't lock you out of it.

   The API docs at `/docs` stay open, and you can authorize them in the browser: click
   **Authorize**, paste the key, and **Try it out** works against your deployment.

> **This key is the only thing in front of your graph.** It's a single shared secret with no
> scopes and no per-user identity — enough to keep a stranger who finds your URL from writing to
> your graph and spending your OpenAI key, and not a substitute for real authorization. The
> rejection limit stops guessing; it does nothing about a caller who *has* the key, and there's
> no cap on what one of those can spend. Before you point production traffic at it, put it behind
> your own auth or an API gateway. Either way, use a dedicated OpenAI key you can revoke and set
> a monthly spend cap in the OpenAI console — that cap is your backstop.

### Using the app

Ingestion is asynchronous — `/messages` queues the episode and returns immediately, then
Graphiti extracts entities and facts in the background. Give it 10–30 seconds before searching.

Set your URL and key once:

```bash
export GRAPHITI_URL=https://graphiti-api.onrender.com
export GRAPHITI_API_KEY=...   # graphiti-api → Environment in the Render Dashboard
```

1. **Check it's up.** The only endpoint that takes no key — Render polls it to decide the
   service is live.

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
     -H "Authorization: Bearer $GRAPHITI_API_KEY" \
     -H 'Content-Type: application/json' \
     -d @examples/render/sample-episode.json
   # {"message":"Messages added to processing queue","success":true}
   ```

3. **Watch the episodes land.**

   ```bash
   curl "$GRAPHITI_URL/episodes/demo?last_n=5" \
     -H "Authorization: Bearer $GRAPHITI_API_KEY"
   ```

4. **Search the facts Graphiti extracted.**

   ```bash
   curl -X POST $GRAPHITI_URL/search \
     -H "Authorization: Bearer $GRAPHITI_API_KEY" \
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

If you'd rather not paste multi-line `curl` commands, [`examples/render/demo.sh`](examples/render/demo.sh)
wraps the same endpoints in one-word shell functions. Set `GRAPHITI_URL` and `GRAPHITI_API_KEY`,
`source` the file (it defines functions, so running it won't work), and you get `use_group take1`
to pick a group, `health`, `ingest` to POST the sample episode with its `group_id` rewritten, then
`watch_ingest` to follow the queue draining, `ask "who owns the ledger migration?"` for the
current answer, and `timeline "leads the payments"` for every version of a fact oldest-first with
its relationship label and validity window. It's a thin projection over `/search` — the raw
endpoint returns more fields than it prints. Use a new group per run rather than re-ingesting into
one that already holds facts: they dedupe into what's there and nothing appears to happen.

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

Auth is mandatory locally too — there's no off switch — so compose defaults
`GRAPHITI_API_KEY` to `insecure-local-dev-key`. Export that and every example above works unchanged
against `http://localhost:8001`; set your own value in `.env` to override it. Running the same
authenticated path locally as on Render is the point: nothing about auth is left untested until
you deploy.

That default key is committed to this repo, so it protects nothing — which is why every port in
`docker-compose.yml` is published to `127.0.0.1` rather than to every interface. The stack is
reachable from your machine and nowhere else. If you need to reach it from another host, drop the
`127.0.0.1:` prefix and set a real `GRAPHITI_API_KEY` in the same edit.

## Learn more

This repo is a fork of [getzep/graphiti](https://github.com/getzep/graphiti) with a Render
Blueprint added. For how Graphiti works — custom entity types, search strategies, the MCP
server, the Python library — see the
[upstream README](https://github.com/getzep/graphiti#readme) and
[the paper](https://arxiv.org/abs/2501.13956).

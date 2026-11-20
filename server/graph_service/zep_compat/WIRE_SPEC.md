# Zep Cloud v2 wire contract (the subset MiroFish uses)

Derived mechanically from the `zep-cloud==3.25.0` Python SDK wheel, not from docs.
Everything below is what the SDK actually puts on the wire and what it will accept back.

Base path: the SDK is constructed with `base_url=<X>` and appends these paths, so the
shim must be mounted such that `<X>/graph/...` resolves. MiroFish pins
`ZEP_CLOUD_BASE_URL = "https://api.getzep.com/api/v2"`, therefore the shim serves
under the prefix `/api/v2`.

Auth: SDK sends `Authorization: Api-Key <key>`. The shim accepts any non-empty key.

## Field aliasing — the single biggest trap

Fern-generated models declare fields like:

```python
uuid_: Annotated[str, FieldMetadata(alias="uuid")] = pydantic.Field()
```

The wire key is **`uuid`**; `uuid_` is only the Python attribute name. Verified:
`parse_obj_as(EntityNode, {"uuid": ...})` succeeds and `{"uuid_": ...}` raises
`ValidationError`. **The shim must emit `uuid`, never `uuid_`.**

Timestamps are plain `str` on the wire (not datetime). Emit RFC3339 / ISO-8601,
e.g. `2024-01-01T00:00:00Z`. Unknown extra keys are tolerated by the models.

## Endpoints

| SDK call | Method | Path | Request body / params | Response |
|---|---|---|---|---|
| `graph.create` | POST | `graph/create` | `{description, graph_id, name, time_zone}` | `Graph` |
| `graph.get` | GET | `graph/{graph_id}` | — | `Graph` |
| `graph.delete` | DELETE | `graph/{graph_id}` | — | `SuccessResponse` |
| `graph.add` | POST | `graph` | `{created_at, data, graph_id, metadata, source_description, strict_ontology, type, user_id}` | `Episode` |
| `graph.search` | POST | `graph/search` | `{bfs_origin_node_uuids, center_node_uuid, graph_id, limit, max_characters, mmr_lambda, query, reranker, return_raw_results, scope, search_filters, user_id}` | `GraphSearchResults` |
| `graph.set_ontology` | PUT | `entity-types` | `{edge_types[], entity_types[], graph_ids[], user_ids[]}` | `SuccessResponse` |
| `graph.node.get` | GET | `graph/node/{uuid}` | — | `EntityNode` |
| `graph.node.get_edges` | GET | `graph/node/{node_uuid}/entity-edges` | — | `EntityEdge[]` |
| `graph.node.get_by_graph_id` | **POST** | `graph/node/graph/{graph_id}` | `{cursor, direction, filters, limit, order_by, uuid_cursor}` | `EntityNode[]` + `zep-next-cursor` header |
| `graph.edge.get_by_graph_id` | **POST** | `graph/edge/graph/{graph_id}` | `{cursor, direction, filters, limit, order_by, uuid_cursor}` | `EntityEdge[]` + `zep-next-cursor` header |
| `graph.episode.get` | GET | `graph/episodes/{uuid}` | — | `Episode` |
| `batch.create` | POST | `batches` | `{ignore_roles, metadata}` | `BatchSummary` |
| `batch.add` | POST | `batches/{batch_id}/items` | `{items: BatchAddItem[]}` | `BatchItemDetail[]` |
| `batch.process` | POST | `batches/{batch_id}/process` | — | `BatchSummary` |
| `batch.get` | GET | `batches/{batch_id}` | — | `BatchSummary` |
| `batch.list` | GET | `batches` | query `{limit, cursor, status}` | `BatchListResponse` |
| `batch.list_items` | GET | `batches/{batch_id}/items` | query params | `BatchItemListResponse` |

Note the two `get_by_graph_id` calls are **POST with a JSON body**, despite reading data.

## Pagination (`zep-next-cursor`)

`fetch_all_nodes` / `fetch_all_edges` in MiroFish's `app/utils/zep_paging.py` drive
pagination **entirely from a response header**, not the body:

- send `{"limit": <=100, "cursor": <opaque str>}`
- read the response header `zep-next-cursor`
- absent header => last page, stop
- a cursor that repeats or equals the one just sent => MiroFish raises
  `RuntimeError("... pagination cursor did not advance ...")`

So the shim MUST set `zep-next-cursor` on node/edge list responses when more rows
remain, MUST omit it on the final page, and MUST make it strictly advance.

## Response models (wire keys, from `model_fields`)

```
Graph              created_at? description? graph_id? id?(int) name? project_uuid? time_zone? uuid?
Episode            content* created_at* uuid*  metadata? processed? relevance? role? role_type?
                   score? selection_rank? source? source_description? task_id? thread_id?
                   source ∈ {text, json, message, fact_triple}
EntityNode         created_at* name* summary* uuid*  attributes? labels?[] relevance? score? selection_rank?
EntityEdge         created_at* fact* name* source_node_uuid* target_node_uuid* uuid*
                   attributes? episodes?[] expired_at? invalid_at? valid_at? relevance? scope? score? selection_rank?
EpisodeResponse    episodes?[Episode]
EpisodeMentions    edges?[EntityEdge] nodes?[EntityNode]
GraphSearchResults context? edges?[] episodes?[] nodes?[] observations?[] response? thread_summaries?[]
BatchSummary       batch_id? completed_at? created_at? ignore_roles?[] item_count? metadata?
                   processed_at? progress?(BatchProgress) status? updated_at?
                   status ∈ {draft, invalid, queued, processing, succeeded, partial, failed, canceled}
BatchProgress      canceled_items? failed_items? percent_complete? processing_items?
                   queued_items? skipped_items? succeeded_items? total_items?
BatchItemDetail    created_at? episode_uuid? error? graph_id? graph_uuid? item_id? kind?
                   sequence_index? source_uuid? status? thread_id? updated_at? user_id? user_uuid?
                   kind ∈ {graph_episode, thread_message}
                   status ∈ {pending, queued, processing, succeeded, failed, skipped, canceled}
BatchItemListResponse  items?[BatchItemDetail] next_cursor?(int)
SuccessResponse    message?
```

`*` = required by the SDK's parser (omitting it raises `ValidationError` client-side).
`?` = optional.

## Ontology request models

```
EntityType   name* description* properties?[EntityProperty] identity_properties?[str]
EdgeType     name* description* properties?[EntityProperty] source_targets?[{source, target}]
EntityProperty  name* description* type*   type ∈ {Text, Int, Float, Boolean}
BatchAddItem type* (graph_episode|thread_message)  content? created_at? data? data_type?
             graph_id? metadata? name? role? source_description? thread_id? user_id?
             data_type ∈ {text, json, message, fact_triple}
```

## Error mapping

The SDK raises typed exceptions off HTTP status, and MiroFish catches some of them
(notably `zep_cloud.NotFoundError`). The shim must therefore use:

- `404` -> `NotFoundError`  (MiroFish treats this as "graph/node absent", not a failure)
- `400` -> `BadRequestError`
- `500` -> `InternalServerError`
- `408` / `429` / `5xx` -> retried by MiroFish's `is_retryable_zep_error`

Returning `200` with an error body will be silently misread as success. Use real
status codes.

"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Query text and pure ranking helpers for SpannerSearchOperations. No I/O here.
#
# Shapes (``g`` is the property graph name, ``<where>`` the filters of filters.py):
#
# * Full-text: SQL on the base table,
#   ``SELECT ..., SCORE(x.search_tokens, @query) AS score FROM <table> AS x
#   WHERE SEARCH(x.search_tokens, @query) AND <where> ORDER BY score DESC LIMIT @limit``.
# * Similarity: SQL on the base table, scored like Neo4j's ``vector.similarity.cosine``,
#   i.e. normalised cosine ``(1 + cos) / 2`` = ``(2 - COSINE_DISTANCE) / 2``, filtered with
#   ``score > @min_score``. ``SAFE.`` turns rows whose stored vector is all zeros (Spanner
#   raises on those) into NULL scores, which the filter drops.
# * BFS: GQL inside ``GRAPH_TABLE`` returning DISTINCT uuids, joined back to the base table
#   on its primary key. GQL's ``RETURN DISTINCT`` rejects ARRAY and JSON columns
#   (``labels``, ``attributes``, ``episodes``), so the graph part returns keys only.
#
# Entity and entity-edge results leave out ``name_embedding``/``fact_embedding``, as the
# Neo4j return clauses do (callers load embeddings separately); community results keep
# ``name_embedding``, as Neo4j's ``COMMUNITY_NODE_RETURN`` does.

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from graphiti_core.driver.spanner.schema import (
    COMMUNITY_TABLE,
    ENTITY_TABLE,
    EPISODIC_TABLE,
    MENTIONS_TABLE,
    RELATES_TO_TABLE,
    Table,
)

# Table aliases used in every query; filters.py predicates use n (node / edge source),
# m (edge target) and e (edge).
ENTITY_ALIAS = 'n'
EDGE_ALIAS = 'e'
EPISODE_ALIAS = 'ep'
COMMUNITY_ALIAS = 'c'


def result_columns(table: Table, alias: str, *, embeddings: bool) -> str:
    """SELECT list for ``table`` under Graphiti field names, qualified with ``alias``.

    Edge tables are renamed as in ``schema.edge_select_columns`` (``edge_uuid AS uuid``,
    ``uuid AS source_node_uuid``). ``embeddings=False`` leaves out the embedding columns.
    """
    skipped = set() if embeddings else set(table.embedding_columns)
    parts: list[str] = []
    if table.is_edge:
        skipped.update(('uuid', 'edge_uuid'))
        parts.extend((f'{alias}.edge_uuid AS uuid', f'{alias}.uuid AS source_node_uuid'))
    parts.extend(
        f'{alias}.{name} AS {name}' for name in table.writable_columns if name not in skipped
    )
    return ', '.join(parts)


ENTITY_COLUMNS = result_columns(ENTITY_TABLE, ENTITY_ALIAS, embeddings=False)
EDGE_COLUMNS = result_columns(RELATES_TO_TABLE, EDGE_ALIAS, embeddings=False)
EPISODE_COLUMNS = result_columns(EPISODIC_TABLE, EPISODE_ALIAS, embeddings=False)
COMMUNITY_COLUMNS = result_columns(COMMUNITY_TABLE, COMMUNITY_ALIAS, embeddings=True)

ENTITY_SOURCE = f'{ENTITY_TABLE.name} AS {ENTITY_ALIAS}'
EPISODE_SOURCE = f'{EPISODIC_TABLE.name} AS {EPISODE_ALIAS}'
COMMUNITY_SOURCE = f'{COMMUNITY_TABLE.name} AS {COMMUNITY_ALIAS}'
EDGE_SOURCE = f'{RELATES_TO_TABLE.name} AS {EDGE_ALIAS}'
# Source (n) and target (m) Entity rows, for label filters on edges.
EDGE_SOURCE_WITH_ENDPOINTS = (
    f'{EDGE_SOURCE}\n'
    f'JOIN {ENTITY_TABLE.name} AS n ON n.uuid = {EDGE_ALIAS}.uuid\n'
    f'JOIN {ENTITY_TABLE.name} AS m ON m.uuid = {EDGE_ALIAS}.target_node_uuid'
)


def edge_source(needs_endpoints: bool) -> str:
    return EDGE_SOURCE_WITH_ENDPOINTS if needs_endpoints else EDGE_SOURCE


def and_all(predicates: Iterable[str]) -> str:
    """AND the predicates (each parenthesised); ``TRUE`` when there are none."""
    parts = [f'({predicate})' for predicate in predicates]
    return '\n  AND '.join(parts) if parts else 'TRUE'


def cosine_score(column: str, vector_param: str = '@search_vector') -> str:
    """Normalised cosine similarity in [0, 1], the scale of Neo4j's vector.similarity.cosine."""
    return f'(2 - SAFE.COSINE_DISTANCE({column}, {vector_param})) / 2'


# ---------------------------------------------------------------------------
# Full-text and similarity (SQL on base tables)
# ---------------------------------------------------------------------------


def fulltext_sql(select: str, source: str, alias: str, predicates: Sequence[str]) -> str:
    """Rank rows of ``source`` matching the rquery ``@query``; params ``query``, ``limit``."""
    where = and_all([f'SEARCH({alias}.search_tokens, @query)', *predicates])
    return (
        f'SELECT {select}, SCORE({alias}.search_tokens, @query) AS score\n'
        f'FROM {source}\n'
        f'WHERE {where}\n'
        'ORDER BY score DESC\n'
        'LIMIT @limit'
    )


def similarity_sql(
    select: str, source: str, alias: str, column: str, predicates: Sequence[str]
) -> str:
    """Exact KNN over ``alias.column``; params ``search_vector``, ``min_score``, ``limit``."""
    qualified = f'{alias}.{column}'
    where = and_all([f'{qualified} IS NOT NULL', *predicates]).replace('\n', '\n  ')
    source = source.replace('\n', '\n  ')
    return (
        'SELECT * FROM (\n'
        f'  SELECT {select}, {cosine_score(qualified)} AS score\n'
        f'  FROM {source}\n'
        f'  WHERE {where}\n'
        ')\n'
        'WHERE score > @min_score\n'
        'ORDER BY score DESC\n'
        'LIMIT @limit'
    )


# ---------------------------------------------------------------------------
# Traversals (GQL inside GRAPH_TABLE, joined back to the base tables)
# ---------------------------------------------------------------------------

# Origins may be Entity or Episodic nodes, as in Neo4j's unlabelled ``(origin {uuid})``:
# only those two kinds have outgoing RELATES_TO or MENTIONS edges.
_ORIGIN = '(origin:Entity|Episodic WHERE origin.uuid IN UNNEST(@origin_uuids))'


def _gql_where(predicates: Sequence[str]) -> str:
    return f'\n  WHERE {and_all(predicates)}' if predicates else ''


def node_bfs_sql(graph_name: str, predicates: Sequence[str]) -> str:
    """Entities 1..``@max_depth`` hops from the origins over RELATES_TO|MENTIONS.

    Params ``origin_uuids``, ``max_depth``, ``limit``. ``predicates`` are GQL over ``origin``
    and ``n``; the caller includes Neo4j's same-group predicate. ``TRAIL`` forbids repeated
    edges, as Cypher's variable-length match does.
    """
    return (
        'SELECT ' + ENTITY_COLUMNS + '\n'
        'FROM GRAPH_TABLE(\n'
        f'  {graph_name}\n'
        f'  MATCH TRAIL {_ORIGIN}-[:RELATES_TO|MENTIONS]->{{1,@max_depth}}(n:Entity)'
        f'{_gql_where(predicates)}\n'
        '  RETURN DISTINCT n.uuid AS uuid\n'
        '  LIMIT @limit\n'
        ') AS hits\n'
        f'JOIN {ENTITY_SOURCE} ON {ENTITY_ALIAS}.uuid = hits.uuid'
    )


def edge_bfs_sql(graph_name: str, predicates: Sequence[str], multi_hop: bool) -> str:
    """RELATES_TO edges on a path of at most ``max_depth`` hops from the origins.

    Equivalent to Cypher's ``(origin)-[:RELATES_TO|MENTIONS*1..D]->()`` followed by every
    RELATES_TO edge on the path: an edge qualifies when its source is an origin (depth 1) or
    is reachable in 1..D-1 hops. Written as a UNION of those two patterns instead of one
    ``{0,D-1}`` quantifier (zero lower bounds are reported as rejected by some emulator
    builds) and because an upper bound of 0 is invalid. ``multi_hop`` (D >= 2) adds the
    second branch, which takes ``@depth_minus_1``. Params ``origin_uuids``, ``limit``.
    ``predicates`` are GQL over ``e`` (source ``n``, target ``m``). The composite query's
    LIMIT would bind only its last branch, so the SQL wrapper applies ``@limit``.
    """
    where = _gql_where(predicates)
    returned = 'e.uuid AS edge_uuid, e.source_node_uuid AS source_uuid'
    direct = (
        '  MATCH TRAIL (n:Entity WHERE n.uuid IN UNNEST(@origin_uuids))'
        f'-[e:RELATES_TO]->(m:Entity){where}\n'
    )
    if multi_hop:
        graph_query = (
            f'{direct}'
            f'  RETURN {returned}\n'
            '  UNION DISTINCT\n'
            f'  MATCH TRAIL {_ORIGIN}-[:RELATES_TO|MENTIONS]->{{1,@depth_minus_1}}(n:Entity)'
            f'-[e:RELATES_TO]->(m:Entity){where}\n'
            f'  RETURN {returned}\n'
        )
    else:
        graph_query = f'{direct}  RETURN DISTINCT {returned}\n'
    return (
        'SELECT ' + EDGE_COLUMNS + '\n'
        'FROM GRAPH_TABLE(\n'
        f'  {graph_name}\n'
        f'{graph_query}'
        ') AS hits\n'
        f'JOIN {EDGE_SOURCE} ON {EDGE_ALIAS}.uuid = hits.source_uuid '
        f'AND {EDGE_ALIAS}.edge_uuid = hits.edge_uuid\n'
        'LIMIT @limit'
    )


# ---------------------------------------------------------------------------
# Rerankers
# ---------------------------------------------------------------------------


def center_neighbors_gql(graph_name: str) -> str:
    """Candidates one undirected RELATES_TO hop from the center; params ``center_uuid``,
    ``node_uuids``."""
    return (
        f'GRAPH {graph_name}\n'
        'MATCH (center:Entity WHERE center.uuid = @center_uuid)-[:RELATES_TO]-'
        '(n:Entity WHERE n.uuid IN UNNEST(@node_uuids))\n'
        'RETURN DISTINCT n.uuid AS uuid'
    )


# MENTIONS edges per candidate entity (served by the MentionsByTarget index).
MENTION_COUNTS_SQL = (
    'SELECT target_node_uuid AS uuid, COUNT(*) AS score\n'
    f'FROM {MENTIONS_TABLE.name}\n'
    'WHERE target_node_uuid IN UNNEST(@node_uuids)\n'
    'GROUP BY target_node_uuid'
)

ENTITIES_BY_UUIDS_SQL = (
    f'SELECT {ENTITY_COLUMNS}\nFROM {ENTITY_SOURCE}\nWHERE {ENTITY_ALIAS}.uuid IN UNNEST(@uuids)'
)


def distance_rerank_order(
    node_uuids: Sequence[str], center_node_uuid: str, neighbors: Iterable[str], min_score: float
) -> list[str]:
    """Neo4j's node_distance_reranker ordering.

    Direct neighbours of the center score 1, other candidates infinity; candidates are
    stably sorted by that distance, the center (when it is a candidate) is put first with
    score 0.1, and a candidate is kept when ``1 / score >= min_score``. Duplicates in
    ``node_uuids`` are kept, as in Neo4j.
    """
    filtered = [uuid for uuid in node_uuids if uuid != center_node_uuid]
    scores: dict[str, float] = {center_node_uuid: 0.0}
    for uuid in neighbors:
        scores[uuid] = 1.0
    for uuid in filtered:
        scores.setdefault(uuid, float('inf'))
    filtered.sort(key=lambda uuid: scores[uuid])
    if center_node_uuid in node_uuids:
        scores[center_node_uuid] = 0.1
        filtered = [center_node_uuid, *filtered]
    return [uuid for uuid in filtered if (1 / scores[uuid]) >= min_score]


def mentions_rerank_order(
    node_uuids: Sequence[str], mention_counts: Mapping[str, int], min_score: float
) -> list[str]:
    """Neo4j's episode_mentions_reranker ordering (``driver/neo4j/operations``).

    Candidates without MENTIONS score infinity; candidates are stably sorted by ASCENDING
    score and kept when ``score >= min_score``. Note: the legacy ``search_utils`` version
    sorts descending and scores unmentioned nodes 0; this mirrors the ops version.
    """
    scores: dict[str, float] = {uuid: float(count) for uuid, count in mention_counts.items()}
    for uuid in node_uuids:
        scores.setdefault(uuid, float('inf'))
    ordered = sorted(node_uuids, key=lambda uuid: scores[uuid])
    return [uuid for uuid in ordered if scores[uuid] >= min_score]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def is_zero_vector(vector: Sequence[float]) -> bool:
    """True for an empty or all-zero vector (Spanner's COSINE_DISTANCE raises on those)."""
    return not any(vector)


def sort_by_score(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rows by descending ``score`` (stable). Guards against unsorted results, seen on the
    emulator for GQL ``ORDER BY``; cheap because the rows are already limited."""
    return sorted(rows, key=lambda row: row['score'], reverse=True)


__all__ = [
    'COMMUNITY_COLUMNS',
    'COMMUNITY_SOURCE',
    'EDGE_COLUMNS',
    'ENTITIES_BY_UUIDS_SQL',
    'ENTITY_COLUMNS',
    'ENTITY_SOURCE',
    'EPISODE_COLUMNS',
    'EPISODE_SOURCE',
    'MENTION_COUNTS_SQL',
    'and_all',
    'center_neighbors_gql',
    'cosine_score',
    'distance_rerank_order',
    'edge_bfs_sql',
    'edge_source',
    'fulltext_sql',
    'is_zero_vector',
    'mentions_rerank_order',
    'node_bfs_sql',
    'result_columns',
    'similarity_sql',
    'sort_by_score',
]

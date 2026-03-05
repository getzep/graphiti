"""Neo4j-backed primitives retrieval for Observational Memory (OM) lane."""

from __future__ import annotations

import re
from typing import Any

OM_QUERY_MAX_CHARS = 240
OM_QUERY_MAX_UNIQUE_TOKENS = 32
OM_QUERY_MIN_TOKEN_LENGTH = 3

OM_NODE_CONTENT_FULLTEXT_INDEX = 'omnode_content_fulltext'
OM_FULLTEXT_MIN_CANDIDATES = 25
OM_FULLTEXT_MAX_CANDIDATES = 500
OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER = 6
OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER = 12

_OM_QUERY_TOKEN_RE = re.compile(r'[a-z0-9]+')
_OM_QUERY_STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'in',
    'is',
    'it',
    'of',
    'on',
    'or',
    'that',
    'the',
    'to',
    'with',
}

OM_RELATION_TYPES = (
    'MOTIVATES',
    'GENERATES',
    'SUPERSEDES',
    'ADDRESSES',
    'RESOLVES',
)


def _tokenize_query(query: str) -> list[str]:
    normalized_query = (query or '').strip().lower()
    if not normalized_query:
        return []

    normalized_query = normalized_query[:OM_QUERY_MAX_CHARS]

    seen: set[str] = set()
    tokens: list[str] = []
    for token in _OM_QUERY_TOKEN_RE.findall(normalized_query):
        if len(token) < OM_QUERY_MIN_TOKEN_LENGTH:
            continue
        if token in _OM_QUERY_STOPWORDS:
            continue
        if token in seen:
            continue

        seen.add(token)
        tokens.append(token)
        if len(tokens) >= OM_QUERY_MAX_UNIQUE_TOKENS:
            break

    return tokens


def _build_fulltext_query(*, group_id: str, tokens: list[str]) -> str:
    # Keep prior "any-token" behavior while applying an index-time lane filter.
    token_query = ' OR '.join(tokens)
    safe_group_id = str(group_id).replace('\\', r'\\').replace('"', r'\"')

    if not token_query:
        return f'group_id:"{safe_group_id}"'

    return f'group_id:"{safe_group_id}" AND ({token_query})'


def _fulltext_candidate_limit(limit: int, *, multiplier: int) -> int:
    bounded_limit = max(1, int(limit or 0))
    candidate_limit = bounded_limit * multiplier
    return max(OM_FULLTEXT_MIN_CANDIDATES, min(OM_FULLTEXT_MAX_CANDIDATES, candidate_limit))


class Neo4jService:
    """Query helper service for OM primitives in Neo4j."""

    async def search_om_nodes(
        self,
        driver: Any,
        *,
        group_id: str,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        tokens = _tokenize_query(query)

        # Guard broad/empty queries to avoid unbounded scan + sort behavior.
        if not tokens:
            return []

        records, _, _ = await driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes(
                $fulltext_index_name,
                $fulltext_query,
                {limit: $candidate_limit}
            )
            YIELD node, score
            WHERE node:OMNode
            WITH node, max(score) AS lexical_score
            RETURN coalesce(node.node_id, '') AS uuid,
                   coalesce(node.content, '') AS content,
                   coalesce(node.last_observed_at, node.created_at) AS created_at,
                   coalesce(node.group_id, $group_id) AS group_id,
                   coalesce(node.status, 'open') AS status,
                   coalesce(node.semantic_domain, '') AS semantic_domain,
                   coalesce(node.urgency_score, 3) AS urgency_score,
                   lexical_score AS lexical_score
            ORDER BY lexical_score DESC, coalesce(node.last_observed_at, node.created_at) DESC
            LIMIT $limit
            """,
            group_id=group_id,
            query_tokens=tokens,
            fulltext_query=_build_fulltext_query(group_id=group_id, tokens=tokens),
            fulltext_index_name=OM_NODE_CONTENT_FULLTEXT_INDEX,
            candidate_limit=_fulltext_candidate_limit(
                limit,
                multiplier=OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER,
            ),
            limit=limit,
            routing_='r',
        )
        return [record.data() if hasattr(record, 'data') else dict(record) for record in records]

    async def search_om_facts(
        self,
        driver: Any,
        *,
        group_id: str,
        query: str,
        limit: int,
        center_node_uuid: str | None = None,
    ) -> list[dict[str, Any]]:
        tokens = _tokenize_query(query)

        # Guard broad/empty queries to avoid unbounded scan + sort behavior.
        # If the caller provides a center node, run a bounded neighborhood scan
        # around that node instead of a whole-lane lexical search.
        if not tokens and center_node_uuid is None:
            return []

        if not tokens:
            records, _, _ = await driver.execute_query(
                """
                MATCH (center:OMNode)
                WHERE center.group_id = $group_id
                  AND (
                    center.node_id = $center_node_uuid
                    OR center.uuid = $center_node_uuid
                  )
                CALL {
                    WITH center
                    MATCH (center)-[rel:MOTIVATES]->(target:OMNode)
                    WHERE rel.group_id = $group_id
                      AND target.group_id = $group_id
                    RETURN center AS source, target AS target, rel

                    UNION

                    WITH center
                    MATCH (center)-[rel:GENERATES]->(target:OMNode)
                    WHERE rel.group_id = $group_id
                      AND target.group_id = $group_id
                    RETURN center AS source, target AS target, rel

                    UNION

                    WITH center
                    MATCH (center)-[rel:SUPERSEDES]->(target:OMNode)
                    WHERE rel.group_id = $group_id
                      AND target.group_id = $group_id
                    RETURN center AS source, target AS target, rel

                    UNION

                    WITH center
                    MATCH (center)-[rel:ADDRESSES]->(target:OMNode)
                    WHERE rel.group_id = $group_id
                      AND target.group_id = $group_id
                    RETURN center AS source, target AS target, rel

                    UNION

                    WITH center
                    MATCH (center)-[rel:RESOLVES]->(target:OMNode)
                    WHERE rel.group_id = $group_id
                      AND target.group_id = $group_id
                    RETURN center AS source, target AS target, rel

                    UNION

                    WITH center
                    MATCH (source:OMNode)-[rel:MOTIVATES]->(center)
                    WHERE rel.group_id = $group_id
                      AND source.group_id = $group_id
                    RETURN source AS source, center AS target, rel

                    UNION

                    WITH center
                    MATCH (source:OMNode)-[rel:GENERATES]->(center)
                    WHERE rel.group_id = $group_id
                      AND source.group_id = $group_id
                    RETURN source AS source, center AS target, rel

                    UNION

                    WITH center
                    MATCH (source:OMNode)-[rel:SUPERSEDES]->(center)
                    WHERE rel.group_id = $group_id
                      AND source.group_id = $group_id
                    RETURN source AS source, center AS target, rel

                    UNION

                    WITH center
                    MATCH (source:OMNode)-[rel:ADDRESSES]->(center)
                    WHERE rel.group_id = $group_id
                      AND source.group_id = $group_id
                    RETURN source AS source, center AS target, rel

                    UNION

                    WITH center
                    MATCH (source:OMNode)-[rel:RESOLVES]->(center)
                    WHERE rel.group_id = $group_id
                      AND source.group_id = $group_id
                    RETURN source AS source, center AS target, rel
                }
                RETURN coalesce(rel.uuid, source.node_id + ':' + type(rel) + ':' + target.node_id) AS uuid,
                       type(rel) AS relation_type,
                       coalesce(source.node_id, '') AS source_node_id,
                       coalesce(target.node_id, '') AS target_node_id,
                       coalesce(rel.created_at, source.created_at, target.created_at) AS created_at,
                       coalesce(rel.group_id, source.group_id, target.group_id, $group_id) AS group_id,
                       coalesce(source.content, '') AS source_content,
                       coalesce(target.content, '') AS target_content,
                       0 AS lexical_score
                ORDER BY coalesce(rel.created_at, source.created_at, target.created_at) DESC
                LIMIT $limit
                """,
                group_id=group_id,
                center_node_uuid=center_node_uuid,
                limit=limit,
                routing_='r',
            )
            return [record.data() if hasattr(record, 'data') else dict(record) for record in records]

        records, _, _ = await driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes(
                $fulltext_index_name,
                $fulltext_query,
                {limit: $candidate_limit}
            )
            YIELD node AS matched_node, score AS matched_score
            WHERE matched_node:OMNode
            CALL {
                WITH matched_node, matched_score
                MATCH (matched_node)-[rel:MOTIVATES|GENERATES|SUPERSEDES|ADDRESSES|RESOLVES]-(neighbor:OMNode)
                WHERE rel.group_id = $group_id
                  AND neighbor.group_id = $group_id
                  AND (
                      $center_node_uuid IS NULL
                      OR matched_node.node_id = $center_node_uuid
                      OR matched_node.uuid = $center_node_uuid
                      OR neighbor.node_id = $center_node_uuid
                      OR neighbor.uuid = $center_node_uuid
                  )
                RETURN CASE
                           WHEN startNode(rel) = matched_node THEN matched_node
                           ELSE neighbor
                       END AS source,
                       CASE
                           WHEN startNode(rel) = matched_node THEN neighbor
                           ELSE matched_node
                       END AS target,
                       rel,
                       matched_score AS lexical_score
            }
            WITH source, target, rel, max(lexical_score) AS lexical_score
            RETURN coalesce(rel.uuid, source.node_id + ':' + type(rel) + ':' + target.node_id) AS uuid,
                   type(rel) AS relation_type,
                   coalesce(source.node_id, '') AS source_node_id,
                   coalesce(target.node_id, '') AS target_node_id,
                   coalesce(rel.created_at, source.created_at, target.created_at) AS created_at,
                   coalesce(rel.group_id, source.group_id, target.group_id, $group_id) AS group_id,
                   coalesce(source.content, '') AS source_content,
                   coalesce(target.content, '') AS target_content,
                   lexical_score AS lexical_score
            ORDER BY lexical_score DESC, coalesce(rel.created_at, source.created_at, target.created_at) DESC
            LIMIT $limit
            """,
            group_id=group_id,
            center_node_uuid=center_node_uuid,
            query_tokens=tokens,
            fulltext_query=_build_fulltext_query(group_id=group_id, tokens=tokens),
            fulltext_index_name=OM_NODE_CONTENT_FULLTEXT_INDEX,
            candidate_limit=_fulltext_candidate_limit(
                limit,
                multiplier=OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER,
            ),
            limit=limit,
            routing_='r',
        )
        return [record.data() if hasattr(record, 'data') else dict(record) for record in records]

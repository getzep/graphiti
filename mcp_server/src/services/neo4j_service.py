"""Neo4j-backed primitives retrieval for Observational Memory (OM) lane."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

OM_QUERY_MAX_CHARS = 240
OM_QUERY_MAX_UNIQUE_TOKENS = 32
OM_QUERY_MIN_TOKEN_LENGTH = 3

OM_NODE_CONTENT_FULLTEXT_INDEX = 'omnode_content_fulltext'
OM_NODE_CONTENT_FULLTEXT_LABEL = 'OMNode'
OM_NODE_CONTENT_FULLTEXT_REQUIRED_PROPERTIES = frozenset({'content', 'group_id'})
OM_FULLTEXT_MIN_CANDIDATES = 25
OM_FULLTEXT_MAX_CANDIDATES = 500
OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER = 6
OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER = 12
OM_FULLTEXT_CREATE_ONLINE_MAX_ATTEMPTS = 6
OM_FULLTEXT_CREATE_ONLINE_RETRY_SECONDS = 0.5
OM_FACT_RELATION_EXPANSION_PER_CANDIDATE_MULTIPLIER = 8
OM_FACT_RELATION_EXPANSION_MIN = 32
OM_FACT_RELATION_EXPANSION_MAX = 256

_OM_FULLTEXT_INDEX_LOOKUP_QUERY = '''
            SHOW INDEXES
            YIELD name, type, state, entityType, labelsOrTypes, properties
            WHERE name = $index_name
            RETURN name, type, state, entityType, labelsOrTypes, properties
            '''

_OM_FULLTEXT_INDEX_CREATE_QUERY = '''
            CREATE FULLTEXT INDEX omnode_content_fulltext IF NOT EXISTS
            FOR (n:OMNode)
            ON EACH [n.content, n.group_id]
            '''

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


def _fact_relation_expansion_limit(limit: int) -> int:
    bounded_limit = max(1, int(limit or 0))
    candidate_limit = bounded_limit * OM_FACT_RELATION_EXPANSION_PER_CANDIDATE_MULTIPLIER
    return max(OM_FACT_RELATION_EXPANSION_MIN, min(OM_FACT_RELATION_EXPANSION_MAX, candidate_limit))


def _om_fulltext_index_health_error(
    *,
    problem: str,
    index_payload: dict[str, Any] | None = None,
) -> RuntimeError:
    details = f' Observed index payload: {index_payload!r}' if index_payload else ''
    return RuntimeError(
        'Neo4j OM full-text index preflight failed for '
        f'"{OM_NODE_CONTENT_FULLTEXT_INDEX}": {problem}.{details}\n'
        'Expected an ONLINE FULLTEXT NODE index on :OMNode including '
        '`content` and `group_id`.\n'
        'Verify/fix with:\n'
        '  SHOW INDEXES YIELD name, type, state, entityType, labelsOrTypes, properties '
        f'WHERE name = "{OM_NODE_CONTENT_FULLTEXT_INDEX}" RETURN *;\n'
        '  CREATE FULLTEXT INDEX omnode_content_fulltext IF NOT EXISTS '
        'FOR (n:OMNode) ON EACH [n.content, n.group_id];'
    )


class Neo4jService:
    """Query helper service for OM primitives in Neo4j."""

    async def _lookup_om_fulltext_index_payload(
        self,
        driver: Any,
        *,
        routing_preference: str = 'r',
    ) -> dict[str, Any] | None:
        records, _, _ = await driver.execute_query(
            _OM_FULLTEXT_INDEX_LOOKUP_QUERY,
            index_name=OM_NODE_CONTENT_FULLTEXT_INDEX,
            routing_=routing_preference,
        )
        if not records:
            return None
        first = records[0]
        return first.data() if hasattr(first, 'data') else dict(first)

    async def _wait_for_om_fulltext_index_online_after_create(
        self,
        driver: Any,
        *,
        routing_preference: str = 'w',
    ) -> dict[str, Any] | None:
        last_payload: dict[str, Any] | None = None

        for attempt in range(OM_FULLTEXT_CREATE_ONLINE_MAX_ATTEMPTS):
            payload = await self._lookup_om_fulltext_index_payload(
                driver,
                routing_preference=routing_preference,
            )
            if payload is not None:
                last_payload = payload
                if str(payload.get('state', '')).upper() == 'ONLINE':
                    return payload

            if attempt < OM_FULLTEXT_CREATE_ONLINE_MAX_ATTEMPTS - 1:
                await asyncio.sleep(OM_FULLTEXT_CREATE_ONLINE_RETRY_SECONDS)

        return last_payload

    async def verify_om_fulltext_index_shape(self, driver: Any) -> None:
        """Validate OM full-text index exists and has required shape.

        This is intended as a startup preflight guard for Neo4j-backed runtimes.
        """

        payload = await self._lookup_om_fulltext_index_payload(driver)
        created_missing_index = False

        if payload is None:
            logger.warning(
                'OM full-text index %r is missing; creating with required shape.',
                OM_NODE_CONTENT_FULLTEXT_INDEX,
            )
            await driver.execute_query(_OM_FULLTEXT_INDEX_CREATE_QUERY, routing_='w')
            created_missing_index = True
            logger.info(
                'Created missing OM full-text index %r; waiting for ONLINE state before preflight verification.',
                OM_NODE_CONTENT_FULLTEXT_INDEX,
            )
            payload = await self._wait_for_om_fulltext_index_online_after_create(driver)

            if payload is None:
                raise _om_fulltext_index_health_error(
                    problem='index is missing after create attempt'
                )

        index_type = str(payload.get('type', '')).upper()
        index_state = str(payload.get('state', '')).upper()
        entity_type = str(payload.get('entityType', '')).upper()

        labels_raw = payload.get('labelsOrTypes') or []
        labels = {str(label) for label in labels_raw}

        properties_raw = payload.get('properties') or []
        properties = {str(prop) for prop in properties_raw}

        shape_issues: list[str] = []

        if created_missing_index and index_state != 'ONLINE':
            shape_issues.append(
                'timed out waiting for index to reach ONLINE after create attempt '
                f'({OM_FULLTEXT_CREATE_ONLINE_MAX_ATTEMPTS} checks at '
                f'{OM_FULLTEXT_CREATE_ONLINE_RETRY_SECONDS:.1f}s interval); '
                f'state={index_state or "<missing>"}'
            )

        if index_type != 'FULLTEXT':
            shape_issues.append(f'type={index_type or "<missing>"}')
        if entity_type != 'NODE':
            shape_issues.append(f'entityType={entity_type or "<missing>"}')
        if index_state != 'ONLINE':
            shape_issues.append(f'state={index_state or "<missing>"}')
        if OM_NODE_CONTENT_FULLTEXT_LABEL not in labels:
            shape_issues.append(
                f'labelsOrTypes missing {OM_NODE_CONTENT_FULLTEXT_LABEL!r} (found={sorted(labels)!r})'
            )

        missing_props = sorted(OM_NODE_CONTENT_FULLTEXT_REQUIRED_PROPERTIES - properties)
        if missing_props:
            shape_issues.append(f'missing properties {missing_props!r} (found={sorted(properties)!r})')

        if shape_issues:
            raise _om_fulltext_index_health_error(
                problem='; '.join(shape_issues),
                index_payload=payload,
            )

        logger.info(
            'OM full-text index preflight passed for %r (state=%s, type=%s, entity=%s).',
            OM_NODE_CONTENT_FULLTEXT_INDEX,
            index_state,
            index_type,
            entity_type,
        )

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
              AND node.group_id = $group_id
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
              AND matched_node.group_id = $group_id
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
                WITH matched_node, matched_score, rel, neighbor
                ORDER BY coalesce(rel.created_at, neighbor.created_at, matched_node.created_at) DESC
                LIMIT $per_candidate_relationship_limit
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
            per_candidate_relationship_limit=_fact_relation_expansion_limit(limit),
            limit=limit,
            routing_='r',
        )
        return [record.data() if hasattr(record, 'data') else dict(record) for record in records]

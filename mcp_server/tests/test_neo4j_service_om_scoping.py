from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from services.neo4j_service import (
    OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER,
    OM_FULLTEXT_MAX_CANDIDATES,
    OM_FULLTEXT_MIN_CANDIDATES,
    OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER,
    OM_NODE_CONTENT_FULLTEXT_INDEX,
    OM_QUERY_MAX_CHARS,
    OM_QUERY_MAX_UNIQUE_TOKENS,
    Neo4jService,
    _fulltext_candidate_limit,
    _tokenize_query,
)


@pytest.mark.asyncio
async def test_search_om_nodes_requires_explicit_group_id_match():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    await service.search_om_nodes(
        driver,
        group_id='s1_observational_memory',
        query='latency',
        limit=5,
    )

    query_text = driver.execute_query.await_args.args[0]
    assert 'CALL db.index.fulltext.queryNodes' in query_text
    assert 'node.group_id = $group_id' not in query_text
    assert 'coalesce(node.group_id, $group_id) = $group_id' not in query_text
    assert 'reduce(' not in query_text
    assert 'CONTAINS token' not in query_text
    assert (
        driver.execute_query.await_args.kwargs['fulltext_index_name']
        == OM_NODE_CONTENT_FULLTEXT_INDEX
    )
    assert (
        driver.execute_query.await_args.kwargs['fulltext_query']
        == 'group_id:"s1_observational_memory" AND (latency)'
    )


@pytest.mark.asyncio
async def test_search_om_facts_requires_rel_and_node_group_matches():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    await service.search_om_facts(
        driver,
        group_id='s1_observational_memory',
        query='latency',
        limit=5,
    )

    query_text = driver.execute_query.await_args.args[0]
    assert 'CALL db.index.fulltext.queryNodes' in query_text
    assert 'matched_node.group_id = $group_id' not in query_text
    assert 'rel.group_id = $group_id' in query_text
    assert 'neighbor.group_id = $group_id' in query_text
    assert '$center_node_uuid IS NULL' in query_text
    assert 'type(rel) IN $relation_types' not in query_text
    assert 'coalesce(source.node_id, source.uuid' not in query_text
    assert 'coalesce(target.node_id, target.uuid' not in query_text
    assert '[rel:MOTIVATES|GENERATES|SUPERSEDES|ADDRESSES|RESOLVES]' in query_text
    assert 'reduce(' not in query_text
    assert 'CONTAINS token' not in query_text
    assert (
        'coalesce(rel.group_id, source.group_id, target.group_id, $group_id) = $group_id'
        not in query_text
    )
    assert (
        driver.execute_query.await_args.kwargs['fulltext_index_name']
        == OM_NODE_CONTENT_FULLTEXT_INDEX
    )
    assert (
        driver.execute_query.await_args.kwargs['fulltext_query']
        == 'group_id:"s1_observational_memory" AND (latency)'
    )
    assert driver.execute_query.await_args.kwargs['center_node_uuid'] is None


@pytest.mark.asyncio
async def test_search_om_facts_applies_center_node_scope_when_provided():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    await service.search_om_facts(
        driver,
        group_id='s1_observational_memory',
        query='latency',
        limit=5,
        center_node_uuid='om-node-123',
    )

    assert driver.execute_query.await_args.kwargs['center_node_uuid'] == 'om-node-123'


def test_tokenize_query_caps_query_length():
    query = 'x' * (OM_QUERY_MAX_CHARS + 37)

    tokens = _tokenize_query(query)

    assert len(tokens) == 1
    assert len(tokens[0]) == OM_QUERY_MAX_CHARS
    assert tokens[0] == ('x' * OM_QUERY_MAX_CHARS)


def test_tokenize_query_bounds_unique_tokens():
    tokens = _tokenize_query(
        ' '.join(f't{i:02d}' for i in range(OM_QUERY_MAX_UNIQUE_TOKENS + 10))
    )

    assert len(tokens) == OM_QUERY_MAX_UNIQUE_TOKENS
    assert tokens == [f't{i:02d}' for i in range(OM_QUERY_MAX_UNIQUE_TOKENS)]


@pytest.mark.asyncio
async def test_search_om_nodes_tokenization_stays_backward_compatible_for_normal_query():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    await service.search_om_nodes(
        driver,
        group_id='s1_observational_memory',
        query='Latency latency node node',
        limit=5,
    )

    assert driver.execute_query.await_args.kwargs['query_tokens'] == ['latency', 'node']
    assert (
        driver.execute_query.await_args.kwargs['fulltext_query']
        == 'group_id:"s1_observational_memory" AND (latency OR node)'
    )
    assert (
        driver.execute_query.await_args.kwargs['candidate_limit']
        >= driver.execute_query.await_args.kwargs['limit']
    )


def test_tokenize_query_filters_stopwords_and_short_tokens():
    tokens = _tokenize_query('the and to a io on neo4j heap')

    assert tokens == ['neo4j', 'heap']


@pytest.mark.asyncio
async def test_search_om_nodes_empty_query_short_circuits_without_db_scan():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    rows = await service.search_om_nodes(
        driver,
        group_id='s1_observational_memory',
        query='   ',
        limit=5,
    )

    assert rows == []
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_om_facts_empty_query_without_center_short_circuits_without_db_scan():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    rows = await service.search_om_facts(
        driver,
        group_id='s1_observational_memory',
        query='',
        limit=5,
    )

    assert rows == []
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_om_facts_empty_query_with_center_uses_bounded_center_path():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([], None, None))

    service = Neo4jService()
    await service.search_om_facts(
        driver,
        group_id='s1_observational_memory',
        query='',
        limit=5,
        center_node_uuid='om-node-123',
    )

    query_text = driver.execute_query.await_args.args[0]
    assert 'MATCH (center:OMNode)' in query_text
    assert 'CALL {' in query_text
    assert 'db.index.fulltext.queryNodes' not in query_text
    assert '$center_node_uuid IS NULL' not in query_text
    assert 'coalesce(center.node_id, center.uuid, \'\') = $center_node_uuid' not in query_text
    assert 'center.node_id = $center_node_uuid' in query_text
    assert 'center.uuid = $center_node_uuid' in query_text
    assert 'type(rel) IN $relation_types' not in query_text
    assert '[rel:MOTIVATES]' in query_text
    assert '[rel:RESOLVES]' in query_text
    assert driver.execute_query.await_args.kwargs['center_node_uuid'] == 'om-node-123'


@pytest.mark.parametrize(
    ('limit', 'multiplier', 'expected'),
    [
        (1, OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER, OM_FULLTEXT_MIN_CANDIDATES),
        (4, OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER, OM_FULLTEXT_MIN_CANDIDATES),
        (5, OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER, 30),
        (83, OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER, 498),
        (84, OM_FULLTEXT_NODE_CANDIDATE_MULTIPLIER, OM_FULLTEXT_MAX_CANDIDATES),
        (1, OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER, OM_FULLTEXT_MIN_CANDIDATES),
        (2, OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER, OM_FULLTEXT_MIN_CANDIDATES),
        (3, OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER, 36),
        (41, OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER, 492),
        (42, OM_FULLTEXT_FACT_CANDIDATE_MULTIPLIER, OM_FULLTEXT_MAX_CANDIDATES),
    ],
)
def test_fulltext_candidate_limit_clamps_min_and_max(limit: int, multiplier: int, expected: int):
    assert _fulltext_candidate_limit(limit, multiplier=multiplier) == expected


@pytest.mark.asyncio
async def test_verify_om_fulltext_index_shape_auto_creates_when_missing():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(
        side_effect=[
            ([], None, None),
            ([], None, None),
            (
                [
                    {
                        'name': 'omnode_content_fulltext',
                        'type': 'FULLTEXT',
                        'state': 'ONLINE',
                        'entityType': 'NODE',
                        'labelsOrTypes': ['OMNode'],
                        'properties': ['content', 'group_id'],
                    }
                ],
                None,
                None,
            ),
        ]
    )

    service = Neo4jService()
    await service.verify_om_fulltext_index_shape(driver)

    assert driver.execute_query.await_count == 3
    assert driver.execute_query.await_args_list[1].args[0].strip().startswith('CREATE FULLTEXT INDEX')


@pytest.mark.asyncio
async def test_verify_om_fulltext_index_shape_rejects_wrong_shape_without_auto_fix():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {
                    'name': 'omnode_content_fulltext',
                    'type': 'FULLTEXT',
                    'state': 'ONLINE',
                    'entityType': 'NODE',
                    'labelsOrTypes': ['OMNode'],
                    'properties': ['content'],
                }
            ],
            None,
            None,
        )
    )

    service = Neo4jService()
    with pytest.raises(RuntimeError, match='missing properties') as exc_info:
        await service.verify_om_fulltext_index_shape(driver)

    message = str(exc_info.value)
    assert 'group_id' in message
    assert 'content' in message
    assert driver.execute_query.await_count == 1
    assert 'CREATE FULLTEXT INDEX' not in driver.execute_query.await_args.args[0]


@pytest.mark.asyncio
async def test_verify_om_fulltext_index_shape_accepts_required_shape():
    driver = AsyncMock()
    driver.execute_query = AsyncMock(
        return_value=(
            [
                {
                    'name': 'omnode_content_fulltext',
                    'type': 'FULLTEXT',
                    'state': 'ONLINE',
                    'entityType': 'NODE',
                    'labelsOrTypes': ['OMNode'],
                    'properties': ['content', 'group_id'],
                }
            ],
            None,
            None,
        )
    )

    service = Neo4jService()
    await service.verify_om_fulltext_index_shape(driver)

    assert driver.execute_query.await_count == 1
    assert driver.execute_query.await_args.kwargs['index_name'] == 'omnode_content_fulltext'

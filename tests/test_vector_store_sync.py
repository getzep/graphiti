"""Unit tests for the vector store backfill utility."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.utils.vector_store_sync import backfill_vector_store


def _mock_driver():
    driver = MagicMock()
    driver.provider = MagicMock()
    driver.provider.value = 'neo4j'
    driver.provider.__eq__ = lambda self, other: False
    driver.execute_query = AsyncMock(return_value=([], None, None))
    return driver


def _mock_vector_store():
    vs = AsyncMock()
    vs.ensure_ready = AsyncMock()
    vs.upsert = AsyncMock()
    vs.collection_name = lambda suffix: f'test_{suffix}'
    return vs


class TestBackfillVectorStore:
    @pytest.mark.asyncio
    async def test_backfill_empty_graph(self):
        driver = _mock_driver()
        vs = _mock_vector_store()

        counts = await backfill_vector_store(driver, vs)

        assert counts == {
            'entity_nodes': 0,
            'entity_edges': 0,
            'episodic_nodes': 0,
            'community_nodes': 0,
        }
        vs.ensure_ready.assert_called_once()
        vs.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_backfill_entity_nodes(self):
        driver = _mock_driver()
        vs = _mock_vector_store()

        # Simulate entity node records from graph DB
        entity_records = [
            {
                'uuid': 'n1',
                'name': 'Alice',
                'group_id': 'g1',
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'summary': 'A person',
                'name_embedding': [0.1] * 128,
                'labels': ['Person', 'Entity'],
                'attributes': {'uuid': 'n1', 'name': 'Alice'},
            },
            {
                'uuid': 'n2',
                'name': 'Bob',
                'group_id': 'g1',
                'created_at': datetime(2024, 1, 2, tzinfo=timezone.utc),
                'summary': 'Another person',
                'name_embedding': [0.2] * 128,
                'labels': ['Person', 'Entity'],
                'attributes': {'uuid': 'n2', 'name': 'Bob'},
            },
        ]

        # First call = entity nodes, rest = empty
        driver.execute_query = AsyncMock(
            side_effect=[
                (entity_records, None, None),
                ([], None, None),  # edges
                ([], None, None),  # episodic
                ([], None, None),  # community
            ]
        )

        counts = await backfill_vector_store(driver, vs)

        assert counts['entity_nodes'] == 2
        assert vs.upsert.call_count == 1
        call_kwargs = vs.upsert.call_args.kwargs
        assert call_kwargs['collection_name'] == 'test_entity_nodes'
        assert len(call_kwargs['data']) == 2

    @pytest.mark.asyncio
    async def test_backfill_skips_nodes_without_embeddings(self):
        driver = _mock_driver()
        vs = _mock_vector_store()

        records = [
            {
                'uuid': 'n1',
                'name': 'Alice',
                'group_id': 'g1',
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'summary': 'A person',
                'name_embedding': None,  # No embedding
                'labels': ['Entity'],
                'attributes': {'uuid': 'n1'},
            },
        ]

        driver.execute_query = AsyncMock(
            side_effect=[
                (records, None, None),
                ([], None, None),
                ([], None, None),
                ([], None, None),
            ]
        )

        counts = await backfill_vector_store(driver, vs)

        assert counts['entity_nodes'] == 0
        vs.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_backfill_entity_edges(self):
        driver = _mock_driver()
        vs = _mock_vector_store()

        edge_records = [
            {
                'uuid': 'e1',
                'group_id': 'g1',
                'source_node_uuid': 'n1',
                'target_node_uuid': 'n2',
                'name': 'KNOWS',
                'fact': 'Alice knows Bob',
                'fact_embedding': [0.3] * 128,
                'episodes': ['ep1'],
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'expired_at': None,
                'valid_at': None,
                'invalid_at': None,
            },
        ]

        driver.execute_query = AsyncMock(
            side_effect=[
                ([], None, None),  # entity nodes
                (edge_records, None, None),  # edges
                ([], None, None),  # episodic
                ([], None, None),  # community
            ]
        )

        counts = await backfill_vector_store(driver, vs)

        assert counts['entity_edges'] == 1

    @pytest.mark.asyncio
    async def test_backfill_with_group_ids_filter(self):
        driver = _mock_driver()
        vs = _mock_vector_store()

        driver.execute_query = AsyncMock(return_value=([], None, None))

        counts = await backfill_vector_store(driver, vs, group_ids=['g1', 'g2'])

        assert counts == {
            'entity_nodes': 0,
            'entity_edges': 0,
            'episodic_nodes': 0,
            'community_nodes': 0,
        }

        # Verify that group_ids was passed to queries
        for call in driver.execute_query.call_args_list:
            assert call.kwargs.get('group_ids') == ['g1', 'g2']

    @pytest.mark.asyncio
    async def test_backfill_batches_upserts(self):
        driver = _mock_driver()
        vs = _mock_vector_store()

        # Create 5 records with batch_size=2 → should make 3 upsert calls
        records = [
            {
                'uuid': f'n{i}',
                'name': f'Node{i}',
                'group_id': 'g1',
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'summary': f'Node {i}',
                'name_embedding': [0.1 * i] * 128,
                'labels': ['Entity'],
                'attributes': {'uuid': f'n{i}'},
            }
            for i in range(5)
        ]

        driver.execute_query = AsyncMock(
            side_effect=[
                (records, None, None),  # entity nodes
                ([], None, None),  # edges
                ([], None, None),  # episodic
                ([], None, None),  # community
            ]
        )

        counts = await backfill_vector_store(driver, vs, batch_size=2)

        assert counts['entity_nodes'] == 5
        # 3 batches: [2, 2, 1]
        entity_upsert_calls = [
            c
            for c in vs.upsert.call_args_list
            if c.kwargs.get('collection_name') == 'test_entity_nodes'
        ]
        assert len(entity_upsert_calls) == 3

    @pytest.mark.asyncio
    async def test_backfill_all_collection_types(self):
        driver = _mock_driver()
        vs = _mock_vector_store()

        entity_records = [
            {
                'uuid': 'n1',
                'name': 'Alice',
                'group_id': 'g1',
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'summary': 'A person',
                'name_embedding': [0.1] * 128,
                'labels': ['Entity'],
                'attributes': {'uuid': 'n1'},
            },
        ]
        edge_records = [
            {
                'uuid': 'e1',
                'group_id': 'g1',
                'source_node_uuid': 'n1',
                'target_node_uuid': 'n2',
                'name': 'KNOWS',
                'fact': 'fact',
                'fact_embedding': [0.2] * 128,
                'episodes': [],
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'expired_at': None,
                'valid_at': None,
                'invalid_at': None,
            },
        ]
        episodic_records = [
            {
                'uuid': 'ep1',
                'group_id': 'g1',
                'name': 'episode1',
                'content': 'content',
                'source': 'text',
                'source_description': 'desc',
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'valid_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'entity_edges': ['e1'],
            },
        ]
        community_records = [
            {
                'uuid': 'c1',
                'group_id': 'g1',
                'name': 'Community',
                'summary': 'A community',
                'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
                'name_embedding': [0.4] * 128,
            },
        ]

        driver.execute_query = AsyncMock(
            side_effect=[
                (entity_records, None, None),
                (edge_records, None, None),
                (episodic_records, None, None),
                (community_records, None, None),
            ]
        )

        counts = await backfill_vector_store(driver, vs)

        assert counts == {
            'entity_nodes': 1,
            'entity_edges': 1,
            'episodic_nodes': 1,
            'community_nodes': 1,
        }
        assert vs.upsert.call_count == 4

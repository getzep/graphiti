"""Unit tests for vector store dual-write hooks in nodes.py, edges.py, and bulk_utils.py."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.nodes import CommunityNode, EntityNode


def _mock_driver(*, with_vector_store: bool = True):
    """Create a mock GraphDriver with optional vector_store."""
    driver = MagicMock()
    driver.graph_operations_interface = None
    driver.provider = MagicMock()
    driver.provider.value = 'neo4j'
    driver.provider.__eq__ = lambda self, other: False  # not KUZU, not NEPTUNE
    driver.execute_query = AsyncMock(return_value=[])

    if with_vector_store:
        driver.vector_store = AsyncMock()
        driver.vector_store.collection_name = lambda suffix: f'test_{suffix}'
        driver.vector_store.ensure_ready = AsyncMock()
        driver.vector_store.upsert = AsyncMock()
        driver.vector_store.delete = AsyncMock()
        driver.vector_store.reset_collections = AsyncMock()
        # Domain-aware methods
        driver.vector_store.save_entity_nodes = AsyncMock()
        driver.vector_store.save_entity_edges = AsyncMock()
        driver.vector_store.save_episodic_nodes = AsyncMock()
        driver.vector_store.save_community_nodes = AsyncMock()
        driver.vector_store.delete_entity_nodes = AsyncMock()
        driver.vector_store.delete_entity_edges = AsyncMock()
        driver.vector_store.delete_nodes_by_uuids = AsyncMock()
        driver.vector_store.delete_community_nodes = AsyncMock()
        driver.vector_store.delete_by_group_ids = AsyncMock()
        driver.vector_store.clear_all = AsyncMock()
    else:
        driver.vector_store = None

    return driver


def _mock_driver_with_session(*, with_vector_store: bool = True):
    """Create a mock GraphDriver with session support (for clear_data tests)."""
    driver = _mock_driver(with_vector_store=with_vector_store)

    mock_tx = AsyncMock()
    mock_session = AsyncMock()
    mock_session.execute_write = AsyncMock(side_effect=lambda fn: fn(mock_tx))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    driver.session = MagicMock(return_value=mock_session)

    return driver


def _make_entity_node(**kwargs):
    return EntityNode(
        uuid=kwargs.get('uuid', 'node-1'),
        name=kwargs.get('name', 'Alice'),
        group_id=kwargs.get('group_id', 'g1'),
        labels=kwargs.get('labels', ['Person']),
        created_at=kwargs.get('created_at', datetime(2024, 1, 15, tzinfo=timezone.utc)),
        summary=kwargs.get('summary', 'A person'),
        name_embedding=kwargs.get('name_embedding', [0.1] * 128),
    )


def _make_community_node(**kwargs):
    return CommunityNode(
        uuid=kwargs.get('uuid', 'comm-1'),
        name=kwargs.get('name', 'Tech Community'),
        group_id=kwargs.get('group_id', 'g1'),
        created_at=kwargs.get('created_at', datetime(2024, 1, 15, tzinfo=timezone.utc)),
        summary=kwargs.get('summary', 'A tech community'),
        name_embedding=kwargs.get('name_embedding', [0.3] * 128),
    )


# ---- EntityNode.save() dual-write ----


class TestEntityNodeDualWrite:
    @pytest.mark.asyncio
    async def test_save_writes_to_graph_and_vector_store(self):
        driver = _mock_driver(with_vector_store=True)
        node = _make_entity_node()

        await node.save(driver)

        driver.execute_query.assert_called_once()
        driver.vector_store.save_entity_nodes.assert_called_once()
        call_args = driver.vector_store.save_entity_nodes.call_args
        assert call_args[0][0][0].uuid == 'node-1'

    @pytest.mark.asyncio
    async def test_save_skips_vector_store_when_not_set(self):
        driver = _mock_driver(with_vector_store=False)
        node = _make_entity_node()

        await node.save(driver)

        driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_succeeds_when_vector_store_fails(self):
        driver = _mock_driver(with_vector_store=True)
        driver.vector_store.save_entity_nodes.side_effect = Exception('connection error')
        node = _make_entity_node()

        await node.save(driver)

        driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_with_graph_ops_interface_skips_dual_write(self):
        """When graph_operations_interface handles the save, no dual-write occurs."""
        driver = _mock_driver(with_vector_store=True)
        driver.graph_operations_interface = MagicMock()
        driver.graph_operations_interface.node_save = AsyncMock()

        node = _make_entity_node()
        await node.save(driver)

        driver.graph_operations_interface.node_save.assert_called_once()
        driver.execute_query.assert_not_called()
        driver.vector_store.save_entity_nodes.assert_not_called()


# ---- CommunityNode.save() dual-write ----


class TestCommunityNodeDualWrite:
    @pytest.mark.asyncio
    async def test_save_writes_to_graph_and_vector_store(self):
        driver = _mock_driver(with_vector_store=True)
        node = _make_community_node()

        await node.save(driver)

        driver.execute_query.assert_called_once()
        driver.vector_store.save_community_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_succeeds_when_vector_store_fails(self):
        driver = _mock_driver(with_vector_store=True)
        driver.vector_store.save_community_nodes.side_effect = Exception('timeout')
        node = _make_community_node()

        await node.save(driver)

        driver.execute_query.assert_called_once()


# ---- EntityEdge.save() dual-write ----


class TestEntityEdgeDualWrite:
    @pytest.mark.asyncio
    async def test_save_writes_to_graph_and_vector_store(self):
        from graphiti_core.edges import EntityEdge

        driver = _mock_driver(with_vector_store=True)
        edge = EntityEdge(
            uuid='edge-1',
            source_node_uuid='src-1',
            target_node_uuid='tgt-1',
            name='WORKS_AT',
            fact='Alice works at Acme',
            group_id='g1',
            episodes=['ep1'],
            created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            fact_embedding=[0.2] * 128,
        )

        await edge.save(driver)

        driver.execute_query.assert_called_once()
        driver.vector_store.save_entity_edges.assert_called_once()
        call_args = driver.vector_store.save_entity_edges.call_args
        assert call_args[0][0][0].uuid == 'edge-1'

    @pytest.mark.asyncio
    async def test_save_succeeds_when_vector_store_fails(self):
        from graphiti_core.edges import EntityEdge

        driver = _mock_driver(with_vector_store=True)
        driver.vector_store.save_entity_edges.side_effect = Exception('nope')
        edge = EntityEdge(
            uuid='edge-2',
            source_node_uuid='src-1',
            target_node_uuid='tgt-1',
            name='WORKS_AT',
            fact='Bob works at Acme',
            group_id='g1',
            episodes=[],
            created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            fact_embedding=[0.2] * 128,
        )

        await edge.save(driver)

        driver.execute_query.assert_called_once()


# ---- bulk_utils dual-write ----


class TestBulkDualWrite:
    @pytest.mark.asyncio
    async def test_bulk_writes_to_vector_store(self):
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EpisodicNode
        from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

        driver = _mock_driver(with_vector_store=True)
        tx = AsyncMock()

        episode = EpisodicNode(
            uuid='ep-1',
            name='episode1',
            group_id='g1',
            content='Meeting notes',
            source='text',
            source_description='test',
            created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            valid_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            entity_edges=['edge-1'],
        )

        node = _make_entity_node()
        edge = EntityEdge(
            uuid='edge-1',
            source_node_uuid='src-1',
            target_node_uuid='tgt-1',
            name='KNOWS',
            fact='Alice knows Bob',
            group_id='g1',
            episodes=['ep-1'],
            created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            fact_embedding=[0.2] * 128,
        )

        await add_nodes_and_edges_bulk_tx(
            tx,
            [episode],
            [],
            [node],
            [edge],
            MagicMock(),
            driver=driver,
        )

        # Vector store should have been called for episodic, entity, and edge
        driver.vector_store.save_episodic_nodes.assert_called_once()
        driver.vector_store.save_entity_nodes.assert_called_once()
        driver.vector_store.save_entity_edges.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_skips_vector_store_when_not_set(self):
        from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

        driver = _mock_driver(with_vector_store=False)
        tx = AsyncMock()

        await add_nodes_and_edges_bulk_tx(tx, [], [], [], [], MagicMock(), driver=driver)

        # No error, no vector store calls

    @pytest.mark.asyncio
    async def test_bulk_succeeds_when_vector_store_fails(self):
        from graphiti_core.edges import EntityEdge
        from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

        driver = _mock_driver(with_vector_store=True)
        driver.vector_store.save_episodic_nodes.side_effect = Exception('vector store down')
        tx = AsyncMock()

        node = _make_entity_node()
        edge = EntityEdge(
            uuid='edge-1',
            source_node_uuid='src-1',
            target_node_uuid='tgt-1',
            name='KNOWS',
            fact='Alice knows Bob',
            group_id='g1',
            episodes=[],
            created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            fact_embedding=[0.2] * 128,
        )

        # Should not raise even though vector store fails
        await add_nodes_and_edges_bulk_tx(tx, [], [], [node], [edge], MagicMock(), driver=driver)

        # Graph DB write still happened
        assert tx.run.call_count >= 1


# ---- Node.delete() dual-write ----


class TestNodeDeleteDualWrite:
    @pytest.mark.asyncio
    async def test_delete_removes_from_vector_store(self):
        driver = _mock_driver(with_vector_store=True)
        node = _make_entity_node()

        await node.delete(driver)

        # Graph DB delete happened
        assert driver.execute_query.call_count >= 1
        # Vector store delete called via domain-aware method
        driver.vector_store.delete_nodes_by_uuids.assert_called_once_with(['node-1'])

    @pytest.mark.asyncio
    async def test_delete_skips_vector_store_when_not_set(self):
        driver = _mock_driver(with_vector_store=False)
        node = _make_entity_node()

        await node.delete(driver)

        assert driver.execute_query.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_succeeds_when_vector_store_fails(self):
        driver = _mock_driver(with_vector_store=True)
        driver.vector_store.delete_nodes_by_uuids.side_effect = Exception('vector store down')
        node = _make_entity_node()

        await node.delete(driver)

        assert driver.execute_query.call_count >= 1


# ---- Node.delete_by_uuids() dual-write ----


class TestNodeDeleteByUuidsDualWrite:
    @pytest.mark.asyncio
    async def test_delete_by_uuids_removes_from_vector_store(self):
        from graphiti_core.nodes import Node

        driver = _mock_driver(with_vector_store=True)

        await Node.delete_by_uuids(driver, ['uuid-1', 'uuid-2'])

        # Vector store delete called via domain-aware method
        driver.vector_store.delete_nodes_by_uuids.assert_called_once_with(['uuid-1', 'uuid-2'])

    @pytest.mark.asyncio
    async def test_delete_by_uuids_skips_empty_list(self):
        from graphiti_core.nodes import Node

        driver = _mock_driver(with_vector_store=True)

        await Node.delete_by_uuids(driver, [])

        # No vector store delete for empty list
        driver.vector_store.delete_nodes_by_uuids.assert_not_called()


# ---- EntityEdge.delete() dual-write ----


class TestEdgeDeleteDualWrite:
    @pytest.mark.asyncio
    async def test_delete_removes_from_vector_store(self):
        from graphiti_core.edges import EntityEdge

        driver = _mock_driver(with_vector_store=True)
        edge = EntityEdge(
            uuid='edge-del-1',
            source_node_uuid='src-1',
            target_node_uuid='tgt-1',
            name='WORKS_AT',
            fact='Alice works at Acme',
            group_id='g1',
            episodes=['ep1'],
            created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            fact_embedding=[0.2] * 128,
        )

        await edge.delete(driver)

        driver.vector_store.delete_entity_edges.assert_called_once_with(['edge-del-1'])

    @pytest.mark.asyncio
    async def test_delete_by_uuids_removes_from_vector_store(self):
        from graphiti_core.edges import EntityEdge

        driver = _mock_driver(with_vector_store=True)

        await EntityEdge.delete_by_uuids(driver, ['e1', 'e2'])

        driver.vector_store.delete_entity_edges.assert_called_once_with(['e1', 'e2'])


# ---- clear_data() dual-write ----


class TestClearDataDualWrite:
    @pytest.mark.asyncio
    async def test_clear_all_resets_vector_store(self):
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        driver = _mock_driver_with_session(with_vector_store=True)

        await clear_data(driver, group_ids=None)

        driver.vector_store.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_by_group_ids_deletes_from_vector_store(self):
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        driver = _mock_driver_with_session(with_vector_store=True)

        await clear_data(driver, group_ids=['g1', 'g2'])

        driver.vector_store.delete_by_group_ids.assert_called_once_with(['g1', 'g2'])

    @pytest.mark.asyncio
    async def test_clear_skips_vector_store_when_not_set(self):
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        driver = _mock_driver_with_session(with_vector_store=False)

        await clear_data(driver, group_ids=None)


# ---- remove_communities() dual-write ----


class TestRemoveCommunitiesDualWrite:
    @pytest.mark.asyncio
    async def test_remove_communities_deletes_from_vector_store(self):
        from graphiti_core.utils.maintenance.community_operations import remove_communities

        driver = _mock_driver(with_vector_store=True)

        await remove_communities(driver)

        driver.vector_store.delete_community_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_communities_skips_vector_store_when_not_set(self):
        from graphiti_core.utils.maintenance.community_operations import remove_communities

        driver = _mock_driver(with_vector_store=False)

        await remove_communities(driver)

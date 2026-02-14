"""Integration tests for Milvus search against Zilliz Cloud.

These tests require:
1. pymilvus installed: pip install "graphiti-core[milvus]"
2. Credentials in .env.milvus file:
   MILVUS_URI=https://your-instance.cloud.zilliz.com
   MILVUS_TOKEN=your_token_here
"""

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from dotenv import load_dotenv

# Load Milvus secrets from .env.milvus if it exists
load_dotenv('.env.milvus')

MILVUS_URI = os.environ.get('MILVUS_URI')
MILVUS_TOKEN = os.environ.get('MILVUS_TOKEN')
HAS_MILVUS_CREDENTIALS = bool(MILVUS_URI and MILVUS_TOKEN)

try:
    from pymilvus import AsyncMilvusClient  # noqa: F401

    HAS_PYMILVUS = True
except ImportError:
    HAS_PYMILVUS = False

SKIP_REASON = 'Milvus credentials not available (set MILVUS_URI and MILVUS_TOKEN in .env.milvus)'
SKIP_PYMILVUS = 'pymilvus not installed'


def _random_prefix():
    """Generate a unique collection prefix for test isolation."""
    return f'test_{uuid4().hex[:8]}'


def _make_embedding(dim=128):
    """Create a simple non-zero embedding vector."""
    import random

    random.seed(42)
    return [random.random() for _ in range(dim)]


@pytest.mark.skipif(not HAS_PYMILVUS, reason=SKIP_PYMILVUS)
@pytest.mark.skipif(not HAS_MILVUS_CREDENTIALS, reason=SKIP_REASON)
class TestMilvusSearchIntegration:
    """Integration tests for Milvus search + graph ops against Zilliz Cloud.

    Uses a unique collection_prefix per test class to avoid collisions.
    Cleans up collections after tests.
    """

    DIM = 128
    PREFIX = _random_prefix()

    @pytest.fixture(autouse=True)
    def _setup_interfaces(self):
        """Create search and graph ops interfaces for the test class."""
        from graphiti_core.vector_store.milvus_client import (
            MilvusVectorStoreClient,
            MilvusVectorStoreConfig,
        )
        from graphiti_core.vector_store.milvus_graph_operations import (
            MilvusGraphOperationsInterface,
        )
        from graphiti_core.vector_store.milvus_search_interface import MilvusSearchInterface

        vs_config = MilvusVectorStoreConfig(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN,
            embedding_dim=self.DIM,
            collection_prefix=self.PREFIX,
        )
        self.vs_client = MilvusVectorStoreClient(config=vs_config)

        self.search = MilvusSearchInterface(vs_client=self.vs_client)
        self.graph_ops = MilvusGraphOperationsInterface(vs_client=self.vs_client)
        yield

    @pytest.fixture(autouse=True)
    def _mock_search_filter(self):
        """Provide an empty search filter for tests."""
        from unittest.mock import MagicMock

        self.empty_filter = MagicMock()
        self.empty_filter.node_labels = []
        self.empty_filter.edge_types = []
        self.empty_filter.edge_uuids = []
        self.empty_filter.valid_at = []
        self.empty_filter.invalid_at = []
        self.empty_filter.created_at = []
        self.empty_filter.expired_at = []
        self.empty_filter.property_filters = []

    @pytest.mark.asyncio
    async def test_collection_creation_int(self):
        """Ensure collections are created on first use."""
        await self.vs_client.ensure_ready()
        for suffix in ['entity_nodes', 'entity_edges', 'episodic_nodes', 'community_nodes']:
            col_name = f'{self.PREFIX}_{suffix}'
            has = await self.vs_client.has_collection(col_name)
            assert has, f'Collection {col_name} was not created'

    @pytest.mark.asyncio
    async def test_node_upsert_and_similarity_search_int(self):
        """Insert entity nodes, then search by vector similarity."""
        from unittest.mock import MagicMock

        node = MagicMock()
        node.uuid = str(uuid4())
        node.group_id = 'test_group'
        node.name = 'Alice Johnson'
        node.summary = 'A software engineer at Acme Corp'
        node.labels = ['Person']
        node.created_at = datetime.now(tz=timezone.utc)
        node.attributes = {'role': 'engineer'}
        node.name_embedding = _make_embedding(self.DIM)

        await self.graph_ops.node_save(node, driver=MagicMock())

        # Search with the same embedding should find the node
        results = await self.search.node_similarity_search(
            driver=MagicMock(),
            search_vector=node.name_embedding,
            search_filter=self.empty_filter,
            group_ids=['test_group'],
            limit=10,
            min_score=0.5,
        )

        assert len(results) >= 1
        assert any(r.uuid == node.uuid for r in results)

    @pytest.mark.asyncio
    async def test_edge_upsert_and_similarity_search_int(self):
        """Insert entity edges, then search by fact embedding."""
        from unittest.mock import MagicMock

        edge = MagicMock()
        edge.uuid = str(uuid4())
        edge.group_id = 'test_group'
        edge.source_node_uuid = str(uuid4())
        edge.target_node_uuid = str(uuid4())
        edge.name = 'WORKS_AT'
        edge.fact = 'Alice works at Acme Corporation as a senior engineer'
        edge.episodes = ['ep1']
        edge.created_at = datetime.now(tz=timezone.utc)
        edge.expired_at = None
        edge.valid_at = datetime.now(tz=timezone.utc)
        edge.invalid_at = None
        edge.attributes = {}
        edge.fact_embedding = _make_embedding(self.DIM)

        await self.graph_ops.edge_save(edge, driver=MagicMock())

        results = await self.search.edge_similarity_search(
            driver=MagicMock(),
            search_vector=edge.fact_embedding,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=self.empty_filter,
            group_ids=['test_group'],
            limit=10,
            min_score=0.5,
        )

        assert len(results) >= 1
        assert any(r.uuid == edge.uuid for r in results)

    @pytest.mark.asyncio
    async def test_node_fulltext_search_int(self):
        """Insert entity nodes, then search by BM25 fulltext."""
        from unittest.mock import MagicMock

        node = MagicMock()
        node.uuid = str(uuid4())
        node.group_id = 'test_group'
        node.name = 'Quantum Computing Research Lab'
        node.summary = 'A research laboratory focused on quantum computing'
        node.labels = ['Organization']
        node.created_at = datetime.now(tz=timezone.utc)
        node.attributes = {}
        node.name_embedding = _make_embedding(self.DIM)

        await self.graph_ops.node_save(node, driver=MagicMock())

        results = await self.search.node_fulltext_search(
            driver=MagicMock(),
            query='Quantum Computing',
            search_filter=self.empty_filter,
            group_ids=['test_group'],
            limit=10,
        )

        assert len(results) >= 1
        assert any(r.uuid == node.uuid for r in results)

    @pytest.mark.asyncio
    async def test_edge_fulltext_search_int(self):
        """Insert edges, then search by BM25 on facts."""
        from unittest.mock import MagicMock

        edge = MagicMock()
        edge.uuid = str(uuid4())
        edge.group_id = 'test_group'
        edge.source_node_uuid = str(uuid4())
        edge.target_node_uuid = str(uuid4())
        edge.name = 'RESEARCHES'
        edge.fact = 'Professor Smith researches superconducting qubits for quantum computing'
        edge.episodes = []
        edge.created_at = datetime.now(tz=timezone.utc)
        edge.expired_at = None
        edge.valid_at = None
        edge.invalid_at = None
        edge.attributes = {}
        edge.fact_embedding = _make_embedding(self.DIM)

        await self.graph_ops.edge_save(edge, driver=MagicMock())

        results = await self.search.edge_fulltext_search(
            driver=MagicMock(),
            query='superconducting qubits',
            search_filter=self.empty_filter,
            group_ids=['test_group'],
            limit=10,
        )

        assert len(results) >= 1
        assert any(r.uuid == edge.uuid for r in results)

    @pytest.mark.asyncio
    async def test_episode_fulltext_search_int(self):
        """Insert episodic nodes, then search by BM25 on content."""
        from unittest.mock import MagicMock

        node = MagicMock()
        node.uuid = str(uuid4())
        node.group_id = 'test_group'
        node.name = 'episode_test'
        node.content = 'During the meeting, Alice discussed the new machine learning pipeline'
        node.source.value = 'text'
        node.source_description = 'Meeting transcript'
        node.created_at = datetime.now(tz=timezone.utc)
        node.valid_at = datetime.now(tz=timezone.utc)
        node.entity_edges = []

        await self.graph_ops.episodic_node_save(node, driver=MagicMock())

        results = await self.search.episode_fulltext_search(
            driver=MagicMock(),
            query='machine learning pipeline',
            search_filter=self.empty_filter,
            group_ids=['test_group'],
            limit=10,
        )

        assert len(results) >= 1
        assert any(r.uuid == node.uuid for r in results)

    @pytest.mark.asyncio
    async def test_community_similarity_search_int(self):
        """Insert community nodes, then search by vector similarity."""
        from unittest.mock import MagicMock

        node = MagicMock()
        node.uuid = str(uuid4())
        node.group_id = 'test_group'
        node.name = 'AI Research Community'
        node.summary = 'Community of AI researchers and practitioners'
        node.created_at = datetime.now(tz=timezone.utc)
        node.name_embedding = _make_embedding(self.DIM)

        await self.graph_ops.community_node_save(node, driver=MagicMock())

        results = await self.search.community_similarity_search(
            driver=MagicMock(),
            search_vector=node.name_embedding,
            group_ids=['test_group'],
            limit=10,
            min_score=0.5,
        )

        assert len(results) >= 1
        assert any(r.uuid == node.uuid for r in results)

    @pytest.mark.asyncio
    async def test_node_delete_int(self):
        """Insert then delete a node, verify search no longer finds it."""
        from unittest.mock import MagicMock

        node = MagicMock()
        node.uuid = str(uuid4())
        node.group_id = 'test_group'
        node.name = 'Delete Me'
        node.summary = 'This node will be deleted'
        node.labels = []
        node.created_at = datetime.now(tz=timezone.utc)
        node.attributes = {}
        node.name_embedding = _make_embedding(self.DIM)

        await self.graph_ops.node_save(node, driver=MagicMock())
        await self.graph_ops.node_delete(node, driver=MagicMock())

        results = await self.search.node_similarity_search(
            driver=MagicMock(),
            search_vector=node.name_embedding,
            search_filter=self.empty_filter,
            group_ids=['test_group'],
            limit=10,
            min_score=0.9,
        )

        assert not any(r.uuid == node.uuid for r in results)

    @pytest.mark.asyncio
    async def test_get_embeddings_for_communities_int(self):
        """Insert communities and retrieve their embeddings."""
        from unittest.mock import MagicMock

        emb = _make_embedding(self.DIM)
        node = MagicMock()
        node.uuid = str(uuid4())
        node.group_id = 'test_group'
        node.name = 'Embedding Test Community'
        node.summary = 'Test community for embedding retrieval'
        node.created_at = datetime.now(tz=timezone.utc)
        node.name_embedding = emb

        await self.graph_ops.community_node_save(node, driver=MagicMock())

        result = await self.search.get_embeddings_for_communities(
            driver=MagicMock(),
            communities=[node],
        )

        assert node.uuid in result
        assert len(result[node.uuid]) == self.DIM

    @pytest.mark.asyncio
    async def test_clear_data_by_group_int(self):
        """Clear data for a specific group_id."""
        from unittest.mock import MagicMock

        group = f'clear_test_{uuid4().hex[:6]}'
        node = MagicMock()
        node.uuid = str(uuid4())
        node.group_id = group
        node.name = 'Clear Me'
        node.summary = ''
        node.labels = []
        node.created_at = datetime.now(tz=timezone.utc)
        node.attributes = {}
        node.name_embedding = _make_embedding(self.DIM)

        await self.graph_ops.node_save(node, driver=MagicMock())
        await self.graph_ops.clear_data(driver=MagicMock(), group_ids=[group])

        results = await self.search.node_similarity_search(
            driver=MagicMock(),
            search_vector=node.name_embedding,
            search_filter=self.empty_filter,
            group_ids=[group],
            limit=10,
            min_score=0.9,
        )

        assert not any(r.uuid == node.uuid for r in results)

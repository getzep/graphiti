"""Unit tests for graphiti_core.vector_store.milvus_search_interface."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.vector_store.milvus_search_interface import MilvusSearchInterface


@pytest.fixture
def mock_vs_client():
    """Create a mock VectorStoreClient."""
    client = MagicMock()
    client.collection_name = lambda suffix: f'test_{suffix}'
    client.search = AsyncMock(return_value=[[]])
    client.query = AsyncMock(return_value=[])
    return client


@pytest.fixture
def search_interface(mock_vs_client):
    """Create a MilvusSearchInterface with a mock VectorStoreClient."""
    return MilvusSearchInterface(vs_client=mock_vs_client)


def _make_search_filter(**kwargs):
    """Create a mock SearchFilters."""
    sf = MagicMock()
    sf.node_labels = kwargs.get('node_labels', [])
    sf.edge_types = kwargs.get('edge_types', [])
    sf.edge_uuids = kwargs.get('edge_uuids', [])
    sf.valid_at = kwargs.get('valid_at', [])
    sf.invalid_at = kwargs.get('invalid_at', [])
    sf.created_at = kwargs.get('created_at', [])
    sf.expired_at = kwargs.get('expired_at', [])
    sf.property_filters = kwargs.get('property_filters', [])
    return sf


def _make_node_hit(uuid='node-1', name='Alice', distance=0.9):
    """Create a mock Milvus search hit for entity nodes."""
    return {
        'distance': distance,
        'entity': {
            'uuid': uuid,
            'group_id': 'g1',
            'name': name,
            'summary': f'Summary of {name}',
            'labels': ['Person'],
            'created_at': 1705320000000,
            'attributes': {},
        },
    }


def _make_edge_hit(uuid='edge-1', name='WORKS_AT', fact='Alice works at Acme', distance=0.85):
    """Create a mock Milvus search hit for entity edges."""
    return {
        'distance': distance,
        'entity': {
            'uuid': uuid,
            'group_id': 'g1',
            'source_node_uuid': 'src-1',
            'target_node_uuid': 'tgt-1',
            'name': name,
            'fact': fact,
            'episodes': ['ep1'],
            'created_at': 1705320000000,
            'expired_at': 0,
            'valid_at': 1705320000000,
            'invalid_at': 0,
            'attributes': {},
        },
    }


def _make_episode_hit(uuid='ep-1', content='Alice met Bob'):
    """Create a mock Milvus search hit for episodic nodes."""
    return {
        'distance': 5.0,  # BM25 scores can be > 1
        'entity': {
            'uuid': uuid,
            'group_id': 'g1',
            'name': 'episode1',
            'content': content,
            'source': 'text',
            'source_description': 'conversation',
            'created_at': 1705320000000,
            'valid_at': 1705320000000,
            'entity_edges': ['edge1'],
        },
    }


def _make_community_hit(uuid='comm-1', name='Tech Community', distance=0.8):
    """Create a mock Milvus search hit for community nodes."""
    return {
        'distance': distance,
        'entity': {
            'uuid': uuid,
            'group_id': 'g1',
            'name': name,
            'summary': f'{name} summary',
            'created_at': 1705320000000,
        },
    }


# ---- Similarity Search ----


class TestNodeSimilaritySearch:
    @pytest.mark.asyncio
    async def test_returns_nodes_above_min_score(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [
            [_make_node_hit('n1', 'Alice', 0.9), _make_node_hit('n2', 'Bob', 0.5)]
        ]

        results = await search_interface.node_similarity_search(
            driver=MagicMock(),
            search_vector=[0.1] * 128,
            search_filter=_make_search_filter(),
            group_ids=['g1'],
            limit=10,
            min_score=0.7,
        )

        assert len(results) == 1
        assert results[0].uuid == 'n1'
        assert results[0].name == 'Alice'

    @pytest.mark.asyncio
    async def test_empty_vector_returns_empty(self, search_interface, mock_vs_client):
        results = await search_interface.node_similarity_search(
            driver=MagicMock(),
            search_vector=[],
            search_filter=_make_search_filter(),
        )
        assert results == []
        mock_vs_client.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_applies_group_id_filter(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [[]]

        await search_interface.node_similarity_search(
            driver=MagicMock(),
            search_vector=[0.1] * 128,
            search_filter=_make_search_filter(),
            group_ids=['g1', 'g2'],
        )

        call_kwargs = mock_vs_client.search.call_args.kwargs
        assert 'group_id in ["g1", "g2"]' in call_kwargs['filter_expr']


class TestEdgeSimilaritySearch:
    @pytest.mark.asyncio
    async def test_returns_edges_above_min_score(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [[_make_edge_hit('e1', distance=0.85)]]

        results = await search_interface.edge_similarity_search(
            driver=MagicMock(),
            search_vector=[0.1] * 128,
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=_make_search_filter(),
            min_score=0.7,
        )

        assert len(results) == 1
        assert results[0].uuid == 'e1'

    @pytest.mark.asyncio
    async def test_applies_source_target_filters(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [[]]

        await search_interface.edge_similarity_search(
            driver=MagicMock(),
            search_vector=[0.1] * 128,
            source_node_uuid='src-uuid',
            target_node_uuid='tgt-uuid',
            search_filter=_make_search_filter(),
        )

        call_kwargs = mock_vs_client.search.call_args.kwargs
        assert 'source_node_uuid == "src-uuid"' in call_kwargs['filter_expr']
        assert 'target_node_uuid == "tgt-uuid"' in call_kwargs['filter_expr']


class TestCommunitySimilaritySearch:
    @pytest.mark.asyncio
    async def test_returns_communities_above_min_score(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [
            [_make_community_hit('c1', distance=0.8), _make_community_hit('c2', distance=0.4)]
        ]

        results = await search_interface.community_similarity_search(
            driver=MagicMock(),
            search_vector=[0.1] * 128,
            min_score=0.6,
        )

        assert len(results) == 1
        assert results[0].uuid == 'c1'


# ---- Fulltext (BM25) Search ----


class TestEdgeFulltextSearch:
    @pytest.mark.asyncio
    async def test_returns_edges(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [[_make_edge_hit()]]

        results = await search_interface.edge_fulltext_search(
            driver=MagicMock(),
            query='Alice works',
            search_filter=_make_search_filter(),
            group_ids=['g1'],
        )

        assert len(results) == 1
        assert results[0].fact == 'Alice works at Acme'

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, search_interface, mock_vs_client):
        results = await search_interface.edge_fulltext_search(
            driver=MagicMock(),
            query='',
            search_filter=_make_search_filter(),
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_uses_bm25_params(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [[]]

        await search_interface.edge_fulltext_search(
            driver=MagicMock(),
            query='test',
            search_filter=_make_search_filter(),
        )

        call_kwargs = mock_vs_client.search.call_args.kwargs
        assert call_kwargs['anns_field'] == 'fact_sparse'
        assert call_kwargs['search_params'] == {'metric_type': 'BM25'}


class TestNodeFulltextSearch:
    @pytest.mark.asyncio
    async def test_returns_nodes(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [[_make_node_hit()]]

        results = await search_interface.node_fulltext_search(
            driver=MagicMock(),
            query='Alice',
            search_filter=_make_search_filter(),
        )

        assert len(results) == 1
        assert results[0].name == 'Alice'


class TestEpisodeFulltextSearch:
    @pytest.mark.asyncio
    async def test_returns_episodes(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [[_make_episode_hit()]]

        results = await search_interface.episode_fulltext_search(
            driver=MagicMock(),
            query='Alice met',
            search_filter=_make_search_filter(),
        )

        assert len(results) == 1
        assert results[0].content == 'Alice met Bob'


class TestCommunityFulltextSearch:
    @pytest.mark.asyncio
    async def test_returns_communities(self, search_interface, mock_vs_client):
        mock_vs_client.search.return_value = [[_make_community_hit()]]

        results = await search_interface.community_fulltext_search(
            driver=MagicMock(),
            query='Tech',
        )

        assert len(results) == 1
        assert results[0].name == 'Tech Community'


# ---- Embeddings ----


class TestGetEmbeddingsForCommunities:
    @pytest.mark.asyncio
    async def test_returns_embedding_dict(self, search_interface, mock_vs_client):
        mock_vs_client.query.return_value = [
            {'uuid': 'c1', 'name_embedding': [0.1, 0.2]},
            {'uuid': 'c2', 'name_embedding': [0.3, 0.4]},
        ]

        community1 = MagicMock()
        community1.uuid = 'c1'
        community2 = MagicMock()
        community2.uuid = 'c2'

        result = await search_interface.get_embeddings_for_communities(
            driver=MagicMock(),
            communities=[community1, community2],
        )

        assert result == {'c1': [0.1, 0.2], 'c2': [0.3, 0.4]}

    @pytest.mark.asyncio
    async def test_empty_communities_returns_empty(self, search_interface, mock_vs_client):
        result = await search_interface.get_embeddings_for_communities(
            driver=MagicMock(), communities=[]
        )
        assert result == {}


# ---- Search Filters ----


class TestSearchFilters:
    def test_build_node_search_filters(self, search_interface):
        sf = _make_search_filter(node_labels=['Person'])
        result = search_interface.build_node_search_filters(sf)
        assert 'json_contains(labels, "Person")' in result

    def test_build_edge_search_filters(self, search_interface):
        sf = _make_search_filter(edge_types=['WORKS_AT'])
        result = search_interface.build_edge_search_filters(sf)
        assert 'name in ["WORKS_AT"]' in result


# ---- NotImplementedError methods ----


class TestNotImplementedMethods:
    @pytest.mark.asyncio
    async def test_edge_bfs_search_raises(self, search_interface):
        with pytest.raises(NotImplementedError):
            await search_interface.edge_bfs_search(
                driver=MagicMock(),
                bfs_origin_node_uuids=['uuid1'],
                bfs_max_depth=2,
                search_filter=_make_search_filter(),
            )

    @pytest.mark.asyncio
    async def test_node_bfs_search_raises(self, search_interface):
        with pytest.raises(NotImplementedError):
            await search_interface.node_bfs_search(
                driver=MagicMock(),
                bfs_origin_node_uuids=['uuid1'],
                search_filter=_make_search_filter(),
                bfs_max_depth=2,
            )

    @pytest.mark.asyncio
    async def test_node_distance_reranker_raises(self, search_interface):
        with pytest.raises(NotImplementedError):
            await search_interface.node_distance_reranker(
                driver=MagicMock(),
                node_uuids=['uuid1'],
                center_node_uuid='center',
            )

    @pytest.mark.asyncio
    async def test_episode_mentions_reranker_raises(self, search_interface):
        with pytest.raises(NotImplementedError):
            await search_interface.episode_mentions_reranker(
                driver=MagicMock(),
                node_uuids=[['uuid1']],
            )

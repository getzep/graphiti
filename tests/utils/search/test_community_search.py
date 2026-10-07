from datetime import datetime

import pytest

from graphiti_core.nodes import CommunityNode
from graphiti_core.search import search as search_module
from graphiti_core.search.search_config import CommunitySearchConfig, CommunitySearchMethod


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('search_method', 'expected_call'),
    [
        (CommunitySearchMethod.bm25, 'bm25'),
        (CommunitySearchMethod.cosine_similarity, 'cosine_similarity'),
    ],
)
async def test_community_search_only_runs_configured_methods(
    monkeypatch, search_method, expected_call
):
    community = CommunityNode(
        name='test community',
        group_id='test-group',
        created_at=datetime.now(),
    )
    calls = []

    async def fulltext_search(*args, **kwargs):
        calls.append('bm25')
        return [community]

    async def similarity_search(*args, **kwargs):
        calls.append('cosine_similarity')
        return [community]

    monkeypatch.setattr(search_module, 'community_fulltext_search', fulltext_search)
    monkeypatch.setattr(search_module, 'community_similarity_search', similarity_search)

    results, scores = await search_module.community_search(
        driver=None,
        cross_encoder=None,
        query='test',
        query_vector=[0.0],
        group_ids=None,
        config=CommunitySearchConfig(search_methods=[search_method]),
    )

    assert calls == [expected_call]
    assert results == [community]
    assert len(scores) == 1

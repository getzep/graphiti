import pytest

from graphiti_core.cross_encoder.lexical_reranker_client import LexicalRerankerClient


@pytest.mark.asyncio
async def test_lexical_reranker_prefers_overlapping_passage():
    results = await LexicalRerankerClient().rank(
        'Neo4j 增量同步',
        ['今天北京天气晴朗', 'Neo4j 支持知识图谱增量同步'],
    )

    assert results[0][0] == 'Neo4j 支持知识图谱增量同步'
    assert results[0][1] > results[1][1]

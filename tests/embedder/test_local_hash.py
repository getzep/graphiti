import math

import pytest

from graphiti_core.embedder.local_hash import LocalHashEmbedder, LocalHashEmbedderConfig


@pytest.mark.asyncio
async def test_local_hash_embedder_is_deterministic_and_normalized():
    embedder = LocalHashEmbedder(LocalHashEmbedderConfig(embedding_dim=64))

    first = await embedder.create('Graphiti 支持增量知识图谱')
    second = await embedder.create('Graphiti 支持增量知识图谱')

    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


@pytest.mark.asyncio
async def test_local_hash_embedder_batch_matches_single_create():
    embedder = LocalHashEmbedder(LocalHashEmbedderConfig(embedding_dim=32))
    values = ['alpha beta', '飞书项目']

    batch = await embedder.create_batch(values)

    assert batch == [await embedder.create(value) for value in values]


@pytest.mark.parametrize(
    'kwargs',
    [
        {'embedding_dim': 0},
        {'min_ngram': 0},
        {'min_ngram': 4, 'max_ngram': 2},
    ],
)
def test_local_hash_embedder_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LocalHashEmbedderConfig(**kwargs)

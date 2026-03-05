from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import graphiti_mcp_server as srv


@pytest.mark.asyncio
async def test_graphiti_service_initialize_runs_neo4j_om_index_preflight(
    monkeypatch: pytest.MonkeyPatch,
    config,
):
    config.database.provider = 'neo4j'

    service = srv.GraphitiService(config, srv.SEMAPHORE_LIMIT)
    fake_client = SimpleNamespace(driver=SimpleNamespace())

    service._build_client = AsyncMock(return_value=fake_client)

    monkeypatch.setattr(srv.LLMClientFactory, 'create', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv.EmbedderFactory, 'create', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        srv.DatabaseDriverFactory,
        'create_config',
        lambda *_args, **_kwargs: {'uri': 'bolt://localhost:7687'},
    )

    verify_preflight = AsyncMock(return_value=None)
    monkeypatch.setattr(
        srv,
        'search_service',
        SimpleNamespace(
            neo4j_service=SimpleNamespace(
                verify_om_fulltext_index_shape=verify_preflight,
            )
        ),
    )

    await service.initialize()

    verify_preflight.assert_awaited_once_with(fake_client.driver)


@pytest.mark.asyncio
async def test_graphiti_service_initialize_surfaces_preflight_error(
    monkeypatch: pytest.MonkeyPatch,
    config,
):
    config.database.provider = 'neo4j'

    service = srv.GraphitiService(config, srv.SEMAPHORE_LIMIT)
    fake_client = SimpleNamespace(driver=SimpleNamespace())

    service._build_client = AsyncMock(return_value=fake_client)

    monkeypatch.setattr(srv.LLMClientFactory, 'create', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv.EmbedderFactory, 'create', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        srv.DatabaseDriverFactory,
        'create_config',
        lambda *_args, **_kwargs: {'uri': 'bolt://localhost:7687'},
    )

    verify_preflight = AsyncMock(
        side_effect=RuntimeError(
            'Neo4j OM full-text index preflight failed for "omnode_content_fulltext": index is missing.'
        )
    )
    monkeypatch.setattr(
        srv,
        'search_service',
        SimpleNamespace(
            neo4j_service=SimpleNamespace(
                verify_om_fulltext_index_shape=verify_preflight,
            )
        ),
    )

    with pytest.raises(RuntimeError, match='omnode_content_fulltext'):
        await service.initialize()


@pytest.mark.asyncio
async def test_graphiti_service_initialize_skips_neo4j_preflight_for_falkordb(
    monkeypatch: pytest.MonkeyPatch,
    config,
):
    config.database.provider = 'falkordb'

    service = srv.GraphitiService(config, srv.SEMAPHORE_LIMIT)
    fake_client = SimpleNamespace(driver=SimpleNamespace())

    service._build_client = AsyncMock(return_value=fake_client)

    monkeypatch.setattr(srv.LLMClientFactory, 'create', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(srv.EmbedderFactory, 'create', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        srv.DatabaseDriverFactory,
        'create_config',
        lambda *_args, **_kwargs: {'uri': 'redis://localhost:6379'},
    )

    verify_preflight = AsyncMock(return_value=None)
    monkeypatch.setattr(
        srv,
        'search_service',
        SimpleNamespace(
            neo4j_service=SimpleNamespace(
                verify_om_fulltext_index_shape=verify_preflight,
            )
        ),
    )

    await service.initialize()

    verify_preflight.assert_not_awaited()

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graphiti_core.nodes import EpisodicNode

import graphiti_mcp_server as server


class DummyGraphitiService:
    def __init__(self, client):
        self._client = client
        self.entity_types = None

    async def get_client(self):
        return self._client


@pytest.fixture
def patched_server(monkeypatch):
    monkeypatch.setattr(
        server,
        "config",
        SimpleNamespace(graphiti=SimpleNamespace(group_id="default-group")),
        raising=False,
    )


@pytest.mark.asyncio
async def test_search_nodes_accepts_scalar_group_id(monkeypatch, patched_server):
    client = SimpleNamespace(search_=AsyncMock(return_value=SimpleNamespace(nodes=[])))
    monkeypatch.setattr(server, "graphiti_service", DummyGraphitiService(client))

    result = await server.search_nodes(query="workspace memory", group_ids="ideadb")

    assert result["message"] == "No relevant nodes found"
    assert client.search_.await_args.kwargs["group_ids"] == ["ideadb"]


@pytest.mark.asyncio
async def test_search_memory_facts_accepts_scalar_group_id(monkeypatch, patched_server):
    client = SimpleNamespace(search=AsyncMock(return_value=[]))
    monkeypatch.setattr(server, "graphiti_service", DummyGraphitiService(client))

    result = await server.search_memory_facts(query="workspace memory", group_ids="ideadb")

    assert result["message"] == "No relevant facts found"
    assert client.search.await_args.kwargs["group_ids"] == ["ideadb"]


@pytest.mark.asyncio
async def test_get_episodes_accepts_scalar_group_id(monkeypatch, patched_server):
    get_by_group_ids = AsyncMock(return_value=[])
    monkeypatch.setattr(EpisodicNode, "get_by_group_ids", get_by_group_ids)
    monkeypatch.setattr(
        server, "graphiti_service", DummyGraphitiService(SimpleNamespace(driver=object()))
    )

    result = await server.get_episodes(group_ids="ideadb")

    assert result["message"] == "No episodes found"
    assert get_by_group_ids.await_args.args[1] == ["ideadb"]


@pytest.mark.asyncio
async def test_clear_graph_accepts_scalar_group_id(monkeypatch, patched_server):
    clear_data = AsyncMock(return_value=None)
    monkeypatch.setattr(server, "clear_data", clear_data)
    monkeypatch.setattr(
        server, "graphiti_service", DummyGraphitiService(SimpleNamespace(driver=object()))
    )

    result = await server.clear_graph(group_ids="ideadb")

    assert result["message"] == "Graph data cleared successfully for group IDs: ideadb"
    assert clear_data.await_args.kwargs["group_ids"] == ["ideadb"]

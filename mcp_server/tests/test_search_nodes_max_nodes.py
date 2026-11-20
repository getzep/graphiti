"""Regression: `search_nodes` must honour `max_nodes` above the recipe default.

The node-search recipe caps its own result list at `SearchConfig.limit`, which
defaults to 10. `search_nodes` sliced the result with `[:max_nodes]` but never
passed `max_nodes` into the recipe, so any `max_nodes` above 10 was silently
truncated to 10 (`search_memory_facts` has no such cap, because
`Graphiti.search` copies the limit into the recipe).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: E402
from graphiti_core.driver.driver import GraphDriver, GraphProvider  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.graphiti import Graphiti  # noqa: E402
from graphiti_core.graphiti_types import GraphitiClients  # noqa: E402
from graphiti_core.llm_client.client import LLMClient  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF  # noqa: E402
from graphiti_core.tracer import NoOpTracer  # noqa: E402

import graphiti_mcp_server as server  # noqa: E402

NODE_COUNT = 25


def _node_record(index: int) -> dict:
    return {
        'uuid': f'uuid-{index:03d}',
        'name': f'node {index:03d}',
        'group_id': 'g',
        'created_at': datetime(2024, 1, 1, tzinfo=timezone.utc),
        'summary': '',
        'labels': ['Entity'],
        'attributes': {},
    }


class _FakeDriver(GraphDriver):
    """Returns NODE_COUNT node records for every node search."""

    provider = GraphProvider.NEO4J
    fulltext_syntax = '@'
    default_group_id = ''
    graph_operations_interface = None
    search_interface = None

    async def execute_query(self, cypher_query_, **kwargs):
        return [_node_record(index) for index in range(NODE_COUNT)], None, None

    async def close(self):
        return None

    async def build_indices_and_constraints(self):
        return None

    async def delete_all_indexes(self):
        return None

    def session(self, database=None):
        raise NotImplementedError('sessions are not used by this test')


class _StubEmbedder(EmbedderClient):
    async def create(self, input_data):
        return [0.1, 0.2, 0.3]

    async def create_batch(self, input_data_list):
        return [[0.1, 0.2, 0.3] for _ in input_data_list]


class _StubCrossEncoder(CrossEncoderClient):
    async def rank(self, query, passages):
        return [(passage, 1.0) for passage in passages]


class _StubLLM(LLMClient):
    def __init__(self):
        super().__init__(LLMConfig())

    async def _generate_response(
        self, messages, response_model=None, max_tokens=0, model_size=None
    ):
        return {}


def _graphiti_client() -> Graphiti:
    driver = _FakeDriver()
    client = Graphiti.__new__(Graphiti)
    client.driver = driver
    client.llm_client = _StubLLM()
    client.embedder = _StubEmbedder()
    client.cross_encoder = _StubCrossEncoder()
    client.max_coroutines = 4
    client.tracer = NoOpTracer()
    client.clients = GraphitiClients(
        driver=driver,
        llm_client=client.llm_client,
        embedder=client.embedder,
        cross_encoder=client.cross_encoder,
        tracer=NoOpTracer(),
        max_coroutines=4,
    )
    return client


class _Service:
    def __init__(self, client: Graphiti) -> None:
        self._client = client

    async def get_client(self) -> Graphiti:
        return self._client


def _nodes_of(response) -> list:
    return response['nodes'] if isinstance(response, dict) else response.nodes


@pytest.fixture
def patched_server(monkeypatch):
    monkeypatch.setattr(server, 'graphiti_service', _Service(_graphiti_client()))
    # `config` is only annotated at module level; `main()` assigns it.
    monkeypatch.setattr(
        server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='g')),
        raising=False,
    )
    return server


@pytest.mark.asyncio
async def test_search_nodes_honours_max_nodes_above_recipe_default(patched_server):
    response = await patched_server.search_nodes(query='anything', group_ids=['g'], max_nodes=25)

    nodes = _nodes_of(response)
    assert len(nodes) == NODE_COUNT


@pytest.mark.asyncio
async def test_search_nodes_does_not_mutate_the_shared_recipe(patched_server):
    await patched_server.search_nodes(query='anything', group_ids=['g'], max_nodes=25)

    assert NODE_HYBRID_SEARCH_RRF.limit == 10


@pytest.mark.asyncio
async def test_search_nodes_still_caps_below_the_recipe_default(patched_server):
    response = await patched_server.search_nodes(query='anything', group_ids=['g'], max_nodes=3)

    assert len(_nodes_of(response)) == 3

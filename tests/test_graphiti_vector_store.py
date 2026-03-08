"""Unit tests for Graphiti vector_store constructor integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client.client import LLMClient


def _make_graphiti(*, vector_store=None):
    """Create a Graphiti instance with a mock driver and mock LLM clients.

    Uses spec= on mocks so they pass Pydantic isinstance validation
    in GraphitiClients, and avoids requiring OPENAI_API_KEY in CI.
    """
    mock_driver = MagicMock(spec=GraphDriver)
    mock_driver.close = AsyncMock()
    mock_driver.search_interface = None
    mock_driver.graph_operations_interface = None
    mock_driver.vector_store = None
    mock_driver.build_indices_and_constraints = AsyncMock()

    g = Graphiti(
        graph_driver=mock_driver,
        llm_client=MagicMock(spec=LLMClient),
        embedder=MagicMock(spec=EmbedderClient),
        cross_encoder=MagicMock(spec=CrossEncoderClient),
        vector_store=vector_store,
    )
    return g


class TestGraphitiVectorStoreInit:
    def test_vector_store_attached_to_driver(self):
        vs = MagicMock()
        g = _make_graphiti(vector_store=vs)

        assert g.driver.vector_store is vs

    def test_no_vector_store_leaves_driver_default(self):
        g = _make_graphiti(vector_store=None)

        assert g.driver.vector_store is None

    def test_generic_vector_store_does_not_attach_search(self):
        """A generic VectorStoreClient does not auto-attach a search interface."""
        vs = MagicMock()
        g = _make_graphiti(vector_store=vs)

        assert g.driver.vector_store is vs
        assert g.driver.search_interface is None


class TestGraphitiVectorStoreLifecycle:
    @pytest.mark.asyncio
    async def test_close_closes_vector_store(self):
        vs = AsyncMock()
        vs.close = AsyncMock()
        g = _make_graphiti(vector_store=vs)

        await g.close()

        g.driver.close.assert_called_once()
        vs.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_vector_store(self):
        g = _make_graphiti(vector_store=None)

        await g.close()

        g.driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_indices_initializes_vector_store(self):
        vs = AsyncMock()
        vs.ensure_ready = AsyncMock()
        g = _make_graphiti(vector_store=vs)

        await g.build_indices_and_constraints()

        g.driver.build_indices_and_constraints.assert_called_once()
        vs.ensure_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_indices_without_vector_store(self):
        g = _make_graphiti(vector_store=None)

        await g.build_indices_and_constraints()

        g.driver.build_indices_and_constraints.assert_called_once()

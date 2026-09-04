"""Group routing must not mutate the shared ``Graphiti`` client.

Providers that keep each ``group_id`` in its own database (FalkorDB) need a
write to run against that group's database. Doing that by assigning
``self.driver = self.driver.clone(database=group_id)`` reroutes the *shared*
client, so on a long-lived multi-group client -- the MCP server is exactly that
-- every later read runs inside whichever group wrote last, matches nothing, and
returns empty with no error.

These tests pin the side-effect surface: the routed driver is correct, and
``self.driver`` / ``self.clients.driver`` are the same objects afterwards. They
use ``spec=`` mocks, so they need no live database.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodeType, EpisodicNode, SagaNode

DEFAULT_DB = 'default_db'


def _make_graphiti():
    """A ``Graphiti`` whose driver is a FalkorDB-shaped mock rooted at ``default_db``."""
    child = MagicMock(spec=GraphDriver)
    child.provider = GraphProvider.FALKORDB
    child._database = 'group-a'

    base = MagicMock(spec=GraphDriver)
    base.provider = GraphProvider.FALKORDB
    base._database = DEFAULT_DB
    base.clone = MagicMock(return_value=child)

    with patch.object(Graphiti, '_capture_initialization_telemetry', lambda self: None):
        graphiti = Graphiti(
            graph_driver=base,
            llm_client=MagicMock(spec=LLMClient),
            embedder=MagicMock(spec=EmbedderClient),
            cross_encoder=MagicMock(spec=CrossEncoderClient),
        )
    return graphiti, base, child


def test_routing_to_another_group_clones_the_driver():
    graphiti, base, child = _make_graphiti()

    routed = graphiti._driver_for_group('group-a')

    base.clone.assert_called_once_with(database='group-a')
    assert routed is child


def test_routing_to_another_group_leaves_the_shared_client_untouched():
    """The regression: routing a write must not rebind the shared client."""
    graphiti, base, child = _make_graphiti()

    graphiti._driver_for_group('group-a')

    assert graphiti.driver is base
    assert graphiti.clients.driver is base
    assert graphiti.driver._database == DEFAULT_DB


def test_routing_to_the_current_group_does_not_clone():
    graphiti, base, _child = _make_graphiti()

    routed = graphiti._driver_for_group(DEFAULT_DB)

    assert routed is base
    base.clone.assert_not_called()


def test_clients_for_a_routed_driver_is_a_copy_carrying_that_driver():
    graphiti, base, child = _make_graphiti()

    routed_clients = graphiti._clients_for_driver(child)

    # The copy carries the routed driver...
    assert routed_clients.driver is child
    # ...every other client is shared, not rebuilt...
    assert routed_clients.llm_client is graphiti.clients.llm_client
    assert routed_clients.embedder is graphiti.clients.embedder
    assert routed_clients.cross_encoder is graphiti.clients.cross_encoder
    assert routed_clients.tracer is graphiti.clients.tracer
    # ...and the shared model is untouched.
    assert routed_clients is not graphiti.clients
    assert graphiti.clients.driver is base


def test_clients_for_the_base_driver_reuses_the_shared_model():
    graphiti, base, _child = _make_graphiti()

    assert graphiti._clients_for_driver(base) is graphiti.clients


@pytest.mark.parametrize('group_id', ['group-a', DEFAULT_DB])
def test_repeated_routing_never_drifts_the_shared_client(group_id):
    """Two groups written in sequence: the client still points at its own database."""
    graphiti, base, _child = _make_graphiti()

    for _ in range(3):
        graphiti._driver_for_group(group_id)
        graphiti._clients_for_driver(graphiti._driver_for_group(group_id))

    assert graphiti.driver is base
    assert graphiti.clients.driver is base
    assert graphiti.driver._database == DEFAULT_DB


@pytest.mark.asyncio
async def test_add_episode_does_not_rebind_the_shared_driver():
    """Drive the public API far enough to observe the routing side effect.

    ``add_episode`` resolves the group's driver and then opens a tracing span.
    Failing the span aborts the call right after routing, so whatever routing
    did to the shared client is all that is left to observe -- no LLM, embedder
    or database work needed. Before the fix this failed: the write path had
    already reassigned ``self.driver``/``self.clients.driver`` to the clone.
    """
    graphiti, base, _child = _make_graphiti()
    graphiti.tracer.start_span = MagicMock(side_effect=RuntimeError('stop here'))

    with pytest.raises(RuntimeError, match='stop here'):
        await graphiti.add_episode(
            name='ep',
            episode_body='body',
            source_description='desc',
            reference_time=datetime.now(timezone.utc),
            group_id='group-a',
        )

    assert graphiti.driver is base
    assert graphiti.clients.driver is base
    assert graphiti.driver._database == DEFAULT_DB


@pytest.mark.asyncio
async def test_add_episode_without_a_group_id_does_not_clone():
    """``group_id=None`` keeps the driver's preset database, exactly as before.

    ``get_default_group_id`` returns ``'_'`` on FalkorDB (``''`` elsewhere) --
    never the driver's own database name -- so routing on the resolved default
    would start cloning where the original code deliberately did not.
    """
    graphiti, base, _child = _make_graphiti()
    graphiti.tracer.start_span = MagicMock(side_effect=RuntimeError('stop here'))

    with pytest.raises(RuntimeError, match='stop here'):
        await graphiti.add_episode(
            name='ep',
            episode_body='body',
            source_description='desc',
            reference_time=datetime.now(timezone.utc),
        )

    base.clone.assert_not_called()
    assert graphiti.driver is base
    assert graphiti.clients.driver is base


# ---------------------------------------------------------------------------
# Routing must reach the *persistence* boundary, not just the early reads.
#
# The first version of this fix routed the reads in ``add_episode`` but still
# handed ``self.driver`` to ``add_nodes_and_edges_bulk`` and to every saga
# write, so one operation could dedupe in ``group-a`` and persist in the base
# database. These tests observe the driver each write actually receives.
# ---------------------------------------------------------------------------


def _episode(group_id: str = 'group-a') -> EpisodicNode:
    return EpisodicNode(
        name='ep',
        group_id=group_id,
        labels=[],
        source=EpisodeType.message,
        content='body',
        source_description='desc',
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_process_episode_data_persists_through_the_routed_driver():
    """The regression the review caught: the final save must use the routed driver."""
    graphiti, _base, child = _make_graphiti()

    with patch('graphiti_core.graphiti.add_nodes_and_edges_bulk', new_callable=AsyncMock) as save:
        await graphiti._process_episode_data(
            _episode(), [], [], datetime.now(timezone.utc), 'group-a', driver=child
        )

    save.assert_awaited_once()
    assert save.await_args.args[0] is child


@pytest.mark.asyncio
async def test_process_episode_data_defaults_to_the_shared_driver():
    """Control group: with no routing, the behaviour is exactly what it was."""
    graphiti, base, _child = _make_graphiti()

    with patch('graphiti_core.graphiti.add_nodes_and_edges_bulk', new_callable=AsyncMock) as save:
        await graphiti._process_episode_data(
            _episode(), [], [], datetime.now(timezone.utc), 'group-a'
        )

    assert save.await_args.args[0] is base


@pytest.mark.asyncio
async def test_process_episode_data_routes_every_saga_write():
    """Saga lookup, NEXT_EPISODE, HAS_EPISODE and the saga node itself."""
    graphiti, base, child = _make_graphiti()

    saga_node = MagicMock(spec=SagaNode)
    saga_node.uuid = 'saga-uuid'
    saga_node.first_episode_uuid = None
    saga_node.save = AsyncMock()

    edge = MagicMock()
    edge.save = AsyncMock()

    with (
        patch('graphiti_core.graphiti.add_nodes_and_edges_bulk', new_callable=AsyncMock),
        patch.object(
            Graphiti, '_get_or_create_saga', new_callable=AsyncMock, return_value=saga_node
        ) as get_saga,
        patch.object(
            Graphiti,
            '_saga_get_previous_episode_uuid',
            new_callable=AsyncMock,
            return_value='previous-uuid',
        ) as get_previous,
        patch('graphiti_core.graphiti.NextEpisodeEdge', return_value=edge),
        patch('graphiti_core.graphiti.HasEpisodeEdge', return_value=edge),
    ):
        await graphiti._process_episode_data(
            _episode(),
            [],
            [],
            datetime.now(timezone.utc),
            'group-a',
            saga='my-saga',
            driver=child,
        )

    assert get_saga.await_args.kwargs['driver'] is child
    assert get_previous.await_args.kwargs['driver'] is child
    # NEXT_EPISODE and HAS_EPISODE both save, and neither may touch the base graph.
    assert edge.save.await_count == 2
    assert all(call.args[0] is child for call in edge.save.await_args_list)
    assert saga_node.save.await_args.args[0] is child
    assert not base.execute_query.called


@pytest.mark.asyncio
async def test_extract_and_resolve_edges_uses_the_routed_clients():
    """Edge extraction and resolution read the graph; they must read the routed one."""
    graphiti, _base, child = _make_graphiti()
    routed_clients = graphiti._clients_for_driver(child)

    with (
        patch(
            'graphiti_core.graphiti.extract_edges', new_callable=AsyncMock, return_value=[]
        ) as extract,
        patch(
            'graphiti_core.graphiti.resolve_extracted_edges',
            new_callable=AsyncMock,
            return_value=([], [], []),
        ) as resolve,
    ):
        await graphiti._extract_and_resolve_edges(
            _episode(), [], [], {}, 'group-a', None, [], {}, clients=routed_clients
        )

    assert extract.await_args.args[0] is routed_clients
    assert resolve.await_args.args[0] is routed_clients
    assert graphiti.clients.driver is _base


@pytest.mark.asyncio
async def test_extract_and_dedupe_nodes_bulk_uses_the_routed_clients():
    graphiti, _base, child = _make_graphiti()
    routed_clients = graphiti._clients_for_driver(child)

    with (
        patch(
            'graphiti_core.graphiti.extract_nodes_and_edges_bulk',
            new_callable=AsyncMock,
            return_value=([], []),
        ) as extract,
        patch(
            'graphiti_core.graphiti.dedupe_nodes_bulk',
            new_callable=AsyncMock,
            return_value=({}, {}),
        ) as dedupe,
    ):
        await graphiti._extract_and_dedupe_nodes_bulk(
            [], {}, None, None, None, clients=routed_clients
        )

    assert extract.await_args.args[0] is routed_clients
    assert dedupe.await_args.args[0] is routed_clients

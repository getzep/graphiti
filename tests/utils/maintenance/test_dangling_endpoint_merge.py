from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now


def _make_graphiti() -> Graphiti:
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )
    return graphiti


def _make_episode() -> EpisodicNode:
    return EpisodicNode(
        uuid='ep_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='AMI is pinned at version 2.18.1.',
        valid_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_extract_and_resolve_edges_preserves_dedup_uuid_map(monkeypatch):
    """Materialized endpoint merge must not clobber dedup remaps on extracted_nodes."""
    graphiti = _make_graphiti()

    extracted_ami = EntityNode(
        uuid='extracted-ami',
        name='AMI',
        group_id='group_1',
        labels=['Entity'],
    )
    canonical_ami = EntityNode(
        uuid='canonical-ami',
        name='AMI',
        group_id='group_1',
        labels=['Entity'],
    )
    version = EntityNode(
        uuid='version-uuid',
        name='version 2.18.1',
        group_id='group_1',
        labels=['Entity'],
    )

    extracted_nodes = [extracted_ami]
    nodes = [canonical_ami]
    uuid_map = {'extracted-ami': 'canonical-ami'}

    async def fake_extract_edges(
        _clients,
        _episode,
        nodes_arg,
        _previous_episodes,
        _edge_type_map,
        group_id='',
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        nodes_arg.append(version)
        edge = EntityEdge(
            source_node_uuid='extracted-ami',
            target_node_uuid='version-uuid',
            name='PINNED_AT',
            group_id=group_id or 'group_1',
            fact='AMI is pinned at version 2.18.1.',
            episodes=['ep_uuid'],
            created_at=utc_now(),
        )
        return [edge], [version]

    async def fake_resolve_extracted_edges(
        _clients,
        edges,
        _episode,
        resolved_nodes,
        _edge_types,
        _edge_type_map,
    ):
        assert {n.uuid for n in resolved_nodes} == {'canonical-ami', 'version-uuid'}
        assert 'extracted-ami' not in {n.uuid for n in resolved_nodes}
        return edges, [], edges

    async def fake_resolve_extracted_nodes(
        _clients,
        extracted,
        _episode=None,
        _previous_episodes=None,
        _entity_types=None,
    ):
        return list(extracted), {node.uuid: node.uuid for node in extracted}, []

    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_edges',
        fake_extract_edges,
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_edges',
        fake_resolve_extracted_edges,
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_nodes',
        fake_resolve_extracted_nodes,
    )

    resolved, invalidated, new_edges = await graphiti._extract_and_resolve_edges(
        _make_episode(),
        extracted_nodes,
        [],
        {},
        'group_1',
        None,
        nodes,
        uuid_map,
    )

    assert uuid_map['extracted-ami'] == 'canonical-ami'
    assert uuid_map['version-uuid'] == 'version-uuid'
    assert {n.uuid for n in nodes} == {'canonical-ami', 'version-uuid'}
    assert len(resolved) == 1
    assert resolved[0].source_node_uuid == 'canonical-ami'
    assert resolved[0].target_node_uuid == 'version-uuid'
    assert invalidated == []
    assert new_edges == resolved


@pytest.mark.asyncio
async def test_materialized_endpoint_resolves_onto_existing_graph_node(monkeypatch):
    """A repeated value endpoint must dedup onto the canonical node already in the graph."""
    graphiti = _make_graphiti()

    ami = EntityNode(uuid='ami-uuid', name='AMI', group_id='group_1', labels=['Entity'])
    # Same name as the endpoint the LLM omitted from extracted entities on a prior episode.
    canonical_version = EntityNode(
        uuid='canonical-version',
        name='version 2.18.1',
        group_id='group_1',
        labels=['Entity'],
    )
    materialized_version = EntityNode(
        uuid='materialized-version',
        name='version 2.18.1',
        group_id='group_1',
        labels=['Entity'],
    )

    nodes = [ami]
    uuid_map = {'ami-uuid': 'ami-uuid'}
    resolve_calls: list[list[EntityNode]] = []

    async def fake_extract_edges(
        _clients,
        _episode,
        _nodes,
        _previous_episodes,
        _edge_type_map,
        group_id='',
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        edge = EntityEdge(
            source_node_uuid='ami-uuid',
            target_node_uuid='materialized-version',
            name='PINNED_AT',
            group_id=group_id or 'group_1',
            fact='AMI is pinned at version 2.18.1.',
            episodes=['ep_uuid'],
            created_at=utc_now(),
        )
        return [edge], [materialized_version]

    async def fake_resolve_extracted_nodes(
        _clients,
        extracted,
        _episode=None,
        _previous_episodes=None,
        _entity_types=None,
    ):
        resolve_calls.append(list(extracted))
        return (
            [canonical_version],
            {node.uuid: canonical_version.uuid for node in extracted},
            [],
        )

    async def fake_resolve_extracted_edges(
        _clients,
        edges,
        _episode,
        _resolved_nodes,
        _edge_types,
        _edge_type_map,
    ):
        return edges, [], edges

    monkeypatch.setattr('graphiti_core.graphiti.extract_edges', fake_extract_edges)
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_nodes', fake_resolve_extracted_nodes
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_edges', fake_resolve_extracted_edges
    )

    resolved, _invalidated, _new_edges = await graphiti._extract_and_resolve_edges(
        _make_episode(),
        [ami],
        [],
        {},
        'group_1',
        None,
        nodes,
        uuid_map,
    )

    assert [node.uuid for node in resolve_calls[0]] == ['materialized-version']
    assert uuid_map['materialized-version'] == 'canonical-version'
    assert {node.uuid for node in nodes} == {'ami-uuid', 'canonical-version'}
    assert resolved[0].target_node_uuid == 'canonical-version'


@pytest.mark.asyncio
async def test_materialized_endpoint_resolving_onto_source_drops_self_edge(monkeypatch):
    """Dedup collapsing both endpoints onto one node must not yield a self-edge."""
    graphiti = _make_graphiti()

    postgres = EntityNode(
        uuid='postgres-uuid', name='Postgres', group_id='group_1', labels=['Entity']
    )
    materialized = EntityNode(
        uuid='materialized-pg16', name='Postgres 16', group_id='group_1', labels=['Entity']
    )

    nodes = [postgres]
    uuid_map = {'postgres-uuid': 'postgres-uuid'}

    async def fake_extract_edges(
        _clients,
        _episode,
        _nodes,
        _previous_episodes,
        _edge_type_map,
        group_id='',
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        edge = EntityEdge(
            source_node_uuid='postgres-uuid',
            target_node_uuid='materialized-pg16',
            name='PINNED_AT',
            group_id=group_id or 'group_1',
            fact='Postgres is pinned at Postgres 16.',
            episodes=['ep_uuid'],
            created_at=utc_now(),
        )
        return [edge], [materialized]

    async def fake_resolve_extracted_nodes(
        _clients,
        extracted,
        _episode=None,
        _previous_episodes=None,
        _entity_types=None,
    ):
        return [postgres], {node.uuid: postgres.uuid for node in extracted}, []

    async def fake_resolve_extracted_edges(
        _clients,
        edges,
        _episode,
        _resolved_nodes,
        _edge_types,
        _edge_type_map,
    ):
        assert edges == []
        return edges, [], edges

    monkeypatch.setattr('graphiti_core.graphiti.extract_edges', fake_extract_edges)
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_nodes', fake_resolve_extracted_nodes
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_edges', fake_resolve_extracted_edges
    )

    resolved, _invalidated, _new_edges = await graphiti._extract_and_resolve_edges(
        _make_episode(),
        [postgres],
        [],
        {},
        'group_1',
        None,
        nodes,
        uuid_map,
    )

    assert resolved == []
    assert [node.uuid for node in nodes] == ['postgres-uuid']


@pytest.mark.asyncio
async def test_materialized_endpoint_reuses_same_name_node_from_this_episode(monkeypatch):
    """An endpoint matching a node already resolved for this episode must not be duplicated."""
    graphiti = _make_graphiti()

    ami = EntityNode(uuid='ami-uuid', name='AMI', group_id='group_1', labels=['Entity'])
    resolved_version = EntityNode(
        uuid='resolved-version',
        name='version 2.18.1',
        group_id='group_1',
        labels=['Entity'],
    )
    materialized_version = EntityNode(
        uuid='materialized-version',
        name='Version 2.18.1 ',
        group_id='group_1',
        labels=['Entity'],
    )

    nodes = [ami, resolved_version]
    uuid_map = {'ami-uuid': 'ami-uuid', 'resolved-version': 'resolved-version'}

    async def fake_extract_edges(
        _clients,
        _episode,
        _nodes,
        _previous_episodes,
        _edge_type_map,
        group_id='',
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        edge = EntityEdge(
            source_node_uuid='ami-uuid',
            target_node_uuid='materialized-version',
            name='PINNED_AT',
            group_id=group_id or 'group_1',
            fact='AMI is pinned at version 2.18.1.',
            episodes=['ep_uuid'],
            created_at=utc_now(),
        )
        return [edge], [materialized_version]

    async def fail_resolve_extracted_nodes(*_args, **_kwargs):
        raise AssertionError('endpoint already present in this episode should not be re-resolved')

    async def fake_resolve_extracted_edges(
        _clients,
        edges,
        _episode,
        _resolved_nodes,
        _edge_types,
        _edge_type_map,
    ):
        return edges, [], edges

    monkeypatch.setattr('graphiti_core.graphiti.extract_edges', fake_extract_edges)
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_nodes', fail_resolve_extracted_nodes
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_edges', fake_resolve_extracted_edges
    )

    resolved, _invalidated, _new_edges = await graphiti._extract_and_resolve_edges(
        _make_episode(),
        [ami],
        [],
        {},
        'group_1',
        None,
        nodes,
        uuid_map,
    )

    assert uuid_map['materialized-version'] == 'resolved-version'
    assert [node.uuid for node in nodes] == ['ami-uuid', 'resolved-version']
    assert resolved[0].target_node_uuid == 'resolved-version'

"""Unit tests for the episodic node save/return queries.

Neptune has no list properties, so `get_episode_node_save_query` and
`get_episode_node_save_bulk_query` store the `entity_edges` list of an episode as one
string joined with a separator. `EPISODIC_NODE_RETURN_NEPTUNE` rebuilds the list, so it
must split on the separator the save queries use: otherwise every episode read back from
Neptune carries the joined string as a single bogus edge uuid, and consumers such as
`Graphiti.remove_episode` (which loads `EntityEdge` records by `episode.entity_edges`)
find no edges at all.
"""

import re

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN_NEPTUNE,
    get_episode_node_save_bulk_query,
    get_episode_node_save_query,
)

_SAVE_JOIN = re.compile(r"entity_edges: join\(.*?, '(.*?)'\)", re.DOTALL)
_RETURN_SPLIT = re.compile(r'split\(e\.entity_edges, "(.*?)"\)')


def _save_separator(query: str) -> str:
    match = _SAVE_JOIN.search(query)
    assert match is not None, 'the Neptune save query must join entity_edges into a string'
    return match.group(1)


def _return_separator() -> str:
    match = _RETURN_SPLIT.search(EPISODIC_NODE_RETURN_NEPTUNE)
    assert match is not None, 'the Neptune return query must split entity_edges'
    return match.group(1)


@pytest.mark.parametrize(
    'save_query',
    [
        get_episode_node_save_query(GraphProvider.NEPTUNE),
        get_episode_node_save_bulk_query(GraphProvider.NEPTUNE),
    ],
    ids=['save', 'save_bulk'],
)
def test_neptune_return_query_splits_entity_edges_with_the_save_separator(save_query):
    assert _return_separator() == _save_separator(save_query)


def test_neptune_entity_edges_survive_a_save_and_load_round_trip():
    save_separator = _save_separator(get_episode_node_save_query(GraphProvider.NEPTUNE))
    load_separator = _return_separator()
    entity_edges = [
        '3f1a6c4e-0000-4000-8000-000000000001',
        '3f1a6c4e-0000-4000-8000-000000000002',
    ]

    stored = save_separator.join(entity_edges)  # what the save query writes to the graph
    assert stored.split(load_separator) == entity_edges


def test_neo4j_and_falkordb_store_entity_edges_as_a_list():
    for provider in (GraphProvider.NEO4J, GraphProvider.FALKORDB):
        assert _SAVE_JOIN.search(get_episode_node_save_query(provider)) is None

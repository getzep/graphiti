from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN_NEPTUNE,
    get_episode_node_save_bulk_query,
    get_episode_node_save_query,
)


def test_neptune_episodic_entity_edges_read_matches_write_delimiter():
    """On Neptune, EpisodicNode.entity_edges is serialized to a delimiter-joined
    string on save and split back into a list on read; the two delimiters must
    match, otherwise a multi-edge list round-trips into a single mashed element
    (e.g. ['uuid1|uuid2|uuid3']). The save side joins on '|', so the read query
    must split on '|', not ','."""
    save = get_episode_node_save_query(GraphProvider.NEPTUNE)
    bulk = get_episode_node_save_bulk_query(GraphProvider.NEPTUNE)

    # Save side joins entity_edges with '|'.
    assert "], '|')" in save
    assert "], '|')" in bulk

    # Read side must split on the same '|' delimiter, not ','.
    assert 'split(e.entity_edges, "|")' in EPISODIC_NODE_RETURN_NEPTUNE
    assert 'split(e.entity_edges, ",")' not in EPISODIC_NODE_RETURN_NEPTUNE

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query


@pytest.mark.parametrize('provider', list(GraphProvider))
def test_entity_edge_return_query_selects_reference_time(provider):
    """reference_time is saved by EntityEdge.save() on every provider and the
    record parser reads it from the top-level record. If the read query does not
    alias it, reference_time is silently dropped on every read-back, so the
    return query must select it alongside the other temporal fields."""
    query = get_entity_edge_return_query(provider)

    assert 'reference_time AS reference_time' in query
    # It should be returned just like its sibling temporal fields.
    for field in (
        'created_at AS created_at',
        'expired_at AS expired_at',
        'valid_at AS valid_at',
        'invalid_at AS invalid_at',
    ):
        assert field in query

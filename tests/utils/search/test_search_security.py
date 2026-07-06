from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import _build_falkor_fulltext_query
from graphiti_core.driver.neo4j.operations.search_ops import _build_neo4j_fulltext_query
from graphiti_core.errors import (
    GroupIdValidationError,
    NodeLabelValidationError,
    PropertyNameValidationError,
)
from graphiti_core.helpers import get_default_group_id, validate_group_id
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    PropertyFilter,
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)
from graphiti_core.search.search_utils import fulltext_query


def test_search_filters_reject_unsafe_node_labels():
    with pytest.raises(ValidationError, match='node_labels must start with a letter or underscore'):
        SearchFilters(node_labels=['Entity`) WITH n MATCH (x) DETACH DELETE x //'])


def test_node_search_filter_constructor_keeps_valid_label_expression():
    filters = SearchFilters(node_labels=['Person', 'Organization'])

    filter_queries, filter_params = node_search_filter_query_constructor(
        filters, GraphProvider.NEO4J
    )

    assert filter_queries == ['n:Person|Organization']
    assert filter_params == {}


def test_node_search_filter_constructor_rejects_unsafe_labels_bypassing_pydantic():
    filters = SearchFilters.model_construct(node_labels=['Entity`) DETACH DELETE x //'])

    with pytest.raises(
        NodeLabelValidationError, match='node_labels must start with a letter or underscore'
    ):
        node_search_filter_query_constructor(filters, GraphProvider.NEO4J)


def test_edge_search_filter_constructor_rejects_unsafe_labels_bypassing_pydantic():
    filters = SearchFilters.model_construct(node_labels=['Entity`) DETACH DELETE x //'])

    with pytest.raises(
        NodeLabelValidationError, match='node_labels must start with a letter or underscore'
    ):
        edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)


def test_node_search_filter_constructor_applies_property_filters():
    filters = SearchFilters(
        property_filters=[
            PropertyFilter(
                property_name='group_id',
                property_value='confidential-tenant',
                comparison_operator=ComparisonOperator.equals,
            )
        ]
    )

    filter_queries, filter_params = node_search_filter_query_constructor(
        filters, GraphProvider.NEO4J
    )

    assert filter_queries == ['n.group_id = $node_prop_0']
    assert filter_params == {'node_prop_0': 'confidential-tenant'}


def test_edge_search_filter_constructor_applies_property_filters():
    filters = SearchFilters(
        property_filters=[
            PropertyFilter(
                property_name='weight',
                property_value=5,
                comparison_operator=ComparisonOperator.greater_than,
            )
        ]
    )

    filter_queries, filter_params = edge_search_filter_query_constructor(
        filters, GraphProvider.NEO4J
    )

    assert 'e.weight > $edge_prop_0' in filter_queries
    assert filter_params['edge_prop_0'] == 5


def test_property_filter_null_operators_emit_no_params():
    filters = SearchFilters(
        property_filters=[
            PropertyFilter(
                property_name='deleted_at',
                comparison_operator=ComparisonOperator.is_null,
            )
        ]
    )

    filter_queries, filter_params = node_search_filter_query_constructor(
        filters, GraphProvider.NEO4J
    )

    assert filter_queries == ['n.deleted_at IS NULL']
    assert filter_params == {}


def test_search_filters_reject_unsafe_property_names():
    with pytest.raises(ValidationError):
        SearchFilters(
            property_filters=[
                PropertyFilter(
                    property_name='group_id`) DETACH DELETE n //',
                    property_value='x',
                    comparison_operator=ComparisonOperator.equals,
                )
            ]
        )


def test_property_filter_constructor_rejects_unsafe_names_bypassing_pydantic():
    filters = SearchFilters.model_construct(
        property_filters=[
            PropertyFilter.model_construct(
                property_name='group_id`) DETACH DELETE n //',
                property_value='x',
                comparison_operator=ComparisonOperator.equals,
            )
        ]
    )

    with pytest.raises(PropertyNameValidationError):
        node_search_filter_query_constructor(filters, GraphProvider.NEO4J)


def test_fulltext_query_rejects_invalid_group_ids():
    driver = SimpleNamespace(provider=GraphProvider.NEO4J, fulltext_syntax='')

    with pytest.raises(GroupIdValidationError, match='must contain only alphanumeric'):
        fulltext_query('test', ['bad"group'], driver)


def test_build_neo4j_fulltext_query_rejects_invalid_group_ids():
    with pytest.raises(GroupIdValidationError, match='must contain only alphanumeric'):
        _build_neo4j_fulltext_query('test', ['bad"group'])


def test_falkordb_fulltext_query_rejects_invalid_group_ids():
    # Import inside the test so collection still works when FalkorDB extras are unavailable.
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    driver = MagicMock(spec=FalkorDriver)
    driver.sanitize.return_value = 'test'

    with pytest.raises(GroupIdValidationError, match='must contain only alphanumeric'):
        FalkorDriver.build_fulltext_query(driver, 'test', ['bad"group'])


@pytest.mark.asyncio
async def test_shared_search_rejects_invalid_group_ids():
    clients = SimpleNamespace(
        driver=SimpleNamespace(),
        embedder=SimpleNamespace(),
        cross_encoder=SimpleNamespace(),
    )

    with pytest.raises(GroupIdValidationError, match='must contain only alphanumeric'):
        await search(
            clients,
            query='test',
            group_ids=['bad"group'],
            config=SearchConfig(),
            search_filter=SearchFilters(),
        )


def test_falkordb_default_group_id_passes_validation():
    # Regression: the FalkorDB default group_id must satisfy validate_group_id.
    # Otherwise add_episode/search with the default group_id raises
    # GroupIdValidationError once search re-validates the group_ids.
    default = get_default_group_id(GraphProvider.FALKORDB)
    assert validate_group_id(default) is True


def test_falkordb_fulltext_query_escapes_default_group_id():
    # The default group_id ('_') must be escaped for RediSearch (-> '\\_').
    # An unescaped '_' produces a RediSearch fulltext syntax error.
    default = get_default_group_id(GraphProvider.FALKORDB)
    built = _build_falkor_fulltext_query('hello', [default])
    assert '@group_id:"\\_"' in built
    assert '@group_id:"_"' not in built


def test_entity_node_rejects_unsafe_group_id_on_write():
    # Regression: the write path must reject group_ids that the read/delete path
    # would later refuse to match, otherwise records become unreachable.
    from graphiti_core.nodes import EntityNode

    with pytest.raises(ValidationError):
        EntityNode(name='test', group_id='bad"group')


def test_entity_node_accepts_valid_and_default_group_ids_on_write():
    from graphiti_core.nodes import EntityNode

    # Empty (Neo4j default) and safe values must still be accepted.
    EntityNode(name='test', group_id='')
    EntityNode(name='test', group_id='valid-group_1')


def test_entity_edge_rejects_unsafe_group_id_on_write():
    from datetime import datetime

    from graphiti_core.edges import EntityEdge

    with pytest.raises(ValidationError):
        EntityEdge(
            name='RELATES_TO',
            group_id='bad"group',
            source_node_uuid='source',
            target_node_uuid='target',
            fact='fact',
            created_at=datetime(2024, 1, 1),
        )

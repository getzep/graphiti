"""
Integration tests for Neptune Gremlin support.

These tests require a Neptune Database instance and OpenSearch cluster.
Set the following environment variables:
- NEPTUNE_HOST: Neptune endpoint (e.g., neptune-db://your-cluster.cluster-xxx.us-east-1.neptune.amazonaws.com)
- NEPTUNE_AOSS_HOST: OpenSearch endpoint
"""

import os
import pytest
from datetime import datetime

from graphiti_core.driver.driver import QueryLanguage
from graphiti_core.driver.neptune_driver import NeptuneDriver
from graphiti_core.graph_queries import (
    gremlin_delete_all_nodes,
    gremlin_match_node_by_property,
    gremlin_match_nodes_by_uuids,
)


@pytest.fixture
def neptune_host():
    """Get Neptune host from environment."""
    host = os.getenv('NEPTUNE_HOST')
    if not host:
        pytest.skip('NEPTUNE_HOST environment variable not set')
    return host


@pytest.fixture
def aoss_host():
    """Get AOSS host from environment."""
    host = os.getenv('NEPTUNE_AOSS_HOST')
    if not host:
        pytest.skip('NEPTUNE_AOSS_HOST environment variable not set')
    return host


@pytest.fixture
async def gremlin_driver(neptune_host, aoss_host):
    """Create a Neptune driver with Gremlin query language."""
    driver = NeptuneDriver(
        host=neptune_host,
        aoss_host=aoss_host,
        query_language=QueryLanguage.GREMLIN,
    )
    yield driver
    await driver.close()


@pytest.fixture
async def cypher_driver(neptune_host, aoss_host):
    """Create a Neptune driver with Cypher query language (for comparison)."""
    driver = NeptuneDriver(
        host=neptune_host,
        aoss_host=aoss_host,
        query_language=QueryLanguage.CYPHER,
    )
    yield driver
    await driver.close()


@pytest.mark.asyncio
async def test_gremlin_driver_initialization(neptune_host, aoss_host):
    """Test that Gremlin driver initializes correctly."""
    driver = NeptuneDriver(
        host=neptune_host,
        aoss_host=aoss_host,
        query_language=QueryLanguage.GREMLIN,
    )

    assert driver.query_language == QueryLanguage.GREMLIN
    assert hasattr(driver, 'gremlin_client')

    await driver.close()


@pytest.mark.asyncio
async def test_gremlin_analytics_raises_error(aoss_host):
    """Test that Gremlin with Neptune Analytics raises appropriate error."""
    with pytest.raises(ValueError, match='Neptune Analytics does not support Gremlin'):
        NeptuneDriver(
            host='neptune-graph://g-12345',
            aoss_host=aoss_host,
            query_language=QueryLanguage.GREMLIN,
        )


@pytest.mark.asyncio
async def test_gremlin_delete_all_nodes(gremlin_driver):
    """Test deleting all nodes with Gremlin."""
    # Clean up any existing data
    query = gremlin_delete_all_nodes()
    result, _, _ = await gremlin_driver.execute_query(query)

    # The result should be successful (no errors)
    assert result is not None


@pytest.mark.asyncio
async def test_gremlin_create_and_query_node(gremlin_driver):
    """Test creating and querying a node with Gremlin."""
    # Clean up first
    await gremlin_driver.execute_query(gremlin_delete_all_nodes())

    # Create a test node
    create_query = (
        "g.addV('Entity')"
        ".property('uuid', test_uuid)"
        ".property('name', test_name)"
        ".property('group_id', test_group)"
    )

    test_uuid = 'test-uuid-123'
    test_name = 'Test Entity'
    test_group = 'test-group'

    await gremlin_driver.execute_query(
        create_query,
        test_uuid=test_uuid,
        test_name=test_name,
        test_group=test_group,
    )

    # Query the node
    query = gremlin_match_node_by_property('Entity', 'uuid', 'test_uuid')
    query += '.valueMap(true)'

    result, _, _ = await gremlin_driver.execute_query(query, test_uuid=test_uuid)

    assert result is not None
    assert len(result) > 0


@pytest.mark.asyncio
async def test_cypher_vs_gremlin_compatibility(neptune_host, aoss_host):
    """Test that both Cypher and Gremlin can work with the same Neptune instance."""
    cypher_driver = NeptuneDriver(
        host=neptune_host,
        aoss_host=aoss_host,
        query_language=QueryLanguage.CYPHER,
    )

    gremlin_driver = NeptuneDriver(
        host=neptune_host,
        aoss_host=aoss_host,
        query_language=QueryLanguage.GREMLIN,
    )

    # Clean with Cypher
    await cypher_driver.execute_query('MATCH (n) DETACH DELETE n')

    # Verify empty with Gremlin
    result, _, _ = await gremlin_driver.execute_query('g.V().count()')
    assert result[0]['value'] == 0 or result[0] == 0

    await cypher_driver.close()
    await gremlin_driver.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

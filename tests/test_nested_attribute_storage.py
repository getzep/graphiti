"""Attribute values written to a graph store must be storable by that store.

Neo4j (like FalkorDB and Neptune) rejects a property whose value is a Map or a
list of Maps: ``Neo.ClientError.Statement.TypeError - Property values can only be
of primitive types or arrays thereof``. Kuzu already avoids this by serialising
``attributes`` to a JSON string, but the other providers spread each attribute
into its own property, so a nested value reaches the driver unchanged and the
whole episode fails.

Two distinct real-world inputs produce a nested value, and both must survive:

* A legitimately nested extraction — the model correctly returned an object.
* A schema echo — the model returned the JSON Schema field definition instead of
  a value. This is malformed, but it must not take down the ingestion pipeline.

These tests assert the storage invariant (no Map-valued property reaches the
driver), not any particular encoding, so an implementation may satisfy them by
serialising, flattening, or rejecting the value as long as the write is safe.
"""

from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.datetime_utils import utc_now

# A value the model extracted correctly; the source data really is nested.
LEGITIMATE_NESTED_ATTRIBUTE: dict[str, Any] = {
    'document': {
        'title': 'compose.yml.bak.qworkers.*',
        'description': 'Compose file backup used for rolling back the queue worker patch.',
    }
}

# A value where the model echoed the JSON Schema field definition back instead of
# filling it in. Malformed, but it must not crash the write path.
SCHEMA_ECHO_ATTRIBUTE: dict[str, Any] = {
    'description': {
        'description': 'Brief description of the object. Only use information mentioned in the context.',
        'title': 'Description',
        'type': 'string',
    }
}

NESTED_LIST_ATTRIBUTE: dict[str, Any] = {
    'contacts': [{'kind': 'email', 'value': 'a@example.com'}],
}

PROPERTY_SAFE_PROVIDERS = [
    GraphProvider.NEO4J,
    GraphProvider.FALKORDB,
    GraphProvider.NEPTUNE,
]


def _assert_property_values_are_storable(payload: dict[str, Any]) -> None:
    """Fail if any property value is a Map or an array containing a Map."""
    for key, value in payload.items():
        assert not isinstance(value, dict), (
            f'property {key!r} is a Map; the store accepts only primitives or arrays of them'
        )
        if isinstance(value, list):
            assert not any(isinstance(item, dict) for item in value), (
                f'property {key!r} is an array containing a Map'
            )


class _RecordingDriver:
    """Capture the parameters a save would send without touching a database."""

    def __init__(self, provider: GraphProvider) -> None:
        self.provider = provider
        self.graph_operations_interface = None
        self.captured: dict[str, Any] = {}

    async def execute_query(self, query: str, **kwargs: Any) -> Any:
        self.captured = kwargs
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', PROPERTY_SAFE_PROVIDERS)
@pytest.mark.parametrize(
    'attributes',
    [LEGITIMATE_NESTED_ATTRIBUTE, SCHEMA_ECHO_ATTRIBUTE, NESTED_LIST_ATTRIBUTE],
    ids=['legitimate_nested', 'schema_echo', 'nested_list'],
)
async def test_entity_node_save_never_sends_map_valued_property(provider, attributes):
    node = EntityNode(
        name='Alice',
        group_id='group',
        labels=['Entity'],
        summary='summary',
        attributes=dict(attributes),
    )
    node.name_embedding = [0.1, 0.2]
    driver = _RecordingDriver(provider)

    await node.save(driver)  # type: ignore[arg-type]

    entity_data = driver.captured.get('entity_data', driver.captured)
    _assert_property_values_are_storable(entity_data)


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', PROPERTY_SAFE_PROVIDERS)
@pytest.mark.parametrize(
    'attributes',
    [LEGITIMATE_NESTED_ATTRIBUTE, SCHEMA_ECHO_ATTRIBUTE, NESTED_LIST_ATTRIBUTE],
    ids=['legitimate_nested', 'schema_echo', 'nested_list'],
)
async def test_entity_edge_save_never_sends_map_valued_property(provider, attributes):
    edge = EntityEdge(
        source_node_uuid='source-uuid',
        target_node_uuid='target-uuid',
        name='RELATES_TO',
        group_id='group',
        fact='a fact',
        episodes=[],
        created_at=utc_now(),
        attributes=dict(attributes),
    )
    edge.fact_embedding = [0.1, 0.2]
    driver = _RecordingDriver(provider)

    await edge.save(driver)  # type: ignore[arg-type]

    edge_data = driver.captured.get('edge_data', driver.captured)
    _assert_property_values_are_storable(edge_data)

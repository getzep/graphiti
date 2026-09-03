"""Contract tests: our responses must parse in the real zep-cloud SDK.

These do not need a graph database, an LLM, or a network. They serialize the
zep_compat response models exactly as FastAPI would and feed the JSON through
zep-cloud's own `parse_obj_as`, which is the code path MiroFish executes.

The point is to catch the aliasing trap: zep-cloud declares
`uuid_: Annotated[str, FieldMetadata(alias='uuid')]`, so emitting `uuid_`
instead of `uuid` fails client-side with a ValidationError that would otherwise
only show up against real hardware.

Run:  uv run --extra dev pytest tests/test_zep_compat_contract.py
"""

from __future__ import annotations

import json

import pytest

zep_cloud = pytest.importorskip('zep_cloud', reason='zep-cloud is a dev-only dependency')
from zep_cloud.core.pydantic_utilities import parse_obj_as  # noqa: E402

from graph_service.zep_compat import models as m  # noqa: E402
from graph_service.zep_compat.ontology import (  # noqa: E402
    build_edge_types,
    build_entity_types,
    safe_field_name,
)


def roundtrip(model, zep_type):
    """Serialize like FastAPI does, then parse like the SDK does."""
    payload = json.loads(model.model_dump_json())
    return parse_obj_as(zep_type, payload)


def test_entity_node_roundtrip():
    node = m.EntityNode(
        uuid='n-1',
        name='Alice',
        summary='A person',
        created_at=m.now_iso(),
        labels=['Entity', 'Person'],
        attributes={'role': 'journalist'},
    )
    parsed = roundtrip(node, zep_cloud.EntityNode)
    assert parsed.uuid_ == 'n-1'
    assert parsed.name == 'Alice'
    assert parsed.labels == ['Entity', 'Person']
    assert parsed.attributes == {'role': 'journalist'}


def test_entity_edge_roundtrip_preserves_temporal_fields():
    edge = m.EntityEdge(
        uuid='e-1',
        name='WORKS_FOR',
        fact='Alice works for Acme',
        source_node_uuid='n-1',
        target_node_uuid='n-2',
        created_at=m.now_iso(),
        episodes=['ep-1'],
        valid_at=m.now_iso(),
        invalid_at=None,
        expired_at=None,
        attributes={'since': '2020'},
    )
    parsed = roundtrip(edge, zep_cloud.EntityEdge)
    assert parsed.uuid_ == 'e-1'
    assert parsed.source_node_uuid == 'n-1'
    assert parsed.valid_at is not None
    # MiroFish's zep_tools reads is_expired/is_invalid off these two.
    assert parsed.expired_at is None
    assert parsed.invalid_at is None


def test_graph_roundtrip():
    graph = m.Graph(
        uuid='g-uuid', graph_id='g-1', name='n', description='d', created_at=m.now_iso()
    )
    parsed = roundtrip(graph, zep_cloud.Graph)
    assert parsed.graph_id == 'g-1'
    assert parsed.uuid_ == 'g-uuid'


def test_episode_roundtrip():
    episode = m.Episode(
        uuid='ep-1',
        content='hello',
        created_at=m.now_iso(),
        source='text',
        source_description='chunk',
        processed=True,
    )
    parsed = roundtrip(episode, zep_cloud.Episode)
    assert parsed.uuid_ == 'ep-1'
    assert parsed.source == 'text'
    assert parsed.processed is True


def test_batch_summary_roundtrip_exposes_polled_fields():
    summary = m.BatchSummary(
        batch_id='b-1',
        status='processing',
        item_count=10,
        metadata={'mirofish_operation_id': 'op-1', 'graph_id': 'g-1'},
        progress=m.BatchProgress(
            total_items=10, succeeded_items=4, failed_items=0, percent_complete=40.0
        ),
        created_at=m.now_iso(),
        updated_at=m.now_iso(),
    )
    parsed = roundtrip(summary, zep_cloud.BatchSummary)
    assert parsed.batch_id == 'b-1'
    assert parsed.status == 'processing'
    # _wait_for_batch reads exactly these two off progress.
    assert parsed.progress is not None
    assert parsed.progress.percent_complete == 40.0
    assert parsed.progress.succeeded_items == 4
    # _find_batch_by_operation_id matches on these metadata keys.
    assert parsed.metadata['mirofish_operation_id'] == 'op-1'


def test_batch_item_detail_roundtrip_has_episode_uuid_and_index():
    item = m.BatchItemDetail(
        item_id='i-1',
        batch_id='b-1',
        status='pending',
        sequence_index=0,
        graph_id='g-1',
        episode_uuid='ep-1',
        created_at=m.now_iso(),
    )
    parsed = roundtrip(item, zep_cloud.BatchItemDetail)
    # add_text_batches collects episode_uuid, and reconciliation asserts on
    # sequence_index, so both must survive the wire.
    assert parsed.episode_uuid == 'ep-1'
    assert parsed.sequence_index == 0


def test_batch_list_response_roundtrip():
    payload = m.BatchListResponse(
        batches=[m.BatchSummary(batch_id='b-1', status='draft')], next_cursor=7
    )
    parsed = roundtrip(payload, zep_cloud.types.batch_list_response.BatchListResponse)
    assert parsed.batches is not None
    assert parsed.batches[0].batch_id == 'b-1'
    assert parsed.next_cursor == 7


def test_batch_item_list_response_roundtrip():
    payload = m.BatchItemListResponse(
        items=[m.BatchItemDetail(item_id='i-1', sequence_index=3, status='succeeded')],
        next_cursor=3,
    )
    parsed = roundtrip(payload, zep_cloud.BatchItemListResponse)
    assert parsed.items is not None
    assert parsed.items[0].sequence_index == 3
    assert parsed.next_cursor == 3


def test_graph_search_results_roundtrip():
    results = m.GraphSearchResults(
        nodes=[
            m.EntityNode(uuid='n-1', name='A', summary='s', created_at=m.now_iso())
        ],
        edges=[
            m.EntityEdge(
                uuid='e-1',
                name='R',
                fact='f',
                source_node_uuid='n-1',
                target_node_uuid='n-2',
                created_at=m.now_iso(),
            )
        ],
        episodes=[m.Episode(uuid='ep-1', content='c', created_at=m.now_iso())],
    )
    parsed = roundtrip(results, zep_cloud.GraphSearchResults)
    assert parsed.nodes and parsed.nodes[0].uuid_ == 'n-1'
    assert parsed.edges and parsed.edges[0].uuid_ == 'e-1'
    assert parsed.episodes and parsed.episodes[0].uuid_ == 'ep-1'


def test_success_response_roundtrip():
    parsed = roundtrip(m.SuccessResponse(message='done'), zep_cloud.SuccessResponse)
    assert parsed.message == 'done'


def test_emitted_json_uses_uuid_not_uuid_underscore():
    """The trap, asserted directly."""
    payload = json.loads(
        m.EntityNode(
            uuid='n-1', name='A', summary='s', created_at=m.now_iso()
        ).model_dump_json()
    )
    assert 'uuid' in payload
    assert 'uuid_' not in payload


# ---------------------------------------------------------------------------
# ontology translation
# ---------------------------------------------------------------------------


def test_ontology_request_parses_what_the_sdk_sends():
    """Shape produced by zep_cloud's set_ontology -> PUT entity-types."""
    body = {
        'entity_types': [
            {
                'name': 'Journalist',
                'description': 'A reporter',
                'properties': [
                    {'name': 'outlet', 'description': 'Employer', 'type': 'Text'},
                    {'name': 'years_active', 'description': 'Tenure', 'type': 'Int'},
                ],
            }
        ],
        'edge_types': [
            {
                'name': 'REPORTS_ON',
                'description': 'Covers a topic',
                'properties': [],
                'source_targets': [{'source': 'Journalist', 'target': 'Entity'}],
            }
        ],
        'graph_ids': ['g-1'],
        'user_ids': None,
    }
    request = m.SetEntityTypesRequest.model_validate(body)
    entity_types = build_entity_types([t.model_dump() for t in request.entity_types])
    edge_types, edge_map = build_edge_types([t.model_dump() for t in request.edge_types])

    assert set(entity_types) == {'Journalist'}
    fields = entity_types['Journalist'].model_fields
    assert set(fields) == {'outlet', 'years_active'}
    assert fields['years_active'].annotation == int | None
    # The docstring is fed to Graphiti's extraction prompt as the definition.
    assert entity_types['Journalist'].__doc__ == 'A reporter'

    assert set(edge_types) == {'REPORTS_ON'}
    assert edge_map == {('Journalist', 'Entity'): ['REPORTS_ON']}


@pytest.mark.parametrize(
    'reserved',
    ['uuid', 'name', 'group_id', 'labels', 'created_at', 'summary', 'attributes',
     'name_embedding'],
)
def test_reserved_attribute_names_are_renamed(reserved):
    """Graphiti raises EntityTypeValidationError when a custom attribute shadows
    an EntityNode field. MiroFish's own reserved list misses `labels` and
    `attributes`, so the shim must defend itself."""
    assert safe_field_name(reserved) == f'attr_{reserved}'


def test_colliding_normalized_names_do_not_overwrite_each_other():
    entity_types = build_entity_types(
        [
            {
                'name': 'T',
                'description': 'd',
                'properties': [
                    {'name': 'a-b', 'description': 'x', 'type': 'Text'},
                    {'name': 'a b', 'description': 'y', 'type': 'Text'},
                ],
            }
        ]
    )
    assert len(entity_types['T'].model_fields) == 2


def test_entity_label_is_not_redefined():
    """'Entity' is Graphiti's built-in default label."""
    assert build_entity_types([{'name': 'Entity', 'description': 'd'}]) == {}

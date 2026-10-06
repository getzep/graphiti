from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from graphiti_core.ingestion_trace import IngestionTraceRecord
from graphiti_core.utils.maintenance.ingestion_trace import (
    IdentifiedDecisionExplanation,
    _build_trace_messages,
    build_trace_model,
    merge_decision_explanations,
    trace_step_output,
)


class SampleOutput(BaseModel):
    value: str


class SampleEdge(BaseModel):
    uuid: str
    fact: str
    created_at: str
    episodes: list[str]
    summary: str = ''
    reference_time: str | None = None
    fact_embedding: list[float] | None = None


class NestedOutput(BaseModel):
    edges: list[SampleEdge]


class ExtractNodesOutput(BaseModel):
    extracted_nodes: list[SampleEdge]
    node_episode_index_map: dict[str, list[int]]


class ExtractEdgesOutput(BaseModel):
    extracted_edges: list[SampleEdge]


class CombinedExtractionTraceOutput(BaseModel):
    nodes: ExtractNodesOutput
    edges: ExtractEdgesOutput


class ResolveLikeOutput(BaseModel):
    nodes: list[SampleEdge]
    node_duplicates: list[tuple[SampleEdge, SampleEdge]]
    uuid_map: dict[str, str]


class TraceSink:
    def __init__(self):
        self.records: list[IngestionTraceRecord] = []

    async def write_trace(self, record: IngestionTraceRecord) -> None:
        self.records.append(record)


class ExplainingLLM:
    model = 'medium-model'
    small_model = 'small-model'

    async def generate_response(self, *args, **kwargs):
        return {
            'decision_explanations': {
                'value': 'The episode states the person is Alice, so the extractor kept that name.'
            }
        }


class FailingLLM:
    model = 'medium-model'
    small_model = 'small-model'

    def __init__(self):
        self.calls = 0

    async def generate_response(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError('llm trace failed')


class NestedExplainingLLM:
    model = 'medium-model'
    small_model = 'small-model'

    def __init__(self):
        self.prompts: list[str] = []

    async def generate_response(self, messages, *args, **kwargs):
        self.prompts.append('\n'.join(message.content for message in messages))
        return {
            'decision_explanations': {
                'edges': [
                    {
                        'uuid': 'edge-1',
                        'explanation': 'The current episode says Alice works at Zep.',
                    }
                ]
            }
        }


def _schema_node(schema: dict, node: dict) -> dict:
    ref = node.get('$ref')
    if not ref:
        return node
    name = ref.rsplit('/', 1)[-1]
    return schema.get('$defs', {}).get(name) or schema.get('definitions', {})[name]


def _make_edge(uuid: str, fact: str = 'Alice works at Zep') -> SampleEdge:
    return SampleEdge(
        uuid=uuid,
        fact=fact,
        created_at='2026-07-02T15:47:31.533304Z',
        episodes=['episode-1'],
        summary='SUMMARY_SENTINEL',
        reference_time='2023-05-20T02:21:00Z',
        fact_embedding=[0.1, 0.2],
    )


def test_merge_decision_explanations_matches_uuid_lists():
    original = {
        'nodes': [
            {'uuid': 'node-1', 'name': 'Alice'},
            {'uuid': 'node-2', 'name': 'Zep'},
        ]
    }
    explanations = {
        'nodes': [
            {'uuid': 'node-2', 'explanation': 'The episode names Zep.'},
            {'uuid': 'node-1', 'explanation': 'The episode names Alice.'},
        ]
    }

    merged = merge_decision_explanations(original, explanations)

    assert merged == {
        'nodes': [
            {
                'uuid': 'node-1',
                'name': 'Alice',
                'explanation': 'The episode names Alice.',
            },
            {
                'uuid': 'node-2',
                'name': 'Zep',
                'explanation': 'The episode names Zep.',
            },
        ]
    }
    assert original['nodes'][0] == {'uuid': 'node-1', 'name': 'Alice'}


def test_merge_decision_explanations_matches_map_keys():
    original = {'uuid_map': {'new-node': 'existing-node'}}
    explanations = {'uuid_map': {'new-node': 'The names and attributes match.'}}

    assert merge_decision_explanations(original, explanations) == {
        'uuid_map': {
            'new-node': {
                'value': 'existing-node',
                'explanation': 'The names and attributes match.',
            }
        }
    }


def test_merge_decision_explanations_keeps_values_without_explanations():
    original = {
        'nodes': [
            {'uuid': 'node-1', 'name': 'Alice'},
            {'uuid': 'node-2', 'name': 'Zep'},
        ],
        'status': 'complete',
    }
    explanations = {'nodes': [{'uuid': 'node-1', 'explanation': 'The episode names Alice.'}]}

    assert merge_decision_explanations(original, explanations) == {
        'nodes': [
            {
                'uuid': 'node-1',
                'name': 'Alice',
                'explanation': 'The episode names Alice.',
            },
            {'uuid': 'node-2', 'name': 'Zep'},
        ],
        'status': 'complete',
    }


def test_merge_decision_explanations_preserves_node_episode_indices_without_explanation():
    original = {'node_episode_index_map': {'node-1': [0]}}
    merged = merge_decision_explanations(original, {})

    assert merged == {'node_episode_index_map': {'node-1': [0]}}
    assert isinstance(merged['node_episode_index_map']['node-1'][0], int)


def test_merge_decision_explanations_handles_nested_extraction_output():
    original = {
        'nodes': [{'uuid': 'node-1', 'name': 'Alice', 'summary': 'A person'}],
        'edges': [
            {
                'uuid': 'edge-1',
                'fact': 'Alice works at Zep',
                'source_node_uuid': 'node-1',
                'target_node_uuid': 'node-2',
            }
        ],
    }
    explanations = {
        'nodes': [{'uuid': 'node-1', 'explanation': 'Alice is a named person.'}],
        'edges': [
            {
                'uuid': 'edge-1',
                'explanation': 'The episode states that Alice works at Zep.',
            }
        ],
    }

    merged = merge_decision_explanations(original, explanations)

    assert merged['nodes'][0] == {
        'uuid': 'node-1',
        'name': 'Alice',
        'summary': 'A person',
        'explanation': 'Alice is a named person.',
    }
    assert merged['edges'][0] == {
        'uuid': 'edge-1',
        'fact': 'Alice works at Zep',
        'source_node_uuid': 'node-1',
        'target_node_uuid': 'node-2',
        'explanation': 'The episode states that Alice works at Zep.',
    }
    assert merged['nodes'][0]['summary'] == 'A person'


def test_merge_decision_explanations_handles_resolve_nodes_output():
    original = {
        'nodes': [
            {'uuid': 'new-node', 'name': 'Alice'},
            {'uuid': 'existing-node', 'name': 'Alice Smith'},
        ],
        'uuid_map': {'new-node': 'existing-node'},
        'node_duplicates': [
            [
                {'uuid': 'new-node', 'name': 'Alice'},
                {'uuid': 'existing-node', 'name': 'Alice Smith'},
            ]
        ],
    }
    explanations = {
        'nodes': [
            {'uuid': 'new-node', 'explanation': 'The extracted node is retained.'},
            {'uuid': 'existing-node', 'explanation': 'The existing node is reused.'},
        ],
        'uuid_map': {'new-node': 'The nodes refer to the same person.'},
        'node_duplicates': [
            [
                {'uuid': 'new-node', 'explanation': 'This is the extracted node.'},
                {'uuid': 'existing-node', 'explanation': 'This is the matching node.'},
            ]
        ],
    }

    merged = merge_decision_explanations(original, explanations)

    assert merged['nodes'][0]['explanation'] == 'The extracted node is retained.'
    assert merged['uuid_map']['new-node'] == {
        'value': 'existing-node',
        'explanation': 'The nodes refer to the same person.',
    }
    assert merged['node_duplicates'][0][0]['name'] == 'Alice'
    assert merged['node_duplicates'][0][0]['explanation'] == 'This is the extracted node.'
    assert merged['node_duplicates'][0][1]['explanation'] == 'This is the matching node.'


def test_trace_model_uses_container_keys_and_identified_item_schema():
    output = NestedOutput(edges=[_make_edge('edge-1')])
    trace_model = build_trace_model(output, 'combined_extraction')
    schema = trace_model.model_json_schema()
    explanations = _schema_node(schema, schema['properties']['decision_explanations'])
    assert set(explanations['properties']) == {'edges'}
    assert 'required' in explanations and 'edges' in explanations['required']

    edges = explanations['properties']['edges']
    assert edges.get('minItems') == 1
    assert edges.get('maxItems') == 1
    item = _schema_node(schema, edges['items'])
    assert set(item['properties']) == {'uuid', 'explanation'}
    assert 'summary' not in item['properties']
    assert 'fact' not in item['properties']


def test_combined_extraction_trace_model_skips_node_episode_index_map():
    output = CombinedExtractionTraceOutput(
        nodes=ExtractNodesOutput(
            extracted_nodes=[_make_edge('node-1')],
            node_episode_index_map={'node-1': [0]},
        ),
        edges=ExtractEdgesOutput(extracted_edges=[_make_edge('edge-1')]),
    )

    schema = build_trace_model(output, 'combined_extraction').model_json_schema()
    explanations = _schema_node(schema, schema['properties']['decision_explanations'])
    node_explanations = _schema_node(schema, explanations['properties']['nodes'])

    assert set(node_explanations['properties']) == {'extracted_nodes'}
    extracted_nodes = node_explanations['properties']['extracted_nodes']
    item = _schema_node(schema, extracted_nodes['items'])
    assert set(item['properties']) == {'uuid', 'explanation'}


def test_trace_model_rejects_summary_named_explanation_fields():
    output = NestedOutput(edges=[_make_edge('edge-1')])
    trace_model = build_trace_model(output, 'combined_extraction')
    with pytest.raises(ValidationError):
        trace_model.model_validate(
            {'decision_explanations': {'edges': [{'uuid': 'edge-1', 'summary': 'why it was kept'}]}}
        )


def test_trace_model_uses_uuid_map_keys_from_the_output_instance():
    output = ResolveLikeOutput(
        nodes=[_make_edge('n-new-notion'), _make_edge('n-existing-notion')],
        node_duplicates=[(_make_edge('n-new-notion'), _make_edge('n-existing-notion'))],
        uuid_map={'n-new-notion': 'n-existing-notion'},
    )
    trace_model = build_trace_model(output, 'resolve_nodes', output_model=ResolveLikeOutput)
    schema = trace_model.model_json_schema()
    explanations = _schema_node(schema, schema['properties']['decision_explanations'])
    assert set(explanations['properties']) == {'nodes', 'node_duplicates', 'uuid_map'}

    uuid_map = _schema_node(schema, explanations['properties']['uuid_map'])
    assert set(uuid_map['properties']) == {'n-new-notion'}
    assert uuid_map['properties']['n-new-notion']['type'] == 'string'


def test_trace_model_rejects_domain_fields_on_identified_objects():
    output = NestedOutput(edges=[_make_edge('edge-1')])
    trace_model = build_trace_model(output, 'combined_extraction')
    with pytest.raises(ValidationError):
        trace_model.model_validate(
            {
                'decision_explanations': {
                    'edges': [
                        {
                            'uuid': 'edge-1',
                            'explanation': 'kept',
                            'summary': 'should not be allowed',
                        }
                    ]
                }
            }
        )


@pytest.mark.parametrize(
    ('stage', 'required_text'),
    [
        ('combined_extraction', ('episode span', 'short quote', 'produced the node or edge')),
        ('resolve_nodes', ('candidate entities', 'new-or-merge decision', 'episode evidence')),
        (
            'resolve_edges',
            ('candidate edges', 'invalidate, or expire decision', 'episode evidence'),
        ),
    ],
)
def test_trace_prompt_requires_stage_specific_grounding(stage: str, required_text: tuple[str, ...]):
    messages = _build_trace_messages(
        stage=stage,
        prompt_name='stage.prompt',
        input_context={'episode_content': 'Alice works at Zep.', 'dedupe_candidates': []},
        original_output={'value': 'kept'},
    )
    prompt = '\n'.join(message.content for message in messages)

    for text in required_text:
        assert text in prompt

    schema = build_trace_model(SampleOutput(value='kept'), stage).model_json_schema()
    description = schema['properties']['decision_explanations']['description']
    for text in required_text:
        assert text in description


@pytest.mark.asyncio
async def test_trace_step_output_records_json_explanations_without_changing_output():
    trace_sink = TraceSink()
    clients = SimpleNamespace(llm_client=ExplainingLLM(), ingestion_trace=trace_sink)
    original = SampleOutput(value='Alice')

    result = await trace_step_output(
        clients,
        output=original,
        output_model=SampleOutput,
        stage='extract_nodes',
        prompt_name='extract_nodes.extract_text',
        input_context={
            'episode_content': 'Alice works at Zep.',
            'episodes': [{'uuid': 'episode-1', 'content': 'Alice works at Zep.'}],
        },
        episode_uuids=['episode-1'],
        account_uuid='account-1',
        project_uuid='project-1',
        graph_uuid='graph-1',
        workflow_run_id='workflow-1',
        retry_count=3,
    )

    assert result is original
    assert result == SampleOutput(value='Alice')
    assert len(trace_sink.records) == 1
    record = trace_sink.records[0]
    assert record.validation_status == 'recorded'
    assert record.original_output == {
        'value': {
            'value': 'Alice',
            'explanation': 'The episode states the person is Alice, so the extractor kept that name.',
        }
    }
    assert record.decision_explanations == {
        'value': 'The episode states the person is Alice, so the extractor kept that name.'
    }
    assert record.input_context['episode_content'] == 'Alice works at Zep.'
    assert record.episode_uuids == ['episode-1']
    assert record.created_at <= datetime.now(UTC)


@pytest.mark.asyncio
async def test_trace_step_output_keeps_summaries_but_omits_embeddings():
    trace_sink = TraceSink()
    llm = NestedExplainingLLM()
    clients = SimpleNamespace(llm_client=llm, ingestion_trace=trace_sink)
    original = NestedOutput(edges=[_make_edge('edge-1')])

    result = await trace_step_output(
        clients,
        output=original,
        output_model=NestedOutput,
        stage='combined_extraction',
        prompt_name='extract_nodes_and_edges.extract_message',
        input_context={
            'episode_content': 'Alice works at Zep',
            'dedupe_candidates': [
                {
                    'uuid': 'edge-2',
                    'summary': 'SUMMARY_SENTINEL',
                    'fact_embedding': [9.9],
                }
            ],
        },
        episode_uuids=['episode-1'],
        workflow_run_id='workflow-1',
    )

    assert result is original
    assert original.edges[0].fact_embedding == [0.1, 0.2]
    assert original.edges[0].summary == 'SUMMARY_SENTINEL'

    prompt = llm.prompts[0]
    assert 'SUMMARY_SENTINEL' in prompt
    assert 'fact_embedding' not in prompt

    stored_edge = trace_sink.records[0].original_output['edges'][0]
    assert 'fact_embedding' not in stored_edge
    assert stored_edge['summary'] == 'SUMMARY_SENTINEL'
    assert stored_edge['fact'] == 'Alice works at Zep'
    assert stored_edge['explanation'] == 'The current episode says Alice works at Zep.'
    stored_candidate = trace_sink.records[0].input_context['dedupe_candidates'][0]
    assert 'fact_embedding' not in stored_candidate
    assert stored_candidate['summary'] == 'SUMMARY_SENTINEL'
    assert trace_sink.records[0].decision_explanations == {
        'edges': [
            {
                'uuid': 'edge-1',
                'explanation': 'The current episode says Alice works at Zep.',
            }
        ]
    }


@pytest.mark.asyncio
async def test_trace_step_output_falls_back_after_retries_and_traces_failure():
    trace_sink = TraceSink()
    llm = FailingLLM()
    clients = SimpleNamespace(llm_client=llm, ingestion_trace=trace_sink)
    original = SampleOutput(value='primary')

    result = await trace_step_output(
        clients,
        output=original,
        output_model=SampleOutput,
        stage='resolve_edges',
        prompt_name='dedupe_edges.resolve_edge',
        input_context={'episode_content': 'primary', 'dedupe_candidates': []},
        episode_uuids=['episode-1'],
        workflow_run_id='workflow-1',
        retry_count=3,
    )

    assert result is original
    assert llm.calls == 4
    assert len(trace_sink.records) == 1
    assert trace_sink.records[0].validation_status == 'failed'
    assert trace_sink.records[0].attempt_count == 4
    assert trace_sink.records[0].decision_explanations == {}
    assert trace_sink.records[0].original_output == {'value': 'primary'}
    assert trace_sink.records[0].input_context['episode_content'] == 'primary'
    assert trace_sink.records[0].error == 'llm trace failed'


def test_identified_decision_explanation_schema_keys():
    schema = IdentifiedDecisionExplanation.model_json_schema()
    assert set(schema['properties']) == {'uuid', 'explanation'}

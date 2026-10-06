from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, create_model

from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.classify_entity_subtype import EntitySubtypeClassification
from graphiti_core.utils.maintenance.dedup_helpers import _promote_resolved_node
from graphiti_core.utils.maintenance.entity_type_hierarchy import (
    MAX_ENTITY_TYPE_SUBTYPE_DEPTH,
    PARENT_ENTITY_TYPE_ATTR,
    build_entity_type_hierarchy,
    classify_entity_subtypes,
    entity_type_depth,
    most_specific_entity_type_name,
    parent_entity_type,
    top_level_entity_types,
)
from graphiti_core.utils.maintenance.node_operations import (
    _collapse_exact_duplicate_extracted_nodes,
    _entity_type_for_node,
    _get_entity_type_description,
)


def _entity_type(name: str, parent: str | None = None, description: str | None = None):
    model = create_model(name)
    setattr(model, PARENT_ENTITY_TYPE_ATTR, parent)
    model.__doc__ = description
    return model


def _episode(content: str = 'Animal context') -> EpisodicNode:
    return EpisodicNode(
        name='episode',
        group_id='graph',
        source=EpisodeType.text,
        source_description='test',
        content=content,
        valid_at=datetime.now(UTC),
    )


def _node(labels: list[str] | None = None) -> EntityNode:
    return EntityNode(name='Rex', group_id='graph', labels=labels or ['Entity', 'Animal'])


def _types() -> dict[str, type[BaseModel]]:
    return {
        'Animal': _entity_type('Animal', description='An animal.'),
        'Mammal': _entity_type('Mammal', 'Animal', 'A mammal.'),
        'Canine': _entity_type('Canine', 'Mammal', 'A canine.'),
        'Feline': _entity_type('Feline', 'Mammal', 'A feline.'),
        'Plant': _entity_type('Plant', description='A plant.'),
    }


def test_parent_depth_and_most_specific_type_selection():
    entity_types = _types()
    assert parent_entity_type(entity_types['Mammal']) == 'Animal'
    assert parent_entity_type(None) is None
    assert entity_type_depth('Canine', entity_types) == 2
    assert most_specific_entity_type_name(
        ['Entity', 'Animal', 'Canine', 'Mammal'], entity_types
    ) == ('Canine')
    assert most_specific_entity_type_name(['Entity', 'Feline', 'Canine'], entity_types) == 'Feline'
    assert most_specific_entity_type_name(['Entity', 'First', 'Second'], None) == 'First'
    assert most_specific_entity_type_name(['Entity'], entity_types) == ''


def test_entity_type_depth_stops_for_missing_types_and_cycles():
    entity_types = {
        'A': _entity_type('A', 'B'),
        'B': _entity_type('B', 'A'),
        'Orphan': _entity_type('Orphan', 'Missing'),
    }
    assert entity_type_depth('A', entity_types) == 1
    assert entity_type_depth('Orphan', entity_types) == 0


def test_build_hierarchy_and_filter_top_level_types():
    entity_types = _types()
    assert build_entity_type_hierarchy(entity_types) == {
        'Animal': ['Mammal'],
        'Mammal': ['Canine', 'Feline'],
    }
    assert list(top_level_entity_types(entity_types) or {}) == ['Animal', 'Plant']
    assert top_level_entity_types(None) is None


def test_node_operations_use_the_most_specific_entity_type():
    entity_types = _types()
    node = _node(['Entity', 'Animal', 'Canine'])

    assert _entity_type_for_node(node, entity_types) is entity_types['Canine']
    assert _get_entity_type_description(node.labels, entity_types) == 'A canine.'


@pytest.mark.asyncio
async def test_classify_entity_subtypes_walks_to_leaf_and_passes_prompt_name():
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        side_effect=[{'subtype_index': 1}, {'subtype_index': 2}]
    )
    node = _node()
    episodes = [_episode('Rex barked.'), _episode('The cat slept.')]

    await classify_entity_subtypes(
        llm_client,
        [node],
        episodes,
        {node.uuid: [0]},
        _types(),
        custom_extraction_instructions='Prefer the clearest fit.',
    )

    assert node.labels == ['Entity', 'Animal', 'Mammal', 'Feline']
    assert llm_client.generate_response.await_count == 2
    first_call = llm_client.generate_response.await_args_list[0]
    assert first_call.kwargs['prompt_name'] == 'classify_entity_subtype'
    assert first_call.kwargs['model_size'].value == 'small'
    assert first_call.kwargs['group_id'] == node.group_id
    assert first_call.kwargs['response_model'] is EntitySubtypeClassification
    assert isinstance(first_call.args[0], list)
    assert 'Rex barked.' in first_call.args[0][1].content
    assert 'The cat slept.' not in first_call.args[0][1].content
    assert 'Prefer the clearest fit.' in first_call.args[0][1].content
    assert 'A mammal.' in first_call.args[0][1].content
    assert '1. Mammal: A mammal.' in first_call.args[0][1].content
    second_call = llm_client.generate_response.await_args_list[1]
    assert '2. Feline: A feline.' in second_call.args[0][1].content


@pytest.mark.parametrize(
    ('result', 'expected_calls'),
    [
        ({'subtype_index': 0}, 1),
        ({'subtype_index': 3}, 1),
    ],
)
@pytest.mark.asyncio
async def test_classify_entity_subtypes_stops_when_no_candidate_fits(result, expected_calls):
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value=result)
    node = _node()

    await classify_entity_subtypes(llm_client, [node], [_episode()], {}, _types())

    assert node.labels == ['Entity', 'Animal']
    assert llm_client.generate_response.await_count == expected_calls


@pytest.mark.asyncio
async def test_classify_entity_subtypes_uses_all_episodes_when_attribution_is_missing():
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'subtype_index': 0})
    node = _node()
    episodes = [_episode('Episode one.'), _episode('Episode two.')]

    await classify_entity_subtypes(llm_client, [node], episodes, {}, _types())

    prompt = llm_client.generate_response.await_args.args[0]
    assert 'Episode one.' in prompt[1].content
    assert 'Episode two.' in prompt[1].content


@pytest.mark.asyncio
async def test_classify_entity_subtypes_stops_without_raising_on_exception():
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=RuntimeError('failed'))
    node = _node()

    await classify_entity_subtypes(llm_client, [node], [_episode()], {}, _types())

    assert node.labels == ['Entity', 'Animal']
    llm_client.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_entity_subtypes_skips_types_without_children():
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock()
    node = _node(['Entity', 'Canine'])

    await classify_entity_subtypes(llm_client, [node], [_episode()], {}, _types())

    llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_classify_entity_subtypes_caps_calls_at_four():
    entity_types = {
        name: _entity_type(name, parent)
        for name, parent in [
            ('Root', None),
            ('Level1', 'Root'),
            ('Level2', 'Level1'),
            ('Level3', 'Level2'),
            ('Level4', 'Level3'),
            ('Level5', 'Level4'),
        ]
    }
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'subtype_index': 1})
    node = _node(['Entity', 'Root'])

    await classify_entity_subtypes(llm_client, [node], [_episode()], {}, entity_types)

    assert llm_client.generate_response.await_count == MAX_ENTITY_TYPE_SUBTYPE_DEPTH
    assert node.labels == ['Entity', 'Root', 'Level1', 'Level2', 'Level3', 'Level4']


def test_prompt_library_registers_subtype_prompt():
    messages = prompt_library.classify_entity_subtype.classify(
        {
            'entity_name': 'Rex',
            'chain': [{'name': 'Animal', 'description': 'An animal.'}],
            'candidates': [{'index': 1, 'name': 'Mammal', 'description': 'A mammal.'}],
            'episode_content': ['Rex barked.'],
            'custom_extraction_instructions': None,
        }
    )
    assert 'subtype_index JSON field' in messages[0].content
    assert 'Rex' in messages[1].content
    assert 'Mammal' in messages[1].content
    assert 'Rex barked.' in messages[1].content


def test_promote_resolved_node_merges_strict_subtype_and_keeps_divergent_labels():
    resolved = _node(['Entity', 'Animal'])
    extracted = _node(['Entity', 'Animal', 'Mammal'])
    assert _promote_resolved_node(extracted, resolved) is resolved
    assert resolved.labels == ['Entity', 'Animal', 'Mammal']

    resolved = _node(['Entity', 'Animal'])
    extracted = _node(['Entity', 'Person'])
    _promote_resolved_node(extracted, resolved)
    assert resolved.labels == ['Entity', 'Animal']


def test_collapse_duplicate_extracted_nodes_keeps_longer_type_chain():
    less_specific = _node(['Entity', 'Animal'])
    more_specific = _node(['Entity', 'Animal', 'Mammal'])

    collapsed = _collapse_exact_duplicate_extracted_nodes([less_specific, more_specific])

    assert collapsed == [more_specific]

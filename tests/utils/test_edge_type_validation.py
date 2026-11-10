"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Running tests: pytest -xvs tests/utils/test_edge_type_validation.py
#
# `add_episode` rejects entity types that declare a protected EntityNode field
# but accepted edge types that declare a protected EntityEdge field. Such a
# field is extracted from the LLM and then silently dropped on write, because
# `EntityEdge.save` and `add_nodes_and_edges_bulk` only copy an attribute when
# it does not collide with a real edge field.

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeTypeValidationError, EntityTypeValidationError
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_edge_types


class Authored(BaseModel):
    """A person authored a work."""

    fact: str | None = Field(default=None, description='collides with EntityEdge.fact')


class AuthoredWithValidAt(BaseModel):
    """A person authored a work."""

    valid_at: str | None = None


class AuthoredOk(BaseModel):
    """A person authored a work."""

    role: str | None = None


class PersonWithUuid(BaseModel):
    """A person."""

    uuid: str | None = None


class PersonOk(BaseModel):
    """A person."""

    occupation: str | None = None


def graphiti_without_driver() -> Graphiti:
    return Graphiti.__new__(Graphiti)


def test_validate_edge_types_rejects_protected_field_names():
    with pytest.raises(EdgeTypeValidationError, match='fact cannot be used as an attribute'):
        validate_edge_types({'Authored': Authored})


def test_validate_edge_types_rejects_every_entity_edge_field():
    for field_name in EntityEdge.model_fields:
        model = type('EdgeType', (BaseModel,), {'__annotations__': {field_name: str | None}})
        with pytest.raises(EdgeTypeValidationError):
            validate_edge_types({'EdgeType': model})


def test_validate_edge_types_accepts_custom_field_names():
    assert validate_edge_types({'Authored': AuthoredOk}) is True
    assert validate_edge_types(None) is True
    assert validate_edge_types({}) is True


@pytest.mark.asyncio
async def test_add_episode_rejects_edge_type_with_protected_field():
    graphiti = graphiti_without_driver()

    with pytest.raises(EdgeTypeValidationError, match='valid_at cannot be used as an attribute'):
        await graphiti.add_episode(
            name='episode',
            episode_body='Alice authored the report.',
            source_description='test',
            reference_time=utc_now(),
            source=EpisodeType.text,
            entity_types={'Person': PersonOk},
            edge_types={'Authored': AuthoredWithValidAt},
        )


@pytest.mark.asyncio
async def test_add_episode_still_validates_entity_types():
    graphiti = graphiti_without_driver()

    with pytest.raises(EntityTypeValidationError, match='uuid cannot be used as an attribute'):
        await graphiti.add_episode(
            name='episode',
            episode_body='Alice is a person.',
            source_description='test',
            reference_time=utc_now(),
            source=EpisodeType.text,
            entity_types={'Person': PersonWithUuid},
        )


@pytest.mark.asyncio
async def test_protected_edge_type_field_is_dropped_on_write():
    """Documents the data loss the validation above prevents."""
    captured: dict = {}

    async def execute_query(_query, **params):
        captured.update(params.get('edge_data') or {})
        return [], [], None

    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        graph_operations_interface=None,
        execute_query=execute_query,
    )
    edge = EntityEdge(
        source_node_uuid='source-uuid',
        target_node_uuid='target-uuid',
        name='Authored',
        fact='Alice authored the report.',
        group_id='group',
        created_at=utc_now(),
        attributes={'fact': 'value the LLM extracted', 'role': 'author'},
    )

    await edge.save(driver)

    assert captured['fact'] == 'Alice authored the report.'
    assert captured['role'] == 'author'
    assert 'value the LLM extracted' not in str(captured)

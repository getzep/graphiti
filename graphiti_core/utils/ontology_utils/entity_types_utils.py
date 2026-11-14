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

from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeTypeValidationError, EntityTypeValidationError
from graphiti_core.nodes import EntityNode


def validate_entity_types(
    entity_types: dict[str, type[BaseModel]] | None,
) -> bool:
    if entity_types is None:
        return True

    entity_node_field_names = EntityNode.model_fields.keys()

    for entity_type_name, entity_type_model in entity_types.items():
        entity_type_field_names = entity_type_model.model_fields.keys()
        for entity_type_field_name in entity_type_field_names:
            if entity_type_field_name in entity_node_field_names:
                raise EntityTypeValidationError(entity_type_name, entity_type_field_name)

    return True


def validate_edge_types(
    edge_types: dict[str, type[BaseModel]] | None,
) -> bool:
    """Reject custom edge types that declare a protected ``EntityEdge`` field.

    ``EntityEdge.save`` and ``add_nodes_and_edges_bulk`` only copy an attribute
    onto the relationship when it does not collide with a real edge field, so an
    edge type field named e.g. ``fact`` or ``valid_at`` would be extracted from
    the LLM and then silently dropped on write.
    """
    if edge_types is None:
        return True

    entity_edge_field_names = EntityEdge.model_fields.keys()

    for edge_type_name, edge_type_model in edge_types.items():
        edge_type_field_names = edge_type_model.model_fields.keys()
        for edge_type_field_name in edge_type_field_names:
            if edge_type_field_name in entity_edge_field_names:
                raise EdgeTypeValidationError(edge_type_name, edge_type_field_name)

    return True

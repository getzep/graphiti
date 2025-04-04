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

from graphiti_core.errors import EntityTypeValidationError
from graphiti_core.nodes import EntityNode


def validate_entity_types(
    entity_types: dict[str, BaseModel] | None,
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

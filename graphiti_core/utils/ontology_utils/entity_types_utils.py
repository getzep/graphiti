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
    entity_types: dict[str, type[BaseModel]] | None,
) -> bool:
    if entity_types is None:
        return True

    # Iterate through the provided entity types
    for entity_type_name, entity_type_model in entity_types.items():
        # Convert model fields to set for fast intersection
        entity_type_field_names = set(entity_type_model.model_fields.keys())
        # Intersect to find any clashing field
        conflict_fields = _ENTITY_NODE_FIELD_NAMES & entity_type_field_names
        if conflict_fields:
            # Only raise for the first conflict found, as per original behavior
            raise EntityTypeValidationError(entity_type_name, next(iter(conflict_fields)))

    return True  # Preserve existing comment


_ENTITY_NODE_FIELD_NAMES = set(EntityNode.model_fields.keys())

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from graph_service.ontologies.agent_memory_v1 import EDGE_TYPE_MAP as AGENT_MEMORY_V1_EDGE_TYPE_MAP
from graph_service.ontologies.agent_memory_v1 import EDGE_TYPES as AGENT_MEMORY_V1_EDGE_TYPES
from graph_service.ontologies.agent_memory_v1 import ENTITY_TYPES as AGENT_MEMORY_V1_ENTITY_TYPES
from graph_service.ontologies.agent_memory_v1 import SCHEMA_ID as AGENT_MEMORY_V1_SCHEMA_ID

SchemaId = Literal['agent_memory_v1']


@dataclass(frozen=True)
class Ontology:
    schema_id: SchemaId
    entity_types: dict[str, type[BaseModel]] | None = None
    excluded_entity_types: list[str] | None = None
    edge_types: dict[str, type[BaseModel]] | None = None
    edge_type_map: dict[tuple[str, str], list[str]] | None = None


_ONTOLOGIES: dict[SchemaId, Ontology] = {
    AGENT_MEMORY_V1_SCHEMA_ID: Ontology(
        schema_id=AGENT_MEMORY_V1_SCHEMA_ID,
        entity_types=dict(AGENT_MEMORY_V1_ENTITY_TYPES),
        edge_types=dict(AGENT_MEMORY_V1_EDGE_TYPES),
        edge_type_map=dict(AGENT_MEMORY_V1_EDGE_TYPE_MAP),
    ),
}


def resolve_ontology(schema_id: str | None, message_content: str) -> Ontology | None:
    if schema_id is not None:
        return _ONTOLOGIES.get(schema_id)  # type: ignore[arg-type]

    if '<graphiti_episode' in message_content:
        return _ONTOLOGIES[AGENT_MEMORY_V1_SCHEMA_ID]

    return None


def is_known_schema_id(schema_id: str) -> bool:
    return schema_id in _ONTOLOGIES  # type: ignore[operator]

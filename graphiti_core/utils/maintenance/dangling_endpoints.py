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

import logging

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact

logger = logging.getLogger(__name__)


def materialize_dangling_endpoint_node(
    name: str,
    *,
    group_id: str,
    name_to_node: dict[str, EntityNode],
    nodes: list[EntityNode],
) -> EntityNode:
    """Create an EntityNode for an edge endpoint missing from the provided nodes list.

    The LLM sometimes emits edges whose source/target is not also listed in
    extracted entities (especially value-like endpoints such as version strings).
    Materializing the missing endpoint keeps the fact instead of dropping it.
    """
    normalized = _normalize_string_exact(name)
    existing = name_to_node.get(normalized)
    if existing is not None:
        return existing

    logger.debug(
        'Materializing dangling edge endpoint node for name %r (group_id=%s)',
        name,
        group_id,
    )
    new_node = EntityNode(
        name=name.strip(),
        group_id=group_id,
        labels=['Entity'],
        summary='',
        created_at=utc_now(),
    )
    nodes.append(new_node)
    name_to_node[normalized] = new_node
    return new_node


def prune_unreferenced_materialized_nodes(
    nodes: list[EntityNode],
    *,
    initial_uuids: set[str],
    referenced_uuids: set[str],
) -> list[EntityNode]:
    """Drop materialized nodes that no kept edge references; return survivors."""
    survivors = [
        node for node in nodes if node.uuid in initial_uuids or node.uuid in referenced_uuids
    ]
    materialized = [node for node in survivors if node.uuid not in initial_uuids]
    if len(survivors) != len(nodes):
        nodes[:] = survivors
    return materialized

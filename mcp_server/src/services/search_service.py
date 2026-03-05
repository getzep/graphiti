"""Search adapter service for OM-lane retrieval from OM primitives."""

from __future__ import annotations

from typing import Any

from services.neo4j_service import Neo4jService

DEFAULT_OM_GROUP_ID = 's1_observational_memory'


def _provider_name(service: Any) -> str:
    try:
        return str(service.config.database.provider).lower()
    except Exception:
        return ''


def _allows_om_node_label(entity_types: list[str] | None) -> bool:
    # Keep search_nodes(entity_types=...) contract: OM adapter should return
    # no rows when OMNode label is not in scope.
    if not entity_types:
        return True

    normalized_labels = {
        str(label).strip().lower()
        for label in entity_types
        if str(label).strip()
    }
    return 'omnode' in normalized_labels


class SearchService:
    """Adapter that routes OM-lane search requests to OM primitives."""

    def __init__(
        self,
        *,
        om_group_id: str = DEFAULT_OM_GROUP_ID,
        neo4j_service: Neo4jService | None = None,
    ):
        self.om_group_id = om_group_id
        self.neo4j_service = neo4j_service or Neo4jService()

    def includes_observational_memory(self, group_ids: list[str]) -> bool:
        # Empty group scope means "all lanes" in the MCP server contract.
        return len(group_ids) == 0 or self.om_group_id in group_ids

    def _om_groups_in_scope(self, group_ids: list[str]) -> list[str]:
        # For all-lanes scope ([]), route to the canonical OM lane.
        if len(group_ids) == 0:
            return [self.om_group_id]
        return [group_id for group_id in group_ids if group_id == self.om_group_id]

    async def search_observational_nodes(
        self,
        *,
        graphiti_service: Any,
        query: str,
        group_ids: list[str],
        max_nodes: int,
        entity_types: list[str] | None,
    ) -> list[dict[str, Any]]:
        if not self.includes_observational_memory(group_ids):
            return []
        if not _allows_om_node_label(entity_types):
            return []
        if _provider_name(graphiti_service) != 'neo4j':
            return []

        client = await graphiti_service.get_client()
        nodes: list[dict[str, Any]] = []

        for group_id in self._om_groups_in_scope(group_ids):
            rows = await self.neo4j_service.search_om_nodes(
                client.driver,
                group_id=group_id,
                query=query,
                limit=max_nodes,
            )
            for row in rows:
                node_id = str(row.get('uuid') or '').strip()
                if not node_id:
                    continue
                content = str(row.get('content') or '').strip()
                created_at = row.get('created_at')
                nodes.append(
                    {
                        'uuid': node_id,
                        'name': (content[:120] if content else node_id),
                        'labels': ['OMNode'],
                        'created_at': str(created_at) if created_at is not None else None,
                        'summary': content or None,
                        'group_id': str(row.get('group_id') or group_id),
                        'attributes': {
                            'source': 'om_primitive',
                            'status': row.get('status'),
                            'semantic_domain': row.get('semantic_domain'),
                            'urgency_score': row.get('urgency_score'),
                            'lexical_score': row.get('lexical_score'),
                        },
                    }
                )
                if len(nodes) >= max_nodes:
                    return nodes
        return nodes

    async def search_observational_facts(
        self,
        *,
        graphiti_service: Any,
        query: str,
        group_ids: list[str],
        max_facts: int,
        center_node_uuid: str | None,
    ) -> list[dict[str, Any]]:
        if not self.includes_observational_memory(group_ids):
            return []
        if _provider_name(graphiti_service) != 'neo4j':
            return []

        client = await graphiti_service.get_client()
        facts: list[dict[str, Any]] = []

        for group_id in self._om_groups_in_scope(group_ids):
            rows = await self.neo4j_service.search_om_facts(
                client.driver,
                group_id=group_id,
                query=query,
                limit=max_facts,
                center_node_uuid=center_node_uuid,
            )
            for row in rows:
                relation_type = str(row.get('relation_type') or '').strip()
                source_node_id = str(row.get('source_node_id') or '').strip()
                target_node_id = str(row.get('target_node_id') or '').strip()
                fact_uuid = str(row.get('uuid') or '').strip()
                if not relation_type or not source_node_id or not target_node_id or not fact_uuid:
                    continue

                source_content = str(row.get('source_content') or '').strip()
                target_content = str(row.get('target_content') or '').strip()
                created_at = row.get('created_at')

                facts.append(
                    {
                        'uuid': fact_uuid,
                        'name': relation_type,
                        'fact': f'{relation_type}: {source_content} -> {target_content}',
                        'group_id': str(row.get('group_id') or group_id),
                        'source_node_uuid': source_node_id,
                        'target_node_uuid': target_node_id,
                        'created_at': str(created_at) if created_at is not None else None,
                        'valid_at': None,
                        'invalid_at': None,
                        'expired_at': None,
                        'episodes': [],
                        'attributes': {
                            'source': 'om_primitive',
                            'lexical_score': row.get('lexical_score'),
                            'source_content': source_content,
                            'target_content': target_content,
                        },
                    }
                )
                if len(facts) >= max_facts:
                    return facts
        return facts

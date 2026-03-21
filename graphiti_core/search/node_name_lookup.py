"""
Helpers for exact and contains node-name fallback lookups.
"""

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.helpers import validate_group_ids
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import EntityNode, get_entity_node_from_record
from graphiti_core.search.search_filters import (
    SearchFilters,
    node_search_filter_query_constructor,
)


async def search_nodes_by_name_fallback(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = 10,
) -> list[EntityNode]:
    """Try exact node-name matching first, then fall back to contains matching."""
    if query.strip() == '':
        return []

    if driver.provider == GraphProvider.KUZU:
        return []

    exact_matches = await _lookup_nodes_by_name(
        driver=driver,
        query=query,
        search_filter=search_filter,
        group_ids=group_ids,
        limit=limit,
        exact=True,
    )
    if exact_matches:
        return exact_matches[:limit]

    contains_matches = await _lookup_nodes_by_name(
        driver=driver,
        query=query,
        search_filter=search_filter,
        group_ids=group_ids,
        limit=limit,
        exact=False,
    )
    return contains_matches[:limit]


async def _lookup_nodes_by_name(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None,
    limit: int,
    *,
    exact: bool,
) -> list[EntityNode]:
    validate_group_ids(group_ids)

    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )
    if group_ids is not None:
        filter_queries.append('n.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    comparator = '=' if exact else 'CONTAINS'
    filter_queries.insert(0, f'toLower(n.name) {comparator} toLower($query)')
    where_clause = ' WHERE ' + ' AND '.join(filter_queries) if filter_queries else ''

    cypher = (
        'MATCH (n:Entity)'
        + where_clause
        + """
        RETURN
        """
        + get_entity_node_return_query(driver.provider)
        + """
        ORDER BY n.created_at DESC
        LIMIT $limit
        """
    )

    records, _, _ = await driver.execute_query(
        cypher,
        query=query,
        limit=limit,
        routing_='r',
        **filter_params,
    )
    return [get_entity_node_from_record(record, driver.provider) for record in records]

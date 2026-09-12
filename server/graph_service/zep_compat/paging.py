"""Offset-based paging over a graph's nodes and edges.

Why this exists instead of `EntityNode/EntityEdge.get_by_group_ids(uuid_cursor=...)`:

graphiti_core 0.29.3 pages with `AND n.uuid < $uuid` plus `ORDER BY n.uuid DESC`.
On FalkorDB that WHERE clause is silently not applied — verified against a live
FalkorDB 4.x, where even a literal `n.uuid < '0005-u'` returns every row while
`ORDER BY` and `SKIP`/`LIMIT` behave correctly. The result is that every page
comes back identical, so a drain never terminates and rows repeat.

MiroFish's `fetch_all_nodes` / `fetch_all_edges` treat `zep-next-cursor` as an
opaque string, so we are free to make it a row offset and page with SKIP/LIMIT,
which the same live test showed to be exact.

Caveat: offset paging is only stable if the result set does not change mid-drain.
MiroFish reads a graph after ingestion has reached a terminal batch status, so
that holds. Concurrent writes during a drain could shift rows across pages.
"""

from __future__ import annotations

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import EntityNode, get_entity_node_from_record


def parse_cursor(cursor: str | None) -> int:
    """Cursors are stringified row offsets; anything unparseable starts over."""
    if not cursor:
        return 0
    try:
        return max(0, int(str(cursor).strip()))
    except (TypeError, ValueError):
        return 0


async def page_nodes(
    driver: GraphDriver, group_id: str, limit: int, offset: int
) -> tuple[list[EntityNode], int | None]:
    """Return (nodes, next_offset). next_offset is None on the last page."""
    records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity)
        WHERE n.group_id IN $group_ids
        RETURN
        """
        + get_entity_node_return_query(driver.provider)
        + """
        ORDER BY n.uuid DESC
        SKIP $skip
        LIMIT $limit
        """,
        group_ids=[group_id],
        skip=offset,
        # One extra row tells us whether another page exists without a count().
        limit=limit + 1,
        routing_='r',
    )
    nodes = [get_entity_node_from_record(record, driver.provider) for record in records]
    has_more = len(nodes) > limit
    return nodes[:limit], (offset + limit if has_more else None)


async def page_edges(
    driver: GraphDriver, group_id: str, limit: int, offset: int
) -> tuple[list[EntityEdge], int | None]:
    """Return (edges, next_offset). next_offset is None on the last page."""
    match_query = """
        MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
    """
    if driver.provider == GraphProvider.KUZU:
        match_query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
        """

    records, _, _ = await driver.execute_query(
        match_query
        + """
        WHERE e.group_id IN $group_ids
        RETURN
        """
        + get_entity_edge_return_query(driver.provider)
        + """
        ORDER BY e.uuid DESC
        SKIP $skip
        LIMIT $limit
        """,
        group_ids=[group_id],
        skip=offset,
        limit=limit + 1,
        routing_='r',
    )
    edges = [get_entity_edge_from_record(record, driver.provider) for record in records]
    has_more = len(edges) > limit
    return edges[:limit], (offset + limit if has_more else None)

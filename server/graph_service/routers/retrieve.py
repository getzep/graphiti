from datetime import datetime, timezone

from fastapi import APIRouter, status
from graphiti_core.driver.driver import GraphProvider  # type: ignore
from graphiti_core.helpers import parse_db_date  # type: ignore
from graphiti_core.nodes import EpisodicNode  # type: ignore

from graph_service.dto import (
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    import logging
    from graph_service.zep_graphiti import get_graphiti_for_group
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Search request received - query: '{query.query}', group_ids: {query.group_ids}, max_facts: {query.max_facts}")
    
    # For multi-tenant FalkorDB, we need to search each organization's graph separately
    # For now, if multiple group_ids are provided, we'll search the first one
    # TODO: Support searching across multiple organizations by combining results
    if query.group_ids and len(query.group_ids) > 0:
        # Use the first group_id as the organization graph
        primary_group_id = query.group_ids[0]
        org_graphiti = get_graphiti_for_group(primary_group_id, graphiti)
        logger.info(f"Using database: {getattr(org_graphiti.driver, '_database', 'unknown')} for search")
        
        # Debug: Check if episodes exist for the group_ids
        for group_id in query.group_ids:
            try:
                # Use organization-specific client for each group_id
                group_graphiti = get_graphiti_for_group(group_id, graphiti)
                episodes = await group_graphiti.retrieve_episodes(
                    group_ids=[group_id],
                    last_n=5,
                    reference_time=datetime.now(timezone.utc)
                )
                episode_names = [ep.name for ep in episodes] if episodes else []
                logger.info(f"Found {len(episodes)} episodes in group_ids=['{group_id}']. Episode names: {episode_names}")
                if episodes:
                    logger.debug(f"Sample episode for {group_id}: uuid={episodes[0].uuid}, name={episodes[0].name}, group_id={episodes[0].group_id}")
                else:
                    logger.warning(f"No episodes found for group_id: {group_id}. This means no data has been indexed yet for this organization.")
            except Exception as ex:
                logger.warning(f"Error checking episodes for group_id {group_id}: {ex}", exc_info=True)
        
        # Search using the primary organization's graph
        relevant_edges = await org_graphiti.search(
            group_ids=query.group_ids,
            query=query.query,
            num_results=query.max_facts,
        )
    else:
        # No group_ids specified, use base client
        relevant_edges = await graphiti.search(
            group_ids=query.group_ids,
            query=query.query,
            num_results=query.max_facts,
        )
    
    logger.info(f"Search returned {len(relevant_edges)} edges")
    if relevant_edges:
        logger.debug(f"First edge sample: uuid={relevant_edges[0].uuid}, name={relevant_edges[0].name}, fact={relevant_edges[0].fact[:100] if relevant_edges[0].fact else 'N/A'}")
    else:
        logger.warning(f"No edges found for query: '{query.query}' with group_ids: {query.group_ids}")
    
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    
    logger.info(f"Returning {len(facts)} facts")
    return SearchResults(
        facts=facts,
    )


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    import logging
    from graph_service.zep_graphiti import get_graphiti_for_group
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Get episodes request - group_id: {group_id}, last_n: {last_n}")
    
    # Get organization-specific Graphiti client (uses group_id as database name for FalkorDB)
    org_graphiti = get_graphiti_for_group(group_id, graphiti)
    
    episodes = await org_graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    logger.info(f"Retrieved {len(episodes)} episodes for group_id: {group_id}")
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


@router.get('/debug/episodes-all', status_code=status.HTTP_200_OK)
async def debug_episodes_all(graphiti: ZepGraphitiDep):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Query all node types to see what's in the database
        from graphiti_core.nodes import EpisodicNode, EntityNode
        from graphiti_core.edges import EntityEdge
        from datetime import datetime, timezone
        
        # Get all episodes
        episode_records, _, _ = await graphiti.driver.execute_query(
            """
            MATCH (e:Episodic)
            RETURN e.uuid, e.name, e.group_id, e.source, e.valid_at
            ORDER BY e.valid_at DESC
            LIMIT 50
            """,
            routing_='r',
        )
        
        # Get all entity nodes
        entity_records, _, _ = await graphiti.driver.execute_query(
            """
            MATCH (e:Entity)
            RETURN e.uuid, e.name, e.group_id
            LIMIT 20
            """,
            routing_='r',
        )
        
        # Get all edges
        edge_records, _, _ = await graphiti.driver.execute_query(
            """
            MATCH ()-[r:RELATES_TO]->()
            RETURN r.uuid, r.name, r.fact
            LIMIT 20
            """,
            routing_='r',
        )
        
        # Count all nodes
        count_result, _, _ = await graphiti.driver.execute_query(
            """
            MATCH (n)
            RETURN labels(n) as labels, count(*) as count
            """,
            routing_='r',
        )
        
        episodes_info = []
        group_ids_found = set()
        for record in episode_records:
            episode_info = {
                'uuid': record.get('e.uuid'),
                'name': record.get('e.name'),
                'group_id': record.get('e.group_id'),
                'source': record.get('e.source'),
                'valid_at': str(record.get('e.valid_at')) if record.get('e.valid_at') else None,
            }
            episodes_info.append(episode_info)
            if record.get('e.group_id'):
                group_ids_found.add(record.get('e.group_id'))
        
        entities_info = []
        for record in entity_records:
            entities_info.append({
                'uuid': record.get('e.uuid'),
                'name': record.get('e.name'),
                'group_id': record.get('e.group_id'),
            })
        
        edges_info = []
        for record in edge_records:
            edges_info.append({
                'uuid': record.get('r.uuid'),
                'name': record.get('r.name'),
                'fact': record.get('r.fact')[:100] if record.get('r.fact') else None,
            })
        
        node_counts = {}
        for record in count_result:
            labels = record.get('labels', [])
            count = record.get('count', 0)
            label_str = ','.join(labels) if labels else 'NoLabel'
            node_counts[label_str] = count
        
        logger.info(f"Debug: Found {len(episodes_info)} episodes, {len(entities_info)} entities, {len(edges_info)} edges. Node counts: {node_counts}")
        
        return {
            'total_episodes': len(episodes_info),
            'total_entities': len(entities_info),
            'total_edges': len(edges_info),
            'group_ids_found': list(group_ids_found),
            'node_counts': node_counts,
            'episodes': episodes_info[:10],
            'entities': entities_info[:10],
            'edges': edges_info[:10],
        }
    except Exception as ex:
        logger.error(f"Error in debug endpoint: {ex}", exc_info=True)
        return {'error': str(ex)}


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query

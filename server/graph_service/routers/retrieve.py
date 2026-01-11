import json
import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, status
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_from_record,
    get_entity_edge_return_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN,
    EPISODIC_NODE_RETURN_NEPTUNE,
    get_episodic_node_from_record,
)
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_utils import calculate_cosine_similarity

from graph_service.dto import (
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
)
from graph_service.zep_graphiti import (
    ZepGraphitiDep,
    get_fact_result_from_edge,
    get_graphiti_for_group,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def build_metadata_filter_conditions(
    meeting_ids: list[str] | None,
    meeting_type_ids: list[str] | None,
    user_ids: list[str] | None,
    provider: GraphProvider,
) -> list[str]:
    """
    Build database filter conditions for meeting_ids, meeting_type_ids, and user_ids.
    These are stored as JSON in source_description, so we use string matching.
    
    For user_ids: matches if user is owner_id OR in direct_access_user_ids array.
    
    Returns list of filter condition strings that can be joined with OR.
    """
    filter_conditions = []
    
    if meeting_ids:
        for meeting_id in meeting_ids:
            # Escape quotes for SQL safety
            escaped_id = meeting_id.replace("'", "''")
            # Match JSON pattern: "meeting_id":"{meeting_id}"
            filter_conditions.append(
                f"e.source_description CONTAINS '\"meeting_id\":\"{escaped_id}\"'"
            )
    
    if meeting_type_ids:
        for meeting_type_id in meeting_type_ids:
            # Escape quotes for SQL safety
            escaped_id = meeting_type_id.replace("'", "''")
            # Match JSON pattern: "meeting_type_id":"{meeting_type_id}"
            filter_conditions.append(
                f"e.source_description CONTAINS '\"meeting_type_id\":\"{escaped_id}\"'"
            )
    
    if user_ids:
        for user_id in user_ids:
            # Escape quotes for SQL safety
            escaped_id = user_id.replace("'", "''")
            # Match if user is owner: "owner_id":"{user_id}"
            filter_conditions.append(
                f"e.source_description CONTAINS '\"owner_id\":\"{escaped_id}\"'"
            )
            # Match if user is in direct_access_user_ids array
            # JSON format: "direct_access_user_ids":["user1","user2","user3"]
            # Check for user_id at start of array: ["user_id"
            filter_conditions.append(
                f"e.source_description CONTAINS '\"direct_access_user_ids\":[\"{escaped_id}\"'"
            )
            # Check for user_id in middle: ,"user_id"
            filter_conditions.append(
                f"e.source_description CONTAINS ',\"{escaped_id}\"'"
            )
            # Check for user_id at end: "user_id"]
            filter_conditions.append(
                f"e.source_description CONTAINS '\"{escaped_id}\"]'"
            )
    
    return filter_conditions


def build_episode_uuid_filter_conditions(
    episode_uuids: list[str],
    provider: GraphProvider,
    property_name: str = "e.episodes",
) -> list[str]:
    """
    Build database filter conditions for episode UUIDs in edges.
    Episodes are stored as comma-separated string: "uuid1,uuid2,uuid3"
    
    Returns list of filter condition strings that can be joined with OR.
    """
    filter_conditions = []
    
    # Limit to avoid query size issues
    for ep_uuid in episode_uuids[:100]:
        # Escape quotes for SQL safety
        escaped_uuid = ep_uuid.replace("'", "''")
        
        if provider == GraphProvider.FALKORDB:
            # FalkorDB: use CONTAINS for string matching
            filter_conditions.append(f"{property_name} CONTAINS '{escaped_uuid}'")
        else:
            # Neo4j/Kuzu: check if episode UUID is in the episodes string/array
            filter_conditions.append(f"'{escaped_uuid}' IN split({property_name}, ',')")
    
    return filter_conditions


def extract_episode_metadata(episode: EpisodicNode) -> dict[str, str | None | list[str]]:
    """
    Extract meeting metadata from episode source_description.
    Returns dict with meeting_id, meeting_type_id, owner_id, and direct_access_user_ids.
    """
    metadata_match = re.search(
        r'METADATA:\s*({.*?})(?:\s*$|\s*\|)', 
        episode.source_description or ''
    )
    
    if not metadata_match:
        return {
            'meeting_id': None,
            'meeting_type_id': None,
            'owner_id': None,
            'direct_access_user_ids': [],
        }
    
    try:
        metadata = json.loads(metadata_match.group(1))
        direct_access = metadata.get('direct_access_user_ids', [])
        # Ensure it's a list
        if not isinstance(direct_access, list):
            direct_access = []
        
        return {
            'meeting_id': metadata.get('meeting_id'),
            'meeting_type_id': metadata.get('meeting_type_id'),
            'owner_id': metadata.get('owner_id'),
            'direct_access_user_ids': direct_access,
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            'meeting_id': None,
            'meeting_type_id': None,
            'owner_id': None,
            'direct_access_user_ids': [],
        }


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):

    logger.info(
        f"Search request received - query: '{query.query}', group_ids: {query.group_ids}, max_facts: {query.max_facts}, meeting_ids: {query.meeting_ids}, meeting_type_ids: {query.meeting_type_ids}, user_ids: {query.user_ids}"
    )

    # Filter episodes by meeting_ids and/or meeting_type_ids if provided
    filtered_episode_uuids = None
    
    # For multi-tenant FalkorDB, we need to search each organization's graph separately
    # For now, if multiple group_ids are provided, we'll search the first one
    # TODO: Support searching across multiple organizations by combining results
    if query.group_ids and len(query.group_ids) > 0:
        # Use the first group_id as the organization graph
        primary_group_id = query.group_ids[0]
        org_graphiti = get_graphiti_for_group(primary_group_id, graphiti)
        logger.info(
            f'Using database: {getattr(org_graphiti.driver, "_database", "unknown")} for search'
        )

        # Filter episodes by meeting_ids, meeting_type_ids, and/or user_ids if provided
        # OPTIMIZED: Filter at database level using string matching before fetching
        # This happens BEFORE the expensive graph/vector search to save operations
        # Note: meeting_id/meeting_type_id/owner_id/direct_access_user_ids are stored in JSON within source_description,
        # so we can't filter them the same way as group_id (which is a direct property)
        if query.meeting_ids or query.meeting_type_ids or query.user_ids:
            try:
                # Build filter conditions using DRY helper function
                episode_filter_conditions = build_metadata_filter_conditions(
                    query.meeting_ids,
                    query.meeting_type_ids,
                    query.user_ids,
                    org_graphiti.driver.provider,
                )
                
                # Query episodes with filters applied at database level
                if episode_filter_conditions:
                    # Use OR logic for database pre-filtering (approximate match)
                    # Exact AND logic is enforced in Python-level filtering below
                    filter_clause = ' OR '.join(episode_filter_conditions)
                    
                    records, _, _ = await org_graphiti.driver.execute_query(
                        f"""
                        MATCH (e:Episodic)
                        WHERE e.group_id IN $group_ids
                        AND ({filter_clause})
                        RETURN DISTINCT
                        """
                        + (
                            EPISODIC_NODE_RETURN_NEPTUNE
                            if org_graphiti.driver.provider == GraphProvider.NEPTUNE
                            else EPISODIC_NODE_RETURN
                        )
                        + """
                        LIMIT 10000
                        """,
                        group_ids=query.group_ids,
                        routing_='r',
                    )
                    
                    matching_episodes = [
                        get_episodic_node_from_record(record) for record in records
                    ]
                    
                    # Now filter in Python to ensure exact matches (database filter is approximate)
                    # This is necessary because meeting_id/meeting_type_id/owner_id/direct_access_user_ids are in JSON, not direct properties
                    filtered_episode_uuids = []
                    for episode in matching_episodes:
                        metadata = extract_episode_metadata(episode)
                        
                        # Check exact matches
                        matches_meeting_id = (
                            not query.meeting_ids or 
                            metadata['meeting_id'] in query.meeting_ids
                        )
                        matches_meeting_type_id = (
                            not query.meeting_type_ids or 
                            metadata['meeting_type_id'] in query.meeting_type_ids
                        )
                        # Check if user is owner OR in direct_access_user_ids
                        matches_user_id = True
                        if query.user_ids:
                            user_id_set = set(query.user_ids)
                            is_owner = metadata['owner_id'] in user_id_set
                            has_direct_access = any(
                                user_id in user_id_set
                                for user_id in metadata.get('direct_access_user_ids', [])
                            )
                            matches_user_id = is_owner or has_direct_access
                        
                        if matches_meeting_id and matches_meeting_type_id and matches_user_id:
                            filtered_episode_uuids.append(episode.uuid)
                    
                    if filtered_episode_uuids:
                        logger.info(
                            f'Filtered to {len(filtered_episode_uuids)} episodes matching meeting_ids={query.meeting_ids}, meeting_type_ids={query.meeting_type_ids}, user_ids={query.user_ids} (from {len(matching_episodes)} database matches)'
                        )
                    else:
                        logger.warning(
                            f'No episodes found matching meeting_ids={query.meeting_ids}, meeting_type_ids={query.meeting_type_ids}, user_ids={query.user_ids}'
                        )
                        # Return empty results if no matching episodes - saves expensive search operations
                        return SearchResults(facts=[])
                else:
                    # No filters provided, skip filtering
                    filtered_episode_uuids = None
            except Exception as ex:
                logger.warning(
                    f'Error filtering episodes by meeting_ids/meeting_type_ids: {ex}', 
                    exc_info=True
                )
                # Continue with search if filtering fails
                filtered_episode_uuids = None

        # Debug code removed for performance optimization

        # If we have filtered episode UUIDs, we need to constrain the search
        # OPTIMIZED: Filter edges at database level by episode UUIDs to avoid expensive operations
        if filtered_episode_uuids:
            # Query edges directly from database that reference our filtered episodes
            # This is much faster than searching the entire graph
            try:
                episode_uuid_set = set(filtered_episode_uuids)
                
                # Build query to get edges that reference our episodes
                # OPTIMIZED: Filter by episode UUIDs in database query
                match_query = """
                    MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
                """
                if org_graphiti.driver.provider == GraphProvider.KUZU:
                    match_query = """
                        MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
                    """
                
                # Build filter conditions using DRY helper function
                episode_filter_conditions = build_episode_uuid_filter_conditions(
                    filtered_episode_uuids,
                    org_graphiti.driver.provider,
                    property_name="e.episodes",
                )
                
                if episode_filter_conditions:
                    episode_filter = ' OR '.join(episode_filter_conditions)
                    
                    # Query edges filtered by episode UUIDs at database level
                    records, _, _ = await org_graphiti.driver.execute_query(
                        match_query
                        + f"""
                        WHERE e.group_id IN $group_ids
                        AND ({episode_filter})
                        RETURN
                        """
                        + get_entity_edge_return_query(org_graphiti.driver.provider)
                        + """
                        LIMIT $limit
                        """,
                        group_ids=query.group_ids,
                        limit=query.max_facts * 20,  # Get enough candidates for semantic search
                        routing_='r',
                    )
                    
                    candidate_edges = [
                        get_entity_edge_from_record(record, org_graphiti.driver.provider) 
                        for record in records
                    ]
                    
                    # Additional Python-level filtering to ensure exact matches
                    # (database filter might match partial UUIDs)
                    filtered_candidate_edges = []
                    for edge in candidate_edges:
                        if edge.episodes:
                            edge_episodes = edge.episodes if isinstance(edge.episodes, list) else str(edge.episodes).split(',')
                            if any(ep.strip() in episode_uuid_set for ep in edge_episodes):
                                filtered_candidate_edges.append(edge)
                    candidate_edges = filtered_candidate_edges
                else:
                    # Fallback if too many episode UUIDs (shouldn't happen with limit)
                    candidate_edges = []
                
                logger.info(
                    f'Found {len(candidate_edges)} candidate edges from {len(filtered_episode_uuids)} filtered episodes (database-filtered)'
                )
                
                if not candidate_edges:
                    return SearchResults(facts=[])
                
                # Now do semantic search ONLY on these filtered candidate edges
                # Get query embedding
                query_vector = await org_graphiti.embedder.embed(query.query)
                
                # Calculate similarity scores for candidate edges
                scored_edges = []
                for edge in candidate_edges:
                    try:
                        # Load embedding if needed
                        if not hasattr(edge, 'fact_embedding') or not edge.fact_embedding:
                            # Edge might need embedding loaded
                            continue
                        
                        edge_embedding = edge.fact_embedding
                        if isinstance(edge_embedding, str):
                            edge_embedding = [float(x.strip()) for x in edge_embedding.split(',')]
                        
                        if len(edge_embedding) == len(query_vector):
                            score = calculate_cosine_similarity(query_vector, edge_embedding)
                            if score > 0.3:  # Minimum similarity threshold
                                scored_edges.append((edge, score))
                    except Exception as ex:
                        logger.debug(f'Error calculating similarity for edge {edge.uuid}: {ex}')
                        continue
                
                # Sort by score and take top results
                scored_edges.sort(key=lambda x: x[1], reverse=True)
                relevant_edges = [edge for edge, _ in scored_edges[:query.max_facts]]
                
                logger.info(
                    f'Semantic search on filtered edges returned {len(relevant_edges)} relevant edges (from {len(candidate_edges)} candidates)'
                )
            except Exception as ex:
                logger.error(
                    f'Error doing optimized search with episode filtering: {ex}', 
                    exc_info=True
                )
                # Fallback: regular search then filter (less efficient but works)
                relevant_edges = await org_graphiti.search(
                    group_ids=query.group_ids,
                    query=query.query,
                    num_results=query.max_facts * 2,
                )
                episode_uuid_set = set(filtered_episode_uuids)
                filtered_edges = []
                for edge in relevant_edges:
                    if edge.episodes:
                        edge_episodes = edge.episodes if isinstance(edge.episodes, list) else str(edge.episodes).split(',')
                        if any(ep.strip() in episode_uuid_set for ep in edge_episodes):
                            filtered_edges.append(edge)
                relevant_edges = filtered_edges[:query.max_facts]
        else:
            # No filtering needed, do regular search
            relevant_edges = await org_graphiti.search(
                group_ids=query.group_ids,
                query=query.query,
                num_results=query.max_facts,
            )
    else:
        # No group_ids specified, use base client
        # Note: Filtering by meeting_ids/meeting_type_ids/user_ids requires group_ids
        if query.meeting_ids or query.meeting_type_ids or query.user_ids:
            logger.warning(
                'meeting_ids, meeting_type_ids, and user_ids filtering requires group_ids to be specified'
            )
        relevant_edges = await graphiti.search(
            group_ids=query.group_ids,
            query=query.query,
            num_results=query.max_facts,
        )

    # Note: Episode filtering is now done BEFORE search (above) to save operations
    # This post-filtering is only needed if the optimized path above failed

    logger.info(f'Search returned {len(relevant_edges)} edges')
    if relevant_edges:
        logger.debug(
            f'First edge sample: uuid={relevant_edges[0].uuid}, name={relevant_edges[0].name}, fact={relevant_edges[0].fact[:100] if relevant_edges[0].fact else "N/A"}'
        )
        if relevant_edges[0].episodes:
            logger.debug(
                f'First edge has {len(relevant_edges[0].episodes)} source episodes: {relevant_edges[0].episodes[:3]}'
            )
    else:
        logger.warning(
            f"No edges found for query: '{query.query}' with group_ids: {query.group_ids}"
        )

    # Convert edges to facts with source episode information
    facts = []
    graphiti_client = org_graphiti if query.group_ids and len(query.group_ids) > 0 else graphiti
    for edge in relevant_edges:
        fact = await get_fact_result_from_edge(edge, graphiti_client)
        facts.append(fact)

    logger.info(f'Returning {len(facts)} facts with source information')
    return SearchResults(
        facts=facts,
    )


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return await get_fact_result_from_edge(entity_edge, graphiti)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):

    logger.info(f'Get episodes request - group_id: {group_id}, last_n: {last_n}')

    # Get organization-specific Graphiti client (uses group_id as database name for FalkorDB)
    org_graphiti = get_graphiti_for_group(group_id, graphiti)

    episodes = await org_graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    logger.info(f'Retrieved {len(episodes)} episodes for group_id: {group_id}')
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    # Use organization-specific graph
    org_graphiti = get_graphiti_for_group(request.group_id, graphiti)

    combined_query = compose_query_from_messages(request.messages)
    result = await org_graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [await get_fact_result_from_edge(edge, org_graphiti) for edge in result]
    return GetMemoryResponse(facts=facts)


@router.get('/debug/episodes-all', status_code=status.HTTP_200_OK)
async def debug_episodes_all(graphiti: ZepGraphitiDep):

    try:
        # Query all node types to see what's in the database

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
            entities_info.append(
                {
                    'uuid': record.get('e.uuid'),
                    'name': record.get('e.name'),
                    'group_id': record.get('e.group_id'),
                }
            )

        edges_info = []
        for record in edge_records:
            edges_info.append(
                {
                    'uuid': record.get('r.uuid'),
                    'name': record.get('r.name'),
                    'fact': record.get('r.fact')[:100] if record.get('r.fact') else None,
                }
            )

        node_counts = {}
        for record in count_result:
            labels = record.get('labels', [])
            count = record.get('count', 0)
            label_str = ','.join(labels) if labels else 'NoLabel'
            node_counts[label_str] = count

        logger.info(
            f'Debug: Found {len(episodes_info)} episodes, {len(entities_info)} entities, {len(edges_info)} edges. Node counts: {node_counts}'
        )

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
        logger.error(f'Error in debug endpoint: {ex}', exc_info=True)
        return {'error': str(ex)}


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
